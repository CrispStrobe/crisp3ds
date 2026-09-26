#!/usr/bin/env python3
"""Bounded foreground-centre training ablation with a frozen heldout set."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colmap_sparse.heldout_v2 import source_camera, training_matches
from scripts.foreground_roi.polygon import in_roi, load_contract
from scripts.sparse_verify import verify as checker

SOURCE = ROOT / "build-opencv/colmap-sparse/heldout-v2-001"
IMAGE_DIR = ROOT / ".local-tools/colmap-sparse/images"
DEFAULT_OUTPUT = ROOT / "build-opencv/colmap-sparse/foreground-001"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_source(manifest, source_db):
    if (manifest.get("schema") != "colmap_sparse_holdout_v3_spatial_components" or
            len(manifest["images"]) != 10 or
            sum(len(i["features"]) for i in manifest["images"]) != 53546 or
            sum(len(i["heldout_ids"]) for i in manifest["images"]) != 10664 or
            len(manifest["heldout_pairs"]) != 1121):
        raise ValueError("unexpected frozen holdout source population")
    with sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True) as conn:
        camera = source_camera(conn)
        rows = conn.execute("SELECT image_id,name,camera_id FROM images").fetchall()
        if {(i,n,c) for i,n,c in rows} != {(r["image_id"],r["name"],r["camera_id"]) for r in manifest["images"]}:
            raise ValueError("source database images differ from frozen manifest")
        if camera != manifest["camera_initial"]:
            raise ValueError("source camera differs from frozen manifest")
        if (conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] != 45 or
                conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] != 45):
            raise ValueError("unexpected source cached pair tables")
    for im in manifest["images"]:
        if Path(im["path"]).resolve() != (IMAGE_DIR / im["name"]).resolve() or sha256(im["path"]) != im["sha256"]:
            raise ValueError(f"source image differs: {im['name']}")
    return camera


def verify_frozen_hashes(manifest_path, source_db):
    report = json.loads((SOURCE / "verification-supervisor.json").read_text())
    if report.get("status") != "pass" or not report.get("heldout_valid"):
        raise ValueError("frozen holdout verification did not pass")
    hashes = report["hashes"]
    if sha256(manifest_path) != hashes["manifest"] or sha256(source_db) != report["database"]["sha256"]:
        raise ValueError("frozen source manifest or database hash differs")
    if sha256(checker.__file__) != hashes["verifier"]:
        raise ValueError("frozen verifier source hash differs")
    for name, digest in hashes["model"].items():
        if sha256(SOURCE / "model_text" / name) != digest:
            raise ValueError(f"frozen model hash differs: {name}")
    for path, digest in hashes["inputs"].items():
        if sha256(path) != digest:
            raise ValueError(f"frozen input hash differs: {path}")
    wal = SOURCE / "database.db-wal"
    if wal.exists() and wal.stat().st_size != 0:
        raise ValueError("source database has uncheckpointed WAL contents")
    return report


def object_subset(manifest, polygons):
    by_name = {im["name"]: im for im in manifest["images"]}
    indices = []
    for index, row in enumerate(manifest["heldout_pairs"]):
        if all(in_roi(polygons[row[f"image{side}"]],
                      by_name[row[f"image{side}"]]["features"][row[f"feature{side}"]])
               for side in (1, 2)):
            indices.append(index)
    return indices


def filter_database(conn, manifest, polygons):
    """Clear stale pair evidence and compact only nonheldout training rows."""
    import numpy as np
    source_camera(conn)
    old_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    old_geometry = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    conn.execute("DELETE FROM matches")
    conn.execute("DELETE FROM two_view_geometries")
    records = []
    for item in manifest["images"]:
        im = copy.deepcopy(item)
        iid, name = im["image_id"], im["name"]
        krow = conn.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (iid,)).fetchone()
        drow = conn.execute("SELECT rows,cols,data FROM descriptors WHERE image_id=?", (iid,)).fetchone()
        if krow is None or drow is None or krow[1] != 6 or drow[1] != 128 or krow[0] != drow[0] or krow[0] != len(im["compact_to_original"]):
            raise ValueError(f"source feature table shape differs: {name}")
        keys = np.frombuffer(krow[2], dtype=np.float32).reshape(krow[0], 6)
        desc = np.frombuffer(drow[2], dtype=np.uint8).reshape(drow[0], 128)
        held = set(im["heldout_ids"])
        keep = []
        for compact, original in enumerate(im["compact_to_original"]):
            if original in held or np.linalg.norm(keys[compact]-im["features"][original]) > 1e-4:
                raise ValueError(f"source compact mapping differs: {name}:{compact}")
            if in_roi(polygons[name], im["features"][original]):
                keep.append(compact)
        im["compact_to_original"] = [im["compact_to_original"][j] for j in keep]
        compact_keys, compact_desc = keys[keep], desc[keep]
        conn.execute("UPDATE keypoints SET rows=?,cols=6,data=? WHERE image_id=?",
                     (len(keep), sqlite3.Binary(compact_keys.tobytes()), iid))
        conn.execute("UPDATE descriptors SET rows=?,cols=128,data=? WHERE image_id=?",
                     (len(keep), sqlite3.Binary(compact_desc.tobytes()), iid))
        records.append(im)
    conn.commit()
    if (conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] or
            conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]):
        raise ValueError("stale pair tables survived filtering")
    return records, {"matches_deleted": old_matches, "geometries_deleted": old_geometry,
                     "training_rows_before": sum(len(i["compact_to_original"]) for i in manifest["images"]),
                     "training_rows_after": sum(len(i["compact_to_original"]) for i in records)}


def score_pairs(manifest, subset_indices, model_dir):
    if not (model_dir / "cameras.txt").exists():
        return {"total_pairs": len(subset_indices), "scored_pairs": 0,
                "finite_pairs": 0, "unscored_pairs": len(subset_indices),
                "greater_than_4px_or_missing": len(subset_indices), "stats": checker.stats([])}
    cameras = checker.cameras(model_dir / "cameras.txt")
    images = {im["name"]: im for im in checker.images(model_dir / "images.txt").values()}
    features = {im["name"]: im["features"] for im in manifest["images"]}
    values = []
    for index in subset_indices:
        row = manifest["heldout_pairs"][index]
        na, nb = row["image1"], row["image2"]
        if na in images and nb in images:
            a, b = images[na], images[nb]
            values.append(checker.pair_sampson(a, b, cameras[a["camera_id"]], cameras[b["camera_id"]],
                                               features[na][row["feature1"]], features[nb][row["feature2"]]))
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {"total_pairs": len(subset_indices), "scored_pairs": len(values),
            "finite_pairs": len(finite), "unscored_pairs": len(subset_indices)-len(values),
            "greater_than_4px_or_missing": len(subset_indices)-sum(v <= 4 for v in finite),
            "stats": checker.stats(values)}


def worker(out, roi_path, approved_hash):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pycolmap
    import numpy as np
    if sha256(roi_path) != approved_hash:
        raise ValueError("ROI bytes differ from approved SHA-256")
    source_manifest_path = SOURCE / "holdout.json"
    source_db = SOURCE / "database.db"
    frozen_report = verify_frozen_hashes(source_manifest_path, source_db)
    source_manifest = json.loads(source_manifest_path.read_text())
    camera = validate_frozen_source(source_manifest, source_db)
    expected_images = {i["name"]: {"width": i["width"], "height": i["height"],
                                   "sha256": i["sha256"]} for i in source_manifest["images"]}
    polygons = load_contract(roi_path, expected_images)
    subset_indices = object_subset(source_manifest, polygons)
    eligibility = {im["name"]: {
        "training_original_ids_inside": [fid for fid in im["compact_to_original"]
                                         if in_roi(polygons[im["name"]], im["features"][fid])],
        "heldout_original_ids_inside": [fid for fid in im["heldout_ids"]
                                        if in_roi(polygons[im["name"]], im["features"][fid])]
        } for im in source_manifest["images"]}
    before = {"source_database": sha256(source_db), "source_manifest": sha256(source_manifest_path),
              "source_provenance": sha256(SOURCE / "provenance.json"),
              "source_verification": sha256(SOURCE / "verification-supervisor.json"),
              "source_model": {n: sha256(SOURCE / "model_text" / n) for n in frozen_report["hashes"]["model"]},
              "roi": sha256(roi_path), "runner": sha256(__file__),
              "heldout_helper": sha256(source_camera.__code__.co_filename),
              "roi_helper": sha256(load_contract.__code__.co_filename),
              "verifier_helper": sha256(checker.__file__),
              "pycolmap_binary": sha256(pycolmap._core.__file__),
              "images": {i["name"]: sha256(i["path"]) for i in source_manifest["images"]}}
    if len(subset_indices) == 0:
        raise ValueError("fixed ROI contains no heldout descriptor pairs")
    (out / "roi-eligibility.json").write_text(json.dumps({"roi_sha256": approved_hash,
        "by_image": eligibility,
        "object_pair_indices": subset_indices, "all_pair_count": len(source_manifest["heldout_pairs"]),
        "object_pair_count": len(subset_indices)}, indent=2) + "\n")
    source = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True)
    dest_path = out / "database.db"
    dest = sqlite3.connect(dest_path)
    source.backup(dest)
    source.close()
    records, preparation = filter_database(dest, source_manifest, polygons)
    dest.close()
    manifest = copy.deepcopy(source_manifest)
    manifest["schema"] = "colmap_sparse_foreground_v1_frozen_holdout"
    manifest["images"] = records
    manifest["training_matches"] = []
    manifest["training_pair_count"] = 0
    if (manifest["heldout_pairs"] != source_manifest["heldout_pairs"] or
            any(a["heldout_ids"] != b["heldout_ids"] or a["features"] != b["features"]
                for a,b in zip(manifest["images"], source_manifest["images"]))):
        raise ValueError("frozen heldout population changed")
    matching = pycolmap.SiftMatchingOptions(); matching.num_threads = 2
    exhaustive = pycolmap.ExhaustiveMatchingOptions()
    verification = pycolmap.TwoViewGeometryOptions()
    mapping = pycolmap.IncrementalPipelineOptions(); mapping.num_threads = 2; mapping.mapper.num_threads = 2
    source_options = json.loads((SOURCE / "provenance.json").read_text())["options"]
    options = {"sift_matching": matching.todict(), "exhaustive_matching": exhaustive.todict(),
               "two_view_geometry": verification.todict(), "incremental_pipeline": mapping.todict()}
    if json.loads(json.dumps(options, default=str)) != source_options:
        raise ValueError("foreground matching or mapper options differ from heldout source")
    provenance = {"schema": "foreground_training_centres_v1", "python": sys.version,
                  "platform": platform.platform(), "pycolmap": pycolmap.__version__,
                  "numpy": np.__version__, "camera_initial": camera,
                  "random_seed_before_matching": 0, "random_seed_before_mapping": 0,
                  "options": options, "preparation": preparation,
                  "input_hashes_before": before, "input_hashes_after": None}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    pycolmap.set_random_seed(0)
    pycolmap.match_exhaustive(str(dest_path), sift_options=matching,
                             matching_options=exhaustive,
                             verification_options=verification,
                             device=pycolmap.Device.cpu)
    with sqlite3.connect(f"file:{dest_path}?mode=ro", uri=True) as conn:
        manifest["training_matches"], manifest["training_pair_count"] = training_matches(conn, records)
    (out / "holdout.json").write_text(json.dumps(manifest, separators=(",", ":")))
    pycolmap.set_random_seed(0)
    models = pycolmap.incremental_mapping(str(dest_path), str(IMAGE_DIR), str(out / "models"), options=mapping)
    summary = {"model_count": len(models), "registered_images": 0, "points3D": 0,
               "image_count": 10, "training_rows_before": preparation["training_rows_before"],
               "training_rows_after": preparation["training_rows_after"],
               "heldout_pair_count": len(manifest["heldout_pairs"]),
               "object_subset_pair_count": len(subset_indices),
               "training_match_count": len(manifest["training_matches"]),
               "training_pair_count": manifest["training_pair_count"],
               "validation": "diagnostic only; independent verifier pending"}
    if models:
        model = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        summary["registered_images"] = model.num_reg_images()
        summary["points3D"] = model.num_points3D()
        text_dir = out / "model_text"; text_dir.mkdir()
        model.write_text(str(text_dir))
    all_indices = list(range(len(manifest["heldout_pairs"])))
    summary["scores"] = {
        "baseline_global": score_pairs(source_manifest, all_indices, SOURCE / "model_text"),
        "baseline_object_subset": score_pairs(source_manifest, subset_indices, SOURCE / "model_text"),
        "foreground_global": score_pairs(manifest, all_indices, out / "model_text"),
        "foreground_object_subset": score_pairs(manifest, subset_indices, out / "model_text")}
    frozen_global = frozen_report["heldout"]["all"]
    baseline_global = summary["scores"]["baseline_global"]
    if (baseline_global["total_pairs"] != frozen_report["heldout"]["total_pairs"] or
            baseline_global["scored_pairs"] != frozen_report["heldout"]["scored_pairs"] or
            abs(baseline_global["stats"]["median_px"] - frozen_global["median_px"]) > 1e-8 or
            abs(baseline_global["stats"]["p90_px"] - frozen_global["p90_px"]) > 1e-8):
        raise ValueError("baseline global heldout scoring differs from frozen verifier")
    after = {"source_database": sha256(source_db), "source_manifest": sha256(source_manifest_path),
             "source_provenance": sha256(SOURCE / "provenance.json"),
             "source_verification": sha256(SOURCE / "verification-supervisor.json"),
             "source_model": {n: sha256(SOURCE / "model_text" / n) for n in frozen_report["hashes"]["model"]},
             "roi": sha256(roi_path), "runner": sha256(__file__),
             "heldout_helper": sha256(source_camera.__code__.co_filename),
             "roi_helper": sha256(load_contract.__code__.co_filename),
             "verifier_helper": sha256(checker.__file__),
             "pycolmap_binary": sha256(pycolmap._core.__file__),
             "images": {i["name"]: sha256(i["path"]) for i in source_manifest["images"]}}
    provenance["input_hashes_after"] = after
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    if before != after:
        raise ValueError("frozen source, ROI, software, or images changed")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--roi", required=True, type=Path)
    parser.add_argument("--approved-roi-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    roi = args.roi.resolve(strict=True)
    output = args.output.absolute()
    if args.worker:
        if not output.is_dir():
            raise ValueError("worker requires guard-created output directory")
        worker(output, roi, args.approved_roi_sha256)
        return
    from scripts.research_job.guard import run_child
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_child([sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--roi", str(roi), "--approved-roi-sha256", args.approved_roi_sha256,
                        "--output", str(output)], cwd=ROOT, output_dir=output,
                       timeout_seconds=300, max_output_bytes=1 << 30,
                       reserve_bytes=10 << 30, max_log_bytes=10 << 20)
    print(json.dumps(result, indent=2))
    if result["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
