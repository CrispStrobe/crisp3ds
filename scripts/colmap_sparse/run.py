#!/usr/bin/env python3
"""Bounded PyCOLMAP sparse run with globally excluded SIFT validation features."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "build-opencv/tree-refine/baseline/scene/conversion.json"
STAGING = ROOT / ".local-tools/colmap-sparse/images"
OUTPUT = ROOT / "build-opencv/colmap-sparse/run-001"
BASELINE_OUTPUT = ROOT / "build-opencv/colmap-sparse/baseline-resized-001"
ORIGINALS_OUTPUT = ROOT / "build-opencv/colmap-sparse/baseline-originals-001"
SEED = "colmap-sparse-v2-20260926-position-groups"
COLMAP_RANDOM_SEED = 20260926
PAIR_BASE = 2147483647


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_images():
    source = json.loads(SOURCE.read_text())
    if len(source["views"]) != 10:
        raise ValueError("expected exactly ten tree views")
    STAGING.mkdir(parents=True, exist_ok=True)
    images = []
    for view in source["views"]:
        src = Path(view["output_image"]).resolve(strict=True)
        if not src.name == "undistorted.png":
            raise ValueError(f"unexpected input {src}")
        # Preserve the source image identity; the existing 768x512 PNG encoding stays intact.
        name = Path(view["name"]).stem + ".png"
        dst = STAGING / name
        if dst.exists():
            if sha256(dst) != sha256(src):
                raise ValueError(f"staged image differs: {dst}")
        else:
            shutil.copyfile(src, dst)
        images.append({"name": name, "original_name": view["name"],
                       "path": str(dst.resolve()), "sha256": sha256(dst),
                       "source_path": str(src), "source_sha256": sha256(src),
                       "width": 768, "height": 512})
    return images


def original_images():
    source = json.loads(SOURCE.read_text())
    if len(source["views"]) != 10:
        raise ValueError("expected exactly ten tree views")
    images = []
    for view in source["views"]:
        path = Path(view["source"]).resolve(strict=True)
        if path.name != view["name"] or path.suffix.lower() != ".jpg":
            raise ValueError(f"unexpected original image: {path}")
        digest = sha256(path)
        images.append({"name": path.name, "original_name": path.name,
                       "path": str(path), "sha256": digest,
                       "source_path": str(path), "source_sha256": digest})
    if len({Path(item["path"]).parent for item in images}) != 1:
        raise ValueError("original images do not share an image directory")
    return images


def holdout_partition(keypoints, name):
    """Assign orientation variants at one location to the same split."""
    validate_keypoint_layout(keypoints)
    groups = {}
    for i, row in enumerate(keypoints):
        # Six-column affine keypoints encode a 2x2 matrix in columns 3-6.
        # Its entries change with orientation, so location alone defines a group.
        end = 2 if keypoints.shape[1] == 6 else 3
        key = tuple(round(float(v), 3) for v in row[:end])
        groups.setdefault(key, []).append(i)
    ranked = sorted(groups, key=lambda key: hashlib.sha256(
        f"{SEED}|{name}|{key}".encode()).digest())
    selected = set(ranked[:round(len(ranked) * 0.20)])
    heldout = [i for key in ranked if key in selected for i in groups[key]]
    heldout_set = set(heldout)
    train = [i for i in range(len(keypoints)) if i not in heldout_set]
    return sorted(heldout), train


def read_feature_blob(conn, table, image_id, dtype):
    import numpy as np
    row = conn.execute(f"SELECT rows,cols,data FROM {table} WHERE image_id=?", (image_id,)).fetchone()
    if row is None:
        raise ValueError(f"missing {table} for image {image_id}")
    return np.frombuffer(row[2], dtype=dtype).reshape(row[0], row[1]).copy()


def validate_keypoint_layout(keypoints):
    """Accept only COLMAP x,y,scale,angle or x,y,affine-matrix layouts."""
    import numpy as np
    if keypoints.ndim != 2 or keypoints.shape[1] not in (4, 6):
        raise ValueError(f"unsupported COLMAP keypoint layout: {keypoints.shape}")
    if not np.isfinite(keypoints).all() or (keypoints.shape[1] == 4 and
                                            (keypoints[:, 2] <= 0).any()):
        raise ValueError("invalid COLMAP keypoint coordinates or scale")


def heldout_correspondences(desc1, ids1, desc2, ids2):
    """Mutual nearest neighbors with a 0.8 Euclidean distance ratio."""
    import numpy as np
    if len(ids1) < 2 or len(ids2) < 2:
        return []
    a = desc1[ids1].astype(np.float32)
    b = desc2[ids2].astype(np.float32)
    # Squared Euclidean distance; clipped to remove roundoff below zero.
    dist = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
    best_b = np.argmin(dist, axis=1)
    best_a = np.argmin(dist, axis=0)
    second = np.partition(dist, 1, axis=1)[:, 1]
    result = []
    for i, j in enumerate(best_b):
        if best_a[j] == i and dist[i, j] < 0.64 * second[i] and second[i] > 0:
            result.append((int(ids1[i]), int(ids2[j])))
    return result


def partition_database(db_path, image_info):
    import numpy as np
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT image_id,name,camera_id FROM images ORDER BY name").fetchall()
    if len(rows) != 10 or len({r[2] for r in rows}) != 1:
        raise ValueError("expected ten images and one shared camera")
    by_name = {i["name"]: i for i in image_info}
    records = {}
    descriptors = {}
    for image_id, name, camera_id in rows:
        if name not in by_name:
            raise ValueError(f"unexpected image {name}")
        keypoints = read_feature_blob(conn, "keypoints", image_id, np.float32)
        validate_keypoint_layout(keypoints)
        desc = read_feature_blob(conn, "descriptors", image_id, np.uint8)
        if len(keypoints) != len(desc):
            raise ValueError("keypoint/descriptor count differs")
        heldout, train = holdout_partition(keypoints, name)
        records[name] = dict(by_name[name], image_id=image_id, camera_id=camera_id,
                             keypoint_columns=keypoints.shape[1],
                             features=[[float(v) for v in row] for row in keypoints],
                             heldout_ids=heldout, compact_to_original=train)
        descriptors[name] = desc
        ktrain = keypoints[train]
        dtrain = desc[train]
        conn.execute("UPDATE keypoints SET rows=?,cols=?,data=? WHERE image_id=?",
                     (len(ktrain), keypoints.shape[1], sqlite3.Binary(ktrain.tobytes()), image_id))
        conn.execute("UPDATE descriptors SET rows=?,cols=?,data=? WHERE image_id=?",
                     (len(dtrain), desc.shape[1], sqlite3.Binary(dtrain.tobytes()), image_id))
    conn.commit()
    conn.close()
    names = sorted(records)
    heldout_pairs = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            for id1, id2 in heldout_correspondences(
                descriptors[n1], records[n1]["heldout_ids"],
                descriptors[n2], records[n2]["heldout_ids"]):
                heldout_pairs.append({"image1": n1, "feature1": id1,
                                      "image2": n2, "feature2": id2})
    return [records[n] for n in names], heldout_pairs


def training_matches(db_path, records):
    import numpy as np
    by_id = {item["image_id"]: item for item in records}
    conn = sqlite3.connect(db_path)
    matches = []
    pair_count = 0
    for pair_id, count, cols, blob in conn.execute("SELECT pair_id,rows,cols,data FROM matches"):
        if not count:
            continue
        pair_count += 1
        id2 = pair_id % PAIR_BASE
        id1 = (pair_id - id2) // PAIR_BASE
        a, b = by_id[id1], by_id[id2]
        pairs = np.frombuffer(blob, dtype=np.uint32).reshape(count, cols)
        for ia, ib in pairs:
            matches.append({"image1": a["name"], "feature1": a["compact_to_original"][int(ia)],
                            "image2": b["name"], "feature2": b["compact_to_original"][int(ib)],
                            "compact1": int(ia), "compact2": int(ib)})
    conn.close()
    return matches, pair_count


def input_hashes(images):
    return {"conversion_json": sha256(SOURCE),
            "images": {item["name"]: {
                "staged": sha256(Path(item["path"])),
                "source": sha256(Path(item["source_path"]))} for item in images}}


def software_hashes(pycolmap):
    return {"runner_script": sha256(Path(__file__)),
            "pycolmap_binary": sha256(Path(pycolmap._core.__file__))}


def assert_unchanged(before, after):
    if before != after:
        raise ValueError("run inputs or software changed during reconstruction")


def build_options(pycolmap):
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = "921.6,384,256,0"
    sift = pycolmap.SiftExtractionOptions()
    sift.max_num_features = 3000
    sift.num_threads = 2
    matching = pycolmap.SiftMatchingOptions()
    matching.num_threads = 2
    exhaustive = pycolmap.ExhaustiveMatchingOptions()
    verification = pycolmap.TwoViewGeometryOptions()
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 2
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    return reader, sift, matching, exhaustive, verification, options


def baseline_options(pycolmap, originals):
    """Conventional COLMAP pipeline, with only the two-thread CPU bound."""
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL" if originals else "SIMPLE_PINHOLE"
    reader.default_focal_length_factor = 1.2
    if reader.camera_params:
        raise ValueError("baseline must not supply camera parameters")
    sift = pycolmap.SiftExtractionOptions()
    sift.num_threads = 2
    matching = pycolmap.SiftMatchingOptions()
    matching.num_threads = 2
    exhaustive = pycolmap.ExhaustiveMatchingOptions()
    verification = pycolmap.TwoViewGeometryOptions()
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 2
    mapping.mapper.num_threads = 2
    camera_mode = pycolmap.CameraMode.AUTO if originals else pycolmap.CameraMode.SINGLE
    return reader, sift, matching, exhaustive, verification, mapping, camera_mode


def baseline_worker(images_path, out, originals):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pycolmap
    import numpy as np
    images = json.loads(images_path.read_text())
    if len(images) != 10:
        raise ValueError("expected ten baseline images")
    image_dir = Path(images[0]["path"]).parent
    if any(Path(item["path"]).parent != image_dir for item in images):
        raise ValueError("baseline images must have one directory")
    out.mkdir(parents=True, exist_ok=True)
    reader, sift, matching, exhaustive, verification, mapping, camera_mode = baseline_options(
        pycolmap, originals)
    before = input_hashes(images)
    software_before = software_hashes(pycolmap)
    for item in images:
        if before["images"][item["name"]] != {
                "staged": item["sha256"], "source": item["source_sha256"]}:
            raise ValueError(f"image changed before run: {item['name']}")
    provenance = {"schema": "colmap_sparse_provenance_v1",
                  "lane": "baseline_originals" if originals else "baseline_resized",
                  "validation": "training reconstruction only; no independent holdout",
                  "pycolmap_version": pycolmap.__version__, "numpy_version": np.__version__,
                  "python_version": sys.version, "platform": platform.platform(),
                  "machine": platform.machine(), "random_seed": {"pycolmap": 0},
                  "options": {"camera_mode": camera_mode.name, "device": "cpu",
                              "image_reader": reader.todict(),
                              "sift_extraction": sift.todict(),
                              "sift_matching": matching.todict(),
                              "exhaustive_matching": exhaustive.todict(),
                              "two_view_geometry": verification.todict(),
                              "incremental_pipeline": mapping.todict()},
                  "input_hashes_before": before, "input_hashes_after": None,
                  "software_hashes_before": software_before,
                  "software_hashes_after": None,
                  "camera_prior_focal_length": None}
    provenance_path = out / "provenance.json"
    def save_provenance():
        provenance_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    save_provenance()
    pycolmap.set_random_seed(0)
    db = out / "database.db"
    pycolmap.extract_features(str(db), str(image_dir), [item["name"] for item in images],
                              camera_mode=camera_mode, reader_options=reader,
                              camera_model=reader.camera_model,
                              sift_options=sift, device=pycolmap.Device.cpu)
    with sqlite3.connect(db) as conn:
        camera_rows = conn.execute(
            "SELECT camera_id,model,width,height,prior_focal_length FROM cameras ORDER BY camera_id"
        ).fetchall()
    provenance["camera_prior_focal_length"] = [
        {"camera_id": row[0], "model_id": row[1], "width": row[2],
         "height": row[3], "prior_focal_length": bool(row[4])} for row in camera_rows]
    save_provenance()
    pycolmap.match_exhaustive(str(db), sift_options=matching,
                             matching_options=exhaustive,
                             verification_options=verification,
                             device=pycolmap.Device.cpu)
    models = pycolmap.incremental_mapping(str(db), str(image_dir), str(out / "models"),
                                           options=mapping)
    report = {"lane": provenance["lane"], "validation": provenance["validation"],
              "model_count": len(models), "registered_images": 0,
              "points3D": 0, "image_count": len(images),
              "units": "arbitrary similarity scale"}
    if models:
        model = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        report["registered_images"] = model.num_reg_images()
        report["points3D"] = model.num_points3D()
        text_dir = out / "model_text"
        text_dir.mkdir()
        model.write_text(str(text_dir))
        model.export_PLY(str(out / "points.ply"))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance["input_hashes_after"] = input_hashes(images)
    provenance["software_hashes_after"] = software_hashes(pycolmap)
    save_provenance()
    assert_unchanged(before, provenance["input_hashes_after"])
    assert_unchanged(software_before, provenance["software_hashes_after"])
    print(json.dumps(report), flush=True)


def worker(images_path, out):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pycolmap
    import numpy as np
    images = json.loads(images_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    reader, sift, matching, exhaustive, verification, options = build_options(pycolmap)
    before = input_hashes(images)
    software_before = software_hashes(pycolmap)
    for item in images:
        if before["images"][item["name"]] != {
                "staged": item["sha256"], "source": item["source_sha256"]}:
            raise ValueError(f"image changed before run: {item['name']}")
    provenance = {"schema": "colmap_sparse_provenance_v1",
                  "pycolmap_version": pycolmap.__version__, "numpy_version": np.__version__,
                  "python_version": sys.version, "platform": platform.platform(),
                  "machine": platform.machine(),
                  "random_seed": {"pycolmap": COLMAP_RANDOM_SEED, "holdout": SEED},
                  "options": {"camera_mode": "SINGLE", "device": "cpu",
                              "image_reader": reader.todict(),
                              "sift_extraction": sift.todict(),
                              "sift_matching": matching.todict(),
                              "exhaustive_matching": exhaustive.todict(),
                              "two_view_geometry": verification.todict(),
                              "incremental_pipeline": options.todict()},
                  "input_hashes_before": before, "input_hashes_after": None,
                  "software_hashes_before": software_before,
                  "software_hashes_after": None,
                  "camera_prior_focal_length": None}
    provenance_path = out / "provenance.json"
    def save_provenance():
        provenance_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    save_provenance()
    pycolmap.set_random_seed(COLMAP_RANDOM_SEED)
    pycolmap.extract_features(str(db), str(STAGING), [i["name"] for i in images],
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model=reader.camera_model,
                              reader_options=reader, sift_options=sift,
                              device=pycolmap.Device.cpu)
    with sqlite3.connect(db) as conn:
        cameras = conn.execute("SELECT prior_focal_length FROM cameras").fetchall()
    if len(cameras) != 1:
        raise ValueError("expected exactly one shared camera after extraction")
    provenance["camera_prior_focal_length"] = bool(cameras[0][0])
    save_provenance()
    records, heldout_pairs = partition_database(db, images)
    manifest = {"schema": "colmap_sparse_holdout_v2", "seed": SEED,
                "camera_initial": {"model": "SIMPLE_RADIAL", "params": [921.6, 384, 256, 0],
                                   "shared": True, "refine_focal": True,
                                   "refine_principal": False, "refine_radial": True},
                "images": records, "heldout_pairs": heldout_pairs,
                "heldout_matching": "mutual nearest SIFT; forward (image1 to image2) Euclidean ratio <0.8 only; no reverse ratio or geometry filter",
                "training_matches": [], "training_pair_count": 0}
    (out / "holdout.json").write_text(json.dumps(manifest, separators=(",", ":")))
    pycolmap.match_exhaustive(str(db), sift_options=matching,
                             matching_options=exhaustive,
                             verification_options=verification,
                             device=pycolmap.Device.cpu)
    manifest["training_matches"], manifest["training_pair_count"] = training_matches(db, records)
    (out / "holdout.json").write_text(json.dumps(manifest, separators=(",", ":")))
    models = pycolmap.incremental_mapping(str(db), str(STAGING), str(out / "models"),
                                           options=options)
    report = {"model_count": len(models), "registered_images": 0,
              "points3D": 0, "image_count": len(images),
              "heldout_feature_count": sum(len(x["heldout_ids"]) for x in records),
              "heldout_pair_count": len(heldout_pairs),
              "training_match_count": len(manifest["training_matches"]),
              "training_pair_count": manifest["training_pair_count"],
              "units": "arbitrary similarity scale"}
    if models:
        model = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        report["registered_images"] = model.num_reg_images()
        report["points3D"] = model.num_points3D()
        text_dir = out / "model_text"
        text_dir.mkdir()
        model.write_text(str(text_dir))
        model.export_PLY(str(out / "points.ply"))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance["input_hashes_after"] = input_hashes(images)
    provenance["software_hashes_after"] = software_hashes(pycolmap)
    save_provenance()
    assert_unchanged(before, provenance["input_hashes_after"])
    assert_unchanged(software_before, provenance["software_hashes_after"])
    print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--baseline", action="store_true",
                       help="full-feature baseline on ten resized PNGs")
    modes.add_argument("--originals", action="store_true",
                       help="full-feature baseline on ten original JPEGs")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if args.images is None or args.output is None:
            parser.error("worker requires --images and --output")
        if not (args.baseline or args.originals):
            parser.error("holdout lane quarantined: v1 validation was contaminated")
        baseline_worker(args.images, args.output, args.originals)
    else:
        from scripts.research_job.guard import run_child
        if not (args.baseline or args.originals):
            parser.error("holdout lane quarantined: use --baseline or --originals")
        default_output = ORIGINALS_OUTPUT if args.originals else (
            BASELINE_OUTPUT if args.baseline else OUTPUT)
        output = (args.output or default_output).absolute()
        images = original_images() if args.originals else stage_images()
        staging_manifest = STAGING / ("original-images.json" if args.originals else "images.json")
        STAGING.mkdir(parents=True, exist_ok=True)
        staging_manifest.write_text(json.dumps(images, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--images", str(staging_manifest), "--output", str(output)]
        if args.baseline:
            command.append("--baseline")
        if args.originals:
            command.append("--originals")
        result = run_child(command,
                           cwd=ROOT, output_dir=output, timeout_seconds=300,
                           max_output_bytes=1 << 30, reserve_bytes=10 << 30,
                           max_log_bytes=10 << 20)
        print(json.dumps(result, indent=2))
        if result["status"] != "succeeded":
            sys.exit(1)


if __name__ == "__main__":
    main()
