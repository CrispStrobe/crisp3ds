#!/usr/bin/env python3
"""Frozen clean observation holdout from the full-feature resized COLMAP DB."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colmap_sparse.feature_split import split_features

SOURCE_DB = ROOT / "build-opencv/colmap-sparse/resized-supervisor-001/database.db"
SOURCE_PROVENANCE = ROOT / "build-opencv/colmap-sparse/resized-supervisor-001/provenance.json"
SPLIT_AUDIT = ROOT / "build-opencv/colmap-sparse/feature-split-audit-001.json"
IMAGE_DIR = ROOT / ".local-tools/colmap-sparse/images"
OUTPUT = ROOT / "build-opencv/colmap-sparse/heldout-v2-001"
SEED = "colmap-spatial-components-v1-20260926"
PAIR_BASE = 2147483647


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_camera(conn):
    rows = conn.execute("SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras").fetchall()
    if len(rows) != 1:
        raise ValueError("expected exactly one shared source camera")
    cid, model, width, height, blob, prior = rows[0]
    if model != 0 or (width, height) != (768, 512) or prior != 0 or len(blob) != 24:
        raise ValueError("source camera is not unknown-focal SIMPLE_PINHOLE 768x512")
    f, cx, cy = struct.unpack("<ddd", blob)
    if not (abs(f-921.6) < 1e-8 and cx == 384 and cy == 256):
        raise ValueError("unexpected initial source camera parameters")
    if conn.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0] != 0:
        raise ValueError("source database has pose priors")
    return dict(camera_id=cid, model="SIMPLE_PINHOLE", model_id=model,
                width=width, height=height, initial_params=[f, cx, cy],
                prior_focal_length=False, pose_prior_count=0)


def _array(conn, table, image_id, dtype):
    import numpy as np
    row = conn.execute(f"SELECT rows,cols,data FROM {table} WHERE image_id=?", (image_id,)).fetchone()
    if row is None or row[2] is None:
        raise ValueError(f"missing {table} for image {image_id}")
    n, width, blob = row
    if len(blob) != n*width*np.dtype(dtype).itemsize:
        raise ValueError(f"invalid {table} blob for image {image_id}")
    return np.frombuffer(blob, dtype=dtype).reshape(n, width).copy()


def mutual_heldout_matches(desc_a, ids_a, desc_b, ids_b):
    """Mutual nearest with Euclidean ratio <.8 independently both ways."""
    import numpy as np
    if len(ids_a) < 2 or len(ids_b) < 2:
        return []
    a = desc_a[list(ids_a)].astype(np.float32)
    b = desc_b[list(ids_b)].astype(np.float32)
    d = np.maximum((a*a).sum(1)[:, None]+(b*b).sum(1)[None, :]-2*a@b.T, 0)
    best_b = np.argmin(d, axis=1)
    best_a = np.argmin(d, axis=0)
    second_b = np.partition(d, 1, axis=1)[:, 1]
    second_a = np.partition(d, 1, axis=0)[1, :]
    result = []
    for ia, ib in enumerate(best_b):
        distance = d[ia, ib]
        if (best_a[ib] == ia and second_b[ia] > 0 and second_a[ib] > 0
                and distance < .64*second_b[ia] and distance < .64*second_a[ib]):
            result.append((int(ids_a[ia]), int(ids_b[ib])))
    return result


def prepare_database(conn, audit, image_dir):
    """Delete old pair evidence, reindex training rows, export original IDs."""
    import numpy as np
    camera = source_camera(conn)
    rows = conn.execute("SELECT image_id,name,camera_id FROM images ORDER BY name").fetchall()
    if len(rows) != 10 or any(cid != camera["camera_id"] for _, _, cid in rows):
        raise ValueError("expected ten images using the shared camera")
    audited = {item["name"]: item for item in audit["images"]}
    if set(audited) != {name for _, name, _ in rows}:
        raise ValueError("split audit image names differ from source database")
    cleared_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    cleared_geometries = conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    conn.execute("DELETE FROM matches")
    conn.execute("DELETE FROM two_view_geometries")
    if (conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] or
            conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]):
        raise ValueError("copied pair tables were not cleared")
    records = []
    full_descriptors = {}
    full_rows = 0
    for iid, name, cid in rows:
        keypoints = _array(conn, "keypoints", iid, np.float32)
        descriptors = _array(conn, "descriptors", iid, np.uint8)
        if keypoints.shape[1] != 6 or descriptors.shape[1] != 128 or len(keypoints) != len(descriptors):
            raise ValueError(f"unexpected full-feature shape: {name}")
        full_rows += len(keypoints)
        split = split_features(keypoints, name, SEED)
        if list(split.heldout_ids) != audited[name]["heldout_ids"]:
            raise ValueError(f"live split differs from audited assignment: {name}")
        image_path = image_dir/name
        if not image_path.is_file():
            raise ValueError(f"missing input image: {image_path}")
        records.append(dict(name=name, path=str(image_path.resolve()),
                            sha256=sha256(image_path), width=768, height=512,
                            image_id=iid, camera_id=cid, keypoint_columns=6,
                            features=keypoints.tolist(),
                            heldout_ids=split.heldout_ids,
                            compact_to_original=split.train_ids,
                            spatial_group_count=len(split.groups)))
        full_descriptors[name] = descriptors
        train = list(split.train_ids)
        compact_keypoints = keypoints[train]
        compact_descriptors = descriptors[train]
        conn.execute("UPDATE keypoints SET rows=?,cols=?,data=? WHERE image_id=?",
                     (len(train), 6, sqlite3.Binary(compact_keypoints.tobytes()), iid))
        conn.execute("UPDATE descriptors SET rows=?,cols=?,data=? WHERE image_id=?",
                     (len(train), 128, sqlite3.Binary(compact_descriptors.tobytes()), iid))
    conn.commit()
    if (conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] or
            conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]):
        raise ValueError("pair tables are nonempty before matching")
    preparation = dict(copied_match_rows_deleted=cleared_matches,
                       copied_two_view_geometries_deleted=cleared_geometries,
                       remaining_match_rows_before_matching=0,
                       remaining_two_view_geometries_before_matching=0,
                       full_keypoint_rows=full_rows,
                       compact_training_rows=sum(len(r["compact_to_original"]) for r in records),
                       excluded_heldout_rows=sum(len(r["heldout_ids"]) for r in records))
    return camera, records, full_descriptors, preparation


def raw_heldout_pairs(records, descriptors):
    pairs = []
    for i, a in enumerate(records):
        for b in records[i+1:]:
            for ia, ib in mutual_heldout_matches(
                    descriptors[a["name"]], a["heldout_ids"],
                    descriptors[b["name"]], b["heldout_ids"]):
                pairs.append(dict(image1=a["name"], feature1=ia,
                                  image2=b["name"], feature2=ib))
    return pairs


def training_matches(conn, records):
    by_id = {record["image_id"]: record for record in records}
    matches = []; pair_count = 0
    for pair_id, rows, cols, blob in conn.execute("SELECT pair_id,rows,cols,data FROM matches"):
        if rows == 0:
            continue
        if cols != 2 or blob is None or len(blob) != rows*8:
            raise ValueError(f"invalid match blob: {pair_id}")
        id2 = pair_id % PAIR_BASE; id1 = (pair_id-id2)//PAIR_BASE
        a, b = by_id[id1], by_id[id2]
        pair_count += 1
        for ia, ib in struct.iter_unpack("<II", blob):
            ma, mb = a["compact_to_original"], b["compact_to_original"]
            if ia >= len(ma) or ib >= len(mb):
                raise ValueError(f"match row out of range: {pair_id}")
            matches.append(dict(image1=a["name"], feature1=ma[ia],
                                image2=b["name"], feature2=mb[ib],
                                compact1=ia, compact2=ib))
    return matches, pair_count


def worker(out):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import numpy as np
    import pycolmap
    audit = json.loads(SPLIT_AUDIT.read_text())
    baseline = json.loads(SOURCE_PROVENANCE.read_text())
    if (audit["status"] != "pass" or audit["seed"] != SEED or
            audit["total_spatial_overlap"] != 0 or
            audit["splitter_sha256"] != sha256(split_features.__code__.co_filename) or
            audit["database_sha256_after"] != sha256(SOURCE_DB)):
        raise ValueError("source DB no longer matches frozen split audit")
    if (baseline["pycolmap_version"] != "3.11.1" or
            baseline["software_hashes_before"] != baseline["software_hashes_after"] or
            baseline["software_hashes_after"]["pycolmap_binary"] != sha256(pycolmap._core.__file__) or
            baseline["input_hashes_before"] != baseline["input_hashes_after"]):
        raise ValueError("baseline extraction provenance or pinned PyCOLMAP changed")
    expected_image_hashes = {name: item["staged"] for name, item in
                             baseline["input_hashes_after"]["images"].items()}
    for name, expected in expected_image_hashes.items():
        if sha256(IMAGE_DIR/name) != expected:
            raise ValueError(f"staged PNG differs from source extraction: {name}")
    input_hashes_before = {"database": sha256(SOURCE_DB),
                           "split_audit": sha256(SPLIT_AUDIT),
                           "baseline_provenance": sha256(SOURCE_PROVENANCE)}
    software_before = {"runner": sha256(__file__),
                       "splitter": sha256(split_features.__code__.co_filename),
                       "pycolmap_binary": sha256(pycolmap._core.__file__)}
    source = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    dest_path = out/"database.db"
    dest = sqlite3.connect(dest_path)
    source.backup(dest)
    source.close()
    camera, records, descriptors, preparation = prepare_database(dest, audit, IMAGE_DIR)
    if {r["name"]: r["sha256"] for r in records} != expected_image_hashes:
        raise ValueError("source PNG names or bytes differ from extracted features")
    if len({r["sha256"] for r in records}) != 10:
        raise ValueError("unexpected duplicate image bytes")
    heldout_pairs = raw_heldout_pairs(records, descriptors)
    del descriptors
    manifest = dict(schema="colmap_sparse_holdout_v3_spatial_components",
                    seed=SEED, radius_px=0.25,
                    camera_initial=camera,
                    heldout_matching="mutual nearest SIFT; Euclidean ratio <0.8 in both directions; no geometry filter",
                    images=records, heldout_pairs=heldout_pairs,
                    training_matches=[], training_pair_count=0)
    (out/"holdout.json").write_text(json.dumps(manifest, separators=(",", ":")))
    matching = pycolmap.SiftMatchingOptions(); matching.num_threads = 2
    exhaustive = pycolmap.ExhaustiveMatchingOptions()
    verification = pycolmap.TwoViewGeometryOptions()
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 2; mapping.mapper.num_threads = 2
    provenance = dict(schema="colmap_heldout_v2_provenance", python=sys.version,
                      platform=platform.platform(), pycolmap=pycolmap.__version__,
                      numpy=np.__version__, device="cpu", threads=2,
                      random_seed_before_matching=0, random_seed_before_mapping=0,
                      source_database=SOURCE_DB.as_posix(),
                      source_camera=camera,
                      preparation=preparation,
                      options=dict(sift_matching=matching.todict(),
                                   exhaustive_matching=exhaustive.todict(),
                                   two_view_geometry=verification.todict(),
                                   incremental_pipeline=mapping.todict()),
                      input_hashes_before=input_hashes_before,
                      software_hashes_before=software_before)
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2, default=str)+"\n")
    dest.close()
    pycolmap.set_random_seed(0)
    pycolmap.match_exhaustive(str(dest_path), sift_options=matching,
                             matching_options=exhaustive,
                             verification_options=verification,
                             device=pycolmap.Device.cpu)
    dest = sqlite3.connect(f"file:{dest_path}?mode=ro", uri=True)
    manifest["training_matches"], manifest["training_pair_count"] = training_matches(dest, records)
    (out/"holdout.json").write_text(json.dumps(manifest, separators=(",", ":")))
    dest.close()
    pycolmap.set_random_seed(0)
    models = pycolmap.incremental_mapping(str(dest_path), str(IMAGE_DIR), str(out/"models"),
                                           options=mapping)
    summary = dict(model_count=len(models), registered_images=0, points3D=0,
                   image_count=len(records), heldout_feature_count=sum(len(r["heldout_ids"]) for r in records),
                   heldout_pair_count=len(heldout_pairs),
                   training_match_count=len(manifest["training_matches"]),
                   training_pair_count=manifest["training_pair_count"],
                   diagnostic_screen="pending independent verifier",
                   quality_accepted=False, units="arbitrary similarity scale")
    if models:
        model = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        summary["registered_images"] = model.num_reg_images()
        summary["points3D"] = model.num_points3D()
        text_dir = out/"model_text"; text_dir.mkdir()
        model.write_text(str(text_dir))
    input_hashes_after = {"database": sha256(SOURCE_DB),
                          "split_audit": sha256(SPLIT_AUDIT),
                          "baseline_provenance": sha256(SOURCE_PROVENANCE)}
    software_after = {"runner": sha256(__file__),
                      "splitter": sha256(split_features.__code__.co_filename),
                      "pycolmap_binary": sha256(pycolmap._core.__file__)}
    image_hashes_after = {r["name"]: sha256(r["path"]) for r in records}
    if (input_hashes_before != input_hashes_after or software_before != software_after or
            any(image_hashes_after[r["name"]] != r["sha256"] for r in records)):
        raise ValueError("input, software, or images changed during run")
    provenance.update(input_hashes_after=input_hashes_after,
                      software_hashes_after=software_after,
                      image_hashes_after=image_hashes_after)
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2, default=str)+"\n")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        if not output.is_dir():
            raise ValueError("worker requires guard-created output directory")
        worker(output)
    else:
        from scripts.research_job.guard import run_child
        result = run_child([sys.executable, str(Path(__file__).resolve()),
                            "--worker", "--output", str(output)],
                           cwd=ROOT, output_dir=output, timeout_seconds=300,
                           max_output_bytes=1 << 30, reserve_bytes=10 << 30,
                           max_log_bytes=10 << 20)
        print(json.dumps(result, indent=2))
        if result["status"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
