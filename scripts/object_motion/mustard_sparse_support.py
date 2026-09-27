#!/usr/bin/env python3
"""Read-only exact-PNG support membership for sealed mustard SfM feature DBs."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image

from scripts.object_motion import ycb_object_masks as common


MAX_DB = 2 * 1024**3
MAX_KEYPOINTS_PER_IMAGE = 100_000
MASK_SIZE = (1280, 1024)


def bound_json(path, expected_sha, limit=2 * 1024**2):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > limit or
            common.digest(path) != expected_sha):
        raise ValueError("metadata missing, linked, oversize or SHA mismatch")
    return json.loads(path.read_text())


def load_masks(stage_root, stage_report):
    stage_root = Path(stage_root)
    report = stage_report
    names = report.get("train_names")
    hashes = report.get("train_photo_sha256")
    entries = report.get("cleaned_masks")
    if (report.get("schema") != "mustard_train_only_stage_v1" or
            report.get("status") != "complete" or
            report.get("package_sha256") != common.PACKAGE_SHA256["006_mustard_bottle"] or
            not isinstance(names, list) or len(names) != 48 or len(set(names)) != 48 or
            not isinstance(hashes, dict) or set(hashes) != set(names) or
            not isinstance(entries, dict) or set(entries) != set(names)):
        raise ValueError("stage report not exact 48 TRAIN photos/masks")
    photos, masks = stage_root / "images", stage_root / "masks"
    if (photos.is_symlink() or masks.is_symlink() or not photos.is_dir() or not masks.is_dir() or
            {p.name for p in photos.iterdir()} != set(names) or
            {p.name for p in masks.iterdir()} != {name + ".png" for name in names}):
        raise ValueError("stage photo/mask directory missing, extra or linked")
    pixels = {}
    for name in names:
        photo, mask = photos / name, masks / (name + ".png")
        if (photo.is_symlink() or mask.is_symlink() or not photo.is_file() or not mask.is_file() or
                common.digest(photo) != hashes[name] or
                common.digest(mask) != entries[name].get("sha256") or
                mask.stat().st_size > 2 * 1024**2):
            raise ValueError(f"staged photo/mask bytes changed: {name}")
        with Image.open(mask) as image:
            if image.mode != "L" or image.size != MASK_SIZE:
                raise ValueError(f"mask not full-resolution grayscale: {name}")
            array = np.array(image)
        if array.dtype != np.uint8 or not np.isin(array, (0, 255)).all():
            raise ValueError(f"mask not binary: {name}")
        pixels[name] = array
    return names, pixels


def sample_membership(xy, mask):
    """Measured COLMAP XY -> exact mask pixel by floor, with OOB separate."""
    values = np.asarray(xy)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("keypoint XY must be finite Nx2")
    x, y = np.floor(values[:, 0]).astype(np.int64), np.floor(values[:, 1]).astype(np.int64)
    valid = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    inside = np.zeros(len(x), dtype=bool)
    inside[valid] = mask[y[valid], x[valid]] != 0
    return {"denominator": int(len(x)), "inside_coarse_pose_mask": int(inside.sum()),
            "outside_coarse_pose_mask": int(valid.sum() - inside.sum()),
            "outside_image_grid": int((~valid).sum())}


def database_keypoint_support(database, names, masks):
    database = Path(database)
    sidecars = [database.with_name(database.name + suffix) for suffix in ("-wal", "-shm")]
    if (database.is_symlink() or not database.is_file() or database.stat().st_size > MAX_DB or
            any(path.exists() or path.is_symlink() for path in sidecars)):
        raise ValueError("feature database requires sidecar-free sealed snapshot")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        rows = db.execute("SELECT i.name,k.rows,k.cols,k.data FROM images i LEFT JOIN keypoints k "
                          "ON i.image_id=k.image_id").fetchall()
    if len(rows) != len(names) or {row[0] for row in rows} != set(names):
        raise ValueError("database image names differ from exact TRAIN list")
    by_name = {}
    for name, count, columns, blob in rows:
        if (not isinstance(count, int) or not 0 <= count <= MAX_KEYPOINTS_PER_IMAGE or
                columns not in (2, 4, 6) or not isinstance(blob, bytes) or
                len(blob) != count * columns * 4):
            raise ValueError(f"missing/invalid COLMAP keypoint blob: {name}")
        xy = np.frombuffer(blob, dtype="<f4").reshape(count, columns)[:, :2]
        by_name[name] = sample_membership(xy, masks[name])
    sums = {key: sum(row[key] for row in by_name.values()) for key in
            ("denominator", "inside_coarse_pose_mask", "outside_coarse_pose_mask", "outside_image_grid")}
    if sums["denominator"] != sum(sums[key] for key in
                                  ("inside_coarse_pose_mask", "outside_coarse_pose_mask", "outside_image_grid")):
        raise ValueError("keypoint support denominator mismatch")
    return {"by_image": {name: by_name[name] for name in names}, "total": sums}


def source_sidecars(database):
    """A producer may retain an empty WAL/SHM; no uncheckpointed WAL is accepted."""
    database = Path(database)
    record = {}
    for suffix in ("-wal", "-shm"):
        path = database.with_name(database.name + suffix)
        if path.exists() or path.is_symlink():
            if (path.is_symlink() or not path.is_file() or
                    path.stat().st_size > (0 if suffix == "-wal" else 64 * 1024)):
                raise ValueError("source database sidecar has uncheckpointed/unsafe content")
            record[suffix] = {"bytes": path.stat().st_size, "sha256": common.digest(path)}
        else:
            record[suffix] = None
    return record


def evaluate(run, result_sha, stage_root, stage_sha, snapshot, source_db_sha):
    run, stage_root = Path(run), Path(stage_root)
    runner_sha = common.digest(__file__)
    result = bound_json(run / "result.json", result_sha)
    stage = bound_json(stage_root / "stage-report.json", stage_sha)
    if (result.get("schema") != "classical_backend_v1" or
            result.get("status") not in ("failed", "sparse_complete") or
            result.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap"):
        raise ValueError("run is not a sealed image-only classical SfM arm")
    names, masks = load_masks(stage_root, stage)
    inputs = result.get("inputs")
    if (not isinstance(inputs, list) or [entry.get("name") for entry in inputs] != names or
            {entry.get("name"): entry.get("sha256") for entry in inputs} != stage["train_photo_sha256"]):
        raise ValueError("run source image hashes/order differ from staged TRAIN data")
    for entry in inputs:
        photo = run / "images" / entry["name"]
        if photo.is_symlink() or not photo.is_file() or common.digest(photo) != entry["sha256"]:
            raise ValueError("run image bytes changed")
    database = run / "database.db"
    snapshot = Path(snapshot)
    sidecars = source_sidecars(database)
    if (database.is_symlink() or not database.is_file() or snapshot.is_symlink() or
            not snapshot.is_file() or not isinstance(source_db_sha, str) or
            len(source_db_sha) != 64 or
            common.digest(database) != source_db_sha or common.digest(snapshot) != source_db_sha or
            snapshot.resolve() == database.resolve()):
        raise ValueError("sidecar-free audit snapshot differs from sealed producer database")
    support = database_keypoint_support(snapshot, names, masks)
    for name in names:
        if (common.digest(stage_root / "images" / name) != stage["train_photo_sha256"][name] or
                common.digest(stage_root / "masks" / (name + ".png")) !=
                stage["cleaned_masks"][name]["sha256"] or
                common.digest(run / "images" / name) != stage["train_photo_sha256"][name]):
            raise ValueError("staged/run photo or mask changed during keypoint audit")
    if (common.digest(run / "result.json") != result_sha or
            common.digest(stage_root / "stage-report.json") != stage_sha or
            common.digest(database) != source_db_sha or common.digest(snapshot) != source_db_sha or
            source_sidecars(database) != sidecars or
            common.digest(__file__) != runner_sha):
        raise ValueError("input source changed during keypoint audit")
    return {"schema": "mustard_sparse_support_audit_v1", "scope": "sealed TRAIN keypoints only",
            "status": "diagnostic_only", "run_status": result["status"],
            "run_result_sha256": result_sha, "stage_report_sha256": stage_sha,
            "source_database_sha256": source_db_sha, "audit_snapshot_sha256": source_db_sha,
            "source_sidecars": sidecars,
            "runner_sha256": runner_sha,
            "pixel_mapping": "floor measured COLMAP XY; OOB separate",
            "mask_role": "accepted coarse pose support, not ground-truth object silhouette",
            "keypoints": support}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "result-sha256", "stage-root", "stage-sha256",
                 "snapshot", "source-db-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if (output.exists() or output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir() or shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES):
        raise ValueError("audit output must be fresh on a volume with 10 GiB free")
    report = evaluate(args.run, args.result_sha256, args.stage_root, args.stage_sha256,
                      args.snapshot, args.source_db_sha256)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > 1024**2:
        raise ValueError("keypoint audit report above 1 MiB")
    with output.open("xb") as file:
        file.write(payload)


if __name__ == "__main__":
    main()
