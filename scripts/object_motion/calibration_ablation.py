#!/usr/bin/env python3
"""Bounded cached-feature YCB SfM ablation; supplied intrinsics are oracle-assisted.

The sealed photo-derived foreground database supplies *all* keypoints, descriptors,
and raw matches. Each fresh arm re-verifies those raw matches after its camera
change, then estimates poses/structure from images. No supplied pose or mesh enters.
"""

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.object_motion.run import digest, run_trial
from scripts.object_motion import run as original_run
from scripts.object_motion.ycb_camera_reference import dataset, reference_cameras, sha256
from scripts.upstream_control.camera_compare import fit_centers
from scripts.object_motion.ycb_camera_reference import rotation_errors

SOURCE = ROOT / "build-opencv/object-motion/foreground"
PREPARED = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
CALIBRATION = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/metadata/calibration.h5"
CAMERA_REFERENCE = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/report.json"
OUTPUT = ROOT / "build-opencv/object-motion/calibration-ablation-002"
PYTHON = ROOT / ".local-tools/colmap-sparse/venv/bin/python"
CALIBRATION_SHA256 = "15f724c10131129183e97cb3ae6fdb3f13ff7ad8c5a50812313f06652a7f6c07"
MAX_IMAGE_ID = 2147483647
ARMS = ("image_only_reverified", "calibrated_shifted", "calibrated_unshifted")
SEED_NAMES = ("NP3_192.jpg", "NP3_162.jpg")
SEED = 20260927
ARM_CAP = 100 * 1024 * 1024


def table_digest(connection, table):
    """Deterministically hash cell types, lengths, and values in primary-key order."""
    allowed = {"keypoints", "descriptors", "matches", "two_view_geometries"}
    if table not in allowed:
        raise ValueError("unsupported table digest")
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY 1")
    hasher = hashlib.sha256()
    count = 0
    for row in cursor:
        count += 1
        for value in row:
            if value is None:
                encoded, tag = b"", b"N"
            elif isinstance(value, int):
                encoded, tag = struct.pack("<q", value), b"I"
            elif isinstance(value, float):
                encoded, tag = struct.pack("<d", value), b"F"
            else:
                encoded, tag = bytes(value), b"B"
            hasher.update(tag + struct.pack("<Q", len(encoded)) + encoded)
    return {"rows": count, "sha256": hasher.hexdigest()}


def feature_digest(connection):
    return {table: table_digest(connection, table) for table in ("keypoints", "descriptors")}


def combined_feature_sha256(details):
    return hashlib.sha256(json.dumps(details, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_match_digest(connection):
    return table_digest(connection, "matches")


def full_opencv_params(calibration, shift):
    if sha256(calibration) != CALIBRATION_SHA256:
        raise ValueError("supplied NP3 calibration H5 changed")
    k = dataset(calibration, "/NP3_rgb_K", (3, 3))
    d = dataset(calibration, "/NP3_rgb_d", (5,))
    if not math.isclose(float(k[2, 2]), 1, abs_tol=1e-12) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("invalid NP3 intrinsic matrix")
    params = [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]) + shift,
              float(k[1, 2]) + shift, *map(float, d), 0.0, 0.0, 0.0]
    if len(params) != 12 or not all(math.isfinite(value) for value in params):
        raise ValueError("FULL_OPENCV parameter vector invalid")
    return params


def source_identity(source, prepared):
    source, prepared = Path(source), Path(prepared)
    provenance_path, summary_path = source / "provenance.json", source / "summary.json"
    provenance, summary = json.loads(provenance_path.read_text()), json.loads(summary_path.read_text())
    manifest = json.loads(prepared.read_text())
    images = original_run.validate(manifest)
    names = {item["name"]: item["sha256"] for item in images}
    if (provenance.get("arm") != "foreground" or provenance.get("image_hashes") != names or
            summary.get("registered") != 60 or summary.get("producer_provenance_sha256") != digest(provenance_path) or
            summary.get("input_manifest_sha256") != digest(prepared)):
        raise ValueError("cached producer/photo identity changed")
    for name, expected in summary["model_files_sha256"].items():
        if digest(Path(summary["model_dir"]) / name) != expected:
            raise ValueError("cached producer model changed")
    db = source / "database.db"
    if db.is_symlink() or not db.is_file() or db.stat().st_size > 20_000_000:
        raise ValueError("cached feature database absent, linked, or oversized")
    return {"source_database_sha256": digest(db), "source_provenance_sha256": digest(provenance_path),
            "source_summary_sha256": digest(summary_path), "source_manifest_sha256": digest(prepared),
            "source_image_hashes": names, "image_dir": str(Path(images[0]["path"]).parent),
            "source_database": str(db)}


def pair_names(connection):
    names = {key: name for key, name in connection.execute("SELECT image_id,name FROM images")}
    if len(names) != 60 or len(set(names.values())) != 60 or not set(SEED_NAMES) <= set(names.values()):
        raise ValueError("unexpected cached image inventory")
    pairs = []
    for pair_id, rows in connection.execute("SELECT pair_id,rows FROM matches ORDER BY pair_id"):
        if rows < 1:
            continue
        first, second = divmod(pair_id, MAX_IMAGE_ID)
        if first not in names or second not in names or first >= second:
            raise ValueError("invalid cached match pair ID")
        pairs.append((names[first], names[second]))
    if not 100 <= len(pairs) <= 1770:
        raise ValueError("unexpected raw-match coverage")
    return pairs, {name: image_id for image_id, name in names.items()}


def serial_reverify(db_path, pairs, ids, options):
    """Avoid 3.11.1's threaded verify_matches worker; use raw matches only."""
    import numpy as np
    import pycolmap
    db = pycolmap.Database()
    db.open(str(db_path))
    try:
        cameras = {}
        keypoints = {}
        for name, image_id in ids.items():
            image = db.read_image(image_id)
            cameras[name] = db.read_camera(image.camera_id)
            keypoints[name] = np.asarray(db.read_keypoints(image_id)[:, :2], dtype=np.float64)
        for left, right in pairs:
            raw = db.read_matches(ids[left], ids[right])
            geometry = pycolmap.estimate_two_view_geometry(
                cameras[left], keypoints[left], cameras[right], keypoints[right], raw, options)
            db.write_two_view_geometry(ids[left], ids[right], geometry)
    finally:
        db.close()


def training_geometry(model):
    import numpy as np
    residuals, parallaxes, track_lengths = [], [], []
    for image in model.images.values():
        if not image.has_pose:
            continue
        for point in image.points2D:
            if not point.has_point3D():
                continue
            xyz = model.points3D[point.point3D_id].xyz
            projection = image.project_point(xyz)
            if projection is not None:
                residuals.append(float(np.linalg.norm(np.asarray(projection) - np.asarray(point.xy))))
    for point in model.points3D.values():
        centers = [np.asarray(model.images[element.image_id].cam_from_world.inverse().translation)
                   for element in point.track.elements]
        track_lengths.append(len(centers))
        if len(centers) < 2:
            continue
        rays = np.asarray(centers) - np.asarray(point.xyz)
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        cosine = np.clip(rays @ rays.T, -1, 1)
        parallaxes.append(float(np.degrees(np.arccos(np.min(cosine)))))
    def summary(values):
        return {"count": len(values), "median": float(np.median(values)) if values else None,
                "p95": float(np.quantile(values, .95)) if values else None}
    return {"training_reprojection_px": summary(residuals),
            "max_pair_track_parallax_degrees": summary(parallaxes),
            "track_length": summary(track_lengths),
            "held_out": False}


def map_worker(source, prepared, calibration, camera_reference, output, arm):
    import pycolmap
    import numpy as np
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    identity = source_identity(source, prepared)
    if sha256(calibration) != CALIBRATION_SHA256:
        raise ValueError("calibration H5 changed")
    oracle_report = json.loads(Path(camera_reference).read_text())
    if oracle_report.get("source_archive_sha256") != "15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5" or (
            oracle_report["metadata"]["members"]["003_cracker_box/calibration.h5"]["sha256"] != CALIBRATION_SHA256):
        raise ValueError("camera oracle metadata provenance differs")
    metadata_folder = Path(camera_reference).parent / "metadata"
    for angle in range(0, 360, 6):
        member = f"003_cracker_box/poses/NP5_{angle}_pose.h5"
        expected = oracle_report["metadata"]["members"][member]["sha256"]
        if sha256(metadata_folder / "poses" / f"NP5_{angle}_pose.h5") != expected:
            raise ValueError(f"camera oracle pose metadata changed: {member}")
    output = Path(output)
    db = output / "database.db"
    shutil.copyfile(identity["source_database"], db)
    if digest(db) != identity["source_database_sha256"]:
        raise ValueError("cached DB copy differs")
    with closing(sqlite3.connect(db)) as connection:
        before_features, before_matches = feature_digest(connection), raw_match_digest(connection)
        pairs, ids = pair_names(connection)
        row = connection.execute("SELECT model,width,height,params,prior_focal_length FROM cameras").fetchall()
        if len(row) != 1 or row[0][:3] != (2, 1280, 1024):
            raise ValueError("unexpected cached shared camera")
        params = None
        if arm != "image_only_reverified":
            shift = 0.5 if arm == "calibrated_shifted" else 0.0
            params = full_opencv_params(calibration, shift)
            connection.execute("UPDATE cameras SET model=6,params=?,prior_focal_length=1",
                               (sqlite3.Binary(struct.pack("<12d", *params)),))
        expected_camera_row = ((6, 1280, 1024, struct.pack("<12d", *params), 1)
                               if params is not None else row[0])
        connection.execute("DELETE FROM two_view_geometries")
        connection.commit()
    pairs_path = output / "raw_match_pairs.txt"
    pairs_path.write_text("".join(f"{left} {right}\n" for left, right in pairs))
    pycolmap.set_random_seed(SEED)
    verification = pycolmap.TwoViewGeometryOptions()
    serial_reverify(db, pairs, ids, verification)
    with closing(sqlite3.connect(db)) as connection:
        after_features, after_matches = feature_digest(connection), raw_match_digest(connection)
        verified = table_digest(connection, "two_view_geometries")
        camera_rows = connection.execute("SELECT model,width,height,params,prior_focal_length FROM cameras").fetchall()
    if before_features != after_features or before_matches != after_matches or verified["rows"] != len(pairs):
        raise ValueError("cached feature/match rows changed or re-verification incomplete")
    if len(camera_rows) != 1 or camera_rows[0] != expected_camera_row:
        raise ValueError("copied DB camera changed unexpectedly")
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 2
    options.init_image_id1 = ids[SEED_NAMES[0]]
    options.init_image_id2 = ids[SEED_NAMES[1]]
    if params is not None:
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
    start = time.monotonic()
    models = pycolmap.incremental_mapping(str(db), identity["image_dir"], str(output / "models"), options=options)
    mapping_seconds = time.monotonic() - start
    model = max(models.values(), key=lambda item: (item.num_reg_images(), item.num_points3D())) if models else None
    registered = model.num_reg_images() if model else 0
    points3d = model.num_points3D() if model else 0
    model_dir = output / "models" / "0"
    hashes = {name: digest(model_dir / name) for name in ("cameras.bin", "images.bin", "points3D.bin")} if model and model_dir.is_dir() else {}
    trajectory = None
    orientation = None
    geometry = None
    if model and registered >= 3:
        source_centers = {image.name: np.asarray(image.cam_from_world.inverse().translation)
                          for image in model.images.values() if image.has_pose}
        target_centers, target_rotations, _, _ = reference_cameras(metadata_folder)
        common = set(source_centers) & set(target_centers)
        if len(common) >= 4:
            matrix, trajectory = fit_centers({name: source_centers[name] for name in common},
                                             {name: target_centers[name] for name in common})
            source_rotations = {image.name: np.asarray(image.cam_from_world.rotation.matrix())
                                for image in model.images.values() if image.has_pose and image.name in common}
            errors = rotation_errors(source_rotations,
                                     {name: target_rotations[name] for name in common}, matrix)
            orientation = {"median_degrees": float(np.median(list(errors.values()))),
                           "p95_degrees": float(np.quantile(list(errors.values()), .95)),
                           "by_name": errors}
        geometry = training_geometry(model)
    status = "complete" if registered >= 3 and hashes else "insufficient_sparse"
    report = {"schema": "ycb_calibrated_intrinsics_sfm_v1" if params else "ycb_image_only_reverified_sfm_v1",
              "lane": "calibrated_intrinsics_only" if params else "image_only_cached_feature_replay",
              "provenance_class": "oracle-assisted intrinsics; image-estimated poses" if params else "image-only intrinsics and poses",
              "status": status, "arm": arm, "registered": registered, "points3D": points3d,
              "model_count": len(models), "model_dir": str(model_dir.resolve()) if hashes else None,
              "model_files_sha256": hashes, **identity,
              "calibration_h5_sha256": CALIBRATION_SHA256 if params else None,
              "camera_reference_report_sha256": digest(camera_reference),
              "full_opencv_params": params,
              "pixel_center_convention": ("assumed OpenCV integer-center source; cx/cy shifted +0.5 for COLMAP"
                                          if arm == "calibrated_shifted" else
                                          "unshifted source sensitivity; no mesh-score selection" if params else
                                          "image-only SIMPLE_RADIAL source camera"),
              "cached_features_sha256_before": combined_feature_sha256(before_features),
              "cached_features_sha256_after": combined_feature_sha256(after_features),
              "raw_matches_sha256_before": before_matches["sha256"],
              "raw_matches_sha256_after": after_matches["sha256"],
              "verified_geometry_sha256_after": verified["sha256"],
              "table_evidence": {"features_before": before_features, "features_after": after_features,
                                 "raw_matches_before": before_matches, "raw_matches_after": after_matches,
                                 "verified_geometry_after": verified},
              "reverification_options": verification.todict(),
              "reverification_method": "serial per-pair pycolmap.estimate_two_view_geometry from cached raw matches; no feature rematching",
              "reverification_thread_limit": "single Python loop; pinned 3.11.1 verify_matches threaded worker crashed in retained first attempt",
              "seed_pair": list(SEED_NAMES), "random_seed": SEED,
              "mapping_options": options.todict(),
              "pycolmap_version": pycolmap.__version__,
              "pycolmap_binary_sha256": digest(pycolmap._core.__file__),
              "runner_sha256": digest(Path(__file__)),
              "mapping_seconds": mapping_seconds,
              "camera_center_fit_to_berkeley": trajectory,
              "orientation_error_to_berkeley": orientation,
              "training_geometry": geometry,
              "reference_mesh_used": False, "supplied_poses_used_for_mapping": False,
              "metric_google_mesh_accuracy_claim_allowed": False}
    if digest(identity["source_database"]) != identity["source_database_sha256"] or (
            source_identity(source, prepared)["source_image_hashes"] != identity["source_image_hashes"]):
        raise ValueError("source database or photos changed during mapper")
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (output / "summary.json").write_text(json.dumps({"model_count": len(models),
                                                     "registered": registered, "points3D": points3d}) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--camera-reference", type=Path, default=CAMERA_REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        report = map_worker(args.source, args.prepared, args.calibration, args.camera_reference,
                            args.output, args.arm)
        print(json.dumps({"arm": args.arm, "status": report["status"],
                          "registered": report["registered"], "points3D": report["points3D"]}))
        return 0 if report["status"] == "complete" else 1
    identity = source_identity(args.source, args.prepared)
    if sha256(args.calibration) != CALIBRATION_SHA256:
        raise ValueError("calibration H5 changed")
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / args.arm
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    original_run.OUTPUT_CAP = ARM_CAP
    cmd = [str(PYTHON), str(Path(__file__)), "--worker", "--arm", args.arm,
           "--source", str(args.source), "--prepared", str(args.prepared),
           "--calibration", str(args.calibration), "--camera-reference", str(args.camera_reference),
           "--output", str(dest)]
    trial = run_trial(cmd, dest, 590, args.output)
    print(json.dumps({"arm": args.arm, **trial}))
    return 0 if trial["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
