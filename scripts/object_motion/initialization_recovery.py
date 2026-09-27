#!/usr/bin/env python3
"""One predeclared cached image-only YCB mapper recovery; no seed sweep."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.object_motion import calibration_ablation as ablation
from scripts.object_motion import run as bounded


SOURCE = ROOT / "build-opencv/object-motion/foreground"
PREPARED = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
DIAGNOSTIC = ROOT / "build-opencv/object-motion/initialization-diagnostic-002/report.json"
AUDIT = ROOT / "build-opencv/object-motion/initialization-recovery-cache-audit-002/report.json"
OUTPUT = ROOT / "build-opencv/object-motion/initialization-recovery-003"
PYTHON = ROOT / ".local-tools/colmap-sparse/venv/bin/python"
SEED_NAMES = ("NP3_018.jpg", "NP3_030.jpg")
DIAGNOSTIC_SHA256 = "962c55e7e75c6f6cdbaebcc5f065075615e6b519113d486e882f704e219fd979"
HISTORICAL_DATABASE_SHA256 = "a36cc364bf9b7d7a8357c12ee3619002dd2c85696f09b5921d1f9e926f2e3ca7"
REBOUND_DATABASE_SHA256 = "5850e6c508df3b296169a25334abdbae2f5f7cdfc3628a78d7fd21717e457562"
FIRST_PREMAPPER_COPY = ROOT / "build-opencv/object-motion/initialization-recovery-001/cached_seed_trial/database.db"
TABLES = ("cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries")
RANDOM_SEED = 20260927
TIMEOUT_SECONDS = 180
OUTPUT_CAP = 100 * 1024 * 1024


def plausible_intrinsics(model_name, width, height, params):
    """Predeclared conservative gate for this shared 1280x1024 SIMPLE_RADIAL arm."""
    if model_name != "SIMPLE_RADIAL" or (width, height) != (1280, 1024) or len(params) != 4:
        return False
    if not all(math.isfinite(float(value)) for value in params):
        return False
    focal, cx, cy, k = map(float, params)
    return (0.5 * width <= focal <= 1.5 * width and
            0 <= cx <= width and 0 <= cy <= height and abs(k) <= 0.5)


def logical_digest(connection, table):
    if table not in TABLES:
        raise ValueError("unsupported cache table")
    hasher = hashlib.sha256()
    count = 0
    for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1"):
        count += 1
        for value in row:
            if value is None:
                tag, encoded = b"N", b""
            elif isinstance(value, int):
                tag, encoded = b"I", struct.pack("<q", value)
            elif isinstance(value, float):
                tag, encoded = b"F", struct.pack("<d", value)
            elif isinstance(value, str):
                tag, encoded = b"S", value.encode("utf-8")
            else:
                tag, encoded = b"B", bytes(value)
            hasher.update(tag + struct.pack("<Q", len(encoded)) + encoded)
    return {"rows": count, "sha256": hasher.hexdigest()}


def read_cache_logical(path):
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("cached database integrity check failed")
        tables = {name: logical_digest(connection, name) for name in TABLES}
        schemas = {name: connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0]
                   for name in TABLES}
        camera_rows = connection.execute("SELECT model,width,height,params,prior_focal_length FROM cameras").fetchall()
        names = dict(connection.execute("SELECT image_id,name FROM images"))
        return {"tables": tables, "schemas_sha256": hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest(),
                "camera_rows": camera_rows, "names": names}
    finally:
        connection.close()


def audit_cache(output, source=SOURCE, prepared=PREPARED, diagnostic=DIAGNOSTIC):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free")
    identity = ablation.source_identity(source, prepared)
    current_path = Path(identity["source_database"])
    source_wal = Path(str(current_path) + "-wal")
    if source_wal.exists() and source_wal.stat().st_size != 0:
        raise ValueError("source DB has nonempty WAL; main-file hash is not a complete snapshot")
    if (identity["source_database_sha256"] != REBOUND_DATABASE_SHA256 or
            bounded.digest(diagnostic) != DIAGNOSTIC_SHA256 or
            bounded.digest(FIRST_PREMAPPER_COPY) != "3ca3e78656a8346e816dff23a8a6248eac3a0ba6c3266ce96762ec16660693de"):
        raise ValueError("cache mutation evidence changed")
    original_diagnostic = json.loads(Path(diagnostic).read_text())
    if original_diagnostic["source_databases_sha256"]["foreground"] != HISTORICAL_DATABASE_SHA256:
        raise ValueError("historical source SHA not in original diagnostic")
    current = read_cache_logical(current_path)
    first_copy = read_cache_logical(FIRST_PREMAPPER_COPY)
    if current["tables"] != first_copy["tables"] or current["schemas_sha256"] != first_copy["schemas_sha256"]:
        raise ValueError("modified source and preserved pre-mapper copy differ logically")
    if (len(current["camera_rows"]) != 1 or current["camera_rows"][0] !=
            (2, 1280, 1024, struct.pack("<4d", 1536.0, 640.0, 512.0, 0.0), 0) or
            len(current["names"]) != 60 or
            set(current["names"].values()) != set(identity["source_image_hashes"])):
        raise ValueError("changed cache camera/image inventory not the frozen image-only input")
    known_original = {"keypoints": "2a38e2c1386af47c1f588fac039e9fdbb73a6ede22ad13fe3403361b1ac7607d",
                      "descriptors": "f5eee590f63544692ae675e891af148f53b7a12995c9006e0d5df62d74a25421",
                      "matches": "669e1ea65e3ac55fbdb22fbf84b6c4a7e4df6ee4d34105bf1055beca43e0467d",
                      "two_view_geometries": "c24775c3a016e22de6c520faec1e9ce0f1667fcb0b66dedaa28e381ff571af1c"}
    for name, expected in known_original.items():
        if current["tables"][name]["sha256"] != expected:
            raise ValueError(f"original logical evidence differs: {name}")
    report = {"schema": "ycb_cache_mutation_audit_v1", "status": "historical_database_bytes_changed_by_agent_api_fixture",
              "historical_source_database_sha256": HISTORICAL_DATABASE_SHA256,
              "current_source_database_sha256": identity["source_database_sha256"],
              "first_failed_premapper_copy_sha256": bounded.digest(FIRST_PREMAPPER_COPY),
              "source_database_path": str(current_path), "first_failed_premapper_copy_path": str(FIRST_PREMAPPER_COPY),
              "source_database_size": current_path.stat().st_size,
              "source_sidecars": {
                  suffix: {"bytes": path.stat().st_size, "sha256": bounded.digest(path)}
                  for suffix in ("-wal", "-shm")
                  for path in (Path(str(current_path) + suffix),) if path.is_file()},
              "first_failed_premapper_copy_sidecars": {
                  suffix: {"bytes": path.stat().st_size, "sha256": bounded.digest(path)}
                  for suffix in ("-wal", "-shm")
                  for path in (Path(str(FIRST_PREMAPPER_COPY) + suffix),) if path.is_file()},
              "current_logical_tables": current["tables"], "copy_logical_tables": first_copy["tables"],
              "current_schema_sha256": current["schemas_sha256"], "copy_schema_sha256": first_copy["schemas_sha256"],
              "known_original_logical_sha256": known_original,
              "historical_cameras_images_logical_sha256": "unknown; not recorded before accidental writable PyCOLMAP fixture open",
              "camera_row": {"model": "SIMPLE_RADIAL", "width": 1280, "height": 1024,
                             "params": [1536, 640, 512, 0], "prior_focal_length": 0},
              "image_count": len(current["names"]), "raw_match_rows": current["tables"]["matches"]["rows"],
              "verified_geometry_rows": current["tables"]["two_view_geometries"]["rows"],
              "photo_manifest_sha256": identity["source_manifest_sha256"],
              "source_provenance_sha256": identity["source_provenance_sha256"],
              "source_summary_sha256": identity["source_summary_sha256"],
              "source_image_hashes": identity["source_image_hashes"],
              "original_diagnostic_sha256": DIAGNOSTIC_SHA256,
              "cause_observed": "pinned PyCOLMAP Database.open on source during a test changed database file bytes; logical source/copy tables listed above agree",
              "not_claimed": "byte-identical replay of the historical a36cc database"}
    if bounded.digest(current_path) != REBOUND_DATABASE_SHA256:
        raise RuntimeError("source database changed during audit")
    output.mkdir()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_frozen_inputs(source, prepared, diagnostic):
    identity = ablation.source_identity(source, prepared)
    diagnostic = Path(diagnostic)
    if bounded.digest(diagnostic) != DIAGNOSTIC_SHA256:
        raise ValueError("frozen image-only seed diagnostic changed")
    evidence = json.loads(diagnostic.read_text())
    if (evidence.get("schema") != "ycb_initialization_read_only_v1" or
            evidence.get("source_databases_sha256", {}).get("foreground") != HISTORICAL_DATABASE_SHA256 or
            identity["source_database_sha256"] != REBOUND_DATABASE_SHA256 or
            evidence.get("top_candidates", [{}])[0].get("pair") != list(SEED_NAMES) or
            evidence.get("reference_metadata_used_for_ranking") is not False):
        raise ValueError("frozen seed no longer matches cached image-derived evidence")
    audit = json.loads(AUDIT.read_text())
    if (audit.get("schema") != "ycb_cache_mutation_audit_v1" or
            audit.get("current_source_database_sha256") != REBOUND_DATABASE_SHA256 or
            audit.get("historical_source_database_sha256") != HISTORICAL_DATABASE_SHA256 or
            audit.get("photo_manifest_sha256") != identity["source_manifest_sha256"]):
        raise ValueError("rebound cache audit absent or inconsistent")
    return identity


def worker(source, prepared, diagnostic, output):
    import pycolmap

    identity = verify_frozen_inputs(source, prepared, diagnostic)
    output = Path(output)
    db = output / "database.db"
    source_wal = Path(str(identity["source_database"]) + "-wal")
    if source_wal.exists() and source_wal.stat().st_size != 0:
        raise ValueError("source DB has nonempty WAL; cannot copy main file alone")
    shutil.copyfile(identity["source_database"], db)
    copied_before = bounded.digest(db)
    if copied_before != identity["source_database_sha256"]:
        raise ValueError("cached source DB copy differs")
    logical_before = read_cache_logical(db)
    ids = {name: image_id for image_id, name in logical_before["names"].items()}
    if len(ids) != 60 or not set(SEED_NAMES) <= set(ids):
        raise ValueError("unexpected cached image IDs")
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 2
    options.init_image_id1 = ids[SEED_NAMES[0]]
    options.init_image_id2 = ids[SEED_NAMES[1]]
    pycolmap.set_random_seed(RANDOM_SEED)
    start = time.monotonic()
    models = pycolmap.incremental_mapping(str(db), identity["image_dir"],
                                          str(output / "models"), options=options)
    mapping_seconds = time.monotonic() - start
    model = max(models.values(), key=lambda value: (value.num_reg_images(), value.num_points3D())) if models else None
    registered = model.num_reg_images() if model else 0
    points = model.num_points3D() if model else 0
    camera = next(iter(model.cameras.values())) if model and len(model.cameras) == 1 else None
    camera_params = list(map(float, camera.params)) if camera is not None else None
    plausible = (plausible_intrinsics(camera.model.name, camera.width, camera.height, camera_params)
                 if camera is not None else False)
    model_dir = output / "models" / "0"
    hashes = ({name: bounded.digest(model_dir / name) for name in
               ("cameras.bin", "images.bin", "points3D.bin")}
              if model and model_dir.is_dir() else {})
    status = "complete" if registered >= 3 and plausible and hashes else (
        "no_model" if not model else "insufficient_three_view" if registered < 3 else "implausible_intrinsics")
    copied_after = bounded.digest(db)
    logical_after = read_cache_logical(db)
    cache_tables_unchanged = (logical_after["tables"] == logical_before["tables"] and
                              logical_after["schemas_sha256"] == logical_before["schemas_sha256"])
    if not cache_tables_unchanged:
        status = "copied_cache_logically_changed"
    again = verify_frozen_inputs(source, prepared, diagnostic)
    if again["source_database_sha256"] != identity["source_database_sha256"]:
        raise RuntimeError("cached source changed during trial")
    report = {"schema": "ycb_image_only_seed_recovery_v1", "lane": "image_only_cached_foreground_features",
              "status": status, "registered": registered, "points3D": points, "model_count": len(models),
              "model_dir": str(model_dir.resolve()) if hashes else None,
              "model_files_sha256": hashes, "seed_pair": list(SEED_NAMES),
              "seed_image_ids": [ids[name] for name in SEED_NAMES], "random_seed": RANDOM_SEED,
              "source_database_sha256": identity["source_database_sha256"],
              "copied_database_sha256_before": copied_before,
              "copied_database_sha256_after": copied_after,
              "copied_cache_tables_unchanged": cache_tables_unchanged,
              "copied_cache_logical_before": logical_before["tables"],
              "copied_cache_logical_after": logical_after["tables"],
              "copied_cache_sidecars_after": {
                  suffix: {"bytes": path.stat().st_size, "sha256": bounded.digest(path)}
                  for suffix in ("-wal", "-shm")
                  for path in (Path(str(db) + suffix),) if path.is_file()},
              "cache_mutation_audit_sha256": bounded.digest(AUDIT),
              "source_image_hashes": identity["source_image_hashes"],
              "source_manifest_sha256": identity["source_manifest_sha256"],
              "source_provenance_sha256": identity["source_provenance_sha256"],
              "source_summary_sha256": identity["source_summary_sha256"],
              "seed_diagnostic_sha256": bounded.digest(diagnostic),
              "camera_model": camera.model.name if camera else None,
              "camera_size": [camera.width, camera.height] if camera else None,
              "camera_params": camera_params,
              "intrinsics_gate": {"passed": plausible, "rule": "SIMPLE_RADIAL 1280x1024, focal 640..1920 px, cx/cy inside image, |k|<=0.5, all finite"},
              "three_view_gate": {"passed": registered >= 3, "registered": registered},
              "mapping_options": options.todict(), "mapping_seconds": mapping_seconds,
              "pycolmap_version": pycolmap.__version__,
              "pycolmap_binary_sha256": bounded.digest(pycolmap._core.__file__),
              "runner_sha256": bounded.digest(Path(__file__)),
              "cache_accounting": "source foreground photos/features/descriptors/raw matches/verified geometry copied; mapping only; no feature extraction, descriptor matching, or geometric re-verification in this trial",
              "reference_mesh_used": False, "supplied_intrinsics_used": False,
              "supplied_poses_used": False, "general_improvement_claim_allowed": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (output / "summary.json").write_text(json.dumps({"model_count": len(models), "registered": registered,
                                                    "points3D": points, "status": status}) + "\n")
    print(json.dumps({"status": status, "registered": registered, "points3D": points,
                      "mapping_seconds": mapping_seconds}))
    return 0 if status == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audit-cache", action="store_true")
    args = parser.parse_args()
    if args.audit_cache:
        report = audit_cache(AUDIT.parent, args.source, args.prepared, args.diagnostic)
        print(json.dumps({"status": report["status"],
                          "current_source_database_sha256": report["current_source_database_sha256"]}))
        return 0
    if args.worker:
        return worker(args.source, args.prepared, args.diagnostic, args.output)
    verify_frozen_inputs(args.source, args.prepared, args.diagnostic)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "cached_seed_trial"
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    bounded.OUTPUT_CAP = OUTPUT_CAP
    cmd = [str(PYTHON), str(Path(__file__)), "--worker", "--source", str(args.source),
           "--prepared", str(args.prepared), "--diagnostic", str(args.diagnostic),
           "--output", str(dest)]
    result = bounded.run_trial(cmd, dest, TIMEOUT_SECONDS, args.output)
    print(json.dumps(result))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
