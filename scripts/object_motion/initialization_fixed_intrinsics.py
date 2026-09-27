#!/usr/bin/env python3
"""One exploratory image-only YCB mapper with frozen heuristic initial intrinsics."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.object_motion import initialization_recovery as base
from scripts.object_motion import run as bounded


OUTPUT = ROOT / "build-opencv/object-motion/initialization-recovery-004-fixed-intrinsics"
AUDIT_SHA256 = "14116ca9768a40d5f3265cbd22b14a7043f216be8f3f671e2bdbba1caacc0db5"
INITIAL_PARAMS = [1536.0, 640.0, 512.0, 0.0]
TIMEOUT_SECONDS = 180
OUTPUT_CAP = 100 * 1024 * 1024


def fixed_options(ids):
    import pycolmap
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = False
    options.max_num_models = 1
    options.min_model_size = 2
    options.init_image_id1 = ids[base.SEED_NAMES[0]]
    options.init_image_id2 = ids[base.SEED_NAMES[1]]
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    return options


def fixed_intrinsics(model_name, width, height, params):
    return (base.plausible_intrinsics(model_name, width, height, params) and
            all(math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-9)
                for actual, expected in zip(params, INITIAL_PARAMS)))


def worker(output):
    import pycolmap
    identity = base.verify_frozen_inputs(base.SOURCE, base.PREPARED, base.DIAGNOSTIC)
    if bounded.digest(base.AUDIT) != AUDIT_SHA256:
        raise ValueError("rebound source mutation audit changed")
    source_db = Path(identity["source_database"])
    source_wal = Path(str(source_db) + "-wal")
    if source_wal.exists() and source_wal.stat().st_size != 0:
        raise ValueError("source DB has nonempty WAL; cannot copy main file alone")
    output = Path(output)
    db = output / "database.db"
    shutil.copyfile(source_db, db)
    copied_before = bounded.digest(db)
    if copied_before != base.REBOUND_DATABASE_SHA256:
        raise ValueError("rebound cached DB bytes changed")
    logical_before = base.read_cache_logical(db)
    ids = {name: image_id for image_id, name in logical_before["names"].items()}
    if len(ids) != 60 or not set(base.SEED_NAMES) <= set(ids):
        raise ValueError("unexpected cached image inventory")
    options = fixed_options(ids)
    pycolmap.set_random_seed(base.RANDOM_SEED)
    start = time.monotonic()
    models = pycolmap.incremental_mapping(str(db), identity["image_dir"],
                                          str(output / "models"), options=options)
    mapping_seconds = time.monotonic() - start
    model = max(models.values(), key=lambda item: (item.num_reg_images(), item.num_points3D())) if models else None
    registered = model.num_reg_images() if model else 0
    points = model.num_points3D() if model else 0
    camera = next(iter(model.cameras.values())) if model and len(model.cameras) == 1 else None
    params = list(map(float, camera.params)) if camera is not None else None
    fixed_gate = (fixed_intrinsics(camera.model.name, camera.width, camera.height, params)
                  if camera is not None else False)
    model_dir = output / "models" / "0"
    hashes = ({name: bounded.digest(model_dir / name) for name in
               ("cameras.bin", "images.bin", "points3D.bin")}
              if model and model_dir.is_dir() else {})
    copied_after = bounded.digest(db)
    copy_wal = Path(str(db) + "-wal")
    wal_empty = not copy_wal.exists() or copy_wal.stat().st_size == 0
    logical_after = base.read_cache_logical(db) if wal_empty else None
    tables_unchanged = (wal_empty and logical_before["tables"] == logical_after["tables"] and
                        logical_before["schemas_sha256"] == logical_after["schemas_sha256"])
    status = "complete" if registered >= 3 and fixed_gate and hashes and tables_unchanged else (
        "no_model" if not model else "insufficient_three_view" if registered < 3 else
        "fixed_intrinsics_gate_failed" if not fixed_gate else "copied_cache_changed")
    again = base.verify_frozen_inputs(base.SOURCE, base.PREPARED, base.DIAGNOSTIC)
    if again["source_database_sha256"] != identity["source_database_sha256"]:
        raise RuntimeError("rebound source database changed during mapping")
    report = {"schema": "ycb_image_only_fixed_initial_intrinsics_v1",
              "lane": "image_only_cached_foreground_features_fixed_heuristic_intrinsics",
              "status": status, "registered": registered, "points3D": points,
              "model_count": len(models), "model_dir": str(model_dir.resolve()) if hashes else None,
              "model_files_sha256": hashes, "seed_pair": list(base.SEED_NAMES),
              "seed_image_ids": [ids[name] for name in base.SEED_NAMES], "random_seed": base.RANDOM_SEED,
              "source_database_sha256": identity["source_database_sha256"],
              "copied_database_sha256_before": copied_before,
              "copied_database_sha256_after": copied_after,
              "copied_cache_tables_unchanged": tables_unchanged,
              "copied_cache_logical_before": logical_before["tables"],
              "copied_cache_logical_after": logical_after["tables"] if logical_after else None,
              "cache_mutation_audit_sha256": AUDIT_SHA256,
              "seed_diagnostic_sha256": bounded.digest(base.DIAGNOSTIC),
              "source_image_hashes": identity["source_image_hashes"],
              "source_manifest_sha256": identity["source_manifest_sha256"],
              "source_provenance_sha256": identity["source_provenance_sha256"],
              "source_summary_sha256": identity["source_summary_sha256"],
              "camera_model": camera.model.name if camera else None,
              "camera_size": [camera.width, camera.height] if camera else None,
              "camera_params": params,
              "intrinsics_gate": {"passed": fixed_gate, "initial_and_required_final_params": INITIAL_PARAMS,
                                  "origin": "photo-only source DB heuristic, not Berkeley calibration"},
              "three_view_gate": {"passed": registered >= 3, "registered": registered},
              "mapping_options": options.todict(), "mapping_seconds": mapping_seconds,
              "pycolmap_version": pycolmap.__version__,
              "pycolmap_binary_sha256": bounded.digest(pycolmap._core.__file__),
              "runner_sha256": bounded.digest(Path(__file__)),
              "changed_from_003": "same seed/cache/mapper settings except global/local BA focal/principal/extra and absolute-pose focal/extra refinement all disabled",
              "cache_accounting": "cached original foreground photos/features/descriptors/raw matches/verified geometry; mapping only",
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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.output)
    base.verify_frozen_inputs(base.SOURCE, base.PREPARED, base.DIAGNOSTIC)
    if bounded.digest(base.AUDIT) != AUDIT_SHA256:
        raise ValueError("rebound cache audit changed")
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "cached_seed_trial"
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    bounded.OUTPUT_CAP = OUTPUT_CAP
    cmd = [str(base.PYTHON), str(Path(__file__)), "--worker", "--output", str(dest)]
    result = bounded.run_trial(cmd, dest, TIMEOUT_SECONDS, args.output)
    print(json.dumps(result))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
