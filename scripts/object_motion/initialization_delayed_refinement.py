#!/usr/bin/env python3
"""One two-phase image-only YCB experiment: 60-view fixed-K model, then BA."""

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

from scripts.object_motion import calibration_ablation as ablation
from scripts.object_motion import initialization_recovery as recovery
from scripts.object_motion import run as bounded


PRODUCER = ROOT / "build-opencv/object-motion/initialization-recovery-004-fixed-intrinsics/cached_seed_trial/report.json"
PRODUCER_SHA256 = "9a7ddeac8b99ab3ec41a512aec0b852b555aa6176860d2047460e3f87f910b63"
OUTPUT = ROOT / "build-opencv/object-motion/initialization-recovery-005-delayed-refinement"
TIMEOUT_SECONDS = 60
OUTPUT_CAP = 100 * 1024 * 1024


def ba_options():
    import pycolmap
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = True
    options.refine_extrinsics = True
    options.use_gpu = False
    options.print_summary = True
    options.solver_options.max_num_iterations = 50
    options.solver_options.max_solver_time_in_seconds = 50.0
    options.solver_options.num_threads = 2
    options.solver_options.trust_region_problem_dump_directory = str(ROOT / ".local-tools/tmp")
    return options


def finite_training(report):
    residual = report["training_reprojection_px"]
    return (residual["count"] > 0 and residual["median"] is not None and
            residual["p95"] is not None and all(math.isfinite(float(value))
            for value in (residual["median"], residual["p95"])))


def verified_producer():
    if bounded.digest(PRODUCER) != PRODUCER_SHA256:
        raise ValueError("sealed fixed-intrinsics producer report changed")
    report = json.loads(PRODUCER.read_text())
    if (report.get("schema") != "ycb_image_only_fixed_initial_intrinsics_v1" or
            report.get("status") != "complete" or report.get("registered") != 60 or
            report.get("source_database_sha256") != recovery.REBOUND_DATABASE_SHA256 or
            report.get("cache_mutation_audit_sha256") != bounded.digest(recovery.AUDIT) or
            report.get("seed_pair") != list(recovery.SEED_NAMES) or
            report.get("camera_params") != [1536.0, 640.0, 512.0, 0.0] or
            report.get("reference_mesh_used") is not False or
            report.get("supplied_intrinsics_used") is not False):
        raise ValueError("fixed-intrinsics producer contract mismatch")
    model = Path(report["model_dir"])
    expected = report.get("model_files_sha256", {})
    if set(expected) != {"cameras.bin", "images.bin", "points3D.bin"}:
        raise ValueError("incomplete fixed model hashes")
    for name, value in expected.items():
        if bounded.digest(model / name) != value:
            raise ValueError(f"fixed model changed: {name}")
    return report, model


def worker(output):
    import pycolmap
    producer, source_model = verified_producer()
    output = Path(output)
    input_copy = output / "input_model_copy"
    input_copy.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copyfile(source_model / name, input_copy / name)
        if bounded.digest(input_copy / name) != producer["model_files_sha256"][name]:
            raise ValueError("input sparse model copy differs")
    model = pycolmap.Reconstruction(str(input_copy))
    if model.num_reg_images() != 60 or model.num_points3D() != producer["points3D"]:
        raise ValueError("loaded sparse model count differs")
    camera = next(iter(model.cameras.values())) if len(model.cameras) == 1 else None
    if camera is None or not recovery.plausible_intrinsics(
            camera.model.name, camera.width, camera.height, camera.params):
        raise ValueError("input fixed heuristic camera is not plausible")
    before_params = list(map(float, camera.params))
    before_geometry = ablation.training_geometry(model)
    if not finite_training(before_geometry):
        raise ValueError("input sparse training reprojection is invalid")
    options = ba_options()
    start = time.monotonic()
    pycolmap.bundle_adjustment(model, options)
    ba_seconds = time.monotonic() - start
    after_camera = next(iter(model.cameras.values())) if len(model.cameras) == 1 else None
    after_params = list(map(float, after_camera.params)) if after_camera else None
    after_geometry = ablation.training_geometry(model)
    counts_preserved = model.num_reg_images() == 60 and model.num_points3D() == producer["points3D"]
    plausible = (recovery.plausible_intrinsics(after_camera.model.name, after_camera.width,
                                               after_camera.height, after_params)
                 if after_camera else False)
    principal_fixed = after_params is not None and after_params[1:3] == before_params[1:3]
    residual_valid = finite_training(after_geometry)
    status = "complete" if counts_preserved and plausible and principal_fixed and residual_valid else "failed_gate"
    refined = output / "refined_model"
    refined.mkdir()
    model.write(str(refined))
    hashes = {name: bounded.digest(refined / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}
    verified_producer()  # Detect a source change during BA before sealing.
    report = {"schema": "ycb_image_only_delayed_self_calibration_v1",
              "lane": "exploratory_image_only_cached_sparse_refinement",
              "status": status, "registered": model.num_reg_images(), "points3D": model.num_points3D(),
              "source_producer_report_sha256": PRODUCER_SHA256,
              "source_model_files_sha256": producer["model_files_sha256"],
              "input_model_copy": str(input_copy.resolve()),
              "input_model_copy_sha256": {name: bounded.digest(input_copy / name) for name in hashes},
              "refined_model_dir": str(refined.resolve()), "refined_model_files_sha256": hashes,
              "source_database_sha256": producer["source_database_sha256"],
              "source_cache_mutation_audit_sha256": producer["cache_mutation_audit_sha256"],
              "seed_pair": producer["seed_pair"], "before_camera_params": before_params,
              "after_camera_params": after_params,
              "before_training_geometry": before_geometry,
              "after_training_geometry": after_geometry,
              "gates": {"registered_and_point_counts_preserved": counts_preserved,
                        "finite_plausible_intrinsics": plausible,
                        "principal_point_fixed": principal_fixed,
                        "finite_training_residuals": residual_valid},
              "bundle_adjustment_options": options.todict(), "bundle_adjustment_seconds": ba_seconds,
              "pycolmap_version": pycolmap.__version__,
              "pycolmap_binary_sha256": bounded.digest(pycolmap._core.__file__),
              "runner_sha256": bounded.digest(Path(__file__)),
              "features_or_matches_recomputed": False, "mapping_rerun": False,
              "supplied_intrinsics_or_poses_used": False, "reference_mesh_used": False,
              "held_out_quality_claim_allowed": False, "shape_accuracy_claim_allowed": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (output / "summary.json").write_text(json.dumps({"model_count": 1, "registered": model.num_reg_images(),
                                                    "points3D": model.num_points3D(), "status": status}) + "\n")
    print(json.dumps({"status": status, "registered": model.num_reg_images(),
                      "before_f": before_params[0], "after_f": after_params[0],
                      "ba_seconds": ba_seconds}))
    return 0 if status == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.output)
    verified_producer()
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "ba_trial"
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    bounded.OUTPUT_CAP = OUTPUT_CAP
    cmd = [str(recovery.PYTHON), str(Path(__file__)), "--worker", "--output", str(dest)]
    result = bounded.run_trial(cmd, dest, TIMEOUT_SECONDS, args.output)
    print(json.dumps(result))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
