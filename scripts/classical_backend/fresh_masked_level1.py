"""Frozen level-1, 10-minute masked OpenMVS fallback for the reviewed 60-view model.

This is a separate runner so a concurrent level-0 run can keep its source seal.
It shares the exact source/camera gate and mask-warp implementation with that run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

from PIL import Image

from scripts.classical_backend import fresh_masked_complete as base
from scripts.classical_backend.calibrated_control import (
    checked_dmap_count, mask_count, native_depth_size, validate_undistorted)
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes
from scripts.classical_backend.run import (
    BIN, RESERVE, StageError, TOOLS, checked_ply, digest, folder_bytes,
    nonempty_bytes, obj_counts, reserve_paths_for_devices, stage, tool_path)


PROFILE = "fresh_masked_level1_bounded_10m"
CAP = 3 << 29  # 1.5 GiB
RSS_CAP = 6 << 30
TIMEOUT_SECONDS = 10 * 60
LOG_CAP = 32 << 20
LEVEL = 1
MIN_RESOLUTION = 600
MAX_RESOLUTION = 1600
GEOMETRIC_ITERS = 2


def depth_preflight(sizes: dict[str, tuple[int, int]], existing_bytes: int) -> dict:
    if set(sizes) != base.EXPECTED_NAMES:
        raise ValueError("undistorted image inventory differs from 60 source views")
    rows = {name: native_depth_size(*size, LEVEL, MIN_RESOLUTION, MAX_RESOLUTION)
            for name, size in sizes.items()}
    if any(row["actual_level"] != LEVEL for row in rows.values()):
        raise ValueError("native minimum would undo requested level-1 downsampling")
    pixels = sum(row["pixel_budget_size"][0] * row["pixel_budget_size"][1]
                 for row in rows.values())
    predicted = existing_bytes + 2 * (20 * pixels + 4096 * len(rows)) + (96 << 20)
    if predicted > CAP:
        raise ValueError("predicted level-1 DMAP peak exceeds 1.5 GiB cap")
    return {"requested_level": LEVEL, "minimum": MIN_RESOLUTION,
            "maximum": MAX_RESOLUTION,
            "predicted_two_generation_peak_bytes": predicted,
            "hard_output_cap_bytes": CAP, "images": rows}


def run(args: argparse.Namespace) -> dict:
    output = args.output
    if output.is_symlink() or output.exists() or output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be fresh under a real existing directory")
    if any(getattr(args, key).is_symlink() for key in
           ("model", "images", "pose_masks", "manifest", "source_report", "camera_gate")):
        raise ValueError("linked source paths are unsupported")
    output = output.resolve()
    inputs = {key: getattr(args, key).resolve() for key in
              ("model", "images", "pose_masks", "manifest", "source_report", "camera_gate")}
    expected_model = dict(zip(MODEL_FILES, args.model_sha256))
    bound = base.input_binding(inputs["model"], inputs["images"], inputs["pose_masks"],
                               inputs["manifest"], inputs["source_report"], inputs["camera_gate"],
                               args.source_report_sha256, args.camera_gate_sha256, expected_model)
    binaries = args.binary_dir.resolve()
    binary_hashes = {name: digest(tool_path(binaries, name)) for name in TOOLS}
    import pycolmap
    python = Path(sys.executable)
    reserve_paths = reserve_paths_for_devices(output.parent.stat().st_dev,
                                             Path(__file__).resolve().parents[2].stat().st_dev)
    if (shutil.disk_usage(output.parent).free < RESERVE + CAP + (256 << 20) or
            any(shutil.disk_usage(path).free < RESERVE for path in reserve_paths)):
        raise ValueError("dual 10 GiB floors or 1.5 GiB output headroom unavailable")
    output.mkdir()
    report = {"schema": base.SCHEMA, "profile": PROFILE, "status": "running",
              "source": {**bound, "model": str(inputs["model"]),
                         "images": str(inputs["images"]),
                         "pose_masks": str(inputs["pose_masks"]),
                         "manifest": str(inputs["manifest"]),
                         "producer": str(inputs["source_report"]),
                         "camera_gate": str(inputs["camera_gate"])},
              "software": {"runner_sha256": digest(Path(__file__)),
                           "base_gate_sha256": digest(Path(base.__file__)),
                           "shared_runner_sha256": digest(Path(__file__).with_name("run.py")),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_binary_sha256": digest(Path(pycolmap._core.__file__)),
                           "openmvs_binaries": binary_hashes},
              "native_options": {"resolution_level": LEVEL,
                                 "min_resolution": MIN_RESOLUTION,
                                 "max_resolution": MAX_RESOLUTION,
                                 "geometric_iters": GEOMETRIC_ITERS,
                                 "tower_mode": 4, "ignore_mask_label": 0,
                                 "compute_device": "cpu", "max_threads": 2,
                                 "refine_resolution_level": 1, "refine_scales": 1},
              "limits": {"max_output_bytes": CAP, "max_child_rss_bytes": RSS_CAP,
                         "timeout_seconds": TIMEOUT_SECONDS, "max_log_bytes": LOG_CAP,
                         "min_free_bytes_each_volume": RESERVE,
                         "extra_reserve_paths": [str(path) for path in reserve_paths]},
              "stages": [], "quality_accepted": False, "metric_scale_verified": False,
              "shipping_approved": False,
              "provenance_scope": "fresh sparse producer plus separate level-1 masked native continuation"}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(name: str, command: list, validate) -> None:
        result = stage(output, name, [str(part) for part in command], deadline,
                       CAP, LOG_CAP, RSS_CAP, extra_reserve_paths=reserve_paths)
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    deadline = time.monotonic() + TIMEOUT_SECONDS
    save()
    common = ["--max-threads", "2", "--working-folder", output]
    try:
        (output / "images").mkdir()
        sparse = output / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in sorted(base.EXPECTED_NAMES):
            target = output / "images" / name
            shutil.copyfile(inputs["images"] / name, target)
            if digest(target) != bound["photo_sha256"][name]:
                raise ValueError(f"copied photograph differs: {name}")
            if (folder_bytes(output) > CAP or shutil.disk_usage(output).free < RESERVE or
                    any(shutil.disk_usage(path).free < RESERVE for path in reserve_paths)):
                raise ValueError("input copy exceeded byte cap or disk floor")
        for name in MODEL_FILES:
            shutil.copyfile(inputs["model"] / name, sparse / name)
        if model_hashes(sparse) != bound["model_sha256"]:
            raise ValueError("copied sparse model differs")
        (output / "inputs.json").write_text(json.dumps(
            [{"name": name} for name in sorted(base.EXPECTED_NAMES)]))
        report["stages"].append({"name": "copy_inputs", "status": "complete", "photos": 60})
        save()
        execute("undistort", [python, "-m", "scripts.classical_backend.run",
                              "--worker", "undistort", "--output", output,
                              "--max-image-size", "1600"],
                lambda: validate_undistorted(output, sorted(base.EXPECTED_NAMES)))
        execute("warp_masks", [python, "-m", "scripts.classical_backend.dense_masks",
                               "--source-model", sparse,
                               "--undistorted-model", output / "dense" / "sparse",
                               "--source-masks", inputs["pose_masks"],
                               "--undistorted-images", output / "dense" / "images",
                               "--output", output / "masks", "--manifest", inputs["manifest"]],
                lambda: {"count": mask_count(output / "masks" / "report.json", 60),
                         "report_sha256": digest(output / "masks" / "report.json")})
        sizes = {}
        for name in base.EXPECTED_NAMES:
            with Image.open(output / "dense" / "images" / name) as image:
                sizes[name] = image.size
        report["native_resolution_preflight"] = depth_preflight(sizes, folder_bytes(output))
        save()
        dense = output / "dense"
        execute("import", [tool_path(binaries, "InterfaceCOLMAP"), "-i", dense,
                           "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", [tool_path(binaries, "DensifyPointCloud"), "-i", output / "scene.mvs",
                            "-o", output / "dense.mvs", "--resolution-level", str(LEVEL),
                            "--min-resolution", str(MIN_RESOLUTION),
                            "--max-resolution", str(MAX_RESOLUTION),
                            "--geometric-iters", str(GEOMETRIC_ITERS), "--tower-mode", "4",
                            "--mask-path", output / "masks", "--ignore-mask-label", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("dmap_mask_check", [python, "-m", "scripts.classical_backend.calibrated_control",
                                    "--worker", "dmap_check", "--output", output],
                lambda: {"views": checked_dmap_count(output / "dmap-camera-check.json", 60),
                         "positive_depth_outside_mask": json.loads(
                             (output / "dmap-camera-check.json").read_text())["positive_depth_outside_mask"]})
        execute("mesh", [tool_path(binaries, "ReconstructMesh"), "-i", output / "dense.mvs",
                         "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        execute("refine", [tool_path(binaries, "RefineMesh"), "-i", output / "dense.mvs",
                           "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                           "--resolution-level", "1", "--scales", "1", *common],
                lambda: {"faces": checked_ply(output / "refined.ply", "face")})
        execute("texture", [tool_path(binaries, "TextureMesh"), "-i", output / "dense.mvs",
                            "-m", output / "refined.ply", "-o", output / "textured.mvs",
                            "--export-type", "obj", *common],
                lambda: dict(zip(("vertices", "faces"), obj_counts(output / "textured.obj"))))
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = shutil.disk_usage(output).free
    report["extra_free_bytes_after"] = {str(path): shutil.disk_usage(path).free
                                        for path in reserve_paths}
    artifact_names = {"scene.mvs", "dense.mvs", "dense.ply", "mesh.mvs", "mesh.ply",
                      "refined.mvs", "refined.ply", "dmap-camera-check.json"}
    try:
        report["artifact_sha256"] = {path.name: digest(path) for path in sorted(output.iterdir())
                                     if path.is_file() and
                                     (path.name in artifact_names or path.name.startswith("textured") and
                                      path.suffix.lower() in {".mvs", ".obj", ".mtl", ".png", ".jpg", ".jpeg"})}
    except Exception as error:
        report.update(status="failed", failure=f"could not seal native artifact hashes: {error}")
    try:
        rebound = base.input_binding(inputs["model"], inputs["images"], inputs["pose_masks"],
                                     inputs["manifest"], inputs["source_report"], inputs["camera_gate"],
                                     args.source_report_sha256, args.camera_gate_sha256, expected_model)
        report["sources_unchanged"] = (bound == rebound and
            digest(Path(__file__)) == report["software"]["runner_sha256"] and
            digest(Path(base.__file__)) == report["software"]["base_gate_sha256"] and
            digest(Path(__file__).with_name("run.py")) == report["software"]["shared_runner_sha256"] and
            digest(Path(pycolmap._core.__file__)) == report["software"]["pycolmap_binary_sha256"] and
            all(digest(tool_path(binaries, name)) == value for name, value in binary_hashes.items()))
    except Exception:
        report["sources_unchanged"] = False
    if (not report["sources_unchanged"] or report["output_bytes"] > CAP or
            report["free_bytes_after"] < RESERVE or
            any(value < RESERVE for value in report["extra_free_bytes_after"].values())):
        report.update(status="failed", failure="source changed or output/disk hard limit exceeded")
    save()
    report["output_bytes"] = folder_bytes(output)
    if report["output_bytes"] > CAP:
        report.update(status="failed", failure="output byte hard limit exceeded after final report")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "images", "pose-masks", "manifest", "source-report",
                 "camera-gate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-report-sha256", type=base.sha256_arg, required=True)
    parser.add_argument("--camera-gate-sha256", type=base.sha256_arg, required=True)
    parser.add_argument("--model-sha256", nargs=3, type=base.sha256_arg, required=True,
                        metavar=("CAMERAS", "IMAGES", "POINTS3D"))
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
