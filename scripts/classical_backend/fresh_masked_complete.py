"""Bounded masked OpenMVS continuation of a separately sealed fresh sparse run.

The independent camera review supplies a complete, dense-eligible JSON gate.
This wrapper does not infer camera accuracy from the number of registered views.
It never reads reference geometry, supplied poses, or supplied calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from scripts.classical_backend.calibrated_control import (checked_dmap_count,
    mask_count, native_depth_size, validate_undistorted)
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes
from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS,
    checked_ply, digest, folder_bytes, nonempty_bytes, obj_counts,
    reserve_paths_for_devices, stage, tool_path)


SCHEMA = "classical_fresh_masked_complete_v1"
CAP = 4 << 30
RSS_CAP = 6 << 30
TIMEOUT_SECONDS = 20 * 60
LOG_CAP = 32 << 20
EXPECTED_NAMES = {f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)}


def sha256_arg(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 hex digest")
    return value


def input_binding(model_dir: Path, photos: Path, pose_masks: Path,
                  manifest_path: Path, producer_path: Path, gate_path: Path,
                  expected_producer: str, expected_gate: str,
                  expected_model: dict[str, str]) -> dict:
    """Bind exact source bytes and reject any changed or unreviewed sparse set."""
    for path in (model_dir, photos, pose_masks, manifest_path, producer_path, gate_path):
        if path.is_symlink() or not path.exists():
            raise ValueError(f"missing or linked input: {path}")
    if digest(producer_path) != expected_producer or digest(gate_path) != expected_gate:
        raise ValueError("sparse producer or camera-review report hash differs")
    producer = json.loads(producer_path.read_text())
    gate = json.loads(gate_path.read_text())
    status = producer.get("status", {})
    profile = producer.get("profile")
    command = producer.get("command", [])
    if (profile not in {"prior_foreground_60_jpeg_replay",
                        "fresh_masked_fixed_intrinsics_delayed_ba"} or
            not isinstance(command, list) or
            (profile == "fresh_masked_fixed_intrinsics_delayed_ba" and
             ("--fresh-fixed" not in command or "--source-database" in command)) or
            producer.get("schema") != "turntable_sparse_profile_v1" or
            producer.get("independent_camera_gate_passed") is not None or
            status.get("status") != "completed" or
            status.get("inputs_unchanged") is not True or
            status.get("registered") != 60 or
            status.get("sparse_coverage_passed") is not True or
            status.get("model_hashes_verified") is not True):
        raise ValueError("fresh sparse profile is incomplete or failed coverage")
    if gate.get("schema") != "ycb_stock_berkeley_camera_diagnostic_v1":
        raise ValueError("independent rig camera diagnostic is missing")
    actual_model = model_hashes(model_dir)
    if actual_model != expected_model:
        raise ValueError("sparse binary model differs from expected hashes")
    summary_path = model_dir.parent.parent / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("fresh sparse summary is missing or linked")
    summary = json.loads(summary_path.read_text())
    if (producer_path.parent != summary_path.parent or
            status.get("model_files_sha256") != actual_model or
            digest(summary_path) != status.get("summary_sha256") or
            Path(summary.get("model_dir", "")).resolve() != model_dir or
            summary.get("model_files_sha256") != actual_model or
            gate.get("candidate_model_sha256") != actual_model or
            Path(gate.get("candidate_model_dir", "")).resolve() != model_dir or
            gate.get("coverage", {}).get("registered") != 60 or
            gate.get("coverage", {}).get("selected") != 60):
        raise ValueError("fresh producer/summary/rig gate do not bind the same 60-view model")
    comparison = gate.get("comparison", {})
    if (not isinstance(comparison.get("center_rms_over_reference_radius"), (int, float)) or
            not isinstance(comparison.get("orientation_p95_degrees"), (int, float)) or
            not np.isfinite(comparison["center_rms_over_reference_radius"]) or
            not np.isfinite(comparison["orientation_p95_degrees"]) or
            comparison["center_rms_over_reference_radius"] >= 0.05 or
            comparison["orientation_p95_degrees"] >= 10):
        raise ValueError("independent rig camera diagnostic fails frozen 5%/10 degree gate")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("images", [])
    if len(rows) != 60 or {row.get("name") for row in rows} != EXPECTED_NAMES:
        raise ValueError("original photo/mask manifest must cover exact 60-view set")
    seal = producer.get("input_seal", {})
    if (seal.get("manifest") != digest(manifest_path) or
            seal.get("photos") != {row["name"]: row["sha256"] for row in rows} or
            seal.get("masks") != {row["name"]: row["mask_sha256"] for row in rows}):
        raise ValueError("fresh sparse profile does not bind exact source photo/mask manifest")
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    names = {image.name for image in reconstruction.images.values() if image.has_pose}
    if names != EXPECTED_NAMES or reconstruction.num_points3D() < 1:
        raise ValueError("sparse binary lacks 60 registered source views or points")
    if len(reconstruction.cameras) != 1:
        raise ValueError("expected a single image-derived camera")
    for camera in reconstruction.cameras.values():
        if (camera.model.name not in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE") or
                (camera.width, camera.height) != (1280, 1024) or
                not np.isfinite(camera.params).all()):
            raise ValueError("sparse camera is unsupported for exact mask reprojection")
    photo_hashes, mask_hashes = {}, {}
    for row in rows:
        name = row["name"]
        photo, mask = photos / name, pose_masks / (name + ".png")
        if (photo.is_symlink() or mask.is_symlink() or not photo.is_file() or
                not mask.is_file() or digest(photo) != row["sha256"] or
                digest(mask) != row["mask_sha256"]):
            raise ValueError(f"source photograph or mask differs from manifest: {name}")
        with Image.open(photo) as image:
            if image.size != (1280, 1024):
                raise ValueError(f"source photograph size differs: {name}")
        photo_hashes[name], mask_hashes[name] = row["sha256"], row["mask_sha256"]
    return {"producer_sha256": expected_producer, "summary_sha256": digest(summary_path),
            "camera_gate_sha256": expected_gate,
            "manifest_sha256": digest(manifest_path), "model_sha256": actual_model,
            "photo_sha256": photo_hashes, "pose_mask_sha256": mask_hashes,
            "sparse_points": reconstruction.num_points3D()}


def depth_preflight(sizes: dict[str, tuple[int, int]], existing_bytes: int) -> dict:
    if set(sizes) != EXPECTED_NAMES:
        raise ValueError("undistorted image inventory differs from 60 source views")
    rows = {name: native_depth_size(*size, 0, 600, 1600)
            for name, size in sizes.items()}
    pixels = sum(row["pixel_budget_size"][0] * row["pixel_budget_size"][1]
                 for row in rows.values())
    # Two raw/geometric DMAP generations at 20 bytes/pixel plus scene slack.
    predicted = existing_bytes + 2 * (20 * pixels + 4096 * len(rows)) + (96 << 20)
    if predicted > CAP:
        raise ValueError("predicted full-resolution DMAP peak exceeds 4 GiB cap")
    return {"requested_level": 0, "minimum": 600, "maximum": 1600,
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
    bound = input_binding(inputs["model"], inputs["images"], inputs["pose_masks"],
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
        raise ValueError("dual 10 GiB floors or 4 GiB output headroom unavailable")
    output.mkdir()
    report = {"schema": SCHEMA, "status": "running", "source": {**bound,
              "model": str(inputs["model"]), "images": str(inputs["images"]),
              "pose_masks": str(inputs["pose_masks"]), "manifest": str(inputs["manifest"]),
              "producer": str(inputs["source_report"]), "camera_gate": str(inputs["camera_gate"])},
              "software": {"runner_sha256": digest(Path(__file__)),
                           "shared_runner_sha256": digest(Path(__file__).with_name("run.py")),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_binary_sha256": digest(Path(pycolmap._core.__file__)),
                           "openmvs_binaries": binary_hashes},
              "native_options": {"resolution_level": 0, "min_resolution": 600,
                                 "max_resolution": 1600, "geometric_iters": 2,
                                 "tower_mode": 4, "ignore_mask_label": 0,
                                 "compute_device": "cpu", "max_threads": 2,
                                 "refine_resolution_level": 1,
                                 "refine_scales": 1},
              "limits": {"max_output_bytes": CAP, "max_child_rss_bytes": RSS_CAP,
                         "timeout_seconds": TIMEOUT_SECONDS, "max_log_bytes": LOG_CAP,
                         "min_free_bytes_each_volume": RESERVE,
                         "extra_reserve_paths": [str(path) for path in reserve_paths]},
              "stages": [], "quality_accepted": False, "metric_scale_verified": False,
              "shipping_approved": False,
              "provenance_scope": "fresh sparse producer plus this composed masked native continuation"}

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
        for name in sorted(EXPECTED_NAMES):
            source, target = inputs["images"] / name, output / "images" / name
            shutil.copyfile(source, target)
            if digest(target) != bound["photo_sha256"][name]:
                raise ValueError(f"copied photograph differs: {name}")
            if (folder_bytes(output) > CAP or shutil.disk_usage(output).free < RESERVE or
                    any(shutil.disk_usage(path).free < RESERVE for path in reserve_paths)):
                raise ValueError("input copy exceeded byte cap or disk floor")
        for name in MODEL_FILES:
            shutil.copyfile(inputs["model"] / name, sparse / name)
        if model_hashes(sparse) != bound["model_sha256"]:
            raise ValueError("copied sparse model differs")
        (output / "inputs.json").write_text(json.dumps([{"name": name} for name in sorted(EXPECTED_NAMES)]))
        report["stages"].append({"name": "copy_inputs", "status": "complete", "photos": 60})
        save()
        execute("undistort", [python, "-m", "scripts.classical_backend.run",
                              "--worker", "undistort", "--output", output,
                              "--max-image-size", "1600"],
                lambda: validate_undistorted(output, sorted(EXPECTED_NAMES)))
        execute("warp_masks", [python, "-m", "scripts.classical_backend.dense_masks",
                               "--source-model", sparse,
                               "--undistorted-model", output / "dense" / "sparse",
                               "--source-masks", inputs["pose_masks"],
                               "--undistorted-images", output / "dense" / "images",
                               "--output", output / "masks", "--manifest", inputs["manifest"]],
                lambda: {"count": mask_count(output / "masks" / "report.json", 60),
                         "report_sha256": digest(output / "masks" / "report.json")})
        sizes = {}
        for name in EXPECTED_NAMES:
            with Image.open(output / "dense" / "images" / name) as image:
                sizes[name] = image.size
        report["native_resolution_preflight"] = depth_preflight(sizes, folder_bytes(output))
        save()
        dense = output / "dense"
        execute("import", [tool_path(binaries, "InterfaceCOLMAP"), "-i", dense,
                           "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", [tool_path(binaries, "DensifyPointCloud"), "-i", output / "scene.mvs",
                            "-o", output / "dense.mvs", "--resolution-level", "0",
                            "--min-resolution", "600", "--max-resolution", "1600",
                            "--geometric-iters", "2", "--tower-mode", "4",
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
        rebound = input_binding(inputs["model"], inputs["images"], inputs["pose_masks"],
                                inputs["manifest"], inputs["source_report"], inputs["camera_gate"],
                                args.source_report_sha256, args.camera_gate_sha256, expected_model)
        report["sources_unchanged"] = (bound == rebound and
            digest(Path(__file__)) == report["software"]["runner_sha256"] and
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
    parser.add_argument("--source-report-sha256", type=sha256_arg, required=True)
    parser.add_argument("--camera-gate-sha256", type=sha256_arg, required=True)
    parser.add_argument("--model-sha256", nargs=3, type=sha256_arg, required=True,
                        metavar=("CAMERAS", "IMAGES", "POINTS3D"))
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
