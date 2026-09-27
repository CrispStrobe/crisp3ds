"""Bounded rough-mesh control from the sealed 005 image-only recovered cameras.

This is a composed cache-rebound SfM -> masked OpenMVS experiment, not a fresh
photo-only pipeline or an independent geometry measurement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from scripts.classical_backend import calibrated_control as audit
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes
from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, stage, tool_path)


SCHEMA = "classical_recovered_dense_v1"
PRODUCER_SCHEMA = "ycb_image_only_delayed_self_calibration_v1"
SOURCE_SCHEMA = "ycb_image_only_fixed_initial_intrinsics_v1"
SEALED_REPORT_SHA256 = "80241c6c8d72097ce38a9a046aadfa495c1027bbfbd05a6a9480ad09b02c7e10"
SEALED_SOURCE_SHA256 = "9a7ddeac8b99ab3ec41a512aec0b852b555aa6176860d2047460e3f87f910b63"
SEALED_CACHE_AUDIT_SHA256 = "14116ca9768a40d5f3265cbd22b14a7043f216be8f3f671e2bdbba1caacc0db5"
REQUIRED_GATES = {"registered_and_point_counts_preserved", "finite_plausible_intrinsics",
                  "principal_point_fixed", "finite_training_residuals"}
CAP = 1 << 30
TIMEOUT = 600
LOG_CAP = 32 << 20
RSS_CAP = 10 << 30


def validate_chain(producer: dict, source: dict, producer_path: Path, source_path: Path,
                   model_dir: Path, images: Path, masks: Path, manifest: Path,
                   cache_audit: Path) -> dict:
    """Verify the delayed-BA binary, its parent, and exact original photo/mask bytes."""
    if (producer.get("schema") != PRODUCER_SCHEMA or producer.get("status") != "complete" or
            producer.get("lane") != "exploratory_image_only_cached_sparse_refinement" or
            producer.get("supplied_intrinsics_or_poses_used") is not False or
            producer.get("features_or_matches_recomputed") is not False or
            producer.get("reference_mesh_used") is not False or
            producer.get("mapping_rerun") is not False or
            set(producer.get("gates", {})) != REQUIRED_GATES or
            any(producer["gates"][name] is not True for name in REQUIRED_GATES)):
        raise ValueError("005 is not the sealed image-only delayed-refinement model")
    if (source.get("schema") != SOURCE_SCHEMA or source.get("status") != "complete" or
            source.get("supplied_intrinsics_used") is not False or
            source.get("supplied_poses_used") is not False or
            source.get("reference_mesh_used") is not False):
        raise ValueError("004 is not the sealed image-only heuristic-initialization model")
    if (digest(producer_path) != SEALED_REPORT_SHA256 or
            digest(source_path) != SEALED_SOURCE_SHA256 or
            digest(cache_audit) != SEALED_CACHE_AUDIT_SHA256 or
            producer.get("source_producer_report_sha256") != digest(source_path) or
            producer.get("source_model_files_sha256") != source.get("model_files_sha256") or
            producer.get("refined_model_files_sha256") != model_hashes(model_dir) or
            Path(producer.get("refined_model_dir", "")).resolve() != model_dir or
            producer.get("source_database_sha256") != source.get("source_database_sha256") or
            producer.get("source_cache_mutation_audit_sha256") != digest(cache_audit) or
            source.get("cache_mutation_audit_sha256") != digest(cache_audit) or
            source.get("source_manifest_sha256") != digest(manifest)):
        raise ValueError("005/004/model/cache-audit/manifest hashes do not bind")
    if producer.get("registered") != 60 or source.get("registered") != 60 or producer.get("points3D", 0) < 1:
        raise ValueError("recovered model must contain all 60 cameras and sparse points")
    entries = json.loads(manifest.read_text()).get("images", [])
    if len(entries) != 60:
        raise ValueError("expected 60 original YCB manifest entries")
    photo_hashes, mask_hashes = {}, {}
    for entry in entries:
        name = audit.simple_name(entry["name"])
        if name in photo_hashes:
            raise ValueError("duplicate original image")
        photo_hashes[name], mask_hashes[name] = entry["sha256"], entry["mask_sha256"]
        photo, mask = images / name, masks / (name + ".png")
        if (photo.is_symlink() or mask.is_symlink() or
                not photo.is_file() or not mask.is_file() or
                digest(photo) != photo_hashes[name] or digest(mask) != mask_hashes[name]):
            raise ValueError(f"original photo/pose mask changed: {name}")
        with Image.open(photo) as image:
            if image.size != (1280, 1024):
                raise ValueError(f"unexpected original photo dimensions: {name}")
    expected_names = {f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)}
    if (set(photo_hashes) != expected_names or
            source.get("source_image_hashes") != photo_hashes):
        raise ValueError("004 photo hashes or complete turntable names differ")
    import pycolmap
    model = pycolmap.Reconstruction(str(model_dir))
    registered = {image.name: image for image in model.images.values() if image.has_pose}
    if (set(registered) != expected_names or len(model.cameras) != 1 or
            model.num_points3D() != producer["points3D"]):
        raise ValueError("005 binary camera/point inventory differs")
    for image in registered.values():
        camera = model.cameras[image.camera_id]
        rotation = image.cam_from_world.rotation.matrix()
        if (camera.model.name != "SIMPLE_RADIAL" or camera.width != 1280 or
                camera.height != 1024 or len(camera.params) != 4 or
                not np.isfinite(camera.params).all() or
                not isinstance(producer.get("after_camera_params"), list) or
                len(producer["after_camera_params"]) != 4 or
                not 640 <= camera.params[0] <= 3200 or
                not np.allclose(camera.params, producer.get("after_camera_params", []),
                                atol=1e-9, rtol=0) or
                not np.isfinite(image.cam_from_world.translation).all() or
                not np.isfinite(rotation).all() or
                np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-5 or
                abs(np.linalg.det(rotation) - 1) > 1e-5):
            raise ValueError("005 binary model has invalid camera intrinsics or poses")
    if any(not np.isfinite(point.xyz).all() for point in model.points3D.values()):
        raise ValueError("005 binary model has nonfinite points")
    return {"registered_names": sorted(registered), "photo_hashes": photo_hashes,
            "mask_hashes": mask_hashes, "points3D": model.num_points3D(),
            "camera_model": "SIMPLE_RADIAL", "camera_params": list(model.cameras.values())[0].params.tolist()}


def preflight_existing(image_sizes: dict[str, tuple[int, int]], existing_bytes: int) -> dict:
    return audit.native_depth_preflight(image_sizes, existing_bytes, output_cap=CAP, minimum=600)


def run(args: argparse.Namespace) -> dict:
    inputs = {"producer": args.producer_report, "source": args.source_report,
              "model": args.model, "images": args.images, "masks": args.pose_masks,
              "manifest": args.manifest, "cache_audit": args.cache_audit}
    if any(path.is_symlink() for path in (*inputs.values(), args.output, args.output.parent)):
        raise ValueError("symlink input/output paths are unsupported")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be fresh under an existing real parent")
    paths = {name: path.resolve() for name, path in inputs.items()}
    output, binaries = args.output.resolve(), args.binary_dir.resolve()
    producer = json.loads(paths["producer"].read_text())
    source = json.loads(paths["source"].read_text())
    bound = validate_chain(producer, source, paths["producer"], paths["source"],
                           paths["model"], paths["images"], paths["masks"],
                           paths["manifest"], paths["cache_audit"])
    import pycolmap
    binary_hashes = {name: digest(tool_path(binaries, name)) for name in TOOLS[:3]}
    input_bytes = sum((paths["images"] / name).stat().st_size for name in bound["registered_names"])
    input_bytes += sum((paths["model"] / name).stat().st_size for name in MODEL_FILES)
    if (input_bytes >= CAP or
            shutil.disk_usage(output.parent).free < RESERVE + CAP + (256 << 20)):
        raise ValueError("1 GiB cap or 10 GiB reserve plus 256 MiB margin unavailable")
    output.mkdir()
    report = {"schema": SCHEMA, "status": "running",
              "lane": "exploratory_image_only_cached_sparse_refinement",
              "provenance_class": "image-only poses/intrinsics; copied and rebound foreground feature cache; delayed sparse BA",
              "source_reports": {name: {"path": str(paths[name]), "sha256": digest(paths[name])}
                                 for name in ("producer", "source", "manifest", "cache_audit")},
              "source_model": str(paths["model"]), "source_model_sha256": model_hashes(paths["model"]),
              "source_photo_sha256": bound["photo_hashes"],
              "source_pose_mask_sha256": bound["mask_hashes"],
              "registered_images": 60, "sparse_points": bound["points3D"],
              "camera_model": bound["camera_model"], "camera_params": bound["camera_params"],
              "software": {"runner_sha256": digest(Path(__file__)), "pycolmap_version": pycolmap.__version__,
                           "pycolmap_binary_sha256": digest(Path(pycolmap._core.__file__)),
                           "openmvs_binaries": binary_hashes},
              "native_options": {"resolution_level": 2, "min_resolution": 600,
                                 "max_resolution": 1600, "geometric_iters": 2,
                                 "tower_mode": 4, "ignore_mask_label": 0, "max_threads": 2},
              "limits": {"max_output_bytes": CAP, "timeout_seconds": TIMEOUT,
                         "max_log_bytes": LOG_CAP, "max_child_rss_bytes": RSS_CAP,
                         "min_free_bytes": RESERVE},
              "stages": [], "quality_accepted": False, "metric_scale_verified": False,
              "shipping_approved": False}

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(name, command, validate):
        result = stage(output, name, [str(x) for x in command], deadline,
                       CAP, LOG_CAP, RSS_CAP)
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    save()
    deadline = time.monotonic() + TIMEOUT
    common = ["--max-threads", "2", "--working-folder", str(output)]
    try:
        copy = {"name": "copy_inputs", "status": "running"}
        report["stages"].append(copy)
        save()
        (output / "images").mkdir()
        sparse = output / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in bound["registered_names"]:
            photo = paths["images"] / name
            if shutil.disk_usage(output).free < RESERVE + photo.stat().st_size:
                raise ValueError("disk reserve reached during photo copy")
            shutil.copyfile(photo, output / "images" / name)
            if digest(output / "images" / name) != bound["photo_hashes"][name]:
                raise ValueError(f"copied photo changed: {name}")
        for name in MODEL_FILES:
            shutil.copyfile(paths["model"] / name, sparse / name)
        if model_hashes(sparse) != report["source_model_sha256"]:
            raise ValueError("copied binary model changed")
        if folder_bytes(output) > CAP or shutil.disk_usage(output).free < RESERVE:
            raise ValueError("copy exceeded output cap or reserve")
        copy.update(status="complete", artifact={"photos": 60, "model_files": 3})
        save()
        python = Path(sys.executable)
        execute("undistort", [python, "-m", "scripts.classical_backend.calibrated_control",
                              "--worker", "undistort", "--output", output],
                lambda: audit.validate_undistorted(output, bound["registered_names"]))
        execute("warp_masks", [python, "-m", "scripts.classical_backend.dense_masks",
                               "--source-model", sparse, "--undistorted-model", output / "dense" / "sparse",
                               "--source-masks", paths["masks"],
                               "--undistorted-images", output / "dense" / "images",
                               "--manifest", paths["manifest"], "--output", output / "masks"],
                lambda: {"count": audit.mask_count(output / "masks" / "report.json", 60)})
        sizes = {}
        for name in bound["registered_names"]:
            with Image.open(output / "dense" / "images" / name) as photo:
                sizes[name] = photo.size
        preflight = {"name": "native_depth_preflight", "status": "running"}
        report["stages"].append(preflight)
        save()
        profile = preflight_existing(sizes, folder_bytes(output))
        report["native_resolution_preflight"] = profile
        preflight.update(status="complete", artifact={"views": 60,
                         "predicted_two_generation_peak_bytes": profile["predicted_two_generation_peak_bytes"]})
        save()
        dense = output / "dense"
        execute("import", [tool_path(binaries, "InterfaceCOLMAP"), "-i", dense,
                           "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", [tool_path(binaries, "DensifyPointCloud"), "-i", output / "scene.mvs",
                            "-o", output / "dense.mvs", "--resolution-level", "2",
                            "--min-resolution", "600", "--max-resolution", "1600",
                            "--geometric-iters", "2", "--tower-mode", "4",
                            "--mask-path", output / "masks", "--ignore-mask-label", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("dmap_camera_check", [python, "-m", "scripts.classical_backend.calibrated_control",
                                      "--worker", "dmap_check", "--output", output],
                lambda: {"views": audit.checked_dmap_count(output / "dmap-camera-check.json", 60)})
        execute("rough_mesh", [tool_path(binaries, "ReconstructMesh"), "-i", output / "dense.mvs",
                               "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        if report["stages"] and report["stages"][-1].get("status") == "running":
            report["stages"][-1].update(status="failed", failure=str(error))
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = shutil.disk_usage(output).free
    try:
        validate_chain(producer, source, paths["producer"], paths["source"], paths["model"],
                       paths["images"], paths["masks"], paths["manifest"], paths["cache_audit"])
        report["sources_unchanged"] = all(digest(paths[name]) == value["sha256"] for name, value in
                                          report["source_reports"].items()) and all(
            digest(tool_path(binaries, name)) == value for name, value in binary_hashes.items())
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"] or report["output_bytes"] > CAP or report["free_bytes_after"] < RESERVE:
        report.update(status="failed", failure="source changed or output/disk hard limit exceeded")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("producer-report", "source-report", "model", "images", "pose-masks",
                 "manifest", "cache-audit", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "result": str(args.output / "result.json"),
                      "failure": report.get("failure")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
