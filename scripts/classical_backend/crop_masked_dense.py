"""Bounded common-crop continuation of a separately accepted sparse model.

This is a new experiment. It never rewrites the 001/002 runs or the producer
model. A single mask-derived integer crop is applied to every TRAIN image,
mask and copied camera; no held-out or reference geometry enters the crop.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.classical_backend import calibrated_control as audit
from scripts.classical_backend import sparse_masked_dense as base
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes, native_mask_name
from scripts.classical_backend.run import (RESERVE, StageError, checked_ply, digest,
                                           folder_bytes, nonempty_bytes, stage, tool_path)


PAD = 32
EXPECTED_CROP = (493, 329, 715, 631)


def common_crop(mask_dir: Path, names: list[str], padding: int = PAD) -> tuple[int, int, int, int]:
    """Half-open union of binary TRAIN foreground boxes plus fixed pixel pad."""
    if not names or len(names) != len(set(names)) or padding < 0:
        raise ValueError("invalid crop names or padding")
    boxes = []
    for name in names:
        path = mask_dir / (name + ".png")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"crop mask missing or linked: {name}")
        with Image.open(path) as mask:
            if mask.mode != "L" or mask.size != (1280, 1024):
                raise ValueError(f"crop mask mode or dimensions differ: {name}")
            values = np.asarray(mask)
            if not np.all((values == 0) | (values == 255)):
                raise ValueError(f"crop mask is not binary: {name}")
            box = mask.getbbox()
        if box is None:
            raise ValueError(f"crop mask has no foreground: {name}")
        boxes.append(box)
    return (max(0, min(box[0] for box in boxes) - padding),
            max(0, min(box[1] for box in boxes) - padding),
            min(1280, max(box[2] for box in boxes) + padding),
            min(1024, max(box[3] for box in boxes) + padding))


def png_name(name: str) -> str:
    if not name.endswith(".jpg") or "/" in name or "\\" in name:
        raise ValueError("source image must be a flat JPEG name")
    return name[:-4] + ".png"


def shift_camera(model: str, params: list[float], crop: tuple[int, int, int, int]) -> list[float]:
    if len(crop) != 4 or not (0 <= crop[0] < crop[2] <= 1280 and 0 <= crop[1] < crop[3] <= 1024):
        raise ValueError("invalid half-open crop")
    f_x, f_y, c_x, c_y = base.pinhole_params(model, params)
    return [f_x, f_y, c_x - crop[0], c_y - crop[1]]


def tracked_inside(model, crop: tuple[int, int, int, int]) -> int:
    count = 0
    for point in model.points3D.values():
        for link in point.track.elements:
            image = model.images[int(link.image_id)]
            xy = np.asarray(image.points2D[int(link.point2D_idx)].xy, dtype=float)
            if (not np.isfinite(xy).all() or not crop[0] <= xy[0] < crop[2] or
                    not crop[1] <= xy[1] < crop[3]):
                raise ValueError("a tracked sparse observation lies outside the mask-only crop")
            count += 1
    return count


def crop_model(source_dir: Path, destination: Path, crop: tuple[int, int, int, int],
               names: list[str], expected_observations: int) -> dict:
    """Serialize copied COLMAP model with exact name, principal-point and 2D shifts."""
    import pycolmap
    if destination.exists():
        raise ValueError("cropped model destination must be fresh")
    source_hashes = model_hashes(source_dir)
    original = pycolmap.Reconstruction(str(source_dir))
    candidate = pycolmap.Reconstruction(str(source_dir))
    if len(original.cameras) != 1 or len(candidate.cameras) != 1:
        raise ValueError("common crop requires one shared camera")
    camera_old = next(iter(original.cameras.values()))
    camera_new = next(iter(candidate.cameras.values()))
    if (camera_old.width, camera_old.height) != (1280, 1024):
        raise ValueError("common crop requires original 1280x1024 camera")
    params = shift_camera(camera_old.model.name, list(camera_old.params), crop)
    if tracked_inside(original, crop) != expected_observations:
        raise ValueError("tracked crop observation count differs from sealed repair")
    camera_new.model = pycolmap.CameraModelId.PINHOLE
    camera_new.params = np.asarray(params, dtype=float)
    camera_new.width = crop[2] - crop[0]
    camera_new.height = crop[3] - crop[1]
    name_map = {}
    for image in candidate.images.values():
        if image.name not in names:
            raise ValueError("cropped model includes an unsealed image name")
        old_name = image.name
        image.name = png_name(old_name)
        name_map[old_name] = image.name
        for point in image.points2D:
            point.xy = np.asarray(point.xy, dtype=float) - np.asarray(crop[:2], dtype=float)
    if set(name_map) != set(names) or len(set(name_map.values())) != len(names):
        raise ValueError("cropped image-name map is incomplete or colliding")
    destination.mkdir(parents=True)
    candidate.write_binary(str(destination))
    reopened = pycolmap.Reconstruction(str(destination))
    new_camera = next(iter(reopened.cameras.values()))
    if (new_camera.model.name != "PINHOLE" or
            (new_camera.width, new_camera.height) != (crop[2]-crop[0], crop[3]-crop[1]) or
            not np.array_equal(new_camera.params, params) or
            set(reopened.points3D) != set(original.points3D) or
            set(reopened.images) != set(original.images)):
        raise ValueError("serialized crop camera or inventory differs")
    max_projection_delta = 0.0
    max_ray_delta = 0.0
    checked_track_projections = 0
    shifted_keypoints = 0
    untracked_outside_crop = 0
    for image_id, old in original.images.items():
        new = reopened.images[image_id]
        if (new.name != name_map[old.name] or new.camera_id != old.camera_id or
                not np.array_equal(new.cam_from_world.matrix(), old.cam_from_world.matrix()) or
                len(new.points2D) != len(old.points2D)):
            raise ValueError("crop changes image pose, ID or keypoint inventory")
        for old_point, new_point in zip(old.points2D, new.points2D):
            if new_point.point3D_id != old_point.point3D_id:
                raise ValueError("crop changes point backlink")
            old_xy, new_xy = np.asarray(old_point.xy), np.asarray(new_point.xy)
            if not np.isfinite(old_xy).all() or not np.isfinite(new_xy).all():
                raise ValueError("crop changes or retains a nonfinite keypoint")
            if (not old_point.has_point3D() and
                    not (crop[0] <= old_xy[0] < crop[2] and crop[1] <= old_xy[1] < crop[3])):
                untracked_outside_crop += 1
            delta = new_xy + crop[:2] - old_xy
            if not np.isfinite(delta).all():
                raise ValueError("nonfinite keypoint translation")
            max_projection_delta = max(max_projection_delta, float(np.max(np.abs(delta))))
            shifted_keypoints += 1
    for point_id, old in original.points3D.items():
        new = reopened.points3D[point_id]
        old_links = sorted((int(link.image_id), int(link.point2D_idx)) for link in old.track.elements)
        new_links = sorted((int(link.image_id), int(link.point2D_idx)) for link in new.track.elements)
        if (not np.array_equal(new.xyz, old.xyz) or new_links != old_links or
                not np.array_equal(new.color, old.color) or new.error != old.error):
            raise ValueError("crop changes sparse XYZ, color, error or tracks")
        for image_id, keypoint_id in old_links:
            old_image, new_image = original.images[image_id], reopened.images[image_id]
            old_projection = np.asarray(old_image.project_point(old.xyz))
            new_projection = np.asarray(new_image.project_point(new.xyz))
            if (not np.isfinite(old_projection).all() or not np.isfinite(new_projection).all()):
                raise ValueError("nonfinite tracked point projection")
            max_projection_delta = max(max_projection_delta, float(np.max(np.abs(
                new_projection + crop[:2] - old_projection))))
            measured_old = np.asarray(old_image.points2D[keypoint_id].xy)
            measured_new = np.asarray(new_image.points2D[keypoint_id].xy)
            old_ray = np.asarray(camera_old.cam_from_img(measured_old))
            new_ray = np.asarray(new_camera.cam_from_img(measured_new))
            if not np.isfinite(old_ray).all() or not np.isfinite(new_ray).all():
                raise ValueError("nonfinite tracked point ray")
            max_ray_delta = max(max_ray_delta, float(np.max(np.abs(old_ray - new_ray))))
            checked_track_projections += 1
    if (checked_track_projections != expected_observations or
            digest(destination / "points3D.bin") != source_hashes["points3D.bin"]):
        raise ValueError("cropped model changes 3D payload or observation count")
    if (not math.isfinite(max_projection_delta) or max_projection_delta > 1e-9 or
            not math.isfinite(max_ray_delta) or max_ray_delta > 1e-9 or
            model_hashes(source_dir) != source_hashes):
        raise ValueError("crop projection/ray difference or source mutation")
    return {"model_sha256": model_hashes(destination), "name_map": name_map,
            "tracked_observations_inside": expected_observations,
            "checked_forward_projections_and_rays": checked_track_projections,
            "shifted_keypoints": shifted_keypoints,
            "untracked_keypoints_outside_crop": untracked_outside_crop,
            "max_measurement_translation_error_pixels": max_projection_delta,
            "max_ray_difference": max_ray_delta,
            "poses_xyz_tracks_ids_preserved": True}


def crop_pixels(images_dir: Path, masks_dir: Path, output: Path,
                names: list[str], crop: tuple[int, int, int, int], deadline: float) -> dict:
    """Write and reload lossless RGB and L crops; produce native stem masks."""
    image_dest = output / "dense" / "images"
    mask_dest = output / "masks"
    image_dest.mkdir(parents=True)
    mask_dest.mkdir()
    rows = []
    for name in names:
        if (time.monotonic() > deadline or
                min(shutil.disk_usage(output).free, shutil.disk_usage(base.ROOT).free) < RESERVE or
                folder_bytes(output) > base.OUTPUT_CAP):
            raise ValueError("crop deadline, output cap or disk floor exceeded")
        source_image = images_dir / name
        source_mask = masks_dir / (name + ".png")
        image_target = image_dest / png_name(name)
        mask_target = mask_dest / native_mask_name(png_name(name))
        with Image.open(source_image) as image, Image.open(source_mask) as mask:
            if (image.mode != "RGB" or image.size != (1280, 1024) or
                    mask.mode != "L" or mask.size != image.size):
                raise ValueError(f"source RGB/mask format differs: {name}")
            rgb = np.asarray(image)[crop[1]:crop[3], crop[0]:crop[2], :]
            label = np.asarray(mask)[crop[1]:crop[3], crop[0]:crop[2]]
            if not np.any(label == 255) or not np.all((label == 0) | (label == 255)):
                raise ValueError(f"cropped mask lacks binary foreground: {name}")
            Image.fromarray(rgb, mode="RGB").save(image_target, format="PNG")
            Image.fromarray(label, mode="L").save(mask_target, format="PNG")
        with Image.open(image_target) as saved_image, Image.open(mask_target) as saved_mask:
            if (saved_image.mode != "RGB" or saved_mask.mode != "L" or
                    not np.array_equal(np.asarray(saved_image), rgb) or
                    not np.array_equal(np.asarray(saved_mask), label)):
                raise ValueError(f"lossless crop bytes differ after reload: {name}")
        rows.append({"name": png_name(name), "source_name": name,
                     "image_sha256": digest(image_target), "mask_sha256": digest(mask_target),
                     "foreground_pixels": int(np.count_nonzero(label))})
    report = {"schema": "classical_dense_masks_v1", "status": "complete",
              "semantics": "mask-derived common integer crop; coarse pose support, not silhouette/GT",
              "ignore_mask_label": 0, "native_filename_basis": "image stem",
              "crop_xyxy_half_open": list(crop), "images": rows}
    (mask_dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if audit.mask_count(mask_dest / "report.json", len(names)) != len(names):
        raise ValueError("cropped native masks incomplete")
    return {"images": len(rows), "pixel_equivalent_after_reload": True,
            "mask_report_sha256": digest(mask_dest / "report.json"),
            "rgb_sha256": {row["name"]: row["image_sha256"] for row in rows},
            "mask_sha256": {row["name"]: row["mask_sha256"] for row in rows}}


def prepared_unchanged(output: Path, model_record: dict, pixel_record: dict) -> bool:
    if (model_hashes(output / "dense" / "sparse") != model_record["model_sha256"] or
            digest(output / "masks" / "report.json") != pixel_record["mask_report_sha256"]):
        return False
    for name, expected in pixel_record["rgb_sha256"].items():
        if digest(output / "dense" / "images" / name) != expected:
            return False
    for name, expected in pixel_record["mask_sha256"].items():
        if digest(output / "masks" / native_mask_name(name)) != expected:
            return False
    return True


def run(args: argparse.Namespace) -> dict:
    """Execute one fresh bounded crop→import→dense→rough trial."""
    started = time.monotonic()
    from scripts.object_motion import sam_m1_parity as mac
    mac.host_preflight(args.output, check_memory=False)
    if (args.output.exists() or args.output.is_symlink() or args.output.parent.is_symlink() or
            not args.output.parent.is_dir()):
        raise ValueError("crop output must be fresh under a real existing parent")
    bound = base.validate_inputs(args)
    names = bound["names"]
    crop = common_crop(bound["paths"]["stage_root"] / "masks", names)
    if crop != EXPECTED_CROP:
        raise ValueError("mask-only common crop differs from frozen 48-view plan")
    import pycolmap
    source_model = pycolmap.Reconstruction(str(bound["paths"]["model"]))
    if tracked_inside(source_model, crop) != bound["report"]["output"]["observations"]:
        raise ValueError("not every repaired track observation is inside the common crop")
    output = args.output.resolve()
    binary_dir = base._real(args.binary_dir, directory=True)
    tool_names = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh")
    tool_hashes = {name: digest(base._real(tool_path(binary_dir, name), directory=False))
                   for name in tool_names}
    source_bytes = sum((bound["paths"]["stage_root"] / "images" / name).stat().st_size +
                       (bound["paths"]["stage_root"] / "masks" / (name + ".png")).stat().st_size
                       for name in names)
    source_bytes += sum((bound["paths"]["model"] / name).stat().st_size for name in MODEL_FILES)
    predicted_crop_bytes = len(names) * (crop[2]-crop[0]) * (crop[3]-crop[1]) * 5
    if (source_bytes + predicted_crop_bytes >= base.OUTPUT_CAP or
            shutil.disk_usage(output.parent).free < RESERVE + base.OUTPUT_CAP + (256 << 20) or
            shutil.disk_usage(base.ROOT).free < RESERVE + (256 << 20)):
        raise ValueError("crop input/cap or dual-disk reserve preflight failed")
    output.mkdir()
    report = {"schema": "classical_verified_sparse_common_crop_dense_v1", "status": "running",
              "provenance_class": "image-only fixed-initial exhaustive sparse model; repaired and root accepted; TRAIN-mask-derived crop",
              "repair_report_sha256": bound["source_hashes"]["repair_report"],
              "supervisor_sha256": bound["source_hashes"]["supervisor"],
              "repair_qa_sha256": bound["source_hashes"]["repair_qa"],
              "baseline_result_sha256": bound["source_hashes"]["baseline_result"],
              "stage_report_sha256": bound["source_hashes"]["stage_report"],
              "source_model_sha256": bound["output_model_hashes"],
              "source_photo_sha256": bound["stage"]["train_photo_sha256"],
              "source_mask_sha256": {name: bound["stage"]["cleaned_masks"][name]["sha256"]
                                     for name in names},
              "crop_xyxy_half_open": list(crop), "crop_padding_pixels": PAD,
              "crop_source": "union of 48 accepted TRAIN cleaned-mask foreground boxes only",
              "source_geometry": bound["geometry"],
              "software": {"runner_sha256": digest(Path(__file__)),
                           "base_runner_sha256": digest(Path(base.__file__)),
                           "python_path": base.worker_python(),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_core_sha256": digest(Path(pycolmap._core.__file__)),
                           "openmvs_binaries": tool_hashes},
              "native_options": {"resolution_level": base.NATIVE_LEVEL,
                                 "min_resolution": base.NATIVE_MIN_RESOLUTION,
                                 "max_resolution": base.NATIVE_MAX_RESOLUTION,
                                 "geometric_iters": 2, "tower_mode": 4,
                                 "ignore_mask_label": 0, "max_threads": 2},
              "limits": {"max_output_bytes": base.OUTPUT_CAP,
                         "max_child_rss_bytes": base.RSS_CAP,
                         "max_log_bytes": base.LOG_CAP,
                         "timeout_seconds": base.TIMEOUT_SECONDS,
                         "min_free_bytes_each_disk": RESERVE},
              "stages": [], "quality_accepted": False,
              "metric_scale_verified": False, "shipping_approved": False}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    deadline = started + base.TIMEOUT_SECONDS
    save()

    def execute(name: str, command: list[str], validate) -> None:
        result = stage(output, name, command, deadline, base.OUTPUT_CAP,
                       base.LOG_CAP, base.RSS_CAP, extra_reserve_paths=(base.ROOT,))
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    try:
        preparation = {"name": "copy_and_crop", "status": "running"}
        report["stages"].append(preparation)
        save()
        source_copy = output / "sparse" / "0"
        source_copy.mkdir(parents=True)
        for name in MODEL_FILES:
            source = bound["paths"]["model"] / name
            target = source_copy / name
            if (time.monotonic() > deadline or
                    shutil.disk_usage(output).free < RESERVE + source.stat().st_size or
                    shutil.disk_usage(base.ROOT).free < RESERVE or
                    folder_bytes(output) + source.stat().st_size > base.OUTPUT_CAP):
                raise ValueError("repaired model copy exceeds deadline/cap/disk floor")
            shutil.copyfile(source, target)
        if model_hashes(source_copy) != bound["output_model_hashes"]:
            raise ValueError("copied repaired model differs")
        preparation["crop_model"] = crop_model(source_copy, output / "dense" / "sparse",
                                                 crop, names,
                                                 bound["report"]["output"]["observations"])
        preparation["crop_pixels"] = crop_pixels(
            bound["paths"]["stage_root"] / "images",
            bound["paths"]["stage_root"] / "masks", output, names, crop, deadline)
        if not prepared_unchanged(output, preparation["crop_model"], preparation["crop_pixels"]):
            raise ValueError("cropped camera, RGB or native mask changed during preparation")
        if (time.monotonic() > deadline or folder_bytes(output) > base.OUTPUT_CAP or
                min(shutil.disk_usage(output).free, shutil.disk_usage(base.ROOT).free) < RESERVE):
            raise ValueError("crop preparation exceeded deadline/cap/disk floor")
        preparation["status"] = "complete"
        save()
        cropped_names = [png_name(name) for name in names]
        with Image.open(output / "dense" / "images" / cropped_names[0]) as sample:
            size = sample.size
        report["native_depth_preflight"] = base.depth_budget(
            {name: size for name in cropped_names}, folder_bytes(output))
        if (shutil.disk_usage(output).free < RESERVE + base.OUTPUT_CAP - folder_bytes(output) or
                shutil.disk_usage(base.ROOT).free < RESERVE + (256 << 20)):
            raise ValueError("native crop preflight lacks reserve plus remaining cap")
        save()
        plan = dict(base.native_plan(output, binary_dir))
        execute("import", plan["import"],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", plan["densify"],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("dmap_camera_check", [base.worker_python(), "-m",
                                      "scripts.classical_backend.calibrated_control",
                                      "--worker", "dmap_check", "--output", str(output)],
                lambda: {"views": audit.checked_dmap_count(output / "dmap-camera-check.json", 48)})
        execute("rough_mesh", plan["rough_mesh"],
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
    report["free_bytes_after"] = {"output": shutil.disk_usage(output).free,
                                  "internal": shutil.disk_usage(base.ROOT).free}
    if (time.monotonic() > deadline or report["output_bytes"] > base.OUTPUT_CAP or
            min(report["free_bytes_after"].values()) < RESERVE):
        report.update(status="failed", failure="postflight time/output/disk cap exceeded")
    try:
        base.validate_inputs(args)
        preparation_record = report["stages"][0]
        report["sources_unchanged"] = (
            digest(Path(__file__)) == report["software"]["runner_sha256"] and
            digest(Path(base.__file__)) == report["software"]["base_runner_sha256"] and
            digest(Path(pycolmap._core.__file__)) == report["software"]["pycolmap_core_sha256"] and
            all(digest(tool_path(binary_dir, name)) == value
                for name, value in tool_hashes.items()) and
            preparation_record.get("status") == "complete" and
            prepared_unchanged(output, preparation_record["crop_model"],
                               preparation_record["crop_pixels"]))
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"]:
        report.update(status="failed", failure="producer, photos, masks, model or native binary changed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repair-report", "repair-qa", "baseline-run", "stage-root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("repair-report-sha256", "supervisor-sha256", "repair-qa-sha256",
                 "baseline-result-sha256", "stage-report-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--binary-dir", type=Path, default=base.BIN)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
