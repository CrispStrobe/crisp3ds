"""Bounded, explicitly intrinsics-assisted YCB COLMAP-to-OpenMVS control.

The sparse poses come from image matching, but the FULL_OPENCV intrinsics are
supplied by the Berkeley calibration. This is not the image-only baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np
from PIL import Image

from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes, native_mask_name
from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, obj_counts,
                                           stage, tool_path)


MODEL_SCHEMA = "ycb_calibrated_intrinsics_sfm_v1"
OUTPUT_SCHEMA = "classical_calibrated_control_v1"
PIXEL_CONVENTION = "assumed OpenCV integer-center source; cx/cy shifted +0.5 for COLMAP"
MAX_OUTPUT = 1 << 30
MAX_TOTAL_NEW = int(1.5 * (1 << 30))
TIMEOUT_SECONDS = 600
LOG_LIMIT = 32 << 20
RSS_LIMIT = 10 << 30
HEADER = struct.Struct("<HBBIIIIff")
NATIVE_RESOLUTION_LEVEL = 2
NATIVE_MIN_RESOLUTION = 600
NATIVE_MAX_RESOLUTION = 1600


def simple_name(name: str) -> str:
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"image name is not a basename: {name}")
    return name


def _sha256_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def calibration_params(calibration: Path) -> list[float]:
    """Read the actual Berkeley NP3 RGB datasets, not report-provided numbers."""
    from scripts.object_motion.ycb_camera_reference import dataset
    K = dataset(calibration, "/NP3_rgb_K", (3, 3))
    d = dataset(calibration, "/NP3_rgb_d", (5,))
    if (not np.isfinite(K).all() or not np.isfinite(d).all() or
            not np.allclose(K[0, 1], 0, atol=1e-12) or
            not np.allclose(K[1, 0], 0, atol=1e-12) or
            not np.allclose(K[2], [0, 0, 1], atol=1e-12) or
            min(K[0, 0], K[1, 1]) <= 0):
        raise ValueError("invalid Berkeley NP3 RGB calibration matrices")
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2] + 0.5),
            float(K[1, 2] + 0.5), *(float(value) for value in d), 0.0, 0.0, 0.0]


def validate_producer(report: dict, model_dir: Path, images: Path,
                      masks: Path, manifest: Path, calibration: Path) -> dict:
    """Bind the new assisted producer to exact photos, calibration and model."""
    if (report.get("schema") != MODEL_SCHEMA or report.get("lane") != "calibrated_intrinsics_only" or
            report.get("provenance_class") != "oracle-assisted intrinsics; image-estimated poses" or
            report.get("status") != "complete" or
            report.get("pixel_center_convention") != PIXEL_CONVENTION or
            Path(report.get("model_dir", "")).resolve() != model_dir):
        raise ValueError("producer is not a sealed calibrated-intrinsics-only image-pose model")
    model_files = {name: model_dir / name for name in MODEL_FILES}
    if any(path.is_symlink() or not path.is_file() for path in model_files.values()):
        raise ValueError("producer binary model is missing or linked")
    if report.get("model_files_sha256") != model_hashes(model_dir):
        raise ValueError("producer binary model hashes differ")
    if report.get("calibration_h5_sha256") != digest(calibration):
        raise ValueError("supplied intrinsics calibration hash differs")
    params = report.get("full_opencv_params")
    if (not isinstance(params, list) or len(params) != 12 or
            not all(isinstance(x, (int, float)) and math.isfinite(x) for x in params) or
            min(params[0], params[1]) <= 0):
        raise ValueError("producer FULL_OPENCV parameters are invalid")
    if not np.allclose(params, calibration_params(calibration), atol=1e-9, rtol=0):
        raise ValueError("producer intrinsics do not match exact calibration HDF5 and +0.5 convention")
    for name in ("source_database_sha256", "cached_features_sha256_before",
                 "cached_features_sha256_after", "raw_matches_sha256_before",
                 "raw_matches_sha256_after", "verified_geometry_sha256_after"):
        if not _sha256_text(report.get(name)):
            raise ValueError(f"missing sealed feature/match evidence: {name}")
    if (report["cached_features_sha256_before"] != report["cached_features_sha256_after"] or
            report["raw_matches_sha256_before"] != report["raw_matches_sha256_after"] or
            not isinstance(report.get("reverification_options"), dict)):
        raise ValueError("cached features/raw matches changed or re-verification undocumented")
    manifest_data = json.loads(manifest.read_text())
    entries = manifest_data.get("images", [])
    if len(entries) != 60:
        raise ValueError("expected the fixed 60-photo YCB manifest")
    photo_hashes = {}
    mask_hashes = {}
    for entry in entries:
        name = simple_name(entry["name"])
        if name in photo_hashes:
            raise ValueError("duplicate manifest image name")
        photo_hashes[name] = entry["sha256"]
        mask_hashes[name] = entry["mask_sha256"]
        image_path, mask_path = images / name, masks / (name + ".png")
        if (image_path.is_symlink() or mask_path.is_symlink() or
                not image_path.is_file() or not mask_path.is_file() or
                digest(image_path) != photo_hashes[name] or digest(mask_path) != mask_hashes[name]):
            raise ValueError(f"photo/pose mask differs from manifest: {name}")
        with Image.open(image_path) as image:
            if image.size != (1280, 1024):
                raise ValueError(f"expected NP3 RGB size 1280x1024: {name}")
    if set(photo_hashes) != {f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)}:
        raise ValueError("manifest does not contain the fixed 60 NP3 turntable images")
    if report.get("source_image_hashes") != photo_hashes:
        raise ValueError("producer photo hashes differ from fixed manifest")
    import pycolmap
    model = pycolmap.Reconstruction(str(model_dir))
    registered = {image.name: image for image in model.images.values() if image.has_pose}
    if (len(registered) != 60 or len(registered) != report.get("registered") or
            model.num_points3D() != report.get("points3D") or
            model.num_points3D() < 1 or registered.keys() != photo_hashes.keys() or
            len(model.cameras) != 1):
        raise ValueError("producer registration/points/names differ from binary model")
    for image in registered.values():
        camera = model.cameras[image.camera_id]
        rotation = image.cam_from_world.rotation.matrix()
        translation = image.cam_from_world.translation
        if (not np.isfinite(rotation).all() or not np.isfinite(translation).all() or
                np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-5 or
                abs(np.linalg.det(rotation) - 1) > 1e-5):
            raise ValueError("binary model contains invalid image poses")
        if (camera.width != 1280 or camera.height != 1024 or
                camera.model.name != "FULL_OPENCV" or len(camera.params) != 12 or
                not np.allclose(camera.params, params, atol=1e-9, rtol=0)):
            raise ValueError("binary model does not preserve supplied FULL_OPENCV intrinsics")
    if any(not np.isfinite(point.xyz).all() for point in model.points3D.values()):
        raise ValueError("binary model contains nonfinite sparse points")
    return {"photo_hashes": photo_hashes, "mask_hashes": mask_hashes,
            "registered_names": sorted(registered), "registered": len(registered),
            "points3D": model.num_points3D()}


def expected_depth_k(params: np.ndarray, image_size: tuple[int, int],
                     depth_size: tuple[int, int]) -> np.ndarray:
    """Pinned importer -0.5, then OpenMVS max-edge-normalized isotropic K."""
    if (len(params) != 4 or not np.isfinite(params).all() or
            min(params[0], params[1]) <= 0 or min(*image_size, *depth_size) <= 0):
        raise ValueError("invalid PINHOLE camera or dimensions")
    sx, sy = depth_size[0] / image_size[0], depth_size[1] / image_size[1]
    if not 0 < sx <= 1 or not 0 < sy <= 1:
        raise ValueError("depth map must not exceed undistorted image dimensions")
    # TImage rounds the short edge independently, but Camera::GetK scales its
    # normalized intrinsics by max(width, height), not by each rounded edge.
    scale = max(depth_size) / max(image_size)
    fx, fy, cx, cy = map(float, params)
    return np.asarray([[fx * scale, 0, cx * scale - 0.5],
                       [0, fy * scale, cy * scale - 0.5], [0, 0, 1]], dtype=np.float64)


def native_depth_size(width: int, height: int, level: int, minimum: int,
                      maximum: int) -> dict:
    """Predict pinned OpenMVS v2.4.0 computeMaxResolution before dense work.

    The native code right-shifts the longest edge, then reduces the requested
    level until the result meets min-resolution. Its final OpenCV resize rounds
    the shorter edge; ceil is used here as a conservative pixel-budget bound.
    """
    if min(width, height, minimum, maximum) <= 0 or minimum > maximum or level < 0:
        raise ValueError("invalid native image-resolution profile")
    longest = max(width, height)
    actual_level = level
    target_long = longest >> actual_level
    if target_long < minimum:
        actual_level = 0
        while longest >> (actual_level + 1) >= minimum:
            actual_level += 1
        target_long = longest >> actual_level
    target_long = min(target_long, maximum)
    scale = min(1.0, target_long / longest)
    predicted = (math.ceil(width * scale), math.ceil(height * scale))
    return {"source_size": [width, height], "requested_level": level,
            "actual_level": actual_level, "predicted_max_resolution": target_long,
            "pixel_budget_size": list(predicted)}


def native_depth_preflight(image_sizes: dict[str, tuple[int, int]],
                           existing_bytes: int, output_cap: int = MAX_OUTPUT,
                           minimum: int = NATIVE_MIN_RESOLUTION) -> dict:
    """Reject unintended full-resolution work and obvious transient DMAP overflow.

    A DMAP currently stores depth, normal and confidence (20 bytes/pixel).
    Two complete generations may coexist during geometric consistency. This
    estimate is a preflight guard, not a replacement for the hard runtime cap.
    """
    if not image_sizes or existing_bytes < 0 or output_cap <= 0:
        raise ValueError("invalid native depth budget")
    rows = {name: native_depth_size(*size, NATIVE_RESOLUTION_LEVEL, minimum,
                                    NATIVE_MAX_RESOLUTION)
            for name, size in image_sizes.items()}
    if any(row["actual_level"] == 0 for row in rows.values()):
        raise ValueError("level-2 native request would run a full-resolution depth image")
    pixels = sum(math.prod(row["pixel_budget_size"]) for row in rows.values())
    # 4 KiB/view allows for camera/header metadata; 96 MiB reserves other files.
    predicted_peak = existing_bytes + 2 * (20 * pixels + 4096 * len(rows)) + (96 << 20)
    if predicted_peak > output_cap:
        raise ValueError("predicted two-generation DMAP peak exceeds output byte cap")
    return {"source": "OpenMVS v2.4.0 TImage::computeMaxResolution",
            "min_resolution": minimum, "max_resolution": NATIVE_MAX_RESOLUTION,
            "requested_level": NATIVE_RESOLUTION_LEVEL,
            "predicted_two_generation_peak_bytes": predicted_peak,
            "hard_output_cap_bytes": output_cap, "images": rows}


def read_dmap_camera(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("truncated DMAP header")
        magic, flags, _, iw, ih, dw, dh, low, high = HEADER.unpack(raw)
        if (magic != 0x5244 or not flags & 1 or not 0 < dw <= iw <= 8192 or
                not 0 < dh <= ih <= 8192 or not np.isfinite([low, high]).all()):
            raise ValueError("invalid DMAP header or image/depth size")
        name_size_raw = stream.read(2)
        if len(name_size_raw) != 2:
            raise ValueError("truncated DMAP image name")
        name_size = struct.unpack("<H", name_size_raw)[0]
        if not 0 < name_size < 4096:
            raise ValueError("invalid DMAP image name size")
        name = stream.read(name_size).decode("utf-8")
        count_raw = stream.read(4)
        if len(count_raw) != 4:
            raise ValueError("truncated DMAP IDs")
        count = struct.unpack("<I", count_raw)[0]
        if not 1 <= count <= 200 or len(stream.read(4 * count)) != 4 * count:
            raise ValueError("invalid DMAP IDs")
        matrices = stream.read(21 * 8)
        if len(matrices) != 21 * 8:
            raise ValueError("truncated DMAP camera")
        values = np.frombuffer(matrices, dtype="<f8")
        k, r, c = values[:9].reshape(3, 3), values[9:18].reshape(3, 3), values[18:]
        if (not np.isfinite(values).all() or abs(np.linalg.det(r) - 1) > 1e-4 or
                np.max(np.abs(r @ r.T - np.eye(3))) > 1e-4):
            raise ValueError("invalid DMAP camera matrices")
        if path.stat().st_size < stream.tell() + dw * dh * 4:
            raise ValueError("truncated DMAP depth payload")
        return {"name": simple_name(Path(name).name), "image_size": (iw, ih),
                "depth_size": (dw, dh), "K": k, "R": r, "C": c}


def validate_dmap_cameras(folder: Path, model_dir: Path) -> dict:
    import pycolmap
    from scripts.classical_backend.dense_masks import cv2
    from scripts.classical_backend.verify_dmap_masks import load_depth
    model = pycolmap.Reconstruction(str(model_dir))
    images = {image.name: image for image in model.images.values() if image.has_pose}
    files = sorted(folder.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    if len(files) != len(images):
        raise ValueError("one complete native base DMAP required per registered camera")
    masks_folder = folder / "masks"
    mask_rows = None
    if masks_folder.is_dir():
        if cv2 is None:
            raise RuntimeError("OpenCV required to verify native depth-mask support")
        mask_report = json.loads((masks_folder / "report.json").read_text())
        if (mask_report.get("status") != "complete" or mask_report.get("ignore_mask_label") != 0 or
                len(mask_report.get("images", [])) != len(images)):
            raise ValueError("native mask report is incomplete")
        mask_rows = {item["name"]: item for item in mask_report["images"]}
    rows = []
    for path in files:
        entry = read_dmap_camera(path)
        name = entry["name"]
        if name not in images or any(row["name"] == name for row in rows):
            raise ValueError("duplicate/foreign DMAP camera")
        image = images[name]
        camera = model.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise ValueError("native DMAP requires undistorted PINHOLE camera")
        expected_k = expected_depth_k(camera.params, (camera.width, camera.height),
                                      entry["image_size"])
        rotation = image.cam_from_world.rotation.matrix()
        center = -rotation.T @ image.cam_from_world.translation
        differences = {"K": float(np.max(np.abs(entry["K"] - expected_k))),
                       "R": float(np.max(np.abs(entry["R"] - rotation))),
                       "C": float(np.max(np.abs(entry["C"] - center)))}
        if (entry["image_size"] != entry["depth_size"] or
                differences["K"] > 1e-4 or differences["R"] > 1e-5 or differences["C"] > 1e-5):
            raise ValueError(f"native depth camera differs from exact COLMAP resize convention: {name}, {differences}")
        outside = None
        if mask_rows is not None:
            if name not in mask_rows:
                raise ValueError(f"native mask missing for DMAP camera: {name}")
            mask_path = masks_folder / native_mask_name(name)
            if mask_path.is_symlink() or digest(mask_path) != mask_rows[name]["mask_sha256"]:
                raise ValueError(f"native mask differs from its report: {name}")
            with Image.open(mask_path) as source_mask:
                mask = np.asarray(source_mask.convert("L"))
            _, depth = load_depth(path)
            resized = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            outside = int(np.count_nonzero((depth > 0) & (resized == 0)))
            if outside:
                raise ValueError(f"native depth exists outside ignore-label-0 mask: {name}: {outside}")
        rows.append({"name": name, "dmap_sha256": digest(path), "image_size": entry["image_size"],
                     "depth_size": entry["depth_size"], "max_abs_difference": differences,
                     "positive_depth_outside_mask": outside})
    return {"schema": "classical_dmap_camera_check_v1", "status": "complete",
            "registered_views": len(images), "checked_views": len(rows),
            "max_abs_difference": {key: max(row["max_abs_difference"][key] for row in rows)
                                   for key in ("K", "R", "C")},
            "positive_depth_outside_mask": (sum(row["positive_depth_outside_mask"] for row in rows)
                                            if mask_rows is not None else None), "images": rows}


def mask_count(report_path: Path, expected: int) -> int:
    report = json.loads(report_path.read_text())
    count = len(report.get("images", []))
    if report.get("schema") != "classical_dense_masks_v1" or report.get("status") != "complete" or count != expected:
        raise ValueError("undistorted native mask set is incomplete")
    return count


def checked_dmap_count(report_path: Path, expected: int) -> int:
    report = json.loads(report_path.read_text())
    count = report.get("checked_views")
    if (report.get("schema") != "classical_dmap_camera_check_v1" or
            report.get("status") != "complete" or count != expected):
        raise ValueError("native DMAP camera consistency check is incomplete")
    return count


def reprojection_pair(source: dict, source_depth: np.ndarray, target: dict,
                      target_depth: np.ndarray, limit: int = 128) -> dict:
    """Image-only depth self-consistency; never an independent shape score."""
    if (source_depth.shape != tuple(reversed(source["depth_size"])) or
            target_depth.shape != tuple(reversed(target["depth_size"])) or limit < 1):
        raise ValueError("depth-camera dimensions or sampling bound differ")
    candidates = np.flatnonzero(source_depth > 0)
    if len(candidates) == 0:
        return {"sampled": 0, "compared": 0, "within_2_percent": 0,
                "median_relative_depth_error": None}
    selected = candidates[np.linspace(0, len(candidates) - 1, min(limit, len(candidates)),
                                      dtype=np.int64)]
    height, width = source_depth.shape
    y, x = np.divmod(selected, width)
    d = source_depth.ravel()[selected]
    K0, K1 = source["K"], target["K"]
    camera_xyz = np.stack([(x - K0[0, 2]) * d / K0[0, 0],
                           (y - K0[1, 2]) * d / K0[1, 1], d], axis=1)
    world = (source["R"].T @ camera_xyz.T).T + source["C"]
    other = (target["R"] @ (world - target["C"]).T).T
    positive = other[:, 2] > 0
    projected_x = K1[0, 0] * other[:, 0] / np.maximum(other[:, 2], 1e-12) + K1[0, 2]
    projected_y = K1[1, 1] * other[:, 1] / np.maximum(other[:, 2], 1e-12) + K1[1, 2]
    u, v = np.rint(projected_x).astype(np.int64), np.rint(projected_y).astype(np.int64)
    inside = (positive & (u >= 0) & (v >= 0) &
              (u < target_depth.shape[1]) & (v < target_depth.shape[0]))
    valid = np.flatnonzero(inside)
    if len(valid) == 0:
        return {"sampled": len(selected), "compared": 0, "within_2_percent": 0,
                "median_relative_depth_error": None}
    neighbor_depth = target_depth[v[valid], u[valid]]
    valid = valid[neighbor_depth > 0]
    neighbor_depth = neighbor_depth[neighbor_depth > 0]
    if len(valid) == 0:
        return {"sampled": len(selected), "compared": 0, "within_2_percent": 0,
                "median_relative_depth_error": None}
    relative = np.abs(other[valid, 2] - neighbor_depth) / np.maximum(other[valid, 2], neighbor_depth)
    return {"sampled": len(selected), "compared": len(relative),
            "within_2_percent": int(np.count_nonzero(relative <= 0.02)),
            "median_relative_depth_error": float(np.median(relative))}


def depth_reprojection_diagnostic(folder: Path) -> dict:
    from scripts.classical_backend.verify_dmap_masks import load_depth
    files = sorted(folder.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    maps = {}
    for path in files:
        camera = read_dmap_camera(path)
        name, depth = load_depth(path)
        if Path(name).name != camera["name"] or camera["name"] in maps:
            raise ValueError("duplicate/mismatched DMAP view in reprojection diagnostic")
        maps[camera["name"]] = (camera, depth)
    names = sorted(maps)
    if len(names) < 3:
        raise ValueError("at least three depth views needed for reprojection diagnostic")
    rows = []
    for index, name in enumerate(names):
        neighbor = names[(index + 1) % len(names)]
        row = reprojection_pair(*maps[name], *maps[neighbor])
        rows.append({"source": name, "neighbor": neighbor, **row})
    compared = sum(row["compared"] for row in rows)
    hits = sum(row["within_2_percent"] for row in rows)
    return {"scope": "adjacent-view native depth self-consistency; not independent pose or shape truth",
            "views": len(names), "sampled": sum(row["sampled"] for row in rows),
            "compared": compared, "within_2_percent": hits,
            "fraction_within_2_percent": hits / compared if compared else None,
            "pairs": rows}


def validate_undistorted(output: Path, expected_names: list[str]) -> dict:
    import pycolmap
    source = pycolmap.Reconstruction(str(output / "sparse" / "0"))
    target = pycolmap.Reconstruction(str(output / "dense" / "sparse"))
    source_images = {image.name: image for image in source.images.values() if image.has_pose}
    target_images = {image.name: image for image in target.images.values() if image.has_pose}
    if source_images.keys() != target_images.keys() or set(target_images) != set(expected_names):
        raise ValueError("undistortion exported a different registered photo set")
    max_pose_difference = 0.0
    for name, image in target_images.items():
        camera = target.cameras[image.camera_id]
        image_path = output / "dense" / "images" / name
        if (camera.model.name != "PINHOLE" or image_path.is_symlink() or
                not image_path.is_file()):
            raise ValueError(f"undistortion did not export PINHOLE image: {name}")
        with Image.open(image_path) as photo:
            if photo.size != (camera.width, camera.height):
                raise ValueError(f"undistorted image dimensions differ from camera: {name}")
        delta = float(np.max(np.abs(image.cam_from_world.matrix() -
                                    source_images[name].cam_from_world.matrix())))
        max_pose_difference = max(max_pose_difference, delta)
    if max_pose_difference > 1e-8:
        raise ValueError("undistortion changed image poses")
    return {"images": len(target_images), "model_sha256": model_hashes(output / "dense" / "sparse"),
            "max_pose_matrix_difference": max_pose_difference}


def worker(kind: str, output: Path) -> None:
    import pycolmap
    if kind == "undistort":
        options = pycolmap.UndistortCameraOptions()
        options.max_image_size = 1600
        pycolmap.undistort_images(str(output / "dense"), str(output / "sparse" / "0"),
                                  str(output / "images"), output_type="COLMAP",
                                  undistort_options=options)
    elif kind == "dmap_check":
        result = validate_dmap_cameras(output, output / "dense" / "sparse")
        result["depth_reprojection"] = depth_reprojection_diagnostic(output)
        (output / "dmap-camera-check.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        raise ValueError("unknown calibrated-control worker")


def run(args: argparse.Namespace) -> dict:
    for label, path in (("producer", args.producer_report), ("model", args.model),
                        ("images", args.images), ("masks", args.pose_masks),
                        ("manifest", args.manifest), ("calibration", args.calibration),
                        ("output", args.output)):
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be fresh under an existing parent")
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    model_dir, images, masks = args.model.resolve(), args.images.resolve(), args.pose_masks.resolve()
    producer_path, manifest_path, calibration_path = (args.producer_report.resolve(),
                                                      args.manifest.resolve(), args.calibration.resolve())
    producer = json.loads(producer_path.read_text())
    bound = validate_producer(producer, model_dir, images, masks, manifest_path, calibration_path)
    import pycolmap
    from scripts.classical_backend.dense_masks import cv2
    from scripts.object_motion import ycb_camera_reference
    binaries = {name: digest(tool_path(binary_dir, name)) for name in TOOLS}
    inputs_size = sum((images / name).stat().st_size + (masks / (name + ".png")).stat().st_size
                      for name in bound["photo_hashes"])
    inputs_size += sum((model_dir / name).stat().st_size for name in MODEL_FILES)
    if inputs_size >= MAX_OUTPUT:
        raise ValueError("source copies alone exceed 1 GiB output cap")
    if shutil.disk_usage(output.parent).free < RESERVE + MAX_TOTAL_NEW + (256 << 20):
        raise ValueError("10 GiB reserve plus 1.5 GiB batch budget unavailable")
    output.mkdir()
    report = {"schema": OUTPUT_SCHEMA, "status": "running", "provenance_class": producer["provenance_class"],
              "lane": producer["lane"], "source_report": str(producer_path),
              "source_report_sha256": digest(producer_path), "source_model": str(model_dir),
              "source_model_sha256": producer["model_files_sha256"],
              "source_manifest_sha256": digest(manifest_path),
              "source_calibration_sha256": digest(calibration_path),
              "source_photo_sha256": bound["photo_hashes"], "source_pose_mask_sha256": bound["mask_hashes"],
              "producer_feature_match_evidence": {key: producer[key] for key in
                  ("source_database_sha256", "cached_features_sha256_before",
                   "cached_features_sha256_after", "raw_matches_sha256_before",
                   "raw_matches_sha256_after", "verified_geometry_sha256_after")},
              "pixel_center_convention": PIXEL_CONVENTION,
              "registered_images": bound["registered"], "sparse_points": bound["points3D"],
              "software": {"openmvs_binaries": binaries, "runner_sha256": digest(Path(__file__)),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_binary_sha256": digest(Path(pycolmap._core.__file__)),
                           "opencv_version": cv2.__version__ if cv2 is not None else None,
                           "calibration_reader_sha256": digest(Path(ycb_camera_reference.__file__))},
              "native_options": {"resolution_level": NATIVE_RESOLUTION_LEVEL,
                                 "min_resolution": NATIVE_MIN_RESOLUTION,
                                 "max_resolution": NATIVE_MAX_RESOLUTION,
                                 "geometric_iters": 2, "tower_mode": 4,
                                 "ignore_mask_label": 0, "max_threads": 2,
                                 "refine_resolution_level": 1, "refine_scales": 1},
              "limits": {"max_output_bytes": MAX_OUTPUT, "total_new_budget_bytes": MAX_TOTAL_NEW,
                         "min_free_bytes": RESERVE, "timeout_seconds": TIMEOUT_SECONDS,
                         "max_child_rss_bytes": RSS_LIMIT, "max_log_bytes": LOG_LIMIT},
              "stages": [], "shipping_approved": False, "metric_scale_verified": False,
              "quality_accepted": False}

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    deadline = time.monotonic() + TIMEOUT_SECONDS
    common = ["--max-threads", "2", "--working-folder", str(output)]

    def execute(name, command, validate):
        result = stage(output, name, [str(x) for x in command], deadline,
                       MAX_OUTPUT, LOG_LIMIT, RSS_LIMIT)
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    try:
        copy_stage = {"name": "copy_inputs", "status": "running"}
        report["stages"].append(copy_stage)
        save()
        (output / "images").mkdir()
        sparse = output / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in bound["registered_names"]:
            image = images / name
            if shutil.disk_usage(output).free < RESERVE + image.stat().st_size:
                raise ValueError("disk reserve reached during photo copy")
            shutil.copyfile(image, output / "images" / name)
            if digest(output / "images" / name) != bound["photo_hashes"][name]:
                raise ValueError(f"copied photo differs: {name}")
        for name in MODEL_FILES:
            shutil.copyfile(model_dir / name, sparse / name)
        if model_hashes(sparse) != report["source_model_sha256"]:
            raise ValueError("copied binary model differs")
        if folder_bytes(output) > MAX_OUTPUT or shutil.disk_usage(output).free < RESERVE:
            raise ValueError("copy exceeded output cap or disk reserve")
        copy_stage.update(status="complete", artifact={"photos": len(bound["registered_names"]),
                                                       "model_files": len(MODEL_FILES)})
        save()
        python = Path(sys.executable)
        execute("undistort", [python, "-m", "scripts.classical_backend.calibrated_control",
                              "--worker", "undistort", "--output", output],
                lambda: validate_undistorted(output, bound["registered_names"]))
        execute("warp_masks", [python, "-m", "scripts.classical_backend.dense_masks",
                               "--source-model", sparse, "--undistorted-model", output / "dense" / "sparse",
                               "--source-masks", masks, "--undistorted-images", output / "dense" / "images",
                               "--output", output / "masks"],
                lambda: {"count": mask_count(output / "masks" / "report.json", bound["registered"])})
        dense = output / "dense"
        preflight_stage = {"name": "native_depth_preflight", "status": "running"}
        report["stages"].append(preflight_stage)
        save()
        image_sizes = {}
        for name in bound["registered_names"]:
            with Image.open(dense / "images" / name) as photo:
                image_sizes[name] = photo.size
        profile = native_depth_preflight(image_sizes, folder_bytes(output))
        report["native_resolution_preflight"] = profile
        preflight_stage.update(status="complete", artifact={
            "images": len(image_sizes),
            "actual_levels": sorted({row["actual_level"] for row in profile["images"].values()}),
            "predicted_two_generation_peak_bytes": profile["predicted_two_generation_peak_bytes"]})
        save()
        execute("import", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                           "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", [tool_path(binary_dir, "DensifyPointCloud"), "-i", output / "scene.mvs",
                            "-o", output / "dense.mvs", "--resolution-level", str(NATIVE_RESOLUTION_LEVEL),
                            "--min-resolution", str(NATIVE_MIN_RESOLUTION),
                            "--max-resolution", str(NATIVE_MAX_RESOLUTION),
                            "--geometric-iters", "2", "--tower-mode", "4",
                            "--mask-path", output / "masks", "--ignore-mask-label", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("dmap_camera_check", [python, "-m", "scripts.classical_backend.calibrated_control",
                                      "--worker", "dmap_check", "--output", output],
                lambda: {"views": checked_dmap_count(output / "dmap-camera-check.json", bound["registered"])})
        geometry_scene = output / "dense.mvs"
        execute("mesh", [tool_path(binary_dir, "ReconstructMesh"), "-i", geometry_scene,
                         "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        execute("refine", [tool_path(binary_dir, "RefineMesh"), "-i", geometry_scene,
                           "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                           "--resolution-level", "1", "--scales", "1", *common],
                lambda: {"faces": checked_ply(output / "refined.ply", "face")})
        execute("texture", [tool_path(binary_dir, "TextureMesh"), "-i", geometry_scene,
                            "-m", output / "refined.ply", "-o", output / "textured.mvs",
                            "--export-type", "obj", *common],
                lambda: dict(zip(("vertices", "faces"), obj_counts(output / "textured.obj"))))
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
        validate_producer(producer, model_dir, images, masks, manifest_path, calibration_path)
        report["sources_unchanged"] = (digest(producer_path) == report["source_report_sha256"] and
                                       digest(manifest_path) == report["source_manifest_sha256"] and
                                       all(digest(tool_path(binary_dir, name)) == value
                                           for name, value in binaries.items()))
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"]:
        report.update(status="failed", failure="calibrated model/photos/masks/calibration/binaries changed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("undistort", "dmap_check"))
    parser.add_argument("--producer-report", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--pose-masks", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    if not all((args.producer_report, args.model, args.images, args.pose_masks,
                args.manifest, args.calibration)):
        parser.error("sealed producer, model, images, pose masks, manifest and calibration required")
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
