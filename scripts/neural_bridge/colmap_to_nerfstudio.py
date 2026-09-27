#!/usr/bin/env python3
"""Strict camera-only COLMAP text to Nerfstudio transforms.json bridge."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAX_CAMERAS_BYTES = 2_000_000
MAX_IMAGES_BYTES = 64_000_000
MAX_FRAMES = 10_000
MIN_FREE_BYTES = 10 * 1024**3
CV_TO_GL = np.diag([1., -1., -1., 1.])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_source(root, name):
    """Resolve a POSIX COLMAP name without traversal, absolute paths or symlink escape."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid source-relative name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("name escapes or ambiguously addresses source root")
    root = Path(root).resolve(strict=True)
    target = (root / Path(*pure.parts)).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("source file escapes root or is not a regular file")
    return target


def parse_cameras(path):
    path = Path(path)
    if path.stat().st_size > MAX_CAMERAS_BYTES:
        raise ValueError("cameras.txt exceeds bound")
    cameras = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError("malformed COLMAP camera row")
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        if model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"distorted or unsupported COLMAP camera model: {model}")
        params = np.asarray([float(value) for value in parts[4:]], dtype=float)
        if len(params) != (4 if model == "PINHOLE" else 3):
            raise ValueError("wrong pinhole parameter count")
        fx, fy, cx, cy = (params if model == "PINHOLE" else (params[0], params[0], params[1], params[2]))
        if (camera_id < 0 or camera_id in cameras or width <= 0 or height <= 0
                or width > 100000 or height > 100000 or not np.isfinite(params).all()
                or fx <= 0 or fy <= 0 or not 0 <= cx <= width or not 0 <= cy <= height):
            raise ValueError("invalid pinhole camera")
        cameras[camera_id] = {"fl_x": float(fx), "fl_y": float(fy), "cx": float(cx),
                              "cy": float(cy), "w": width, "h": height,
                              "k1": 0., "k2": 0., "p1": 0., "p2": 0.}
    if not cameras:
        raise ValueError("empty COLMAP camera file")
    return cameras


def rotation_from_qvec(qvec):
    q = np.asarray(qvec, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(q).all() or not 0.999 <= norm <= 1.001:
        raise ValueError("invalid COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def opengl_c2w(qvec, tvec):
    """COLMAP/OpenCV world-to-camera to OpenGL camera-to-same-world."""
    rotation = rotation_from_qvec(qvec)
    tvec = np.asarray(tvec, dtype=float)
    if tvec.shape != (3,) or not np.isfinite(tvec).all():
        raise ValueError("invalid COLMAP camera translation")
    c2w = np.eye(4)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ tvec
    return c2w @ CV_TO_GL


def parse_images(path, cameras):
    path = Path(path)
    if path.stat().st_size > MAX_IMAGES_BYTES:
        raise ValueError("images.txt exceeds bound")
    images, ids = {}, set()
    with path.open(encoding="utf-8") as stream:
        iterator = iter(stream)
        for line in iterator:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("malformed COLMAP image pose row")
            image_id, camera_id = int(fields[0]), int(fields[8])
            name = fields[9]
            if (image_id <= 0 or image_id in ids or name in images or camera_id not in cameras
                    or len(images) >= MAX_FRAMES):
                raise ValueError("duplicate image ID/name, missing camera or frame limit")
            pose = opengl_c2w([float(value) for value in fields[1:5]],
                               [float(value) for value in fields[5:8]])
            observation_line = next(iterator, None)  # COLMAP text has two lines per image.
            if observation_line is None or observation_line.lstrip().startswith("#"):
                raise ValueError("missing image observation line")
            images[name] = (image_id, camera_id, pose)
            ids.add(image_id)
    if not images:
        raise ValueError("empty COLMAP image file")
    return images


def validate_splits(images, train, val, test):
    splits = {"train": list(train), "val": list(val), "test": list(test)}
    seen = set()
    for split, names in splits.items():
        if not names or len(names) != len(set(names)) or any(name not in images for name in names):
            raise ValueError(f"{split} split empty, duplicated or outside model")
        if seen.intersection(names):
            raise ValueError("split image names overlap")
        seen.update(names)
    if seen != set(images):
        raise ValueError("every registered image needs exactly one explicit split")
    return splits


def _mask_header(path, width, height):
    with Path(path).open("rb") as stream:
        header = stream.read(26)
    if (len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR" or struct.unpack(">II", header[16:24]) != (width, height)
            or header[24:26] != bytes([8, 0])):
        raise ValueError("mask must be an 8-bit one-channel PNG matching image dimensions")


def _validate_raster(path, width, height, is_mask):
    # Pillow is already a test/runtime dependency elsewhere in this repository.
    from PIL import Image
    if Path(path).stat().st_size > 128_000_000 or width * height > 50_000_000:
        raise ValueError("image or mask exceeds raster bounds")
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError("raster dimensions differ from undistorted camera")
        if is_mask and image.mode != "L":
            raise ValueError("mask must have exactly one grayscale channel")
        if not is_mask and image.mode not in ("RGB", "L"):
            raise ValueError("unsupported image channel mode")
        image.load()  # Detect truncated/compression-invalid inputs before export.
        if is_mask and not np.isin(np.asarray(image), (0, 255)).all():
            raise ValueError("mask pixels must be black or white only")


def build(model, images_root, output, train, val, test, provenance, masks_root=None):
    model, images_root, output = Path(model), Path(images_root), Path(output)
    if provenance not in ("supplied", "estimated") or output.exists() or output.is_symlink():
        raise ValueError("declare supplied/estimated provenance and use a fresh output directory")
    if shutil.disk_usage(output.parent).free < MIN_FREE_BYTES:
        raise OSError("neural camera bridge requires at least 10 GiB free disk")
    camera_file = safe_source(model, "cameras.txt")
    image_file = safe_source(model, "images.txt")
    early_model_hashes = {str(path): sha256(path) for path in (camera_file, image_file)}
    cameras = parse_cameras(camera_file)
    images = parse_images(image_file, cameras)
    splits = validate_splits(images, train, val, test)
    source_paths = [camera_file, image_file]
    masks_root = Path(masks_root) if masks_root is not None else None
    frames, exported_paths = [], {}
    for name in sorted(images):
        image_id, camera_id, pose = images[name]
        image = safe_source(images_root, name)
        _validate_raster(image, cameras[camera_id]["w"], cameras[camera_id]["h"], False)
        source_paths.append(image)
        file_path = Path(os.path.relpath(image, output)).as_posix()
        frame = {"file_path": file_path, "transform_matrix": pose.tolist(),
                 "colmap_im_id": image_id, **cameras[camera_id]}
        if masks_root is not None:
            mask = safe_source(masks_root, name + ".png")
            _mask_header(mask, cameras[camera_id]["w"], cameras[camera_id]["h"])
            _validate_raster(mask, cameras[camera_id]["w"], cameras[camera_id]["h"], True)
            source_paths.append(mask)
            frame["mask_path"] = Path(os.path.relpath(mask, output)).as_posix()
        frames.append(frame)
        exported_paths[name] = file_path
    before = {str(path.resolve()): sha256(path) for path in source_paths}
    if {str(path): before[str(path)] for path in (camera_file, image_file)} != early_model_hashes:
        raise ValueError("model text changed during parsing")
    transform = {"camera_model": "OPENCV", "orientation_override": "none",
                 "frames": frames, **{key + "_filenames": [exported_paths[name] for name in splits[key]]
                                    for key in ("train", "val", "test")}}
    report = {"schema": "colmap_nerfstudio_camera_bridge_v1", "camera_provenance": provenance,
              "frame_count": len(frames), "split_counts": {key: len(value) for key, value in splits.items()},
              "source_sha256": before, "source_camera_convention": "COLMAP/OpenCV world-to-camera",
              "export_camera_convention": "OpenGL camera-to-world; +X right, +Y up, +Z back; unchanged world gauge",
              "image_or_mask_bytes_copied": 0,
              "caveat": "estimated poses may have used all split images in SfM; splits do not imply held-out camera estimation"}
    if {str(path.resolve()): sha256(path) for path in source_paths} != before:
        raise ValueError("source changed during camera export")
    payload = (json.dumps(transform, indent=2, allow_nan=False) + "\n").encode()
    report["transforms_sha256"] = hashlib.sha256(payload).hexdigest()
    report["bridge_sha256"] = sha256(Path(__file__))
    metadata = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) + len(metadata) > 2_000_000:
        raise ValueError("camera export exceeds 2 MB bound")
    output.mkdir(parents=True)
    (output / "transforms.json").write_bytes(payload)
    (output / "provenance.json").write_bytes(metadata)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", choices=("supplied", "estimated"), required=True)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--val", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--masks-root", type=Path)
    args = parser.parse_args()
    report = build(args.model, args.images_root, args.output, args.train, args.val,
                   args.test, args.provenance, args.masks_root)
    print(json.dumps({"frames": report["frame_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
