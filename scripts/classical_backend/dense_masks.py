"""Reproject image-derived pose masks into COLMAP's undistorted image pixels.

OpenMVS v2.4.0 expects IMAGE.mask.png (filename stem, without .jpg) beside an undistorted image (or in
--mask-path) and ignores pixels with --ignore-mask-label 0. These are coarse
photo-derived foreground-support masks, not ground-truth silhouettes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # Feature requires the local PyCOLMAP venv; generic CI may not have OpenCV.
    cv2 = None


MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
RESERVE = 10 << 30


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_hashes(model_dir: Path) -> dict[str, str]:
    files = {name: model_dir / name for name in MODEL_FILES}
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError(f"COLMAP binary model incomplete: {model_dir}")
    return {name: digest(path) for name, path in files.items()}


def native_mask_name(image_name: str) -> str:
    name = Path(image_name)
    if name.name != image_name or not name.stem or name.stem in (".", ".."):
        raise ValueError("native mask image name must be a simple basename")
    return name.stem + ".mask.png"


def camera_intrinsics(camera: pycolmap.Camera) -> tuple[float, float, float, float, float]:
    model = camera.model.name
    params = [float(x) for x in camera.params]
    if model == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k = params
        return f, f, cx, cy, k
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params
        return f, f, cx, cy, 0.0
    if model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        return fx, fy, cx, cy, 0.0
    raise ValueError(f"unsupported mask camera model {model}; refuse approximate warp")


def remap_binary_mask(source: np.ndarray, source_camera: pycolmap.Camera,
                      target_camera: pycolmap.Camera) -> np.ndarray:
    """Mirror COLMAP 3.11.1 WarpImageBetweenCameras pixel-center geometry.

    COLMAP first warps at the distorted source resolution using a rescaled
    target camera, then resizes to the undistorted resolution. Image coordinates
    refer to pixel centers at (x + 0.5, y + 0.5), whereas cv2.remap addresses
    source samples by integer index, hence the final -0.5.
    """
    if source.ndim != 2 or source.shape != (source_camera.height, source_camera.width):
        raise ValueError("source mask dimensions differ from distorted COLMAP camera")
    if cv2 is None:
        raise RuntimeError("dense mask reprojection requires OpenCV in the PyCOLMAP venv")
    if not np.all((source == 0) | (source == 255)):
        raise ValueError("pose mask must contain only 0 and 255 labels")
    fx, fy, cx, cy, k = camera_intrinsics(source_camera)
    ux, uy, ucx, ucy, uk = camera_intrinsics(target_camera)
    if uk != 0 or min(fx, fy, ux, uy) <= 0:
        raise ValueError("target camera must be undistorted with positive focal lengths")
    sx = source_camera.width / target_camera.width
    sy = source_camera.height / target_camera.height
    ux, uy, ucx, ucy = ux * sx, uy * sy, ucx * sx, ucy * sy
    xx, yy = np.meshgrid(np.arange(source_camera.width, dtype=np.float32) + 0.5,
                         np.arange(source_camera.height, dtype=np.float32) + 0.5)
    nx = (xx - ucx) / ux
    ny = (yy - ucy) / uy
    radial = 1.0 + k * (nx * nx + ny * ny)
    map_x = np.asarray(fx * nx * radial + cx - 0.5, dtype=np.float32)
    map_y = np.asarray(fy * ny * radial + cy - 0.5, dtype=np.float32)
    intermediate = cv2.remap(source, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if (target_camera.width, target_camera.height) == (source_camera.width, source_camera.height):
        return intermediate
    return cv2.resize(intermediate, (target_camera.width, target_camera.height),
                      interpolation=cv2.INTER_NEAREST)


def prepare(source_model: Path, undistorted_model: Path, source_masks: Path,
            undistorted_images: Path, output: Path, manifest_path: Path | None = None) -> dict:
    import pycolmap
    if output.exists():
        raise ValueError(f"mask output must be fresh: {output}")
    if shutil.disk_usage(output.parent).free < RESERVE + (32 << 20):
        raise ValueError("insufficient free disk above 10 GiB reserve")
    if source_masks.is_symlink() or not source_masks.is_dir():
        raise ValueError("source mask path must be a real directory")
    source_hashes = model_hashes(source_model)
    target_hashes = model_hashes(undistorted_model)
    source = pycolmap.Reconstruction(str(source_model))
    target = pycolmap.Reconstruction(str(undistorted_model))
    source_images = {image.name: image for image in source.images.values() if image.has_pose}
    target_images = {image.name: image for image in target.images.values() if image.has_pose}
    if not source_images or source_images.keys() != target_images.keys():
        raise ValueError("distorted and undistorted COLMAP image names differ")
    if len({native_mask_name(name) for name in source_images}) != len(source_images):
        raise ValueError("image names collide after OpenMVS mask stem conversion")
    max_pose_matrix_difference = 0.0
    for name in source_images:
        difference = np.max(np.abs(source_images[name].cam_from_world.matrix() -
                                   target_images[name].cam_from_world.matrix()))
        max_pose_matrix_difference = max(max_pose_matrix_difference, float(difference))
    if max_pose_matrix_difference > 1e-8:
        raise ValueError("undistortion changed camera poses; refuse independent mask warp")
    manifest_hashes = None
    if manifest_path:
        manifest = json.loads(manifest_path.read_text())
        manifest_hashes = {item["name"]: item["mask_sha256"] for item in manifest["images"]}
        if manifest_hashes.keys() != source_images.keys():
            raise ValueError("manifest mask names differ from sparse images")
    report = {"schema": "classical_dense_masks_v1", "status": "running",
              "semantics": "photo-derived coarse pose support, not a silhouette or ground truth",
              "ignore_mask_label": 0, "filename_suffix": ".mask.png", "native_filename_basis": "image stem",
              "source_model": str(source_model.resolve()), "source_model_sha256": source_hashes,
              "undistorted_model": str(undistorted_model.resolve()), "undistorted_model_sha256": target_hashes,
              "max_pose_matrix_difference": max_pose_matrix_difference,
              "manifest_sha256": digest(manifest_path) if manifest_path else None,
              "images": []}
    output.mkdir(parents=True)
    for name in sorted(source_images):
        original_mask = source_masks / (name + ".png")
        image_path = undistorted_images / name
        if (original_mask.is_symlink() or image_path.is_symlink() or
                not original_mask.is_file() or not image_path.is_file()):
            raise ValueError(f"missing or symlink mask/image for {name}")
        mask_hash = digest(original_mask)
        if manifest_hashes and mask_hash != manifest_hashes[name]:
            raise ValueError(f"source pose mask hash differs from manifest: {name}")
        distorted_camera = source.cameras[source_images[name].camera_id]
        undistorted_camera = target.cameras[target_images[name].camera_id]
        with Image.open(image_path) as image:
            if image.size != (undistorted_camera.width, undistorted_camera.height):
                raise ValueError(f"undistorted image dimensions differ from camera: {name}")
        with Image.open(original_mask) as image:
            original = np.asarray(image.convert("L"))
        warped = remap_binary_mask(original, distorted_camera, undistorted_camera)
        if not np.any(warped == 255):
            raise ValueError(f"mask warp has no foreground support: {name}")
        destination = output / native_mask_name(name)
        if not cv2.imwrite(str(destination), warped):
            raise ValueError(f"failed to save mask: {destination}")
        report["images"].append({"name": name, "native_mask_name": destination.name,
                                 "source_mask_sha256": mask_hash,
                                 "undistorted_image_sha256": digest(image_path),
                                 "mask_sha256": digest(destination),
                                 "source_size": [distorted_camera.width, distorted_camera.height],
                                 "undistorted_size": [undistorted_camera.width, undistorted_camera.height],
                                 "source_foreground_pixels": int(np.count_nonzero(original)),
                                 "undistorted_foreground_pixels": int(np.count_nonzero(warped))})
    if model_hashes(source_model) != source_hashes or model_hashes(undistorted_model) != target_hashes:
        raise ValueError("COLMAP model changed during mask preparation")
    for item in report["images"]:
        if (digest(source_masks / (item["name"] + ".png")) != item["source_mask_sha256"] or
                digest(undistorted_images / item["name"]) != item["undistorted_image_sha256"]):
            raise ValueError(f"mask or undistorted image changed during preparation: {item['name']}")
    report["status"] = "complete"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--undistorted-model", type=Path, required=True)
    parser.add_argument("--source-masks", type=Path, required=True)
    parser.add_argument("--undistorted-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = prepare(args.source_model, args.undistorted_model, args.source_masks,
                     args.undistorted_images, args.output, args.manifest)
    print(json.dumps({"status": report["status"], "mask_count": len(report["images"]),
                      "report": str((args.output / "report.json").resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
