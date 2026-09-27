#!/usr/bin/env python3
"""Photo-only, object-specific bunny silhouette masks for COLMAP/OpenMVS.

This deliberately exploits the dark figurine on a bright turntable. It is not a
general segmenter and never reads cameras, reference scans, or recovered meshes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = ROOT / "build-opencv/bunny-gamma05-clahe2"
DEFAULT_OUTPUT = ROOT / "build-opencv/bunny-masks-001"
NAME = re.compile(r"frame_(\d{4})\.png\Z")
THRESHOLD = 170
LEFT, RIGHT, BOTTOM = 0.15, 0.85, 0.961
EXPECTED_COUNT = 73
MAX_PIXELS = 12_000_000
MAX_INPUT_BYTES = 200_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(images: Path) -> list[Path]:
    if not images.is_dir() or images.is_symlink():
        raise ValueError("image input must be a real directory")
    paths = sorted(images.glob("frame_*.png"))
    if len(paths) != EXPECTED_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_COUNT} image files, got {len(paths)}")
    if any(not NAME.fullmatch(p.name) or p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("unexpected image name, symlink, or non-file")
    if [int(NAME.fullmatch(p.name).group(1)) for p in paths] != list(range(EXPECTED_COUNT)):
        raise ValueError("image names must be contiguous frame_0000 through frame_0072")
    if sum(p.stat().st_size for p in paths) > MAX_INPUT_BYTES:
        raise ValueError("input images exceed 200 MB bound")
    return paths


def segment(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected uint8 single-channel image")
    height, width = gray.shape
    if height * width > MAX_PIXELS or min(height, width) < 64:
        raise ValueError("image dimensions out of bounds")
    x0, x1, y1 = int(width * LEFT), int(width * RIGHT), int(height * BOTTOM)
    candidate = np.zeros_like(gray, dtype=np.uint8)
    candidate[:y1, x0:x1] = (gray[:y1, x0:x1] < THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n < 2:
        raise ValueError("no dark foreground component")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    fraction = area / (height * width)
    if not 0.08 <= fraction <= 0.30:
        raise ValueError(f"unexpected foreground area fraction: {fraction:.4f}")
    mask = np.where(labels == component, 255, 0).astype(np.uint8)
    box = [int(stats[component, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
    return mask, {"foreground_pixels": area, "foreground_fraction": fraction,
                  "bbox_xywh": box, "touches_bottom_guard": bool(np.any(mask[y1 - 1]))}


def prepare(images: Path = DEFAULT_IMAGES, output: Path = DEFAULT_OUTPUT) -> dict:
    images, output = Path(images), Path(output)
    paths = discover(images)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    mask_dir = output / "masks"
    mask_dir.mkdir()
    report = {"schema": "bunny_photo_only_masks_v1", "status": "running",
              "purpose": "object-specific dark-bunny silhouette, not general segmentation",
              "source_images": str(images.resolve()), "mask_dir": str(mask_dir.resolve()),
              "image_count": len(paths), "method": {"grayscale_below": THRESHOLD,
              "roi_fractions": {"left": LEFT, "right": RIGHT, "bottom": BOTTOM},
              "component": "largest 8-connected; internal holes retained"}, "images": []}
    manifest = output / "manifest.json"
    try:
        for path in paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"cannot read image: {path.name}")
            mask, diagnostic = segment(gray)
            dest = mask_dir / f"{path.name}.png"
            if not cv2.imwrite(str(dest), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                raise OSError(f"failed writing {dest}")
            report["images"].append({"name": path.name, "image_sha256": sha256(path),
                                     "mask_sha256": sha256(dest), "width": gray.shape[1],
                                     "height": gray.shape[0], **diagnostic})
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = prepare(args.images, args.output)
    print(json.dumps({"status": report["status"], "count": len(report["images"]),
                      "mask_dir": report["mask_dir"]}))


if __name__ == "__main__":
    main()
