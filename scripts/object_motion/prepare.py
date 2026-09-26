#!/usr/bin/env python3
"""Prepare conservative photo-derived pose-support masks for YCB NP3 photos.

Run with a Python interpreter containing Pillow. The independent scan, supplied
depth, poses, and dataset masks are never opened.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".local-tools/test-data/ycb-cracker-box"
DEFAULT_OUT = ROOT / "build-opencv/object-motion/prepare-001"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def red_scanlines(image):
    """Detect reddish packaging within a fixed, image-inspected search window.

Rows are filled between the extreme red pixels, so pale print on the carton
can support pose estimation. This is a coarse support region, not segmentation.
"""
    width, height = image.size
    if (width, height) != (1280, 1024):
        raise ValueError(f"unexpected image size {(width, height)}")
    rgb = image.convert("RGB")
    pix = rgb.load()
    lines = []
    for y in range(200, 701):
        xs = []
        for x in range(390, 851, 2):
            r, g, b = pix[x, y]
            if r >= 28 and r - g >= 15 and r - b >= 7 and r >= 1.27 * g:
                xs.append(x)
        if len(xs) >= 4:
            lines.append((y, max(390, min(xs) - 8), min(850, max(xs) + 8)))
    if len(lines) < 150:
        raise ValueError(f"too few packaging-colored rows: {len(lines)}")
    # Bridge pale label rows, but never extend to the fixed camera background.
    first, last = lines[0][0], lines[-1][0]
    by_y = {y: (lo, hi) for y, lo, hi in lines}
    result = []
    for y in range(first, last + 1):
        near = [by_y[k] for k in range(max(first, y - 12), min(last, y + 12) + 1)
                if k in by_y]
        if near:
            result.append((y, min(p[0] for p in near), max(p[1] for p in near)))
    return result


def validate_records(source, records):
    names = [r.get("path") for r in records]
    expected = [f"photos/NP3_{angle:03}.jpg" for angle in range(0, 360, 6)]
    if names != expected or [r.get("turntable_angle_degrees") for r in records] != list(range(0, 360, 6)):
        raise ValueError("expected 60 ordered NP3 photo paths and angles")
    photo_dir = source / "photos"
    if photo_dir.is_symlink():
        raise ValueError("photo directory must not be a symlink")
    for rec in records:
        photo = source / rec["path"]
        if photo.is_symlink() or not photo.is_file() or photo.resolve(strict=True).parent != photo_dir.resolve():
            raise ValueError(f"invalid source photo: {photo}")


def prepare(source, output):
    from PIL import Image, ImageDraw
    source = Path(source).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB free")
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("source manifest must not be a symlink")
    records = json.loads(manifest_path.read_text())["photos"]
    validate_records(source, records)
    output.mkdir(parents=True)
    mask_dir = output / "masks"
    mask_dir.mkdir()
    prepared = []
    try:
        for rec in records:
            photo = (source / rec["path"]).resolve(strict=True)
            if photo.stat().st_size != rec["bytes"] or digest(photo) != rec["sha256"]:
                raise ValueError(f"unverified photo: {photo}")
            with Image.open(photo) as image:
                lines = red_scanlines(image)
            mask = Image.new("L", (1280, 1024), 0)
            drawer = ImageDraw.Draw(mask)
            for y, lo, hi in lines:
                drawer.line((lo, y, hi, y), fill=255)
            name = photo.name + ".png"  # COLMAP mask convention: image basename + .png
            dest = mask_dir / name
            mask.save(dest)
            prepared.append({"name": photo.name, "path": str(photo), "sha256": rec["sha256"],
                             "angle_degrees": rec["turntable_angle_degrees"],
                             "pose_support_scanlines": lines,
                             "pose_support_mask": str(dest.resolve()), "mask_sha256": digest(dest),
                             "support_pixels": sum(hi - lo + 1 for _, lo, hi in lines)})
        result = {"schema": "object_motion_prepare_v1", "source_manifest": str(manifest_path),
                  "source_manifest_sha256": digest(manifest_path),
                  "mask_method": "photo RGB reddish-packaging scanline hull; x390:850 y200:700 search; 12px row bridge",
                  "mask_role": "coarse pose support only; not mesh silhouette or quality ground truth",
                  "images": prepared}
        (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"status": "incomplete", "error": str(exc),
            "completed_masks": len(prepared)}, indent=2) + "\n")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    result = prepare(a.source, a.output)
    print(json.dumps({"images": len(result["images"]), "output": str(a.output)}))


if __name__ == "__main__":
    main()
