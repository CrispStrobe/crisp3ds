#!/usr/bin/env python3
"""Keep one 8-connected object component in sealed SAM mustard smoke masks."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_motion import sam_mask_trial as trial
from scripts.object_motion import ycb_object_masks as common


MAX_REMOVED_FRACTION = 0.10
MAX_RAW_MASK_BYTES = 2 * 1024**2
MAX_OUTPUT_BYTES = 20 * 1024**2


def largest_component(mask):
    """8-connectivity only: no holes filled and no surviving pixel moved."""
    array = np.asarray(mask)
    if array.shape != (1024, 1280) or array.dtype != np.uint8 or not np.isin(array, (0, 255)).all():
        raise ValueError("expected 1280x1024 binary uint8 mask")
    foreground = array != 0
    raw_pixels = int(foreground.sum())
    if raw_pixels == 0:
        raise ValueError("empty raw mask")
    height, width = foreground.shape
    flat = foreground.ravel()
    visited = np.zeros(flat.size, dtype=bool)
    components = []
    for seed in np.flatnonzero(flat):
        seed = int(seed)
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        component = []
        while queue:
            pixel = queue.popleft()
            component.append(pixel)
            y, x = divmod(pixel, width)
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = yy * width + xx
                    if flat[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    if len(components) > 1 and len(components[0]) == len(components[1]):
        raise ValueError("ambiguous equal-size largest components")
    retained = len(components[0])
    removed = raw_pixels - retained
    if removed / raw_pixels > MAX_REMOVED_FRACTION:
        raise ValueError("more than 10% of raw foreground would be removed")
    clean = np.zeros(flat.size, dtype=np.uint8)
    clean[components[0]] = 255
    clean = clean.reshape(foreground.shape)
    ys, xs = np.nonzero(clean)
    return clean, {"raw_pixels": raw_pixels, "cleaned_pixels": retained,
                   "removed_pixels": removed, "removed_fraction": removed / raw_pixels,
                   "raw_components_8_connected": len(components),
                   "removed_component_sizes": [len(c) for c in components[1:]],
                   "cleaned_bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()),
                                                   int(xs.max()) + 1, int(ys.max()) + 1]}


def validated_raw_manifest(path, expected_sha):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("raw manifest must be a small regular file")
    if common.digest(path) != expected_sha:
        raise ValueError("raw manifest hash differs from reviewed input")
    record = json.loads(path.read_text())
    if (record.get("schema") != "sam21_tiny_mustard_smoke_v1" or
            record.get("status") != "complete_unreviewed" or
            record.get("mask_status") != "generated_unreviewed" or
            record.get("package_sha256") != common.PACKAGE_SHA256[trial.OBJECT_ID] or
            record.get("box_xyxy_original_pixels") != list(trial.BOX) or
            record.get("checkpoint_sha256") !=
            "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"):
        raise ValueError("raw manifest does not match frozen SAM2 smoke contract")
    rows = record.get("images")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != list(trial.SMOKE_NAMES):
        raise ValueError("raw manifest must contain exact three smoke training photos")
    for row in rows:
        if not isinstance(row.get("source_sha256"), str) or not isinstance(row.get("mask_sha256"), str):
            raise ValueError("raw manifest lacks image/mask hashes")
    return record


def prepare(raw_dir, raw_manifest_sha, package_path, dataset_root, output):
    raw_dir, package_path, output = Path(raw_dir), Path(package_path), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise ValueError("raw candidate directory must be real")
    raw_manifest_path = raw_dir / "manifest.json"
    raw = validated_raw_manifest(raw_manifest_path, raw_manifest_sha)
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != trial.OBJECT_ID or package_sha != raw["package_sha256"]:
        raise ValueError("source package differs from sealed raw run")
    rows = {Path(row["path"]).name: row for row in common.training_records(package)}
    for row in raw["images"]:
        if rows[row["name"]]["sha256"] != row["source_sha256"]:
            raise ValueError("source image hash differs from sealed raw run")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output volume below 10 GiB free")
    runner_sha = common.digest(__file__)
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    report = {"schema": "sam21_tiny_mustard_cleanup_smoke_v1", "status": "running",
              "mask_status": "generated_unreviewed", "mask_role": "candidate coarse pose support; not exact silhouette",
              "cleanup_policy": "keep sole largest 8-connected component; no fill, erosion, or dilation",
              "maximum_removed_fraction": MAX_REMOVED_FRACTION,
              "raw_manifest_sha256": raw_manifest_sha, "package_sha256": package_sha,
              "runner_sha256": runner_sha, "images": []}
    used = 0
    try:
        sheet = Image.new("RGB", (3 * 320, 2 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(raw["images"]):
            name = row["name"]
            input_mask = raw_dir / "masks" / f"{name}.png"
            if input_mask.is_symlink() or not input_mask.is_file() or input_mask.stat().st_size > MAX_RAW_MASK_BYTES:
                raise ValueError("raw mask must be a bounded regular PNG")
            payload = input_mask.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["mask_sha256"]:
                raise ValueError("raw mask hash mismatch")
            with Image.open(input_mask) as loaded:
                loaded.load()
                if loaded.mode != "L" or loaded.size != common.IMAGE_SIZE:
                    raise ValueError("raw mask mode/size mismatch")
                raw_mask = np.array(loaded)
            clean, stats = largest_component(raw_mask)
            raw_output = output / "raw_masks" / f"{name}.png"
            clean_output = output / "cleaned_masks" / f"{name}.png"
            raw_output.write_bytes(payload)
            used += len(payload)
            used = common.save_bounded_image(Image.fromarray(clean, "L"), clean_output, "PNG", used)
            photo = common.verified_training_photo(dataset_root, rows[name])
            for lane, pixels in enumerate((raw_mask, clean)):
                masked = Image.new("RGB", photo.size, "black")
                masked.paste(photo, (0, 0), Image.fromarray(pixels, "L"))
                crop = masked.crop(trial.BOX)
                crop.thumbnail((310, 245), Image.Resampling.LANCZOS)
                sheet.paste(crop, (index * 320 + (320 - crop.width) // 2, lane * 280))
                draw.text((index * 320 + 8, lane * 280 + 253),
                          name + (" raw" if lane == 0 else " largest component"), fill="black")
            report["images"].append({"name": name, "source_sha256": rows[name]["sha256"],
                                     "raw_mask_sha256": row["mask_sha256"],
                                     "cleaned_mask_sha256": common.digest(clean_output), **stats})
            if common.digest(input_mask) != row["mask_sha256"]:
                raise ValueError("raw mask changed during cleanup")
        sheet_path = output / "raw_vs_cleaned_crop.jpg"
        used = common.save_bounded_image(sheet, sheet_path, "JPEG", used, quality=90)
        report["review_sheet_sha256"] = common.digest(sheet_path)
        if used > MAX_OUTPUT_BYTES - 1024**2:
            raise ValueError("output exceeds 19 MiB artifact reserve")
        if (common.digest(raw_manifest_path) != raw_manifest_sha or common.digest(package_path) != package_sha or
                common.digest(__file__) != runner_sha):
            raise ValueError("source manifest/package/runner changed during cleanup")
        report["status"] = "complete_unreviewed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        raise
    finally:
        encoded = (json.dumps(report, indent=2) + "\n").encode()
        if len(encoded) > 1024**2:
            raise ValueError("manifest exceeds 1 MiB")
        (output / "manifest.json").write_bytes(encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest-sha256", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw_dir, args.raw_manifest_sha256, args.package, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
