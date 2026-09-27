#!/usr/bin/env python3
"""Frozen photo-only mustard-bottle coarse pose-support candidate.

This is a dataset-specific training-image heuristic, not an exact silhouette.
No reference geometry, supplied pose/depth/mask, or held-out photo is opened.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_motion import ycb_object_masks as common


OBJECT_ID = "006_mustard_bottle"
ROI = (480, 300, 760, 650)  # x0,y0,x1,y1 exclusive, original RGB pixels
MARGIN = 3
MAX_GAP = 3
MIN_SEEDS_PER_ROW = 4
MIN_ACTIVE_ROWS, MAX_ACTIVE_ROWS = 60, 330
MIN_BBOX_WIDTH, MAX_BBOX_WIDTH = 35, 240
MIN_SUPPORT, MAX_SUPPORT = 5_000, 100_000


def support_mask(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.shape != (1024, 1280, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected 1024x1280 uint8 RGB photo")
    x0, y0, x1, y1 = ROI
    crop = rgb[y0:y1, x0:x1].astype(np.int16)
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    seeds = ((red >= 35) & (green >= 25) &
             (20 * red > 21 * blue) & (20 * green > 21 * blue) &
             (red - blue >= 10) & (green - blue >= 10))
    rows = {}
    for local_y in range(y1 - y0):
        xs = np.flatnonzero(seeds[local_y])
        if len(xs) >= MIN_SEEDS_PER_ROW:
            lo = max(0, int(xs[0]) - MARGIN)
            hi = min(x1 - x0 - 1, int(xs[-1]) + MARGIN)
            rows[local_y] = (lo, hi)
    if not MIN_ACTIVE_ROWS <= len(rows) <= MAX_ACTIVE_ROWS:
        raise ValueError(f"mustard warm-seed row count out of bounds: {len(rows)}")
    keys = sorted(rows)
    for previous, following in zip(keys, keys[1:]):
        gap = following - previous - 1
        if 0 < gap <= MAX_GAP:
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1)
                rows[previous + offset] = (
                    round((1 - alpha) * rows[previous][0] + alpha * rows[following][0]),
                    round((1 - alpha) * rows[previous][1] + alpha * rows[following][1]))
    min_x = min(lo for lo, _ in rows.values())
    max_x = max(hi for _, hi in rows.values())
    min_y, max_y = min(rows), max(rows)
    width = max_x - min_x + 1
    if not MIN_BBOX_WIDTH <= width <= MAX_BBOX_WIDTH:
        raise ValueError(f"mustard support bbox width out of bounds: {width}")
    if min_x == 0 or max_x == x1 - x0 - 1 or min_y == 0 or max_y == y1 - y0 - 1:
        raise ValueError("mustard support touches fixed ROI boundary")
    mask = np.zeros((1024, 1280), dtype=np.uint8)
    for local_y, (lo, hi) in rows.items():
        mask[y0 + local_y, x0 + lo:x0 + hi + 1] = 255
    pixels = int(np.count_nonzero(mask))
    if not MIN_SUPPORT <= pixels <= MAX_SUPPORT:
        raise ValueError(f"mustard support pixel count out of bounds: {pixels}")
    return mask, {"warm_seed_rows": len(keys), "bridged_rows": len(rows) - len(keys),
                  "support_pixels": pixels,
                  "bbox_xywh": [x0 + min_x, y0 + min_y, width, max_y - min_y + 1]}


def prepare(package_path, dataset_root, output):
    package_path, output = Path(package_path), Path(output)
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != OBJECT_ID:
        raise ValueError("mustard package required")
    if common.digest(package_path) != package_sha:
        raise ValueError("package changed during validation")
    runner_sha = common.digest(__file__)
    common_sha = common.digest(common.__file__)
    rows = common.training_records(package)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("less than 10 GiB free at output")
    output.mkdir()
    mask_dir = output / "masks"
    mask_dir.mkdir()
    report = {"schema": "ycb_mustard_photo_support_v1", "status": "running",
              "object_id": OBJECT_ID, "source_package_sha256": package_sha,
              "runner_sha256": runner_sha, "shared_helper_sha256": common_sha,
              "source_scope": "48 training RGB photos only; no held-out/reference/pose/depth files opened",
              "mask_status": "generated_unreviewed",
              "mask_role": "coarse pose support only, not exact silhouette or mesh mask",
              "fixed_policy": {"roi_xyxy_exclusive": ROI, "red_min": 35, "green_min": 25,
                               "red_and_green_over_blue": ">1.05", "red_and_green_minus_blue_min": 10,
                               "seed_pixels_per_row_min": MIN_SEEDS_PER_ROW,
                               "horizontal_margin_px": MARGIN, "bridge_missing_rows_max": MAX_GAP,
                               "seed_rows_range": [MIN_ACTIVE_ROWS, MAX_ACTIVE_ROWS],
                               "bbox_width_range": [MIN_BBOX_WIDTH, MAX_BBOX_WIDTH],
                               "support_pixel_range": [MIN_SUPPORT, MAX_SUPPORT]},
              "images": []}
    used = 0
    try:
        tile = (200, 210)
        masked_sheet = Image.new("RGB", (8 * tile[0], 6 * tile[1]), "white")
        overlay_sheet = Image.new("RGB", masked_sheet.size, "white")
        masked_draw, overlay_draw = ImageDraw.Draw(masked_sheet), ImageDraw.Draw(overlay_sheet)
        for index, row in enumerate(rows):
            name = Path(row["path"]).name
            image = common.verified_training_photo(dataset_root, row)
            array_mask, stats = support_mask(np.asarray(image))
            mask = Image.fromarray(array_mask, "L")
            destination = mask_dir / f"{name}.png"
            used = common.save_bounded_image(mask, destination, "PNG", used)
            region_mask = mask.crop(ROI)
            source_crop = image.crop(ROI)
            masked = Image.new("RGB", source_crop.size, "black")
            masked.paste(source_crop, (0, 0), region_mask)
            overlay = source_crop.copy()
            overlay.paste(Image.new("RGB", source_crop.size, (255, 0, 0)), (0, 0),
                          region_mask.point(lambda value: value // 3))
            x, y = (index % 8) * tile[0], (index // 8) * tile[1]
            for preview, sheet, draw in ((masked, masked_sheet, masked_draw),
                                         (overlay, overlay_sheet, overlay_draw)):
                preview.thumbnail((tile[0] - 8, tile[1] - 28), Image.Resampling.LANCZOS)
                sheet.paste(preview, (x + (tile[0] - preview.width) // 2, y + 2))
                draw.text((x + 4, y + tile[1] - 22), name, fill="black")
            report["images"].append({"name": name, "image_sha256": row["sha256"],
                                     "mask_sha256": common.digest(destination), **stats})
        for image, filename, field in ((masked_sheet, "training_masked_sheet.jpg", "masked_sheet_sha256"),
                                       (overlay_sheet, "training_overlay_sheet.jpg", "overlay_sheet_sha256")):
            destination = output / filename
            used = common.save_bounded_image(image, destination, "JPEG", used, quality=90)
            report[field] = common.digest(destination)
        if (common.digest(package_path) != package_sha or common.digest(__file__) != runner_sha or
                common.digest(common.__file__) != common_sha):
            raise ValueError("package, runner, or shared helper changed during mask generation")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = str(error)
        raise
    finally:
        manifest = (json.dumps(report, indent=2) + "\n").encode()
        if len(manifest) > 1024 ** 2:
            raise ValueError("output manifest exceeds 1 MiB reserve")
        (output / "manifest.json").write_bytes(manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.package, args.dataset_root, args.output)
    print(json.dumps({"status": report["status"], "mask_status": report["mask_status"],
                      "training_views": len(report["images"])}))


if __name__ == "__main__":
    main()
