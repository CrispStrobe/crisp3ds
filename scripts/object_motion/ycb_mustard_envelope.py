#!/usr/bin/env python3
"""Frozen successor: label-preserving mustard pose-support row envelope.

Photo-only and training-only. The first warm-scanline candidate remains immutable
and was rejected; this separately labeled successor is not an exact silhouette.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_motion import ycb_object_masks as common
from scripts.object_motion import ycb_mustard_support as first


ROI = first.ROI
ROW_RADIUS = 24
HORIZONTAL_MARGIN = 12
MAX_ANCHOR_GAP = 64
MIN_WIDTH, MAX_WIDTH = 35, 250
MIN_AREA, MAX_AREA = 5_000, 50_000
MAX_EXPANSION = 2.5


def support_mask(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.shape != (1024, 1280, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected 1024x1280 uint8 RGB photo")
    x0, y0, x1, y1 = ROI
    crop = rgb[y0:y1, x0:x1].astype(np.int16)
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    seeds = ((red >= 35) & (green >= 25) &
             (20 * red > 21 * blue) & (20 * green > 21 * blue) &
             (red - blue >= 10) & (green - blue >= 10))
    anchors = {}
    for local_y in range(y1 - y0):
        xs = np.flatnonzero(seeds[local_y])
        if len(xs) >= first.MIN_SEEDS_PER_ROW:
            anchors[local_y] = (int(xs[0]), int(xs[-1]))
    if not first.MIN_ACTIVE_ROWS <= len(anchors) <= first.MAX_ACTIVE_ROWS:
        raise ValueError(f"warm-seed row count out of bounds: {len(anchors)}")
    keys = sorted(anchors)
    if any(right - left - 1 > MAX_ANCHOR_GAP for left, right in zip(keys, keys[1:])):
        raise ValueError("warm evidence has unsupported row gap over 64 pixels")
    seed_span_pixels = sum(hi - lo + 1 for lo, hi in anchors.values())
    rows = {}
    interpolated = 0
    for local_y in range(keys[0], keys[-1] + 1):
        neighbors = [anchors[y] for y in keys if abs(y - local_y) <= ROW_RADIUS]
        if neighbors:
            lo, hi = min(pair[0] for pair in neighbors), max(pair[1] for pair in neighbors)
        else:
            left = max(y for y in keys if y < local_y)
            right = min(y for y in keys if y > local_y)
            alpha = (local_y - left) / (right - left)
            lo = round((1 - alpha) * anchors[left][0] + alpha * anchors[right][0])
            hi = round((1 - alpha) * anchors[left][1] + alpha * anchors[right][1])
            interpolated += 1
        rows[local_y] = (max(0, lo - HORIZONTAL_MARGIN),
                         min(x1 - x0 - 1, hi + HORIZONTAL_MARGIN))
    min_x = min(lo for lo, _ in rows.values())
    max_x = max(hi for _, hi in rows.values())
    width = max_x - min_x + 1
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise ValueError(f"envelope bbox width out of bounds: {width}")
    if min_x == 0 or max_x == x1 - x0 - 1 or keys[0] == 0 or keys[-1] == y1 - y0 - 1:
        raise ValueError("envelope touches fixed ROI boundary")
    mask = np.zeros((1024, 1280), dtype=np.uint8)
    for local_y, (lo, hi) in rows.items():
        mask[y0 + local_y, x0 + lo:x0 + hi + 1] = 255
    pixels = int(np.count_nonzero(mask))
    if not MIN_AREA <= pixels <= MAX_AREA:
        raise ValueError(f"envelope support pixel count out of bounds: {pixels}")
    expansion = pixels / seed_span_pixels
    if expansion > MAX_EXPANSION:
        raise ValueError(f"envelope expansion beyond warm evidence: {expansion:.3f}")
    return mask, {"warm_seed_rows": len(keys), "envelope_rows": len(rows),
                  "interpolated_rows_without_local_anchor": interpolated,
                  "warm_seed_span_pixels": seed_span_pixels,
                  "support_pixels": pixels, "expansion_ratio": expansion,
                  "bbox_xywh": [x0 + min_x, y0 + keys[0], width, keys[-1] - keys[0] + 1]}


def prepare(package_path, dataset_root, output):
    package_path, output = Path(package_path), Path(output)
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != first.OBJECT_ID:
        raise ValueError("mustard package required")
    if common.digest(package_path) != package_sha:
        raise ValueError("package changed during validation")
    runner_sha = common.digest(__file__)
    common_sha = common.digest(common.__file__)
    predecessor_sha = common.digest(first.__file__)
    rows = common.training_records(package)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("less than 10 GiB free at output")
    output.mkdir()
    mask_dir = output / "masks"
    mask_dir.mkdir()
    report = {"schema": "ycb_mustard_photo_envelope_v1", "status": "running",
              "object_id": first.OBJECT_ID, "source_package_sha256": package_sha,
              "runner_sha256": runner_sha, "shared_helper_sha256": common_sha,
              "predecessor_helper_sha256": predecessor_sha,
              "predecessor_status": "first warm-scanline candidate rejected by visual QA; preserved separately",
              "source_scope": "48 training RGB photos only; no held-out/reference/pose/depth files opened",
              "mask_status": "generated_unreviewed",
              "mask_role": "label-preserving coarse pose support, not exact silhouette or mesh mask",
              "fixed_policy": {"roi_xyxy_exclusive": ROI, "warm_seed": "R>=35,G>=25,R>1.05B,G>1.05B,R-B>=10,G-B>=10",
                               "seed_pixels_per_row_min": first.MIN_SEEDS_PER_ROW,
                               "neighbor_row_radius": ROW_RADIUS,
                               "horizontal_margin_px": HORIZONTAL_MARGIN,
                               "maximum_anchor_gap_rows": MAX_ANCHOR_GAP,
                               "seed_rows_range": [first.MIN_ACTIVE_ROWS, first.MAX_ACTIVE_ROWS],
                               "bbox_width_range": [MIN_WIDTH, MAX_WIDTH],
                               "support_pixel_range": [MIN_AREA, MAX_AREA],
                               "maximum_expansion_ratio": MAX_EXPANSION},
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
                common.digest(common.__file__) != common_sha or common.digest(first.__file__) != predecessor_sha):
            raise ValueError("package or helper changed during mask generation")
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
