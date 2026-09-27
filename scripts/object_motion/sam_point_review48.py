#!/usr/bin/env python3
"""Train-only, inference-free original-RGB review of common SAM2 point prompts."""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from scripts.object_motion import sam_mask_trial as prior
from scripts.object_motion import ycb_object_masks as common


POINTS = ((600, 525, 1), (720, 620, 0), (520, 630, 0))
PANEL_SIZE = (280, 370)
MAX_OUTPUT = 20 * 1024**2


def validate_policy(rows):
    if len(rows) != 48:
        raise ValueError("exactly 48 frozen training rows required")
    x0, y0, x1, y1 = prior.BOX
    for x, y, label in POINTS:
        if not x0 <= x < x1 or not y0 <= y < y1 or label not in (0, 1):
            raise ValueError("common point outside fixed original-pixel box")
    if [p[2] for p in POINTS] != [1, 0, 0]:
        raise ValueError("one positive and two negatives required")


def prompt_rows(rows):
    validate_policy(rows)
    return [{"name": Path(row["path"]).name, "source_sha256": row["sha256"],
             "points_xy_label": [list(point) for point in POINTS]} for row in rows]


def annotated_crop(photo, points=POINTS):
    crop = photo.crop(prior.BOX).copy()
    drawing = ImageDraw.Draw(crop)
    for x, y, label in points:
        xx, yy = x - prior.BOX[0], y - prior.BOX[1]
        color = "lime" if label else "red"
        drawing.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), outline=color, width=3)
        drawing.text((xx + 8, yy - 9), "+" if label else "-", fill=color,
                     stroke_width=1, stroke_fill="black")
    return crop


def prepare(package_path, dataset_root, output):
    package_path, output = Path(package_path), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != prior.OBJECT_ID:
        raise ValueError("mustard package required")
    rows = common.training_records(package)
    prompt_entries = prompt_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output filesystem below 10 GiB free")
    runner_sha = common.digest(__file__)
    helper_sha = common.digest(common.__file__)
    output.mkdir()
    report = {"schema": "sam21_mustard_common_prompt_review_v1", "status": "running",
              "source_scope": "all 48 training original RGB photos, no held-out/GT/mask/depth/pose",
              "prompt_role": "visual proposal only; no inference or mask quality claim",
              "box_xyxy_original_pixels": list(prior.BOX),
              "common_points_xy_label": [list(p) for p in POINTS],
              "package_sha256": package_sha, "runner_sha256": runner_sha,
              "shared_helper_sha256": helper_sha, "sheets": []}
    try:
        for sheet_index in range(4):
            sheet = Image.new("RGB", (3 * PANEL_SIZE[0], 4 * PANEL_SIZE[1]), "white")
            draw = ImageDraw.Draw(sheet)
            for panel_index, row in enumerate(rows[sheet_index * 12:(sheet_index + 1) * 12]):
                photo = common.verified_training_photo(dataset_root, row)
                crop = annotated_crop(photo)
                x = (panel_index % 3) * PANEL_SIZE[0]
                y = (panel_index // 3) * PANEL_SIZE[1]
                sheet.paste(crop, (x, y))
                draw.text((x + 4, y + 352), Path(row["path"]).name, fill="black")
            name = f"training_prompt_review_{sheet_index + 1:02}.jpg"
            destination = output / name
            sheet.save(destination, "JPEG", quality=94)
            report["sheets"].append({"name": name, "sha256": common.digest(destination),
                                     "photo_names": [Path(r["path"]).name for r in rows[sheet_index * 12:(sheet_index + 1) * 12]]})
        prompt_document = {"schema": "sam21_mustard_common_point_prompts_48_v1",
                           "source": "48 frozen training RGB photos only; review required before inference",
                           "box_xyxy_original_pixels": list(prior.BOX), "images": prompt_entries}
        prompt_path = output / "prompts.json"
        prompt_path.write_text(json.dumps(prompt_document, indent=2) + "\n")
        report["prompt_manifest_sha256"] = common.digest(prompt_path)
        if (sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > MAX_OUTPUT or
                common.digest(package_path) != package_sha or common.digest(__file__) != runner_sha or
                common.digest(common.__file__) != helper_sha):
            raise ValueError("output cap or immutable source hash failed")
        report["status"] = "complete_pending_visual_review"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("review report exceeds 1 MiB")
        (output / "report.json").write_bytes(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.package, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
