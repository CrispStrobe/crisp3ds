"""Explicit dark-object/bright-background cleanup of bounded enclosed mask holes."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_budget(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 0.05
    ):
        raise ValueError(
            "maximum_total_filled_foreground_fraction must be finite in0..0.05"
        )
    return float(value)


def clean_holes(
    rgb,
    mask,
    *,
    dark_object_bright_background=False,
    maximum_total_filled_foreground_fraction=0.01,
):
    """Fill only small enclosed background whose every photo pixel is dark."""
    if dark_object_bright_background is not True:
        raise ValueError("explicit dark_object_bright_background=True prior required")
    budget = validate_budget(maximum_total_filled_foreground_fraction)
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must be uint8 HxWx3")
    if mask.dtype != bool or mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError("mask must be nonempty boolean with matching RGB dimensions")
    gray = np.array(Image.fromarray(rgb).convert("L"))
    labels, count = ndimage.label(~mask)
    boundary = set(
        np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    )
    clean = mask.copy()
    rows = []
    for label in range(1, count + 1):
        if label in boundary:
            continue
        component = labels == label
        area = int(component.sum())
        maximum = int(gray[component].max())
        filled = area <= 2048 and maximum <= 128
        rows.append(
            {
                "label": label,
                "pixels": area,
                "maximum_luminance": maximum,
                "filled": filled,
            }
        )
        if filled:
            clean[component] = True
    pixels = int(clean.sum() - mask.sum())
    if pixels > int(mask.sum()) * budget:
        raise ValueError(
            f"total filled pixels exceed {budget:g} fraction original foreground"
        )
    return clean, {
        "original_foreground_pixels": int(mask.sum()),
        "filled_pixels": pixels,
        "enclosed_components": rows,
    }


def run(
    images,
    masks,
    output,
    *,
    dark_object_bright_background=False,
    maximum_total_filled_foreground_fraction=0.01,
):
    if dark_object_bright_background is not True:
        raise ValueError("explicit dark-object/bright-background prior required")
    budget = validate_budget(maximum_total_filled_foreground_fraction)
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(
        p for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not 1 <= len(paths) <= 255:
        raise ValueError("requires 1..255 RGB photos")
    sealed = (
        [Path(__file__).resolve()] + paths + [masks / (p.name + ".png") for p in paths]
    )
    before = {str(p): digest(p) for p in sealed}
    configuration = {
        "maximum_hole_pixels": 2048,
        "maximum_all_pixel_luminance": 128,
        "maximum_total_filled_foreground_fraction": budget,
        "grayscale": "Pillow RGB.convert L",
        "background_connectivity": 4,
        "dark_object_bright_background": True,
    }
    output.mkdir(parents=True)
    (output / "frozen.json").write_text(
        json.dumps(
            {"configuration": configuration, "source_hashes_before": before}, indent=2
        )
        + "\n"
    )
    report = {
        "schema": "photo_dark_hole_cleanup_v1",
        "status": "running",
        "configuration": configuration,
        "quality_accepted": False,
        "reference_used": False,
        "rows": [],
    }
    prepared = []
    try:
        for p in paths:
            with Image.open(p) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError("upright RGB required")
                rgb = np.array(im.convert("RGB"))
            with Image.open(masks / (p.name + ".png")) as im:
                values = np.array(im)
                if values.ndim != 2 or not np.isin(values, [0, 255]).all():
                    raise ValueError("binary single-channel masks required")
                mask = values > 0
            clean, row = clean_holes(
                rgb,
                mask,
                dark_object_bright_background=True,
                maximum_total_filled_foreground_fraction=budget,
            )
            prepared.append((p.name, clean))
            report["rows"].append({"name": p.name, **row})
        after = {str(p): digest(p) for p in sealed}
        if before != after:
            raise ValueError("source changed during cleanup")
        # All masks must pass the same frozen policy before publishing any mask.
        (output / "masks").mkdir()
        for name, clean in prepared:
            target = output / "masks" / (name + ".png")
            Image.fromarray(clean.astype(np.uint8) * 255).save(target)
        report.update(
            status="complete_unreviewed",
            source_hashes_before=before,
            source_hashes_after=after,
            output_hashes={
                str(p): digest(p) for p in sorted((output / "masks").iterdir())
            },
        )
    except Exception as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "masks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dark-object-bright-background", action="store_true")
    parser.add_argument(
        "--maximum-total-filled-foreground-fraction", type=float, default=0.01
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.images,
                args.masks,
                args.output,
                dark_object_bright_background=args.dark_object_bright_background,
                maximum_total_filled_foreground_fraction=args.maximum_total_filled_foreground_fraction,
            )
        )
    )


if __name__ == "__main__":
    main()
