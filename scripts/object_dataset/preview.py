#!/usr/bin/env python3
"""Render paired orthographic sample previews after an explicit Sim(3) fit."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__:
    from . import evaluate
else:
    import evaluate

REPO = Path(__file__).resolve().parents[2]
MAX_SAMPLES = 20_000
PANEL = 560
MARGIN = 42
PROJECTIONS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))


def panel_bounds(reference, output, axes):
    merged = np.concatenate((reference[:, axes], output[:, axes]), axis=0)
    lo = merged.min(axis=0)
    hi = merged.max(axis=0)
    center = (lo + hi) / 2
    radius = float(max(hi - lo)) / 2
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("projected geometry has no finite extent")
    return center, radius * 1.05


def draw_sample(draw, points, axes, bounds, origin, color):
    center, radius = bounds
    scale = (PANEL - 2 * MARGIN) / (2 * radius)
    x0, y0 = origin
    projected = points[:, axes]
    xy = (projected - center) * scale
    for x, y in xy:
        px = round(x0 + PANEL / 2 + float(x))
        py = round(y0 + PANEL / 2 - float(y))
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)


def make_preview(reference, output, matrix, count=10_000, seed=2030):
    if not 1 <= count <= MAX_SAMPLES:
        raise ValueError("preview sample count exceeds 20,000")
    reference_samples = evaluate.sample_surface(*reference, count, seed)
    output_samples = evaluate.sample_surface(*output, count, seed)
    output_samples = output_samples @ matrix[:3, :3].T + matrix[:3, 3]
    image = Image.new("RGB", (2 * PANEL, len(PROJECTIONS) * PANEL), "white")
    draw = ImageDraw.Draw(image)
    for row, (a, b, label) in enumerate(PROJECTIONS):
        y0 = row * PANEL
        bounds = panel_bounds(reference_samples, output_samples, (a, b))
        draw_sample(draw, reference_samples, (a, b), bounds, (0, y0), "#087f8c")
        draw_sample(draw, output_samples, (a, b), bounds, (PANEL, y0), "#d46812")
        draw.text((12, y0 + 12), f"{label}  reference", fill="black")
        draw.text((PANEL + 12, y0 + 12), f"{label}  transformed reconstruction", fill="black")
        draw.line((PANEL, y0, PANEL, y0 + PANEL), fill="#cccccc", width=2)
        draw.line((0, y0 + PANEL - 1, 2 * PANEL, y0 + PANEL - 1), fill="#cccccc")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--save-preview", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()
    local_root = (REPO / ".local-tools").resolve()
    target = args.save_preview.resolve()
    if not target.is_relative_to(local_root) or args.save_preview.is_symlink():
        parser.error("preview path must be a fresh file under .local-tools")
    if target.exists():
        raise FileExistsError(target)
    _, matrix, _ = evaluate.load_sim3(args.transform, args.reference, args.output)
    _, rv, rf = evaluate.inspect_ply(args.reference, geometry=True)
    _, ov, of = evaluate.inspect_ply(args.output, geometry=True)
    image = make_preview((rv, rf), (ov, of), matrix, count=args.samples)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        image.save(stream, format="PNG")
    print(target)


if __name__ == "__main__":
    main()
