#!/usr/bin/env python3
"""Three shared-bounds orthographic overlays for registration QA, not scoring."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_dataset import evaluate


ROOT = Path(__file__).resolve().parents[2]
PANEL = 560
MARGIN = 42
SEED = 2030
MAX_SAMPLES = 10_000
MIN_FREE = 10 * 1024**3
VIEWS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))


def make_overlay(reference, candidate, matrix, count=10_000):
    """Area-sample both whole meshes and draw both on the SAME axes per view."""
    if not 1 <= count <= MAX_SAMPLES:
        raise ValueError("overlay sample count outside 1..10000")
    ref = evaluate.sample_surface(*reference, count, SEED)
    out = evaluate.sample_surface(*candidate, count, SEED + 1)
    out = out @ matrix[:3, :3].T + matrix[:3, 3]
    image = Image.new("RGBA", (PANEL, len(VIEWS) * PANEL), "white")
    bounds = []
    for row, (a, b, label) in enumerate(VIEWS):
        projected = np.concatenate((ref[:, [a, b]], out[:, [a, b]]))
        lo, hi = projected.min(axis=0), projected.max(axis=0)
        if not np.isfinite(projected).all() or float(max(hi - lo)) <= 0:
            raise ValueError("nonfinite or flat registration preview projection")
        center = (lo + hi) / 2
        radius = 0.525 * float(max(hi - lo))
        scale = (PANEL - 2 * MARGIN) / (2 * radius)
        bounds.append({"view": label, "center": center.tolist(), "radius": radius})
        layer = Image.new("RGBA", (PANEL, PANEL), (255, 255, 255, 255))
        draw = ImageDraw.Draw(layer, "RGBA")
        for point in ref[:, [a, b]]:
            x, y = (point - center) * scale
            px, py = round(PANEL / 2 + float(x)), round(PANEL / 2 - float(y))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(8, 127, 140, 150))
        for point in out[:, [a, b]]:
            x, y = (point - center) * scale
            px, py = round(PANEL / 2 + float(x)), round(PANEL / 2 - float(y))
            draw.line((px - 3, py, px + 3, py), fill=(212, 104, 18, 200), width=1)
            draw.line((px, py - 3, px, py + 3), fill=(212, 104, 18, 200), width=1)
        draw.rectangle((0, 0, PANEL - 1, 27), fill="white")
        draw.text((8, 6), f"{label}  cyan dots: Google ref   orange crosses: candidate", fill="black")
        image.alpha_composite(layer, (0, row * PANEL))
    return image.convert("RGB"), bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--save-preview", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()
    local_root = (ROOT / ".local-tools").resolve()
    target = args.save_preview.absolute()
    report_path = target.with_suffix(target.suffix + ".json")
    if (target.suffix.lower() != ".png" or not target.resolve().is_relative_to(local_root) or
            target.parent.is_symlink() or not target.parent.is_dir() or
            target.exists() or target.is_symlink() or report_path.exists() or report_path.is_symlink()):
        parser.error("preview and sidecar must be fresh PNG/JSON under .local-tools")
    if shutil.disk_usage(target.parent).free < MIN_FREE:
        parser.error("10 GiB disk reserve required")
    reference = evaluate.inspect_ply(args.reference, geometry=True)
    candidate = evaluate.inspect_ply(args.output, geometry=True)
    if reference[0]["nontriangle_faces"] or candidate[0]["nontriangle_faces"]:
        parser.error("triangular meshes required")
    provenance, matrix, scale = evaluate.load_sim3(args.transform, args.reference, args.output)
    image, bounds = make_overlay(reference[1:], candidate[1:], matrix, args.samples)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    png = stream.getvalue()
    if len(png) > 10 * 1024**2:
        raise ValueError("overlay PNG exceeds 10 MiB")
    with target.open("xb") as file:
        file.write(png)
    record = {"schema": "object_dataset_three_view_overlay_v1",
              "scope": "sampled whole-mesh registration visualization; not a quality metric",
              "reference_sha256": evaluate.sha256_file(args.reference),
              "output_sha256": evaluate.sha256_file(args.output),
              "transform_sha256": evaluate.sha256_file(args.transform),
              "renderer_sha256": evaluate.sha256_file(Path(__file__)),
              "registration_basis": provenance["registration_basis"],
              "fitted_scale": scale, "samples_per_mesh": args.samples,
              "reference_seed": SEED, "output_seed": SEED + 1,
              "shared_bounds_by_view": bounds,
              "preview_sha256": hashlib.sha256(png).hexdigest()}
    with report_path.open("xb") as file:
        file.write((json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps({"preview": str(target), "report": str(report_path),
                      "preview_sha256": record["preview_sha256"]}))


if __name__ == "__main__":
    main()
