"""Write a small analytic turntable scene in the dense pipeline's input format.

A unit sphere with a texture fixed to its surface, darker than a plain backdrop,
seen by a ring of pinhole cameras. Exact depth is known, so tests and smoke runs
can check the stereo and the mesh without any dataset. NumPy and Pillow only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def camera(angle, elevation, distance):
    """World-to-camera rotation and translation looking at the origin, y down."""
    a, e = np.radians(angle), np.radians(elevation)
    center = distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    return rotation, -rotation @ center


def texture(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    value = (np.sin(9 * x + 2 * y) + np.sin(11 * y - 3 * z) + np.sin(13 * z + 5 * x)
             + 0.7 * np.sin(31 * x - 17 * y + 23 * z)) / 3.7
    return 0.38 + 0.22 * value


def render(rotation, translation, k, size, radius=1.0):
    """Grey image, mask and exact z-depth of the sphere for one camera."""
    h, w = size[1], size[0]
    yy, xx = np.mgrid[:h, :w].astype(float)
    rays = np.stack(((xx + 0.5 - k[2]) / k[0], (yy + 0.5 - k[3]) / k[1], np.ones_like(xx)), -1)
    center = -rotation.T @ translation
    directions = rays @ rotation  # world directions, one unit of camera z each
    b = directions @ center
    a = (directions * directions).sum(-1)
    discriminant = b * b - a * (center @ center - radius * radius)
    hit = discriminant > 0
    depth = np.where(hit, (-b - np.sqrt(np.maximum(discriminant, 0))) / a, 0)
    points = center + directions * depth[..., None]
    gray = np.where(hit, texture(points), 0.82)
    return gray, hit, depth


def write(output, *, views=24, size=(128, 128), distance=6.0, elevation=10.0, focal=227.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    k = [focal, focal, size[0] / 2, size[1] / 2]
    rows, depths = [], {}
    for n in range(views):
        rotation, translation = camera(360 * n / views, elevation, distance)
        gray, mask, depth = render(rotation, translation, k, size)
        name = f"view_{n:03d}"
        value = (255 * np.clip(gray, 0, 1)).astype(np.uint8)
        Image.fromarray(np.stack((value,) * 3, -1)).save(output / "images" / f"{name}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(output / "masks" / f"{name}.png")
        rows.append({"name": name, "source": f"{name}.png", "image": f"images/{name}.png",
                     "mask": f"masks/{name}.png", "width": size[0], "height": size[1], "k": k,
                     "rotation": rotation.tolist(), "translation": translation.tolist()})
        depths[name] = depth.astype(np.float32)
    generator = np.random.default_rng(0)
    sparse = generator.normal(size=(400, 3))
    sparse /= np.linalg.norm(sparse, axis=1, keepdims=True)
    np.save(output / "sparse_points.npy", sparse)
    np.savez_compressed(output / "exact_depths.npz", **depths)
    (output / "cameras.json").write_text(json.dumps({"views": rows}, indent=1) + "\n")
    return rows


# Settings sized for the synthetic scene; the defaults target megapixel photos.
SMALL = ("sizes=64,128", "grid=96", "planes=48", "neighbours=4", "best_of=2", "vote_neighbours=4",
         "min_votes=2,2", "crop_padding=6", "hull_dilate=1", "windows=5,7", "aggregates=1,1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=24)
    args = parser.parse_args()
    write(args.output, views=args.views)
    print("suggested overrides: " + " ".join("--set " + item for item in SMALL))


if __name__ == "__main__":
    main()
