"""Render photos and labelled STL rows through the same recovered cameras.

Painter-sorted flat shading of the actual STL triangles; visualization only. Every
row uses identical cameras, crop, scale and lighting.
"""

import argparse
from pathlib import Path
import struct

import cv2
import numpy as np

from .multiscale_stereo import load_views


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise ValueError("binary STL required")
    record = np.frombuffer(data, dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")], offset=84)
    return record["vertices"].astype(np.float64)


def shade(triangles, row, center, scale, size, supersample=2):
    width, height = size[0] * supersample, size[1] * supersample
    R, t = np.array(row["rotation"]), np.array(row["translation"])
    fx, fy, cx, cy = row["k"]
    cam = triangles @ R.T + t
    xy = cam[:, :, :2] / cam[:, :, 2:] * [fx, fy] + [cx, cy]
    pixels = ((xy - center) * scale + [size[0] / 2, size[1] / 2]) * supersample
    normals = np.cross(cam[:, 1] - cam[:, 0], cam[:, 2] - cam[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    light = np.array([-0.35, -0.5, -1.0])
    light /= np.linalg.norm(light)
    values = 0.30 + 0.65 * np.maximum(0, normals @ light)
    panel = np.full((height, width, 3), [249, 247, 246], np.uint8)
    for face in np.argsort(cam[:, :, 2].mean(1))[::-1]:
        color = tuple(int(c * values[face]) for c in (204, 186, 170))
        cv2.fillConvexPoly(panel, np.rint(pixels[face]).astype(np.int32), color)
    return cv2.resize(panel, size, interpolation=cv2.INTER_AREA)


def run(cameras, names, meshes, output, size=(560, 430)):
    rows = {r["name"]: r for r in load_views(Path(cameras).parent)}
    rows = [rows[n] for n in names]
    loaded = [(label, read_stl(path)) for label, path in meshes]
    header = 34
    canvas = np.full(((len(loaded) + 1) * (size[1] + header), len(rows) * size[0], 3), [249, 247, 246], np.uint8)
    for col, row in enumerate(rows):
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mask)
        center = np.array([(xs.min() + xs.max()) / 2 + 0.5, (ys.min() + ys.max()) / 2 + 0.5])
        scale = 0.9 * min(size[0] / (xs.max() - xs.min()), size[1] / (ys.max() - ys.min()))
        photo = cv2.imread(row["image"])
        affine = np.array([[scale, 0, size[0] / 2 - scale * center[0]], [0, scale, size[1] / 2 - scale * center[1]]])
        canvas[header : header + size[1], col * size[0] : (col + 1) * size[0]] = cv2.warpAffine(photo, affine, size, borderValue=(249, 247, 246))
        for index, (_, triangles) in enumerate(loaded):
            y = (index + 1) * (size[1] + header) + header
            canvas[y : y + size[1], col * size[0] : (col + 1) * size[0]] = shade(triangles, row, center, scale, size)
    labels = ["Input photographs"] + [f"{label}  ({len(t):,} triangles)" for label, t in loaded]
    for index, text in enumerate(labels):
        cv2.putText(canvas, text, (12, index * (size[1] + header) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (49, 36, 26), 2, cv2.LINE_AA)
    if Path(output).exists():
        raise FileExistsError(output)
    cv2.imwrite(str(output), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--mesh", nargs=2, action="append", metavar=("LABEL", "STL"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cameras, args.views, args.mesh, args.output)


if __name__ == "__main__":
    main()
