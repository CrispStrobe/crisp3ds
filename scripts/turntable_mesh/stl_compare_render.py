"""Render photos and labelled STL rows through the same recovered cameras.

Z-buffered flat shading of the actual STL triangles; visualization only. Every
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


def rasterise(pixels, depth, values, width, height):
    """Z-buffered flat shading. pixels N×3×2, depth N×3, values N; returns H×W float, NaN where empty.

    Triangles are grouped by bounding-box size and tested against every pixel
    centre of their box at once, so a million small triangles need no Python loop.
    """
    zbuffer = np.full(height * width, np.inf)
    image = np.full(height * width, np.nan)
    low = np.floor(pixels.min(1)).astype(np.int64)
    span = np.ceil(pixels.max(1)).astype(np.int64) - low + 1
    size = span.max(1)
    a, b, c = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    usable = np.abs(area) > 1e-12
    edge = 2
    while usable.any():
        chosen = np.flatnonzero(usable & (size <= edge))
        usable[chosen] = False
        oy, ox = np.mgrid[:edge, :edge]
        for start in range(0, len(chosen), max(1, 4_000_000 // (edge * edge))):
            t = chosen[start : start + max(1, 4_000_000 // (edge * edge))]
            px = low[t, 0, None] + ox.reshape(-1)[None]
            py = low[t, 1, None] + oy.reshape(-1)[None]
            x, y = px + 0.0, py + 0.0
            w0 = ((b[t, 0, None] - x) * (c[t, 1, None] - y) - (b[t, 1, None] - y) * (c[t, 0, None] - x)) / area[t, None]
            w1 = ((c[t, 0, None] - x) * (a[t, 1, None] - y) - (c[t, 1, None] - y) * (a[t, 0, None] - x)) / area[t, None]
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
            z = w0 * depth[t, 0, None] + w1 * depth[t, 1, None] + w2 * depth[t, 2, None]
            if not inside.any():
                continue
            index = (py * width + px)[inside]
            z = z[inside]
            value = np.broadcast_to(values[t, None], inside.shape)[inside]
            nearer = z < zbuffer[index]
            index, z, value = index[nearer], z[nearer], value[nearer]
            # Far to near: for repeated indices NumPy keeps the last assignment.
            order = np.argsort(-z, kind="stable")
            zbuffer[index[order]] = z[order]
            image[index[order]] = value[order]
        edge *= 2
    return image.reshape(height, width)


def shade(triangles, row, center, scale, size, supersample=2):
    width, height = size[0] * supersample, size[1] * supersample
    R, t = np.array(row["rotation"]), np.array(row["translation"])
    fx, fy, cx, cy = row["k"]
    cam = triangles @ R.T + t
    xy = cam[:, :, :2] / cam[:, :, 2:] * [fx, fy] + [cx, cy]
    pixels = ((xy - center) * scale + [size[0] / 2, size[1] / 2]) * supersample - 0.5
    normals = np.cross(cam[:, 1] - cam[:, 0], cam[:, 2] - cam[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    light = np.array([-0.35, -0.5, -1.0])
    light /= np.linalg.norm(light)
    values = 0.30 + 0.65 * np.maximum(0, normals @ light)
    visible = (cam[:, :, 2] > 0).all(1) & (pixels.max(1) >= 0).all(1) & (pixels.min(1) < [width, height]).all(1)
    grey = rasterise(pixels[visible], cam[visible][:, :, 2], values[visible], width, height)
    panel = np.where(np.isnan(grey)[..., None], np.array([249.0, 247.0, 246.0]),
                     np.nan_to_num(grey)[..., None] * np.array([204.0, 186.0, 170.0]))
    return cv2.resize(panel.astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def run(cameras, names, meshes, output, size=(560, 430), crops=None):
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
        if crops and row["name"] in crops:
            # Source-image pixels, shared by the photo and every mesh row. Do not
            # enlarge a rendered overview: rasterise the actual triangles again.
            x, y, width, height = crops[row["name"]]
            center = np.array([x + width / 2, y + height / 2])
            scale = min(size[0] / width, size[1] / height)
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
    parser.add_argument("--crop", nargs=5, action="append", metavar=("VIEW", "X", "Y", "WIDTH", "HEIGHT"),
                        help="Detail crop in source-image pixels; repeat for different views")
    args = parser.parse_args()
    crops = {}
    for name, *coordinates in args.crop or []:
        try:
            box = tuple(float(value) for value in coordinates)
        except ValueError:
            parser.error("crop coordinates must be numbers")
        if name not in args.views or not np.isfinite(box).all() or min(box[2:]) <= 0:
            parser.error("each crop needs a selected view, finite coordinates and positive width/height")
        if name in crops:
            parser.error(f"duplicate crop for {name}")
        crops[name] = box
    run(args.cameras, args.views, args.mesh, args.output, crops=crops)


if __name__ == "__main__":
    main()
