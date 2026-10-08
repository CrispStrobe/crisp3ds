"""Comparison sheet for one rendered case: photo / exact mesh / reconstruction coloured by error.

    python compare_sheet.py work/CASE [--title TEXT]

Reads work/CASE/capture/{truth.json,reference.ply,photos}, the STL in work/CASE/run and the
alignment in work/CASE/eval/result.json (scan_evaluate). Four views at the capture's own camera
directions (0, 90, 180, 270 degrees). The reconstruction is mapped into the reference frame with
the evaluator's similarity transform and coloured by its distance to the reference surface:
green at 0, yellow at 1 %, red at 2 % of the reference diagonal and beyond. Writes work/CASE/sheet.png.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.turntable_mesh.scan_evaluate import read_reference, sample_surface  # noqa: E402
from scripts.turntable_mesh.stl_compare_render import rasterise, read_stl  # noqa: E402

BG = np.array([249.0, 247.0, 246.0])


def project(triangles, rotation, centre, focal, size, ss, shift=(0.0, 0.0)):
    cam = (triangles - centre) @ rotation.T
    xy = cam[:, :, :2] / cam[:, :, 2:] * focal + np.array(size) / 2 + np.array(shift)
    return xy * ss - 0.5, cam


def panels(triangles, colours, rotation, centre, focal, size, shift, ss=2):
    """Flat-shaded triangles (per-triangle RGB in 0..1) through a pinhole camera; returns uint8 image."""
    pixels, cam = project(triangles, rotation, centre, focal, size, ss, shift)
    normals = np.cross(cam[:, 1] - cam[:, 0], cam[:, 2] - cam[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    light = np.array([-0.35, -0.5, -1.0]) / np.linalg.norm([-0.35, -0.5, -1.0])
    shade = 0.30 + 0.65 * np.abs(normals @ light)
    w, h = size[0] * ss, size[1] * ss
    index = rasterise(pixels, cam[:, :, 2], np.arange(len(triangles), dtype=np.float64), w, h)
    empty = np.isnan(index)
    i = np.nan_to_num(index).astype(np.int64)
    rgb = colours[i][..., ::-1] * shade[i, None] * 255.0  # OpenCV writes BGR
    image = np.where(empty[..., None], BG, rgb)
    return cv2.resize(image.astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def error_colour(distance):
    """0 green, 1 yellow, 2+ red (distance in units of 1 % of the diagonal)."""
    t = np.clip(distance / 2.0, 0, 1)
    r = np.clip(2 * t, 0, 1)
    g = np.clip(2 - 2 * t, 0, 1) * 0.85
    return np.stack([r * 0.9, g, np.full_like(t, 0.15)], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    case = args.case
    truth = json.loads((case / "capture/truth.json").read_text())
    vertices, faces = read_reference(case / "capture/reference.ply")
    reference = vertices[faces]
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    result = json.loads((case / "score/result.json").read_text())
    al = result["alignment"]
    linear = np.array(al["linear_part_including_mirror"])
    recon = read_stl(case / "run/mesh/mesh.stl") @ linear.T + np.array(al["translation"])
    samples, _ = sample_surface(reference, 400_000, np.random.default_rng(1))
    distance = cKDTree(samples).query(recon.mean(1))[0] / (0.01 * diagonal)
    grey = np.tile([[0.80, 0.73, 0.67]], (len(reference), 1))
    views = truth["views"]
    n = len(views)
    lens = json.loads((case / "capture/lens.json").read_text())
    size = (440, 360)
    header = 30
    picks = [0, n // 4, n // 2, 3 * n // 4]
    canvas = np.full((3 * (size[1] + header) + 40, len(picks) * size[0], 3), BG, np.uint8)
    for col, k in enumerate(picks):
        view = views[k]
        rotation, centre = np.array(view["rotation"]), np.array(view["centre"])
        # Crop: the reference's projected bounding box, with a margin.
        cam = (reference.reshape(-1, 3) - centre) @ rotation.T
        pix = cam[:, :2] / cam[:, 2:]
        lo, hi = pix.min(0), pix.max(0)
        mid, span = (lo + hi) / 2, (hi - lo) * 1.15
        focal = min(size[0] / span[0], size[1] / span[1])
        shift = -focal * mid
        # Same crop of the photo: photo pixel = fx * xy + (cx, cy) (distortion ignored for the crop).
        photo = cv2.imread(str(case / "capture/photos" / view["photo"]))
        if photo is not None:
            s = focal / lens["fx"]
            affine = np.array([[s, 0, size[0] / 2 + shift[0] - s * lens["cx"]],
                               [0, s, size[1] / 2 + shift[1] - s * lens["cy"]]])
            tile = cv2.warpAffine(photo, affine, size, borderValue=BG.tolist())
            canvas[header:header + size[1], col * size[0]:(col + 1) * size[0]] = tile
        for row, (tris, colours) in enumerate([(reference, grey), (recon, error_colour(distance))], start=1):
            tile = panels(tris, colours, rotation, centre, focal, size, shift)
            y = row * (size[1] + header) + header
            canvas[y:y + size[1], col * size[0]:(col + 1) * size[0]] = tile
    f1 = result["metrics"]["all"]["thresholds"]
    scores = " / ".join(f"{v['f1']:.3f}" for v in f1.values())
    labels = [args.title or case.name + "  (rendered photos)",
              f"Exact mesh ({len(reference):,} triangles)",
              f"Reconstruction ({len(recon):,} triangles), error: green 0, yellow 1 %, red 2 % of diagonal;  F1 {scores}"]
    for i, text in enumerate(labels):
        cv2.putText(canvas, text, (10, i * (size[1] + header) + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (49, 36, 26), 1, cv2.LINE_AA)
    cv2.imwrite(str(case / "sheet.png"), canvas)
    print(case / "sheet.png")


if __name__ == "__main__":
    main()
