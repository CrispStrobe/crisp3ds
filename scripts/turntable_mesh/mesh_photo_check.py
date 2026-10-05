"""Compare a reconstructed STL with the photos it came from.

Reports silhouette intersection-over-union of the projected mesh against each
view's mask, and writes a preview sheet (photos above, shaded mesh below) for a
few evenly spaced views. This is photo agreement: it cannot see errors along
the viewing direction and does not certify scanner accuracy or physical scale.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .dense_events import EventLog
from .multiscale_stereo import load_views
from .stl_compare_render import read_stl, run as render


def silhouette_iou(triangles, rows, mask_directories=(None,), overlay=None):
    """IoU per view against each mask set (None = the masks named in the camera table).

    ``overlay`` = (path, names): also write photo crops for those views with the
    last mask set compared to the mesh outline (green both, red mask only, blue mesh only).
    """
    values = [{} for _ in mask_directories]
    tiles = []
    for row in rows:
        R, t = np.array(row["rotation"]), np.array(row["translation"])
        fx, fy, cx, cy = row["k"]
        cam = triangles @ R.T + t
        if (cam[:, :, 2] <= 0).any():
            raise ValueError("mesh reaches behind camera " + row["name"])
        xy = cam[:, :, :2] / cam[:, :, 2:] * [fx, fy] + [cx - 0.5, cy - 0.5]
        drawn = np.zeros((row["height"], row["width"]), np.uint8)
        cv2.fillPoly(drawn, np.rint(xy * 16).astype(np.int32), 255, shift=4)
        for store, directory in zip(values, mask_directories):
            path = Path(directory) / (row["name"] + ".png") if directory else Path(row["mask"])
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != drawn.shape:
                raise ValueError("missing or mismatched mask " + str(path))
            a, b = mask > 127, drawn > 0
            store[row["name"]] = float((a & b).sum() / max((a | b).sum(), 1))
        if overlay and row["name"] in overlay[1]:
            photo = cv2.imread(row["image"])
            for region, colour in ((a & b, (60, 170, 60)), (a & ~b, (40, 40, 230)), (~a & b, (230, 120, 40))):
                photo[region] = (0.45 * photo[region] + 0.55 * np.array(colour)).astype(np.uint8)
            ys, xs = np.nonzero(a | b)
            crop = photo[max(ys.min() - 20, 0) : ys.max() + 20, max(xs.min() - 20, 0) : xs.max() + 20]
            tiles.append(cv2.resize(crop, (600, round(600 * crop.shape[0] / crop.shape[1]))))
    if tiles:
        height = max(t.shape[0] for t in tiles)
        sheet = np.full((height, 600 * len(tiles), 3), 30, np.uint8)
        for n, tile in enumerate(tiles):
            sheet[: tile.shape[0], 600 * n : 600 * (n + 1)] = tile
        cv2.imwrite(str(overlay[0]), sheet)
    return values


def summary(values):
    v = np.array(list(values.values()))
    return {"median": float(np.median(v)), "minimum": float(v.min()), "maximum": float(v.max()),
            "worst_view": min(values, key=values.get)}


def run(inputs, mesh, output, *, repaired_masks=None, preview_views=3, check_views=24, events=None):
    events = events or EventLog(None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_views(inputs)
    triangles = read_stl(mesh)
    checked = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, min(check_views, len(rows))).round().astype(int))]
    sets = [None]
    if repaired_masks is not None and Path(repaired_masks).is_dir():
        sets.append(repaired_masks)
    names = [rows[i]["name"] for i in np.linspace(0, len(rows), max(preview_views, 1), endpoint=False, dtype=int)]
    checked += [r for r in rows if r["name"] in names and r not in checked]
    scores = silhouette_iou(triangles, checked, sets, overlay=(output / "photo-overlay.png", names) if preview_views else None)
    events.progress(0.6, "Silhouettes compared")
    report = {"mesh": str(mesh), "triangles": len(triangles), "views": len(rows), "views_checked": len(checked),
              "silhouette_iou_input_masks": summary(scores[0]),
              "note": "photo agreement only; not scanner accuracy or physical scale"}
    if len(sets) > 1:
        report["silhouette_iou_repaired_masks"] = summary(scores[1])
    events.metric("silhouette_iou_median", report["silhouette_iou_input_masks"]["median"])
    if preview_views:
        events.artifact("photo_overlay", output / "photo-overlay.png",
                        "Mesh outline against masks (green: both, red: mask only, blue: mesh only)")
        render(Path(inputs) / "cameras.json", names, [("Reconstruction", mesh)], output / "preview.png")
        report["preview"] = str(output / "preview.png")
        events.artifact("preview_render", output / "preview.png", "Photos and shaded reconstruction")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    events.artifact("report", output / "result.json", "Photo check")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repaired-masks", type=Path)
    parser.add_argument("--preview-views", type=int, default=3)
    parser.add_argument("--check-views", type=int, default=24, help="evenly spaced views scored for silhouette IoU")
    parser.add_argument("--events", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.mesh, args.output, repaired_masks=args.repaired_masks,
                         preview_views=args.preview_views, check_views=args.check_views,
                         events=EventLog(args.events, "check")), indent=2))


if __name__ == "__main__":
    main()
