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

from .multiscale_stereo import load_views
from .stl_compare_render import read_stl, run as render


def silhouette_iou(triangles, rows, mask_directories=(None,)):
    """IoU per view against each mask set (None = the masks named in the camera table)."""
    values = [{} for _ in mask_directories]
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
    return values


def summary(values):
    v = np.array(list(values.values()))
    return {"median": float(np.median(v)), "minimum": float(v.min()), "maximum": float(v.max()),
            "worst_view": min(values, key=values.get)}


def run(inputs, mesh, output, *, repaired_masks=None, preview_views=3, check_views=24):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_views(inputs)
    triangles = read_stl(mesh)
    checked = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, min(check_views, len(rows))).round().astype(int))]
    sets = [None]
    if repaired_masks is not None and Path(repaired_masks).is_dir():
        sets.append(repaired_masks)
    scores = silhouette_iou(triangles, checked, sets)
    report = {"mesh": str(mesh), "triangles": len(triangles), "views": len(rows), "views_checked": len(checked),
              "silhouette_iou_input_masks": summary(scores[0]),
              "note": "photo agreement only; not scanner accuracy or physical scale"}
    if len(sets) > 1:
        report["silhouette_iou_repaired_masks"] = summary(scores[1])
    if preview_views:
        names = [rows[i]["name"] for i in np.linspace(0, len(rows), preview_views, endpoint=False, dtype=int)]
        render(Path(inputs) / "cameras.json", names, [("Reconstruction", mesh)], output / "preview.png")
        report["preview"] = str(output / "preview.png")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repaired-masks", type=Path)
    parser.add_argument("--preview-views", type=int, default=3)
    parser.add_argument("--check-views", type=int, default=24, help="evenly spaced views scored for silhouette IoU")
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.mesh, args.output, repaired_masks=args.repaired_masks,
                         preview_views=args.preview_views, check_views=args.check_views), indent=2))


if __name__ == "__main__":
    main()
