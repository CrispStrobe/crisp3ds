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


def silhouette_iou(triangles, rows, mask_directory=None):
    values = {}
    for row in rows:
        path = Path(mask_directory) / (row["name"] + ".png") if mask_directory else Path(row["mask"])
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("missing mask " + str(path))
        R, t = np.array(row["rotation"]), np.array(row["translation"])
        fx, fy, cx, cy = row["k"]
        cam = triangles @ R.T + t
        if (cam[:, :, 2] <= 0).any():
            raise ValueError("mesh reaches behind camera " + row["name"])
        xy = cam[:, :, :2] / cam[:, :, 2:] * [fx, fy] + [cx - 0.5, cy - 0.5]
        drawn = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(drawn, np.rint(xy * 16).astype(np.int32), 255, shift=4)
        a, b = mask > 127, drawn > 0
        values[row["name"]] = float((a & b).sum() / max((a | b).sum(), 1))
    return values


def summary(values):
    v = np.array(list(values.values()))
    return {"median": float(np.median(v)), "minimum": float(v.min()), "maximum": float(v.max()),
            "worst_view": min(values, key=values.get)}


def run(inputs, mesh, output, *, repaired_masks=None, preview_views=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_views(inputs)
    triangles = read_stl(mesh)
    report = {"mesh": str(mesh), "triangles": len(triangles), "views": len(rows),
              "silhouette_iou_input_masks": summary(silhouette_iou(triangles, rows)),
              "note": "photo agreement only; not scanner accuracy or physical scale"}
    if repaired_masks is not None and Path(repaired_masks).is_dir():
        report["silhouette_iou_repaired_masks"] = summary(silhouette_iou(triangles, rows, repaired_masks))
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
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.mesh, args.output, repaired_masks=args.repaired_masks,
                         preview_views=args.preview_views), indent=2))


if __name__ == "__main__":
    main()
