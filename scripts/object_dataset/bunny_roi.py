#!/usr/bin/env python3
"""Export the frozen reference-only bunny support-disk ROI; no candidate input."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image, ImageDraw

if __package__:
    from . import evaluate, preview
else:
    import evaluate
    import preview

SOURCE_SHA256 = "28ed4462b6ee84edcc87157b49aee9216ef170ddd1e176928c72c74233ef9520"
CUT_Y = -24.0


def select_roi(vertices, faces, cut_y=CUT_Y):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("invalid triangle faces")
    if (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("face index out of range")
    keep = np.all(vertices[faces, 1] > cut_y, axis=1)
    selected = faces[keep]
    if not len(selected):
        raise ValueError("empty reference ROI")
    ids, inverse = np.unique(selected, return_inverse=True)
    return vertices[ids], inverse.reshape(-1, 3).astype(np.int32), int(np.sum(keep))


def write_ply(path, vertices, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB free")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property double x\nproperty double y\nproperty double z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n").encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        stream.write(np.asarray(vertices, dtype="<f8").tobytes(order="C"))
        packed = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
        packed["count"] = 3
        packed["indices"] = faces
        stream.write(packed.tobytes(order="C"))


def roi_preview(full, roi):
    left = evaluate.sample_surface(*full, 10000, 2030)
    right = evaluate.sample_surface(*roi, 10000, 2030)
    image = Image.new("RGB", (2 * preview.PANEL, 3 * preview.PANEL), "white")
    draw = ImageDraw.Draw(image)
    for row, (a, b, label) in enumerate(preview.PROJECTIONS):
        bounds = preview.panel_bounds(left, right, (a, b))
        y0 = row * preview.PANEL
        preview.draw_sample(draw, left, (a, b), bounds, (0, y0), "#087f8c")
        preview.draw_sample(draw, right, (a, b), bounds, (preview.PANEL, y0), "#d46812")
        draw.text((12, y0 + 12), f"{label} full scanner", fill="black")
        draw.text((preview.PANEL + 12, y0 + 12), f"{label} ROI scanner", fill="black")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    args = parser.parse_args()
    if any(p.exists() or p.is_symlink() for p in (args.output, args.report, args.preview)):
        parser.error("all output paths must be fresh")
    digest = evaluate.sha256_file(args.reference)
    if digest != SOURCE_SHA256:
        parser.error("reference SHA-256 does not match the frozen bunny scanner mesh")
    info, vertices, faces = evaluate.inspect_ply(args.reference, geometry=True)
    roi_vertices, roi_faces, kept = select_roi(vertices, faces)
    try:
        write_ply(args.output, roi_vertices, roi_faces)
        roi_info, check_vertices, check_faces = evaluate.inspect_ply(args.output, geometry=True)
        if not np.array_equal(np.asarray(check_vertices), roi_vertices) or not np.array_equal(np.asarray(check_faces), roi_faces):
            raise ValueError("ROI export did not round-trip")
        image = roi_preview((vertices, faces), (roi_vertices, roi_faces))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        with args.preview.open("xb") as stream:
            image.save(stream, format="PNG")
        record = {"schema": "bunny_scanner_roi_v1", "reference": str(args.reference),
                  "reference_sha256": digest, "roi": str(args.output),
                  "roi_sha256": evaluate.sha256_file(args.output),
                  "rule": "retain triangle iff all three scanner vertices have Y > -24; no cap; no candidate selection",
                  "cut_y": CUT_Y, "source_faces": info["faces"], "retained_faces": kept,
                  "roi_vertices": roi_info["vertices"], "roi_faces": roi_info["faces"],
                  "roi_bbox_diagonal": float(np.linalg.norm(roi_vertices.max(axis=0) - roi_vertices.min(axis=0))),
                  "preview": str(args.preview),
                  "interpretation": "post-hoc reference-only support-disk ROI; may exclude bunny contact/body; not GT"}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(json.dumps(record, indent=2))
    except BaseException:
        args.output.unlink(missing_ok=True)
        args.preview.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
