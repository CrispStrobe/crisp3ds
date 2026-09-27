"""Export the constrained Apple preview Mesh to an untextured PLY and QA image.

This handles the observed one-Mesh, triangle-only USD text emitted by usdcat.
It is not a general USD parser or a geometry-quality evaluator.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess

from PIL import Image, ImageDraw

from scripts.object_dataset import evaluate

RESERVE = 10 << 30
MAX_SOURCE_BYTES = 2 << 20
MAX_USDA_BYTES = 4 << 20
MAX_VERTICES = 100_000
MAX_FACES = 200_000
SAMPLES = 10_000
PANEL = 480
VIEWS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))


def field(source: str, declaration: str):
    matches = re.findall(r"^\s*" + re.escape(declaration) + r" = (\[[^\n]*\])\s*$",
                         source, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {declaration} array")
    return ast.literal_eval(matches[0])


def parse_mesh(source: str):
    if len(re.findall(r'^\s*def Mesh "', source, re.MULTILINE)) != 1:
        raise ValueError("expected exactly one Mesh prim")
    for declaration in ('def Xform "ObjectCapture"', 'def Scope "Geometry"',
                        'def Mesh "Mesh"'):
        if len(re.findall(r'^\s*' + re.escape(declaration) + r'\s*(?:\(|\{)',
                          source, re.MULTILINE)) != 1:
            raise ValueError(f"expected one {declaration}")
    if "xformOp:" in source or "xformOpOrder" in source:
        raise ValueError("authored transforms are not supported")
    if not re.search(r'^\s*metersPerUnit = 1\s*$', source, re.MULTILINE):
        raise ValueError("expected authored metersPerUnit = 1")
    if not re.search(r'^\s*upAxis = "Y"\s*$', source, re.MULTILINE):
        raise ValueError("expected authored Y upAxis")
    if not re.search(r'^\s*uniform token subdivisionScheme = "none"\s*$',
                     source, re.MULTILINE):
        raise ValueError("subdivision is not supported")
    points = field(source, "point3f[] points") if "point3f[] points" in source else field(source, "float3[] points")
    counts = field(source, "int[] faceVertexCounts")
    indices = field(source, "int[] faceVertexIndices")
    if not (0 < len(points) <= MAX_VERTICES and 0 < len(counts) <= MAX_FACES):
        raise ValueError("mesh count exceeds bounds or is empty")
    if any(count != 3 for count in counts) or len(indices) != 3 * len(counts):
        raise ValueError("only explicit triangles are supported")
    vertices = []
    for point in points:
        if len(point) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                      for x in point):
            raise ValueError("invalid point coordinate")
        vertices.append(tuple(float(x) for x in point))
    if any(not isinstance(i, int) or i < 0 or i >= len(vertices) for i in indices):
        raise ValueError("invalid face index")
    faces = [tuple(indices[i:i + 3]) for i in range(0, len(indices), 3)]
    return vertices, faces


def ply_bytes(vertices, faces) -> bytes:
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar uint vertex_indices\nend_header\n").encode("ascii")
    stream = io.BytesIO()
    stream.write(header)
    for point in vertices:
        stream.write(struct.pack("<fff", *point))
    for face in faces:
        stream.write(struct.pack("<BIII", 3, *face))
    return stream.getvalue()


def preview_bytes(vertices, faces) -> bytes:
    samples = evaluate.sample_surface(vertices, faces, SAMPLES, 20260927)
    image = Image.new("RGB", (PANEL * 3, PANEL), "white")
    draw = ImageDraw.Draw(image)
    for column, (a, b, label) in enumerate(VIEWS):
        xy = samples[:, [a, b]]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        extent = float(max(hi - lo))
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError("flat or nonfinite preview projection")
        center = (lo + hi) / 2
        scale = (PANEL - 60) / (extent * 1.05)
        for x, y in xy:
            px = round(column * PANEL + PANEL / 2 + (float(x) - center[0]) * scale)
            py = round(PANEL / 2 - (float(y) - center[1]) * scale)
            draw.point((px, py), fill="#555555")
        draw.text((column * PANEL + 12, 12), f"{label} | untextured surface sample", fill="black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def review(source: Path, run_dir: Path) -> dict:
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("source USDZ missing, linked, or over 2 MiB")
    if run_dir.exists() or run_dir.is_symlink() or not run_dir.parent.is_dir():
        raise ValueError("review directory must be fresh")
    if shutil.disk_usage(run_dir.parent).free < RESERVE + (32 << 20):
        raise ValueError("external volume has less than 10 GiB plus 32 MiB")
    usdcat = Path(shutil.which("usdcat") or "")
    if not usdcat.is_file() or usdcat.is_symlink():
        raise ValueError("installed usdcat is unavailable")
    source_hash = evaluate.sha256_file(source)
    tool_hash = evaluate.sha256_file(usdcat)
    command = [str(usdcat), str(source)]
    result = subprocess.run(command, capture_output=True, timeout=30, check=True)
    if len(result.stdout) > MAX_USDA_BYTES:
        raise ValueError("decoded USD text exceeds 4 MiB")
    vertices, faces = parse_mesh(result.stdout.decode("utf-8"))
    if evaluate.sha256_file(source) != source_hash or evaluate.sha256_file(usdcat) != tool_hash:
        raise ValueError("source USDZ or usdcat changed during review")
    ply = ply_bytes(vertices, faces)
    png = preview_bytes(vertices, faces)
    if len(ply) > 16 << 20 or len(png) > 10 << 20:
        raise ValueError("review artifact exceeds cap")
    run_dir.mkdir()
    mesh_path = run_dir / "mesh-untextured.ply"
    preview_path = run_dir / "preview-neutral.png"
    mesh_path.write_bytes(ply)
    preview_path.write_bytes(png)
    stats = evaluate.inspect_ply(mesh_path)
    report = {"schema": "apple_usdz_mesh_review_v1",
              "scope": "structural geometry and neutral preview only; no quality acceptance",
              "source_usdz": str(source), "source_usdz_sha256": source_hash,
              "usdcat_command": command, "usdcat_sha256": tool_hash,
              "usdcat_stderr": result.stderr.decode("utf-8", errors="replace")[:4096],
              "geometry_prim": "/ObjectCapture/Geometry/Mesh",
              "units": "authored metersPerUnit=1; metric accuracy not independently verified",
              "up_axis": "Y", "transform_handling": "no authored xformOp; coordinates copied unchanged",
              "ply": {"path": str(mesh_path), "sha256": hashlib.sha256(ply).hexdigest(),
                      "bytes": len(ply)},
              "preview": {"path": str(preview_path), "sha256": hashlib.sha256(png).hexdigest(),
                          "bytes": len(png), "samples": SAMPLES,
                          "views": [view[2] for view in VIEWS]},
              "mesh_stats": stats}
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usdz", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = review(args.source_usdz, args.run_dir)
    print(json.dumps({"result": str(args.run_dir / "result.json"),
                      "vertices": report["mesh_stats"]["vertices"],
                      "faces": report["mesh_stats"]["faces"]}))


if __name__ == "__main__":
    main()
