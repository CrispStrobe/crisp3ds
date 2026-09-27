"""Bounded texture/UV audit for a sealed Apple bunny preview USDZ.

Writes only a small atlas thumbnail and JSON beside a prior mesh review.
The reconstruction USDZ remains read-only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZipFile
from collections import Counter

from PIL import Image

from scripts.apple_object_capture import review_mesh
from scripts.classical_backend.run import digest


MAX_USDZ = 2 << 20
MAX_TEXTURE = 2 << 20
MAX_USDA = 4 << 20
MAX_THUMB = 1 << 20
RESERVE = 11 << 30


def _array(text: str, declaration: str) -> list:
    lines = [line for line in text.splitlines() if line.lstrip().startswith(declaration + " = [")]
    if len(lines) != 1:
        raise ValueError(f"expected one {declaration} array")
    value = lines[0].split(" = [", 1)[1].split("]", 1)[0]
    return ast.literal_eval("[" + value + "]")


def topology(vertices: list, faces: list) -> dict:
    parent = list(range(len(faces)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = Counter()
    first_face_by_edge = {}
    for face_index, (a, b, c) in enumerate(faces):
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edges[key] += 1
            if key in first_face_by_edge:
                parent[root(face_index)] = root(first_face_by_edge[key])
            else:
                first_face_by_edge[key] = face_index
    components = Counter(root(index) for index in range(len(faces)))
    return {"face_connected_components": len(components),
            "largest_component_faces": max(components.values(), default=0),
            "boundary_edges": sum(count == 1 for count in edges.values()),
            "nonmanifold_edges": sum(count > 2 for count in edges.values())}


def inspect_usdz(source: Path, usdcat: Path = Path("/usr/bin/usdcat")) -> tuple[dict, bytes]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_USDZ:
        raise ValueError("USDZ missing, linked, or above 2 MiB")
    if usdcat.is_symlink() or not usdcat.is_file():
        raise ValueError("usdcat missing or linked")
    before = digest(source)
    with ZipFile(source) as archive:
        members = archive.infolist()
        if (len(members) != 2 or any(info.is_dir() or info.file_size > MAX_TEXTURE
                                      for info in members)):
            raise ValueError("expected one bounded USD scene and one texture")
        textures = [info for info in members if info.filename.endswith(".png")]
        scenes = [info for info in members if info.filename.endswith(".usdc")]
        if len(textures) != 1 or len(scenes) != 1:
            raise ValueError("USDZ archive contents differ from one-scene one-texture profile")
        texture_name = textures[0].filename
        if texture_name.startswith("/") or ".." in Path(texture_name).parts:
            raise ValueError("unsafe texture archive path")
        texture = archive.read(textures[0])
    converted = subprocess.run([str(usdcat), str(source)], capture_output=True,
                               check=True, timeout=30)
    if len(converted.stdout) > MAX_USDA:
        raise ValueError("decoded USD text exceeds 4 MiB")
    text = converted.stdout.decode("utf-8")
    vertices, faces = review_mesh.parse_mesh(text)
    links = re.findall(r'^\s*asset inputs:file = @([^@\n]+)@\s*$', text, re.MULTILINE)
    if links != [texture_name] or text.count("material:binding =") != 1:
        raise ValueError("mesh material does not bind the archived texture exactly once")
    uv = _array(text, "texCoord2f[] primvars:st")
    uv_indices = _array(text, "int[] primvars:st:indices")
    if (len(uv_indices) != 3 * len(faces) or not uv or
            any(not isinstance(index, int) or index < 0 or index >= len(uv)
                for index in uv_indices)):
        raise ValueError("face-corner UV indices are incomplete or invalid")
    if any(len(pair) != 2 or not all(isinstance(value, (int, float)) and
                                   -0.001 <= value <= 1.001 for value in pair)
           for pair in uv):
        raise ValueError("UV coordinates fall outside bounded atlas")
    with Image.open(BytesIO(texture)) as image:
        image.load()
        if image.format != "PNG" or image.mode not in ("RGB", "RGBA") or any(
                size < 1 or size > 4096 for size in image.size):
            raise ValueError("unsupported texture image")
        size = image.size
        mode = image.mode
        extrema = image.getextrema()
        thumbnail = image.copy()
        thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
        data = BytesIO()
        thumbnail.save(data, format="PNG")
        preview = data.getvalue()
    if len(preview) > MAX_THUMB or digest(source) != before:
        raise ValueError("texture thumbnail cap exceeded or source changed")
    return ({"source_usdz_sha256": before,
             "usdcat_sha256": digest(usdcat),
             "archive_members": [{"name": info.filename, "bytes": info.file_size}
                                 for info in members],
             "texture_entry": texture_name,
             "texture_sha256": hashlib.sha256(texture).hexdigest(),
             "texture_bytes": len(texture), "texture_dimensions": list(size),
             "texture_mode": mode, "channel_extrema": extrema,
             "mesh_vertices": len(vertices), "mesh_faces": len(faces),
             "mesh_topology": topology(vertices, faces),
             "uv_points": len(uv), "face_corner_uv_indices": len(uv_indices),
             "all_face_corners_textured": True}, preview)


def review(source: Path, launch_receipt: Path, mesh_receipt: Path, review_dir: Path) -> dict:
    for path in (launch_receipt, mesh_receipt):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked receipt: {path}")
    if (review_dir.exists() or review_dir.is_symlink() or not review_dir.parent.is_dir() or
            review_dir.parent.is_symlink()):
        raise ValueError("texture review directory must be fresh under a real parent")
    if shutil.disk_usage(review_dir.parent).free < RESERVE + (2 << 20):
        raise ValueError("11 GiB disk reserve plus 2 MiB review headroom unavailable")
    result_path = review_dir / "texture-review.json"
    preview_path = review_dir / "texture-atlas-preview.png"
    launch = json.loads(launch_receipt.read_text())
    mesh = json.loads(mesh_receipt.read_text())
    summary, preview = inspect_usdz(source)
    if (launch.get("schema") != "apple_photogrammetry_probe_v1" or
            launch.get("status") != "complete" or launch.get("mode") != "rgb_to_usdz" or
            launch.get("artifact", {}).get("sha256") != summary["source_usdz_sha256"] or
            mesh.get("schema") != "apple_usdz_mesh_review_v1" or
            mesh.get("source_usdz_sha256") != summary["source_usdz_sha256"] or
            mesh.get("mesh_stats", {}).get("vertices") != summary["mesh_vertices"] or
            mesh.get("mesh_stats", {}).get("faces") != summary["mesh_faces"]):
        raise ValueError("launch, mesh review, and USDZ do not bind the same artifact")
    report = {"schema": "apple_bunny_preview_texture_review_v1",
              "scope": "archive, UV, and atlas structure; texture quality not accepted",
              "source_usdz": str(source),
              "launch_receipt_sha256": digest(launch_receipt),
              "mesh_receipt_sha256": digest(mesh_receipt),
              **summary,
              "atlas_preview": {"path": str(preview_path), "bytes": len(preview),
                                "sha256": hashlib.sha256(preview).hexdigest()}}
    review_dir.mkdir()
    preview_path.write_bytes(preview)
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usdz", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--mesh-receipt", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.source_usdz, args.launch_receipt, args.mesh_receipt, args.review_dir)
    print(json.dumps({"result": str(args.review_dir / "texture-review.json"),
                      "texture_dimensions": result["texture_dimensions"],
                      "face_corner_uv_indices": result["face_corner_uv_indices"]}))


if __name__ == "__main__":
    main()
