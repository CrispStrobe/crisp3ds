"""Remove only exact-zero-area triangles from a validated binary MVE PLY.

Vertex records and retained 13-byte triangle records are copied byte for byte.
No smoothing, decimation, reindexing, or coordinate transform is performed.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct

from scripts.mve_full.run import ply_counts, sha256


def remove_zero_area_faces(source: Path, destination: Path) -> dict:
    if source == destination or destination.exists() or source.is_symlink() or destination.is_symlink():
        raise ValueError("source and fresh destination must be distinct real files")
    vertices, faces = ply_counts(source, allow_degenerate=True)
    with source.open("rb") as stream:
        header = []
        vertex_format = []
        vertex_names = []
        in_vertex = False
        for _ in range(200):
            line = stream.readline()
            if not line:
                raise ValueError("truncated PLY header")
            header.append(line)
            words = line.decode("ascii").split()
            if words[:2] == ["element", "vertex"]:
                in_vertex = True
            elif words[:2] == ["element", "face"]:
                in_vertex = False
            elif words[:1] == ["property"] and in_vertex:
                vertex_format.append({"char": "b", "uchar": "B", "short": "h", "ushort": "H",
                                      "int": "i", "uint": "I", "float": "f", "double": "d"}[words[1]])
                vertex_names.append(words[2])
            if words[:1] == ["end_header"]:
                break
        else:
            raise ValueError("PLY header too long")
        record = struct.Struct("<" + "".join(vertex_format))
        vertex_offset = stream.tell()
        xyz = tuple(vertex_names.index(axis) for axis in ("x", "y", "z"))
        positions = []
        for _ in range(vertices):
            values = record.unpack(stream.read(record.size))
            positions.append(tuple(values[index] for index in xyz))
        face_offset = stream.tell()
        retained = 0
        for _ in range(faces):
            face = stream.read(13)
            a, b, c = struct.unpack("<iii", face[1:])
            p, q, r = positions[a], positions[b], positions[c]
            u = tuple(q[i] - p[i] for i in range(3))
            v = tuple(r[i] - p[i] for i in range(3))
            cross = (u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0])
            retained += math.fsum(component * component for component in cross) > 0
        if retained < 1:
            raise ValueError("all mesh faces have zero area")
        revised = [f"element face {retained}\n".encode("ascii") if line.startswith(b"element face ")
                   else line for line in header]
        try:
            with destination.open("xb") as sink:
                sink.writelines(revised)
                stream.seek(vertex_offset)
                remaining = vertices * record.size
                while remaining:
                    chunk = stream.read(min(1 << 20, remaining))
                    sink.write(chunk)
                    remaining -= len(chunk)
                stream.seek(face_offset)
                for _ in range(faces):
                    face = stream.read(13)
                    a, b, c = struct.unpack("<iii", face[1:])
                    p, q, r = positions[a], positions[b], positions[c]
                    u = tuple(q[i] - p[i] for i in range(3))
                    v = tuple(r[i] - p[i] for i in range(3))
                    cross = (u[1] * v[2] - u[2] * v[1],
                             u[2] * v[0] - u[0] * v[2],
                             u[0] * v[1] - u[1] * v[0])
                    if math.fsum(component * component for component in cross) > 0:
                        sink.write(face)
            if ply_counts(destination) != (vertices, retained):
                raise ValueError("sanitized PLY did not pass strict validation")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    return {"native_mesh_sha256": sha256(source), "mesh_sha256": sha256(destination),
            "native_mesh_vertices": vertices, "native_mesh_faces": faces,
            "removed_zero_area_faces": faces - retained,
            "mesh_vertices": vertices, "mesh_faces": retained}
