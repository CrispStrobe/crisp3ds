"""Strict, bounded validation of Brush's binary Gaussian-splat PLY export."""
from __future__ import annotations

import math
from pathlib import Path
import struct

REQUIRED = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}


def validate(path: Path, *, max_vertices: int = 50_000, max_bytes: int = 512 * 1024**2) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("missing, symlinked, or oversized splat PLY")
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            char = stream.read(1)
            if not char or len(header) >= 64 * 1024:
                raise ValueError("missing/oversized PLY header")
            header.extend(char)
        lines = header.decode("ascii").splitlines()
        if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
            raise ValueError("Brush splat PLY must be binary little-endian")
        vertices = None
        props: list[str] = []
        element = None
        for line in lines:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "element":
                element = fields[1]
                if element == "vertex":
                    vertices = int(fields[2])
                elif int(fields[2]) != 0:
                    raise ValueError("unexpected nonvertex PLY element")
            elif fields[:1] == ["property"] and element == "vertex":
                if len(fields) != 3 or fields[1] != "float":
                    raise ValueError("unsupported non-float vertex property")
                props.append(fields[2])
        if vertices is None or not 1 <= vertices <= max_vertices or len(props) != len(set(props)):
            raise ValueError("invalid vertex count/properties")
        if not REQUIRED.issubset(props):
            raise ValueError("missing Gaussian position/opacity/scale/rotation/color properties")
        expected = vertices * len(props) * 4
        if expected > max_bytes or path.stat().st_size != len(header) + expected:
            raise ValueError("truncated or trailing PLY payload")
        row = struct.Struct("<" + "f" * len(props))
        rotations = [props.index(f"rot_{i}") for i in range(4)]
        for _ in range(vertices):
            values = row.unpack(stream.read(row.size))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite Gaussian property")
            if sum(values[index] ** 2 for index in rotations) <= 1e-12:
                raise ValueError("degenerate Gaussian rotation")
    return {"vertices": vertices, "properties": props, "payload_bytes": expected}
