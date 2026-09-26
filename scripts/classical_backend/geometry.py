"""Bounded OpenMVS binary PLY validation and lossless geometry-only export."""

from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

MAX_BYTES = 512 * 1024 ** 2
MAX_VERTICES = 5_000_000
MAX_FACES = 2_000_000
TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}
FLOATS = frozenset(("f", "d"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _header(stream):
    if stream.readline() != b"ply\n" or stream.readline() != b"format binary_little_endian 1.0\n":
        raise ValueError("PLY must be binary little endian 1.0")
    elements = []
    current = None
    for _ in range(256):
        raw = stream.readline(1025)
        if not raw or len(raw) > 1024:
            raise ValueError("invalid or oversized PLY header")
        words = raw.decode("ascii", "strict").split()
        if words == ["end_header"]:
            break
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "element" and len(words) == 3:
            try:
                count = int(words[2])
            except ValueError as exc:
                raise ValueError("invalid PLY element count") from exc
            if words[1] not in ("vertex", "face") or count < 0:
                raise ValueError("unsupported PLY element or count")
            current = {"name": words[1], "count": count, "props": []}
            elements.append(current)
        elif words[0] == "property" and current is not None:
            if len(words) == 3 and words[1] in TYPES:
                current["props"].append((words[2], TYPES[words[1]], None))
            elif len(words) == 5 and words[1] == "list" and words[2] in TYPES and words[3] in TYPES:
                current["props"].append((words[4], TYPES[words[2]], TYPES[words[3]]))
            else:
                raise ValueError("unsupported PLY property")
        else:
            raise ValueError("unsupported PLY header line")
    else:
        raise ValueError("PLY header too long")
    if [e["name"] for e in elements] not in (["vertex"], ["vertex", "face"]):
        raise ValueError("expected vertex then optional face element")
    vertex = elements[0]
    face = elements[1] if len(elements) == 2 else {"name": "face", "count": 0, "props": []}
    nv, nf = vertex["count"], face["count"]
    if not 0 < nv <= MAX_VERTICES or not 0 <= nf <= MAX_FACES:
        raise ValueError("vertex or face count outside limits")
    for element in elements:
        names = [p[0] for p in element["props"]]
        if len(names) != len(set(names)):
            raise ValueError("duplicate PLY property")
    vp = {p[0]: p for p in vertex["props"]}
    if not all(name in vp and vp[name][2] is None and vp[name][1] in FLOATS
               for name in ("x", "y", "z")):
        raise ValueError("XYZ must be float scalar properties")
    normals = {"nx", "ny", "nz"} & set(vp)
    if normals and (normals != {"nx", "ny", "nz"} or
                    any(vp[n][2] is not None or vp[n][1] not in FLOATS for n in normals)):
        raise ValueError("normals must be three float scalars")
    for name, count_fmt, item_fmt in vertex["props"]:
        if item_fmt is None:
            continue
        if (name == "view_indices" and (count_fmt, item_fmt) == ("B", "I")) or (
                name == "view_weights" and (count_fmt, item_fmt) == ("B", "f")):
            continue
        raise ValueError("unsupported vertex list property")
    if ("view_indices" in vp) != ("view_weights" in vp):
        raise ValueError("view indices and weights must occur together")
    if len(elements) == 2 and face["props"] not in ([("vertex_indices", "B", "I")],
                                                     [("vertex_indices", "B", "i")]):
        raise ValueError("face must have one uchar/int triangle-index list")
    return vertex, face


def _read(stream, fmt, count=1):
    size = struct.calcsize("<" + fmt) * count
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("truncated PLY payload")
    return struct.unpack("<" + fmt * count, raw)


def inspect(path, output_stream=None):
    """Parse every byte; return counts and exact-zero-area triangle count."""
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("missing, symlinked, empty, or oversized PLY")
    with path.open("rb") as stream:
        vertex, face = _header(stream)
        nv, nf = vertex["count"], face["count"]
        if output_stream is not None:
            output_stream.write(("ply\nformat binary_little_endian 1.0\n"
                f"element vertex {nv}\nproperty double x\nproperty double y\nproperty double z\n"
                f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii"))
        positions = array("d")
        for _ in range(nv):
            record = {}
            for name, fmt, item_fmt in vertex["props"]:
                if item_fmt is None:
                    value = _read(stream, fmt)[0]
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError("nonfinite vertex scalar")
                else:
                    length = _read(stream, fmt)[0]
                    values = _read(stream, item_fmt, length)
                    if item_fmt in FLOATS and not all(math.isfinite(v) for v in values):
                        raise ValueError("nonfinite vertex list value")
                    value = values
                record[name] = value
            if "view_indices" in record and len(record["view_indices"]) != len(record["view_weights"]):
                raise ValueError("view index/weight lengths differ")
            xyz = (record["x"], record["y"], record["z"])
            positions.extend(xyz)
            if output_stream is not None:
                output_stream.write(struct.pack("<ddd", *xyz))
        zero_area = 0
        for _ in range(nf):
            length = _read(stream, "B")[0]
            if length != 3:
                raise ValueError("non-triangle PLY face")
            indices = _read(stream, face["props"][0][2], 3)
            if len(set(indices)) != 3 or any(i < 0 or i >= nv for i in indices):
                raise ValueError("invalid triangle indices")
            a, b, c = (positions[3*i:3*i+3] for i in indices)
            ux, uy, uz = (b[j]-a[j] for j in range(3))
            vx, vy, vz = (c[j]-a[j] for j in range(3))
            cross = ((uy*vz-uz*vy), (uz*vx-ux*vz), (ux*vy-uy*vx))
            if sum(t*t for t in cross) == 0:
                zero_area += 1
            if output_stream is not None:
                output_stream.write(struct.pack("<Biii", 3, *indices))
        if stream.read(1):
            raise ValueError("unexpected trailing PLY payload")
    return {"vertices": nv, "faces": nf, "zero_area_faces": zero_area}


def counts(path):
    report = inspect(path)
    return report["vertices"], report["faces"]


def export_geometry(source, destination):
    """Write exact-geometry evaluator-compatible PLY to a fresh destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB free")
    before = digest(source)
    verified = inspect(source)
    try:
        with destination.open("xb") as stream:
            exported = inspect(source, stream)
        if exported != verified or digest(source) != before:
            raise ValueError("source PLY changed during export")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"schema": "classical_geometry_export_v1", **verified,
            "source": str(source.resolve()), "source_sha256": before,
            "output": str(destination.resolve()), "output_sha256": digest(destination),
            "scope": "geometry-only; float32 XYZ promoted exactly to float64; triangles preserved; source kept"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError(args.report)
    report = export_geometry(args.source, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("vertices", "faces", "source_sha256", "output_sha256")}))


if __name__ == "__main__":
    main()
