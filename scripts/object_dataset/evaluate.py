#!/usr/bin/env python3
"""Audit PLY meshes and optionally compare sampled surfaces with a supplied Sim(3)."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

MAX_BYTES = 512 * 1024**2
MAX_VERTICES = 5_000_000
MAX_FACES = 10_000_000
MAX_SCORE_VERTICES = 1_000_000
MAX_SCORE_FACES = 2_000_000
MAX_SAMPLES = 4096
SCALARS = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
           "int": "i", "uint": "I", "float": "f", "double": "d"}


def inspect_ply(path: Path, *, geometry: bool = False):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("PLY exceeds 512 MiB validation limit")
    with path.open("rb") as stream:
        lines = []
        for _ in range(128):
            line = stream.readline()
            if len(line) > 1024 or not line:
                raise ValueError("invalid PLY header")
            decoded = line.decode("ascii").strip()
            lines.append(decoded)
            if decoded == "end_header":
                break
        else:
            raise ValueError("PLY header too long")
        if lines[:2] != ["ply", "format binary_little_endian 1.0"] or lines[-1] != "end_header":
            raise ValueError("only binary little-endian PLY is supported")
        elements = []
        current = None
        for line in lines[2:-1]:
            fields = line.split()
            if not fields or fields[0] in ("comment", "obj_info"):
                continue
            if fields[0] == "element" and len(fields) == 3:
                current = {"name": fields[1], "count": int(fields[2]), "properties": []}
                elements.append(current)
            elif fields[0] == "property" and current is not None:
                current["properties"].append(fields[1:])
            else:
                raise ValueError(f"unsupported PLY header line: {line}")
        if [e["name"] for e in elements] != ["vertex", "face"]:
            raise ValueError("expected vertex then face elements")
        vertex, face = elements
        nv, nf = vertex["count"], face["count"]
        if not 0 < nv <= MAX_VERTICES or not 0 <= nf <= MAX_FACES:
            raise ValueError("vertex or face count outside limits")
        if geometry and (nv > MAX_SCORE_VERTICES or nf > MAX_SCORE_FACES):
            raise ValueError("mesh exceeds bounded scoring geometry limit")
        properties = vertex["properties"]
        if any(len(p) != 2 or p[0] not in SCALARS for p in properties):
            raise ValueError("unsupported vertex property")
        names = [p[1] for p in properties]
        if not all(name in names for name in ("x", "y", "z")) or len(names) != len(set(names)):
            raise ValueError("missing or duplicate coordinate property")
        vertex_fmt = struct.Struct("<" + "".join(SCALARS[p[0]] for p in properties))
        if face["properties"] not in ([["list", "uchar", "uint", "vertex_indices"]],
                                        [["list", "uchar", "int", "vertex_indices"]]):
            raise ValueError("unsupported face representation")
        index_format = "I" if face["properties"][0][2] == "uint" else "i"
        coordinate_indices = [names.index(name) for name in ("x", "y", "z")]
        bounds_min = [math.inf] * 3
        bounds_max = [-math.inf] * 3
        vertices = []
        for _ in range(nv):
            values = vertex_fmt.unpack(stream.read(vertex_fmt.size))
            xyz = tuple(float(values[i]) for i in coordinate_indices)
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError("nonfinite mesh coordinate")
            vertices.append(xyz)
            for i, value in enumerate(xyz):
                bounds_min[i] = min(bounds_min[i], value)
                bounds_max[i] = max(bounds_max[i], value)
        degenerate = 0
        zero_area = 0
        nontriangles = 0
        triangles = [] if geometry else None
        for _ in range(nf):
            raw_count = stream.read(1)
            if len(raw_count) != 1:
                raise ValueError("truncated PLY face")
            count = raw_count[0]
            if count < 3 or count > 32:
                raise ValueError("face vertex count outside limits")
            raw_indices = stream.read(4 * count)
            if len(raw_indices) != 4 * count:
                raise ValueError("truncated PLY face indices")
            indices = struct.unpack("<" + index_format * count, raw_indices)
            if min(indices) < 0 or max(indices) >= nv:
                raise ValueError("face index exceeds vertex count")
            if len(set(indices)) < count:
                degenerate += 1
            if count != 3:
                nontriangles += 1
            else:
                if geometry:
                    triangles.append(indices)
                a, b, c = (vertices[i] for i in indices)
                u = [b[i] - a[i] for i in range(3)]
                v = [c[i] - a[i] for i in range(3)]
                cross = (u[1]*v[2] - u[2]*v[1],
                         u[2]*v[0] - u[0]*v[2],
                         u[0]*v[1] - u[1]*v[0])
                if sum(x*x for x in cross) == 0:
                    zero_area += 1
        if stream.read(1):
            raise ValueError("unexpected bytes after PLY geometry")
    report = {"path": str(path), "bytes": path.stat().st_size, "vertices": nv,
            "faces": nf, "nontriangle_faces": nontriangles,
            "faces_with_repeated_indices": degenerate,
            "zero_area_triangle_faces": zero_area,
            "bbox_min": bounds_min, "bbox_max": bounds_max,
            "bbox_extent": [bounds_max[i] - bounds_min[i] for i in range(3)],
            "all_coordinates_finite": True, "all_face_indices_valid": True}
    if geometry:
        return report, vertices, triangles
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sim3(path: Path, reference: Path, output: Path):
    import numpy as np
    data = json.loads(path.read_text())
    required = ("matrix", "object_id", "source_frame", "target_frame", "reference_units",
                "provenance", "scale_provenance", "registration_basis",
                "reference_sha256", "output_sha256")
    if any(key not in data for key in required):
        raise ValueError("Sim(3) file lacks required transform or provenance fields")
    if any(not isinstance(data[key], str) or not data[key].strip() for key in required[1:]):
        raise ValueError("Sim(3) provenance fields must be nonempty strings")
    if data["registration_basis"] not in ("external-calibration", "manual-landmarks", "reference-fit"):
        raise ValueError("unknown registration basis")
    if data["reference_sha256"] != sha256_file(reference) or data["output_sha256"] != sha256_file(output):
        raise ValueError("transform is not bound to these exact meshes")
    matrix = np.asarray(data["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Sim(3) matrix must be finite 4x4")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("Sim(3) homogeneous row invalid")
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(singular.mean())
    if not 1e-9 < scale < 1e9 or not np.allclose(singular, scale, rtol=1e-6, atol=0):
        raise ValueError("Sim(3) rotation must have uniform positive scale")
    if np.linalg.det(linear) <= 0:
        raise ValueError("Sim(3) reflection or singular rotation is invalid")
    return data, matrix, scale


def sample_surface(vertices, faces, count: int, seed: int):
    import numpy as np
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("surface comparison needs triangle faces")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    valid = np.isfinite(areas) & (areas > 0)
    if not valid.any():
        raise ValueError("surface has no positive-area triangles")
    triangles = triangles[valid]
    areas = areas[valid]
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    r1 = np.sqrt(rng.random(count))[:, None]
    r2 = rng.random(count)[:, None]
    return (1 - r1) * chosen[:, 0] + r1 * (1 - r2) * chosen[:, 1] + r1 * r2 * chosen[:, 2]


def nearest_distances(points, targets):
    import numpy as np
    target_norm = np.einsum("ij,ij->i", targets, targets)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 128):
        block = points[start:start + 128]
        squared = (np.einsum("ij,ij->i", block, block)[:, None]
                   + target_norm[None, :] - 2 * block @ targets.T)
        result[start:start + len(block)] = np.sqrt(np.maximum(0, squared.min(axis=1)))
    return result


def summarize_distances(values):
    import numpy as np
    return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values**2))),
            "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}


def compare_meshes(reference_geometry, output_geometry, matrix, *, threshold: float,
                   count: int = 2048, seed: int = 2026):
    import numpy as np
    if not 1 <= count <= MAX_SAMPLES or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid sample count or threshold")
    reference = sample_surface(*reference_geometry, count, seed)
    output = sample_surface(*output_geometry, count, seed)
    transformed = output @ matrix[:3, :3].T + matrix[:3, 3]
    to_reference = nearest_distances(transformed, reference)
    to_output = nearest_distances(reference, transformed)
    precision = float(np.mean(to_reference <= threshold))
    recall = float(np.mean(to_output <= threshold))
    return {"method": "deterministic area-weighted triangle sampling; nearest sampled point",
            "exact_point_to_triangle_distance": False, "sample_count_per_mesh": count, "seed": seed,
            "threshold": threshold,
            "output_to_reference": summarize_distances(to_reference),
            "reference_to_output": summarize_distances(to_output),
            "precision": precision, "recall": recall,
            "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def reference_sampling_control(reference_geometry, *, threshold: float,
                               count: int = 2048, seed: int = 2026):
    """Same-reference, independent-seed sampling floor; never a reconstruction score."""
    import numpy as np
    if not 1 <= count <= MAX_SAMPLES or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid sample count or threshold")
    if not 0 <= seed < 2**64 - 1:
        raise ValueError("sample seed outside NumPy range")
    first = sample_surface(*reference_geometry, count, seed)
    second = sample_surface(*reference_geometry, count, seed + 1)
    first_to_second = nearest_distances(first, second)
    second_to_first = nearest_distances(second, first)
    precision = float(np.mean(first_to_second <= threshold))
    recall = float(np.mean(second_to_first <= threshold))
    return {"method": "same reference mesh, two independent deterministic area-weighted surface samples",
            "interpretation": "sampling-distance floor only; not reconstruction performance or a correction to primary score",
            "exact_point_to_triangle_distance": False,
            "sample_count_per_draw": count, "first_seed": seed, "second_seed": seed + 1,
            "threshold": threshold,
            "first_to_second": summarize_distances(first_to_second),
            "second_to_first": summarize_distances(second_to_first),
            "precision": precision, "recall": recall,
            "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transform", type=Path, help="JSON output-to-reference Sim(3), bound to exact mesh hashes")
    parser.add_argument("--threshold", type=float, help="precision/recall distance in reference_units")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-self-control", action="store_true",
                        help="add a distinct-seed same-reference sampling floor")
    parser.add_argument("--save-report", type=Path, help="write JSON to a fresh path")
    args = parser.parse_args()
    if args.transform and (not args.output or args.threshold is None):
        parser.error("--transform requires --output and --threshold")
    if args.threshold is not None and not (args.transform or args.reference_self_control):
        parser.error("--threshold requires --transform or --reference-self-control")
    if args.reference_self_control and args.threshold is None:
        parser.error("--reference-self-control requires --threshold")
    need_geometry = bool(args.transform or args.reference_self_control)
    reference = inspect_ply(args.reference, geometry=need_geometry)
    report = {"reference": reference[0] if need_geometry else reference,
              "geometry_quality_score": None,
              "score_status": "unavailable: independently scanned reference lacks verified alignment to reconstruction"}
    if args.output:
        output = inspect_ply(args.output, geometry=bool(args.transform))
        report["reconstruction"] = output[0] if args.transform else output
    if args.transform:
        provenance, matrix, scale = load_sim3(args.transform, args.reference, args.output)
        comparison = compare_meshes(reference[1:], output[1:], matrix,
                                    threshold=args.threshold, count=args.samples, seed=args.seed)
        comparison["reference_units"] = provenance["reference_units"]
        comparison["sim3_scale_output_to_reference"] = scale
        comparison["registration_basis"] = provenance["registration_basis"]
        comparison["transform_provenance"] = provenance["provenance"]
        comparison["scale_provenance"] = provenance["scale_provenance"]
        comparison["object_id"] = provenance["object_id"]
        comparison["reference_faces_excluded_from_sampling"] = (
            report["reference"]["nontriangle_faces"] + report["reference"]["zero_area_triangle_faces"])
        comparison["reconstruction_faces_excluded_from_sampling"] = (
            report["reconstruction"]["nontriangle_faces"] + report["reconstruction"]["zero_area_triangle_faces"])
        comparison["reference_sha256"] = provenance["reference_sha256"]
        comparison["output_sha256"] = provenance["output_sha256"]
        comparison["metric_accuracy_claim_allowed"] = False
        comparison["interpretation"] = ("reference-fitted descriptive comparison; scale and alignment were fitted using the reference"
                                         if provenance["registration_basis"] == "reference-fit" else
                                         "conditional on the supplied, independently unverified transform")
        report["sampled_surface_comparison"] = comparison
        report["score_status"] = "sampled comparison available; transform provenance supplied but not independently verified"
    if args.reference_self_control:
        control = reference_sampling_control(reference[1:], threshold=args.threshold,
                                             count=args.samples, seed=args.seed)
        control["reference_units"] = (provenance["reference_units"] if args.transform else
                                      "reference-coordinate units (physical units unverified)")
        report["reference_sampling_control"] = control
    rendered = json.dumps(report, indent=2) + "\n"
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with args.save_report.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
