#!/usr/bin/env python3
"""Score every frozen sparse point against the aligned ETH3D eval scan."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import xml.etree.ElementTree as ET

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import cv2
import numpy as np
import numpy.core._multiarray_umath as numpy_binary

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
PROTOCOL = ROOT / "docs/PIPES-EVALUATION-PROTOCOL.md"

THRESHOLDS_M = (0.01, 0.02, 0.05, 0.1)
MAX_SCAN_POINTS = 12_000_000
MAX_SPARSE_BYTES = 128 * 1024**2
MAX_REPORT_BYTES = 128 * 1024**2
TIME_LIMIT_S = 300
FREE_FLOOR_BYTES = 10 * 1024**3
TYPE_MAP = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
            "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
            "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded_file(path, maximum=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or (maximum is not None and path.stat().st_size > maximum):
        raise ValueError(f"missing, linked or oversized input: {path}")
    return path


def ply_layout(path):
    with bounded_file(path).open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline(4097)
            if not line or len(line) > 4096 or len(header) + len(line) > 65536:
                raise ValueError("invalid PLY header")
            header.extend(line)
    lines = header.decode("ascii").splitlines()
    if len(lines) < 4 or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("expected little-endian binary PLY")
    elements = []
    for line in lines[2:-1]:
        words = line.split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "element" and len(words) == 3:
            count = int(words[2])
            if count < 0:
                raise ValueError("negative PLY element count")
            elements.append({"name": words[1], "count": count, "props": []})
        elif words[0] == "property" and len(words) == 3 and elements and words[1] in TYPE_MAP:
            elements[-1]["props"].append((words[2], TYPE_MAP[words[1]]))
        else:
            raise ValueError("unsupported PLY property")
    vertices = [e for e in elements if e["name"] == "vertex"]
    if len(vertices) != 1 or elements[0] is not vertices[0]:
        raise ValueError("PLY vertex element must occur first and exactly once")
    vertex = vertices[0]
    if not 0 < vertex["count"] <= MAX_SCAN_POINTS:
        raise ValueError("scan point count outside budget")
    names = [name for name, _ in vertex["props"]]
    if len(names) != len(set(names)) or not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY needs distinct x/y/z fields")
    for name, typ in vertex["props"]:
        if name in ("x", "y", "z") and typ not in ("<f4", "<f8"):
            raise ValueError("PLY xyz must be floating point")
    dtypes = [np.dtype(e["props"]) for e in elements]
    payload = sum(e["count"] * dt.itemsize for e, dt in zip(elements, dtypes))
    if path.stat().st_size != len(header) + payload:
        raise ValueError("PLY payload length mismatch")
    return len(header), vertex["count"], dtypes[0]


def mlp_matrix(path, scan_path):
    root = ET.parse(bounded_file(path, 1 << 20)).getroot()
    meshes = root.findall(".//MLMesh")
    if len(meshes) != 1 or meshes[0].get("filename") != scan_path.name:
        raise ValueError("MLP must identify exactly the supplied scan")
    node = meshes[0].find("MLMatrix44")
    if node is None or not node.text:
        raise ValueError("missing MLP transform")
    values = np.array([float(x) for x in node.text.split()], dtype=np.float64)
    if values.size != 16 or not np.isfinite(values).all():
        raise ValueError("invalid MLP matrix")
    matrix = values.reshape(4, 4)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4, rtol=0):
        raise ValueError("invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-3)):
        raise ValueError("MLP matrix is not a proper rigid transform")
    return matrix


def scan_points(path, matrix, deadline):
    offset, count, dtype = ply_layout(path)
    raw = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    world = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, 250_000):
        if time.monotonic() > deadline:
            raise TimeoutError("scan load exceeded 300 seconds")
        end = min(count, start + 250_000)
        local = np.column_stack([raw[name][start:end] for name in ("x", "y", "z")])
        if not np.isfinite(local).all():
            raise ValueError("nonfinite scan point")
        chunk = local @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(chunk).all():
            raise ValueError("invalid transformed scan point")
        world[start:end] = chunk
    del raw
    return world


def sparse_points(points_path, report_path):
    report = json.loads(bounded_file(report_path, MAX_SPARSE_BYTES).read_text())
    data = json.loads(bounded_file(points_path, MAX_SPARSE_BYTES).read_text())
    if (report.get("schema") != "eth3d_pipes_fixed_camera_sparse_v1"
            or report.get("reference_geometry_used") is not False
            or data.get("schema") != "fixed_camera_sparse_points_v1"):
        raise ValueError("unrecognized or reference-fed sparse run")
    points = data.get("points")
    if not isinstance(points, list) or not points or len(points) != report.get("accepted_tracks"):
        raise ValueError("empty or inconsistent accepted sparse population")
    ids, xyz = [], []
    for record in points:
        ident = record.get("id")
        coord = record.get("xyz")
        if (not isinstance(ident, int) or isinstance(ident, bool) or ident < 1
                or not isinstance(coord, list) or len(coord) != 3):
            raise ValueError("invalid sparse point")
        ids.append(ident)
        xyz.append(coord)
    if len(set(ids)) != len(ids) or ids != list(range(1, len(ids) + 1)):
        raise ValueError("sparse point IDs must be unique and ordered")
    result = np.asarray(xyz, dtype=np.float64)
    if result.shape != (len(ids), 3) or not np.isfinite(result).all():
        raise ValueError("nonfinite sparse point")
    return ids, result, report


def brute_distance(world, query, deadline):
    if not len(world):
        raise ValueError("empty reference cloud")
    best_squared = math.inf
    best_index = -1
    q = query.astype(np.float64)
    for start in range(0, len(world), 250_000):
        if time.monotonic() > deadline:
            raise TimeoutError("full-scan nearest-neighbor search exceeded 300 seconds")
        chunk = world[start:start+250_000].astype(np.float64)
        squared = np.sum((chunk - q) ** 2, axis=1)
        local_index = int(np.argmin(squared))
        local_squared = float(squared[local_index])
        global_index = start + local_index
        if local_squared < best_squared or (local_squared == best_squared and global_index < best_index):
            best_squared = local_squared
            best_index = global_index
    return math.sqrt(best_squared), best_index


def exact_distances(world, queries, deadline):
    distances = np.empty(len(queries), dtype=np.float64)
    nearest = np.empty(len(queries), dtype=np.int32)
    for i, query in enumerate(queries):
        distances[i], nearest[i] = brute_distance(world, query, deadline)
    return distances, nearest


def summarise(distances):
    if not len(distances):
        raise ValueError("empty sparse distance population")
    counts = {f"{threshold:.2f}": int(np.count_nonzero(distances <= threshold))
              for threshold in THRESHOLDS_M}
    return {"denominator": len(distances), "threshold_m": list(THRESHOLDS_M),
            "within_threshold_counts": counts,
            "within_threshold_fractions": {key: value / len(distances) for key, value in counts.items()},
            "median_m": float(np.median(distances)), "p90_m": float(np.percentile(distances, 90)),
            "p95_m": float(np.percentile(distances, 95)), "max_m": float(np.max(distances))}


def configure_threads():
    cv2.setNumThreads(0)
    actual = cv2.getNumThreads()
    if actual != 1:
        raise ValueError("OpenCV serial thread setting did not apply")
    return actual


def run(prepared, sparse, output):
    deadline = time.monotonic() + TIME_LIMIT_S
    configure_threads()
    scan = prepared / "pipes/dslr_scan_eval/scan1.ply"
    alignment = prepared / "pipes/dslr_scan_eval/scan_alignment.mlp"
    prepared_report = prepared / "prepare-metadata.json"
    points = sparse / "points.json"
    sparse_report = sparse / "report.json"
    opencv_binaries = sorted(Path(cv2.__file__).resolve().parent.glob("cv2*.so"))
    if len(opencv_binaries) != 1:
        raise ValueError("cannot uniquely identify OpenCV binary")
    numpy_library = Path(numpy_binary.__file__)
    for path in (scan, alignment, prepared_report, points, sparse_report, SOURCE,
                 PROTOCOL, opencv_binaries[0], numpy_library):
        bounded_file(path)
    input_paths = (scan, alignment, prepared_report, points, sparse_report, SOURCE,
                   PROTOCOL, opencv_binaries[0], numpy_library)
    inputs = {str(p.resolve()): sha(p) for p in input_paths}
    prepared_metadata = json.loads(bounded_file(prepared_report, 2 * 1024**2).read_text())
    if prepared_metadata.get("status") != "validated":
        raise ValueError("prepared reference is not validated")
    offset, count, dtype = ply_layout(scan)
    declared = prepared_metadata.get("ply", {})
    if (declared.get("header_bytes") != offset or declared.get("vertex_count") != count
            or declared.get("vertex_stride_bytes") != dtype.itemsize):
        raise ValueError("prepared PLY metadata differs from scan")
    declared_hashes = prepared_metadata.get("file_sha256", {})
    for path in (scan, alignment):
        relative = path.relative_to(prepared).as_posix()
        if declared_hashes.get(relative) != inputs[str(path.resolve())]:
            raise ValueError("prepared reference hash differs from metadata")
    matrix = mlp_matrix(alignment, scan)
    if not np.allclose(prepared_metadata.get("alignment", {}).get("matrix_row_major"), matrix,
                       atol=1e-10, rtol=0):
        raise ValueError("prepared scan alignment differs from MLP")
    ids, queries, source_report = sparse_points(points, sparse_report)
    if (source_report.get("source_manifest") != str(prepared_report)
            or source_report.get("source_manifest_sha256") != inputs[str(prepared_report.resolve())]):
        raise ValueError("sparse run references different prepared metadata")
    world = scan_points(scan, matrix, deadline)
    distances, nearest = exact_distances(world, queries, deadline)
    if {str(p.resolve()): sha(p) for p in input_paths} != inputs:
        raise ValueError("input changed during scoring")
    result = {"schema": "eth3d_pipes_sparse_laser_proximity_v1", "status": "pass",
              "interpretation": "one-way nearest laser point distances; not official ETH3D accuracy or completeness",
              "scene_unit": "metre", "laser_points": len(world),
              "scorer_source_sha256": inputs[str(SOURCE.resolve())],
              "protocol_sha256": inputs[str(PROTOCOL.resolve())],
              "library_binary_sha256": {"opencv": inputs[str(opencv_binaries[0].resolve())],
                                        "numpy": inputs[str(numpy_library.resolve())]},
              "python_version": sys.version.split()[0], "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "opencv_set_num_threads": 0,
              "opencv_effective_threads": cv2.getNumThreads(),
              "cpu": platform.processor() or platform.machine(),
              "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_rss_unit": "bytes" if sys.platform == "darwin" else "kilobytes",
              "scan_local_to_world_matrix_row_major": matrix.tolist(),
              "source_sparse_schema": source_report["schema"], "input_sha256": inputs,
              "algorithm": {"search": "chunked full-scan brute force for every accepted sparse point",
                            "reference_chunk_points": 250000,
                            "arithmetic": "float64 scan transform and squared Euclidean distances; sqrt after global minimum",
                            "tie_break": "lowest scan vertex index",
                            "deprecated_method": "OpenCV FLANN checks=-1 failed full-scan exactness audit in score-001"},
              "summary": summarise(distances),
              "point_distances_m": [{"id": ident, "distance_m": float(distance),
                                     "nearest_scan_index": int(index)}
                                    for ident, distance, index in zip(ids, distances, nearest)]}
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError("128 MiB scorer output cap exceeded")
    if time.monotonic() > deadline:
        raise TimeoutError("scoring exceeded 300 seconds")
    if not output.is_dir() or output.is_symlink() or (output / "score.json").exists():
        raise ValueError("guard output directory missing or score already exists")
    stat = os.statvfs(output)
    if stat.f_bavail * stat.f_frsize - len(encoded) < FREE_FLOOR_BYTES:
        raise ValueError("10 GiB free-space floor would be crossed")
    with (output / "score.json").open("xb") as stream:
        stream.write(encoded)
    result["score_sha256"] = sha(output / "score.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    prepared, sparse, output = args.prepared.absolute(), args.sparse.absolute(), args.output.absolute()
    if not args.worker:
        from scripts.research_job.guard import run_child
        command = [sys.executable, "-m", "scripts.pipes_score.run", "--worker",
                   "--prepared", str(prepared), "--sparse", str(sparse), "--output", str(output)]
        status = run_child(command, ROOT, output, timeout_seconds=TIME_LIMIT_S,
                           max_output_bytes=256 * 1024**2, reserve_bytes=FREE_FLOOR_BYTES)
        print(json.dumps(status))
        if status["status"] != "succeeded":
            raise SystemExit(1)
        return
    result = run(prepared, sparse, output)
    print(json.dumps({"point_count": result["summary"]["denominator"],
                      "median_m": result["summary"]["median_m"],
                      "score_sha256": result["score_sha256"]}))


if __name__ == "__main__":
    main()
