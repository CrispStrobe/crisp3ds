#!/usr/bin/env python3
"""Bounded, deterministic sampled-surface to exact-triangle comparison.

Sampling approximates the surface integral; each sampled point's distance to the
opposite piecewise-linear mesh is exact up to floating-point arithmetic.
"""

import argparse
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import evaluate
else:
    import evaluate

MAX_SAMPLES = 4096
MAX_SECONDS = 300
LEAF_SIZE = 16


def positive_triangles(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite XYZ rows")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangle index array must have shape (n, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("face indices must be integers")
    faces = faces.astype(np.int64, copy=False)
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face index outside vertex array")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.isfinite(cross).all():
        raise ValueError("triangle cross product overflowed")
    valid = np.linalg.norm(cross, axis=1) > 0
    if not valid.any():
        raise ValueError("mesh has no positive-area triangles")
    return triangles[valid], int((~valid).sum())


def point_triangle_squared(point, triangles):
    """Exact closest-point distance for nonzero-area triangles, vectorized by leaf."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    ap = point - a
    signed = np.einsum("ij,ij->i", ap, normal)
    projected = point - (signed / normal2)[:, None] * normal
    v = projected - a
    u = np.einsum("ij,ij->i", np.cross(v, ac), normal) / normal2
    w = np.einsum("ij,ij->i", np.cross(ab, v), normal) / normal2
    result = np.full(len(triangles), np.inf)
    inside = (u >= -1e-12) & (w >= -1e-12) & (u + w <= 1 + 1e-12)
    result[inside] = signed[inside] ** 2 / normal2[inside]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        parameter = np.clip(np.einsum("ij,ij->i", point - start, edge)
                            / np.einsum("ij,ij->i", edge, edge), 0, 1)
        delta = point - (start + parameter[:, None] * edge)
        result = np.minimum(result, np.einsum("ij,ij->i", delta, delta))
    return result


def triangle_normals(triangles):
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return cross / np.linalg.norm(cross, axis=1)[:, None]


def sampled_face_normals(triangles, count, seed):
    """Reproduce evaluate.sample_surface's face choice without changing its points."""
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    choice = np.random.default_rng(seed).choice(len(triangles), size=count, p=areas / areas.sum())
    return triangle_normals(triangles)[choice]


def mesh_topology(vertices, faces, deadline=None):
    """Exact raw-mesh edge incidence and vertex-connected face components."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or len(vertices) > evaluate.MAX_SCORE_VERTICES or len(faces) > evaluate.MAX_SCORE_FACES):
        raise ValueError("topology geometry outside limits")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("topology face indices must be integers")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("topology face index outside vertex array")
    if not len(faces):
        return {"boundary_edges": 0, "nonmanifold_edges": 0,
                "vertex_connected_components": 0, "largest_component_face_fraction": None,
                "euler_v_minus_e_plus_f": 0, "referenced_vertices": 0, "unique_edges": 0}
    faces = faces.astype(np.int32, copy=False)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    used_vertices = int(np.unique(faces).size)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    del order
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    edge_count = int(len(counts))
    del edges, starts, counts
    parent = np.arange(len(vertices), dtype=np.int32)
    rank = np.zeros(len(vertices), dtype=np.uint8)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a, b):
        a, b = root(int(a)), root(int(b))
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    for i, (a, b, c) in enumerate(faces):
        if deadline is not None and i % 8192 == 0 and time.monotonic() > deadline:
            raise TimeoutError("topology scoring exceeded time limit")
        union(a, b)
        union(a, c)
    roots = np.fromiter((root(int(face[0])) for face in faces), dtype=np.int32, count=len(faces))
    _, component_faces = np.unique(roots, return_counts=True)
    return {"boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
            "vertex_connected_components": int(len(component_faces)),
            "largest_component_face_fraction": float(component_faces.max() / len(faces)) if len(faces) else None,
            "euler_v_minus_e_plus_f": int(used_vertices - edge_count + len(faces)),
            "referenced_vertices": used_vertices, "unique_edges": edge_count}


class TriangleBVH:
    """Median-split AABB tree; best-first search has no approximation cutoff."""

    def __init__(self, triangles, deadline=None):
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not len(self.triangles):
            raise ValueError("need at least one triangle")
        self.lower = self.triangles.min(axis=1)
        self.upper = self.triangles.max(axis=1)
        self.centers = (self.lower + self.upper) / 2
        self.order = np.arange(len(self.triangles), dtype=np.int32)
        self.nodes = []
        self.deadline = deadline
        self._build(0, len(self.order))

    def _build(self, start, end):
        if self.deadline is not None and len(self.nodes) % 1024 == 0 and time.monotonic() > self.deadline:
            raise TimeoutError("surface indexing exceeded time limit")
        indices = self.order[start:end]
        lower = self.lower[indices].min(axis=0)
        upper = self.upper[indices].max(axis=0)
        node = len(self.nodes)
        self.nodes.append((lower, upper, -1, -1, start, end))
        if end - start > LEAF_SIZE:
            axis = int(np.argmax(np.ptp(self.centers[indices], axis=0)))
            middle = (start + end) // 2
            self.order[start:end] = indices[np.argpartition(self.centers[indices, axis], middle - start)]
            left = self._build(start, middle)
            right = self._build(middle, end)
            self.nodes[node] = (lower, upper, left, right, start, end)
        return node

    @staticmethod
    def _box_squared(point, node):
        lower, upper = node[:2]
        delta = np.maximum(0, np.maximum(lower - point, point - upper))
        return float(delta @ delta)

    def distances_and_faces(self, points, deadline=None):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("query points must be finite XYZ rows")
        out = np.empty(len(points))
        nearest = np.empty(len(points), dtype=np.int32)
        for i, point in enumerate(points):
            if deadline is not None and i % 16 == 0 and time.monotonic() > deadline:
                raise TimeoutError("surface scoring exceeded time limit")
            best = math.inf
            best_face = -1
            pending = [(self._box_squared(point, self.nodes[0]), 0)]
            visited = 0
            while pending:
                visited += 1
                if deadline is not None and visited % 256 == 0 and time.monotonic() > deadline:
                    raise TimeoutError("surface scoring exceeded time limit")
                bound, index = heapq.heappop(pending)
                if bound >= best:
                    break
                node = self.nodes[index]
                if node[2] < 0:
                    ids = self.order[node[4]:node[5]]
                    squared = point_triangle_squared(point, self.triangles[ids])
                    local = int(np.argmin(squared))
                    if float(squared[local]) < best:
                        best = float(squared[local])
                        best_face = int(ids[local])
                else:
                    for child in (node[2], node[3]):
                        child_bound = self._box_squared(point, self.nodes[child])
                        if child_bound < best:
                            heapq.heappush(pending, (child_bound, child))
            out[i] = math.sqrt(best)
            nearest[i] = best_face
        return out, nearest

    def distances(self, points, deadline=None):
        return self.distances_and_faces(points, deadline)[0]


def compare(reference_geometry, output_geometry, matrix, *, thresholds, count=2048, seed=2027):
    if not 1 <= count <= MAX_SAMPLES or not 0 <= seed < 2**64:
        raise ValueError("samples or seed outside limits")
    thresholds = list(thresholds)
    if not thresholds or len(thresholds) > 16 or any(not math.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError("provide 1–16 finite positive thresholds")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be distinct")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("invalid transform matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("invalid homogeneous row")
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(singular.mean())
    if (not 1e-9 < scale < 1e9 or not np.allclose(singular, scale, rtol=1e-6, atol=0)
            or np.linalg.det(linear) <= 0):
        raise ValueError("transform must be a proper uniform-scale Sim(3)")
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    ref_vertices, ref_faces = reference_geometry
    out_vertices, out_faces = output_geometry
    ref_triangles, ref_dropped = positive_triangles(ref_vertices, ref_faces)
    out_vertices = np.asarray(out_vertices, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    out_triangles, out_dropped = positive_triangles(out_vertices, out_faces)
    prepared = time.monotonic()
    ref_topology = mesh_topology(ref_vertices, ref_faces, deadline)
    out_topology = mesh_topology(out_vertices, out_faces, deadline)
    topologized = time.monotonic()
    ref_tree = TriangleBVH(ref_triangles, deadline)
    out_tree = TriangleBVH(out_triangles, deadline)
    indexed = time.monotonic()
    # Samples are independent of the index, and both directions use equal counts.
    ref_samples = evaluate.sample_surface(ref_vertices, ref_faces, count, seed)
    out_samples = evaluate.sample_surface(out_vertices, out_faces, count, seed + 1)
    ref_normals = sampled_face_normals(ref_triangles, count, seed)
    out_normals = sampled_face_normals(out_triangles, count, seed + 1)
    sampled = time.monotonic()
    accuracy, ref_nearest = ref_tree.distances_and_faces(out_samples, deadline)
    completeness, out_nearest = out_tree.distances_and_faces(ref_samples, deadline)
    accuracy_normal = np.abs(np.einsum("ij,ij->i", out_normals, triangle_normals(ref_triangles)[ref_nearest]))
    completeness_normal = np.abs(np.einsum("ij,ij->i", ref_normals, triangle_normals(out_triangles)[out_nearest]))
    accuracy_normal = np.clip(accuracy_normal, 0, 1)
    completeness_normal = np.clip(completeness_normal, 0, 1)
    finished = time.monotonic()
    ref_vertices = np.asarray(ref_vertices, dtype=np.float64)
    diagonal = float(np.linalg.norm(ref_vertices.max(axis=0) - ref_vertices.min(axis=0)))
    symmetric_mean = float((accuracy.mean() + completeness.mean()) / 2)
    rows = []
    for threshold in thresholds:
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        rows.append({"threshold": threshold, "precision": precision, "recall": recall,
                     "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                     "accuracy_normal_abs_dot_within_threshold": float(accuracy_normal[accuracy <= threshold].mean()) if precision else None,
                     "completeness_normal_abs_dot_within_threshold": float(completeness_normal[completeness <= threshold].mean()) if recall else None})
    accuracy_summary = evaluate.summarize_distances(accuracy)
    completeness_summary = evaluate.summarize_distances(completeness)
    accuracy_summary["p95"] = float(np.quantile(accuracy, 0.95))
    completeness_summary["p95"] = float(np.quantile(completeness, 0.95))
    return {"method": "area-weighted Monte Carlo query samples; closest target triangle via exhaustive-pruning AABB tree",
            "target_distance_exact_for_samples_up_to_floating_point": True,
            "barycentric_boundary_tolerance": 1e-12,
            "surface_integral_approximate": True,
            "sample_count_per_mesh": count, "reference_seed": seed, "output_seed": seed + 1,
            "reference_zero_area_faces_excluded": ref_dropped,
            "output_zero_area_faces_excluded": out_dropped,
            "output_to_reference_accuracy": accuracy_summary,
            "reference_to_output_completeness": completeness_summary,
            "chamfer_l1_mean": symmetric_mean,
            "chamfer_l1_mean_normalized_by_reference_bbox_diagonal": symmetric_mean / diagonal,
            "reference_bbox_diagonal": diagonal,
            "normal_consistency_abs_dot": {"output_to_reference_mean": float(accuracy_normal.mean()),
                                            "reference_to_output_mean": float(completeness_normal.mean())},
            "topology": {"reference": ref_topology, "output": out_topology,
                         "definition": "raw triangle edges; vertex-connected face components"},
            "threshold_scores": rows,
            "stage_seconds": {"triangle_preparation": prepared - started,
                              "topology": topologized - prepared,
                              "tree_build": indexed - topologized, "sampling": sampled - indexed,
                              "distance_queries": finished - sampled, "total": finished - started},
            "limits": {"samples_per_mesh": MAX_SAMPLES, "time_seconds": MAX_SECONDS,
                       "leaf_triangles": LEAF_SIZE,
                       "geometry_vertices_per_mesh": evaluate.MAX_SCORE_VERTICES,
                       "geometry_faces_per_mesh": evaluate.MAX_SCORE_FACES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transform", type=Path)
    parser.add_argument("--self-control", action="store_true", help="compare reference against itself")
    parser.add_argument("--threshold", type=float, action="append", required=True,
                        help="repeat for multiple absolute thresholds in reference units")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save-report", type=Path)
    args = parser.parse_args()
    if args.self_control:
        if args.output or args.transform:
            parser.error("self-control takes only --reference")
    elif not args.output or not args.transform:
        parser.error("scoring needs --output and exact-mesh-bound --transform")
    reference = evaluate.inspect_ply(args.reference, geometry=True)
    if reference[0]["nontriangle_faces"]:
        parser.error("reference mesh contains nontriangle faces")
    if args.self_control:
        output = reference
        matrix = np.eye(4)
        provenance = None
    else:
        output = evaluate.inspect_ply(args.output, geometry=True)
        if output[0]["nontriangle_faces"]:
            parser.error("output mesh contains nontriangle faces")
        provenance, matrix, scale = evaluate.load_sim3(args.transform, args.reference, args.output)
    result = compare(reference[1:], output[1:], matrix, thresholds=args.threshold,
                     count=args.samples, seed=args.seed)
    result.update({"reference": reference[0], "output": output[0],
                   "reference_sha256": evaluate.sha256_file(args.reference),
                   "output_sha256": evaluate.sha256_file(args.reference if args.self_control else args.output),
                   "reference_units": provenance["reference_units"] if provenance else "reference-coordinate units (physical units unverified)",
                   "registration_basis": provenance["registration_basis"] if provenance else "identity-self-control",
                   "sim3_scale_output_to_reference": scale if provenance else 1.0,
                   "scale_provenance": provenance["scale_provenance"] if provenance else "identity self-control",
                   "transform_provenance": provenance["provenance"] if provenance else "same exact reference mesh",
                   "interpretation": ("same-mesh numerical and sampling control; not reconstruction quality" if args.self_control else
                                      "reference-fitted shape diagnostic; scale and alignment depend on this reference" if provenance["registration_basis"] == "reference-fit" else
                                      "conditional on supplied alignment; independent metric accuracy not certified"),
                   "metric_accuracy_claim_allowed": False})
    rendered = json.dumps(result, indent=2) + "\n"
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with args.save_report.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
