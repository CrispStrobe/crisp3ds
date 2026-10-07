"""Score a reconstructed STL against an independent physical scan, after the fact.

Evaluation only: nothing computed here may be fed back into reconstruction.

The scan's support platform is removed from the REFERENCE ONLY (dominant plane,
keep triangles clearly above it, largest connected component). The mesh is then
mapped onto the remaining reference object by an explicit similarity transform:
many rotation starts (PCA signed permutations plus seeded random rotations),
each run through a symmetric trimmed scaled ICP on area-weighted surface
samples, the best few refined on more samples and the winner finished against
the triangles themselves. The fit uses one random half of
the samples; every reported distance comes from the other half, measured to the
other surface's triangles.

Limits. The registration is shape-only, so scale and pose are whatever
minimises the trimmed residual: a uniformly too-large or too-small
reconstruction cannot be detected, and a systematic shape error is partly
absorbed by the fit. The fit and held-out samples lie on the same two surfaces,
so holding out guards against overfitting sample noise, not against that
absorption. Point-to-surface distance takes the exact distance to the nearest
few triangles by centroid, which is an upper bound that is tight when triangles
are small against the distance. The reconstructions are closed solids with an
invented underside, while the scan is open where the object stood on the
platform and loses the band removed with the platform, so the "all" numbers
include those regions and the "above_margin" numbers exclude them.

Handedness. A proper similarity cannot map a shape onto its mirror image. By
default both the mesh and its mirror image (x negated) are registered, both
residuals are recorded and the better one is scored, with a warning when that
is the mirrored one: the mesh and the scan then disagree in chirality, and this
tool cannot tell which of the two is flipped.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scripts.object_dataset.evaluate import inspect_ply

THRESHOLDS = (0.005, 0.01, 0.02)


def read_binary_stl(path):
    data = np.fromfile(path, np.uint8)
    if len(data) < 84:
        raise ValueError("STL too short")
    count = int(data[80:84].view("<u4")[0])
    if len(data) != 84 + 50 * count or not count:
        raise ValueError("only binary STL with a consistent triangle count is supported")
    record = np.dtype([("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = data[84:].view(record)["points"].astype(np.float64)
    if not np.isfinite(triangles).all():
        raise ValueError("nonfinite STL coordinate")
    return triangles


def read_reference(path):
    report, vertices, faces = inspect_ply(Path(path), geometry=True)
    if report["nontriangle_faces"]:
        raise ValueError("reference must contain only triangles")
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def triangle_areas(triangles):
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def sample_surface(triangles, count, rng):
    """Area-weighted uniform surface samples and the unit normal of each one's triangle."""
    area = triangle_areas(triangles)
    if not area.sum() > 0:
        raise ValueError("surface has no area")
    chosen = rng.choice(len(triangles), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = (triangles[chosen, k] for k in range(3))
    points = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    return points, normal


def fit_plane(points):
    centre = points.mean(0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][2]
    return normal, float(normal @ centre)


def remove_platform(vertices, faces, rng, band=0.002, margin=0.006, trials=600, samples=20000):
    """Drop the dominant plane and everything not clearly above it.

    Heights are n.x - offset with n pointing to the side holding more of the
    remaining surface. A slab platform has two parallel faces; the plane is moved
    to the upper one. band and margin are fractions of the whole scan's diagonal.
    """
    triangles = vertices[faces]
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    tolerance = band * diagonal
    points, _ = sample_surface(triangles, samples, rng)
    best = (-1, None, None)
    for _ in range(trials):
        a, b, c = points[rng.choice(samples, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        inliers = int((np.abs((points - a) @ normal) < tolerance).sum())
        if inliers > best[0]:
            best = (inliers, normal, float(normal @ a))
    _, normal, offset = best
    for _ in range(3):
        normal, offset = fit_plane(points[np.abs(points @ normal - offset) < tolerance])
    height = points @ normal - offset
    if (height > tolerance).sum() < (height < -tolerance).sum():
        normal, offset, height = -normal, -offset, -height
    # Slab platforms: step to the highest parallel level that is still plane-like.
    levels = np.arange(-0.05 * diagonal, 0.05 * diagonal, 0.25 * tolerance)
    mass = np.array([(np.abs(height - level) < tolerance).sum() for level in levels])
    shift = float(levels[np.nonzero(mass >= 0.5 * mass.max())[0][-1]])
    if abs(shift) > tolerance:
        offset += shift
        near = np.abs(points @ normal - offset) < tolerance
        refit_normal, refit_offset = fit_plane(points[near])
        if refit_normal @ normal < 0:
            refit_normal, refit_offset = -refit_normal, -refit_offset
        normal, offset = refit_normal, refit_offset
    height = points @ normal - offset
    plane_fraction = float((np.abs(height) < tolerance).mean())

    cut = margin * diagonal
    vertex_height = vertices @ normal - offset
    above = (vertex_height[faces] > cut).all(1)
    area = triangle_areas(triangles)
    kept_faces = faces[above]
    edges = np.concatenate((kept_faces[:, [0, 1]], kept_faces[:, [1, 2]]))
    graph = sparse.coo_matrix(
        (np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2
    )
    _, label = connected_components(graph, directed=False)
    face_label = label[kept_faces[:, 0]]
    component_area = np.bincount(face_label, weights=area[above])
    largest = int(component_area.argmax())
    keep = np.zeros(len(faces), bool)
    keep[np.nonzero(above)[0][face_label == largest]] = True
    report = {
        "method": "RANSAC dominant plane on area samples, least-squares refit, moved to the upper "
                  "face of a slab; kept triangles with all vertices above the margin; kept the "
                  "largest-area connected component",
        "scan_bbox_diagonal": diagonal,
        "plane_normal": normal.tolist(),
        "plane_offset": float(offset),
        "plane_equation": "height = normal . x - offset, in reference units",
        "inlier_band": tolerance,
        "inlier_band_fraction_of_scan_diagonal": band,
        "slab_shift_applied": shift if abs(shift) > tolerance else 0.0,
        "area_fraction_within_band_of_plane": plane_fraction,
        "margin_above_plane": cut,
        "margin_fraction_of_scan_diagonal": margin,
        "input_vertices": int(len(vertices)),
        "input_triangles": int(len(faces)),
        "input_area": float(area.sum()),
        "removed_triangles_not_above_margin": int((~above).sum()),
        "removed_area_not_above_margin": float(area[~above].sum()),
        "components_above_margin": int(len(np.unique(face_label))),
        "removed_triangles_in_small_components": int(above.sum() - keep.sum()),
        "removed_area_in_small_components": float(area[above].sum() - area[keep].sum()),
        "kept_triangles": int(keep.sum()),
        "kept_vertices": int(len(np.unique(faces[keep]))),
        "kept_area": float(area[keep].sum()),
        "kept_height_range": [float(vertex_height[faces[keep]].min()),
                              float(vertex_height[faces[keep]].max())],
    }
    return keep, normal, float(offset), report


def boundary_report(faces):
    """Open-edge count of a triangle set: how much of the scan surface is unclosed."""
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"edges": int(len(counts)), "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum())}


def umeyama(source, target):
    """Least-squares proper similarity (scale, rotation, translation) source -> target."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / (a * a).sum(1).mean())
    return scale, rotation, target_mean - scale * rotation @ source_mean


def apply(transform, points):
    scale, rotation, translation = transform
    return scale * points @ rotation.T + translation


def apply_inverse(transform, points):
    scale, rotation, translation = transform
    return (points - translation) @ rotation / scale


def trimmed_icp(source, target, source_tree, target_tree, start, iterations, keep=0.8, tolerance=1e-6):
    """Symmetric trimmed scaled ICP.

    Both directions contribute pairs (each keeps its closest `keep` fraction), so
    the scale cannot collapse onto a patch. The residual is the RMS of the kept
    distances of both directions in target units.
    """
    transform = start
    previous = np.inf
    n_source, n_target = int(keep * len(source)), int(keep * len(target))
    residual = np.inf
    used = 0
    for used in range(1, iterations + 1):
        forward, forward_index = target_tree.query(apply(transform, source))
        backward, backward_index = source_tree.query(apply_inverse(transform, target))
        backward = backward * transform[0]
        forward_keep = np.argpartition(forward, n_source - 1)[:n_source]
        backward_keep = np.argpartition(backward, n_target - 1)[:n_target]
        residual = float(np.sqrt(
            (np.square(forward[forward_keep]).sum() + np.square(backward[backward_keep]).sum())
            / (n_source + n_target)
        ))
        if abs(previous - residual) < tolerance * max(residual, 1e-30):
            break
        previous = residual
        transform = umeyama(
            np.concatenate((source[forward_keep], source[backward_index[backward_keep]])),
            np.concatenate((target[forward_index[forward_keep]], target[backward_keep])),
        )
        if not 1e-6 < transform[0] < 1e6:
            return transform, np.inf, used
    return transform, residual, used


def pca_frame(points):
    centred = points - points.mean(0)
    axes = np.linalg.eigh(centred.T @ centred)[1][:, ::-1].copy()
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return axes


def proper_signed_permutations():
    matrices = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[range(3), order] = signs
            if np.linalg.det(matrix) > 0:
                matrices.append(matrix)
    return matrices


def start_rotations(source, target, random_starts, seed):
    source_axes, target_axes = pca_frame(source), pca_frame(target)
    rotations = [target_axes @ m @ source_axes.T for m in proper_signed_permutations()]
    kinds = ["pca"] * len(rotations)
    if random_starts:
        rotations += list(Rotation.random(random_starts, random_state=seed).as_matrix())
        kinds += ["random"] * random_starts
    return rotations, kinds


def rotation_angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def register(source, target, seed, random_starts=96, coarse_points=1500, refine_points=20000,
             keep=0.8, refine_best=4, refine_other=2, same_angle=30.0):
    """Multi-start similarity registration source -> target with a transparent record."""
    rng = np.random.default_rng(seed)

    def subset(points, count):
        return points if len(points) <= count else points[rng.choice(len(points), count, replace=False)]

    coarse_source, coarse_target = subset(source, coarse_points), subset(target, coarse_points)
    source_tree, target_tree = cKDTree(coarse_source), cKDTree(coarse_target)
    source_mean, target_mean = source.mean(0), target.mean(0)
    scale = float(np.sqrt(np.square(target - target_mean).sum(1).mean()
                          / np.square(source - source_mean).sum(1).mean()))
    rotations, kinds = start_rotations(source, target, random_starts, seed)
    coarse = []
    for index, (rotation, kind) in enumerate(zip(rotations, kinds)):
        start = (scale, rotation, target_mean - scale * rotation @ source_mean)
        transform, residual, used = trimmed_icp(
            coarse_source, coarse_target, source_tree, target_tree, start, 40, keep, 1e-4)
        coarse.append({"start": index, "kind": kind, "transform": transform,
                       "residual": residual, "iterations": used})
    coarse.sort(key=lambda item: item["residual"])
    leader = coarse[0]["transform"][1]
    for item in coarse:
        item["angle"] = rotation_angle(leader, item["transform"][1])
    same = [item for item in coarse if item["angle"] <= same_angle]
    other = [item for item in coarse if item["angle"] > same_angle]
    chosen = same[:refine_best] + other[:refine_other]

    fine_source, fine_target = subset(source, refine_points), subset(target, refine_points)
    source_tree, target_tree = cKDTree(fine_source), cKDTree(fine_target)
    refined = []
    for item in chosen:
        transform, residual, used = trimmed_icp(
            fine_source, fine_target, source_tree, target_tree, item["transform"], 100, keep, 1e-5)
        refined.append({"start": item["start"], "kind": item["kind"], "transform": transform,
                        "residual": residual, "iterations": used,
                        "coarse_residual": item["residual"]})
    refined.sort(key=lambda item: item["residual"])
    best = refined[0]
    for item in refined:
        item["angle"] = rotation_angle(best["transform"][1], item["transform"][1])
    distinct = [item for item in refined if item["angle"] > same_angle]

    def public(item, angle_name):
        return {"start": item["start"], "kind": item["kind"],
                "trimmed_rms": item["residual"], "iterations": item["iterations"],
                "scale": item["transform"][0], angle_name: item["angle"]}

    report = {
        "objective": f"RMS of the closest {keep:.0%} nearest-sample distances in each direction "
                     "(mesh->reference and reference->mesh), reference units",
        "starts": len(rotations),
        "start_kinds": {"pca_signed_permutations": kinds.count("pca"),
                        "seeded_random_rotations": kinds.count("random")},
        "initial_scale_from_rms_radius": scale,
        "coarse_points_per_surface": int(len(coarse_source)),
        "refine_points_per_surface": int(len(fine_source)),
        "same_optimum_angle_degrees": same_angle,
        "coarse_starts_at_best_optimum": len(same),
        "coarse_starts_elsewhere": len(other),
        "coarse_best": [public(item, "angle_to_coarse_best_degrees") for item in coarse[:10]],
        "coarse_best_elsewhere": [public(item, "angle_to_coarse_best_degrees") for item in other[:5]],
        "coarse_residual_quantiles_0_25_50_75_100": np.percentile(
            [item["residual"] for item in coarse], [0, 25, 50, 75, 100]).tolist(),
        "refined": [public(item, "angle_to_best_degrees") for item in refined],
        "best_trimmed_rms": best["residual"],
        "best_distinct_trimmed_rms": distinct[0]["residual"] if distinct else None,
        "distinct_to_best_residual_ratio":
            distinct[0]["residual"] / best["residual"] if distinct else None,
    }
    return best["transform"], report


def closest_on_triangles(points, triangles):
    """Closest point of triangles[i] to points[i] (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c

    def dot(x, y):
        return np.einsum("ij,ij->i", x, y)

    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        total = va + vb + vc
        closest = a + ab * (vb / total)[:, None] + ac * (vc / total)[:, None]
        edge_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest[edge_bc] = (b + (c - b) * w[:, None])[edge_bc]
        edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        closest[edge_ac] = (a + ac * (d2 / (d2 - d6))[:, None])[edge_ac]
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        closest[edge_ab] = (a + ab * (d1 / (d1 - d3))[:, None])[edge_ab]
    corner_c = (d6 >= 0) & (d5 <= d6)
    closest[corner_c] = c[corner_c]
    corner_b = (d3 >= 0) & (d4 <= d3)
    closest[corner_b] = b[corner_b]
    corner_a = (d1 <= 0) & (d2 <= 0)
    closest[corner_a] = a[corner_a]
    bad = ~np.isfinite(closest).all(1)
    closest[bad] = a[bad]
    return closest


def surface_closest(points, triangles, tree=None, candidates=12, chunk=20000):
    """Distance and closest point on the `candidates` triangles with the nearest centroids."""
    if tree is None:
        tree = cKDTree(triangles.mean(1))
    candidates = min(candidates, len(triangles))
    result = np.empty(len(points))
    nearest = np.empty((len(points), 3))
    for begin in range(0, len(points), chunk):
        part = points[begin:begin + chunk]
        index = tree.query(part, k=candidates)[1].reshape(len(part), candidates)
        closest = closest_on_triangles(
            np.repeat(part, candidates, axis=0), triangles[index.ravel()]
        ).reshape(len(part), candidates, 3)
        distance = np.linalg.norm(closest - part[:, None], axis=2)
        best = distance.argmin(1)
        rows = np.arange(len(part))
        result[begin:begin + chunk] = distance[rows, best]
        nearest[begin:begin + chunk] = closest[rows, best]
    return result, nearest


def surface_distance(points, triangles, tree=None):
    return surface_closest(points, triangles, tree)[0]


def polish(transform, source, target, source_triangles, target_triangles, source_tree, target_tree,
           keep=0.8, iterations=30, tolerance=1e-5):
    """Finish the fit against the triangles themselves, removing sample-spacing noise.

    Same symmetric trimmed objective as trimmed_icp, with each fit sample paired
    to its closest point on the other surface instead of the nearest sample.
    """
    n_source, n_target = int(keep * len(source)), int(keep * len(target))
    previous, residual, used = np.inf, np.inf, 0
    for used in range(1, iterations + 1):
        forward, on_target = surface_closest(apply(transform, source), target_triangles, target_tree)
        backward, on_source = surface_closest(
            apply_inverse(transform, target), source_triangles, source_tree)
        backward = backward * transform[0]
        forward_keep = np.argpartition(forward, n_source - 1)[:n_source]
        backward_keep = np.argpartition(backward, n_target - 1)[:n_target]
        residual = float(np.sqrt(
            (np.square(forward[forward_keep]).sum() + np.square(backward[backward_keep]).sum())
            / (n_source + n_target)))
        if abs(previous - residual) < tolerance * max(residual, 1e-30):
            break
        previous = residual
        transform = umeyama(
            np.concatenate((source[forward_keep], on_source[backward_keep])),
            np.concatenate((on_target[forward_keep], target[backward_keep])))
    return transform, residual, used


def distance_summary(distance, diagonal):
    if not len(distance):
        return None
    values = {"mean": float(distance.mean()), "median": float(np.median(distance)),
              "p90": float(np.percentile(distance, 90)), "p95": float(np.percentile(distance, 95)),
              "max": float(distance.max())}
    return {"samples": int(len(distance)), "reference_units": values,
            "fraction_of_diagonal": {key: value / diagonal for key, value in values.items()}}


def score(mesh_to_reference, reference_to_mesh, diagonal):
    result = {"accuracy_mesh_to_reference": distance_summary(mesh_to_reference, diagonal),
              "completeness_reference_to_mesh": distance_summary(reference_to_mesh, diagonal),
              "thresholds": {}}
    for fraction in THRESHOLDS:
        precision = float((mesh_to_reference <= fraction * diagonal).mean())
        recall = float((reference_to_mesh <= fraction * diagonal).mean())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["thresholds"][f"{fraction:g}"] = {
            "fraction_of_diagonal": fraction, "distance_reference_units": fraction * diagonal,
            "precision": precision, "recall": recall, "f1": f1}
    return result


def view_axes(normal, points):
    """Right, up and toward-viewer axes for front, side and top views; up is the plane normal."""
    centred = points - points.mean(0)
    flat = centred - np.outer(centred @ normal, normal)
    major = np.linalg.eigh(flat.T @ flat)[1][:, -1]
    major -= (major @ normal) * normal
    major /= np.linalg.norm(major)
    minor = np.cross(normal, major)
    # right x up = toward in every view, so the pictures are not mirrored.
    return (("front", major, normal, -minor), ("side", minor, normal, major),
            ("top", major, minor, normal))


def splat(points, colours, shade, axes, centre, radius, size):
    """Nearest-point-wins orthographic point rendering with 3x3 splats."""
    _, right, up, toward = axes
    scale = 0.46 * size / radius
    x = np.rint((points - centre) @ right * scale + size / 2).astype(np.int64)
    y = np.rint(size / 2 - (points - centre) @ up * scale).astype(np.int64)
    depth = (points - centre) @ toward
    image = np.full((size, size, 3), 255, np.uint8)
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    xs = np.concatenate([x + dx for dx, _ in offsets])
    ys = np.concatenate([y + dy for _, dy in offsets])
    source = np.tile(np.arange(len(points)), len(offsets))
    inside = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    xs, ys, source = xs[inside], ys[inside], source[inside]
    pixel = ys * size + xs
    # One integer key per splat: pixel first, then nearness rank; the smallest per pixel wins.
    rank = np.empty(len(points), np.int64)
    rank[np.argsort(-depth)] = np.arange(len(points))
    key = np.sort(pixel * len(points) + rank[source])
    first = np.ones(len(key), bool)
    first[1:] = key[1:] // len(points) != key[:-1] // len(points)
    winner = np.argsort(-depth)[key[first] % len(points)]
    lit = colours[winner] * shade[winner, None]
    image.reshape(-1, 3)[key[first] // len(points)] = np.clip(lit, 0, 255).astype(np.uint8)
    return image


def render_overlay(path, mesh_points, mesh_normals, mesh_distance, reference_points,
                   reference_normals, reference_distance, plane_normal, diagonal, size=440):
    """3 views x 3 rows: mesh by distance, reference by distance, both in flat colours."""
    limit = 0.02 * diagonal
    centre = 0.5 * (reference_points.min(0) + reference_points.max(0))
    radius = max(np.linalg.norm(reference_points - centre, axis=1).max(),
                 np.percentile(np.linalg.norm(mesh_points - centre, axis=1), 99.5))

    def heat(distance):
        level = np.clip(distance / limit * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(level[:, None], cv2.COLORMAP_TURBO)[:, 0].astype(np.float64)

    rows = []
    both_points = np.concatenate((mesh_points, reference_points))
    both_normals = np.concatenate((mesh_normals, reference_normals))
    flat = np.concatenate((np.tile([40.0, 140.0, 255.0], (len(mesh_points), 1)),
                           np.tile([230.0, 150.0, 40.0], (len(reference_points), 1))))
    layers = (
        ("mesh, colour = distance to reference", mesh_points, mesh_normals, heat(mesh_distance)),
        ("reference, colour = distance to mesh", reference_points, reference_normals,
         heat(reference_distance)),
        ("overlay: mesh orange, reference blue", both_points, both_normals, flat),
    )
    for label, points, normals, colours in layers:
        panels = []
        for axes in view_axes(plane_normal, reference_points):
            shade = 0.45 + 0.55 * np.abs(normals @ axes[3])
            panel = splat(points, colours, shade, axes, centre, radius, size)
            cv2.putText(panel, f"{axes[0]} | {label}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.hstack(panels))
    bar = np.full((34, 3 * size, 3), 255, np.uint8)
    ramp = cv2.applyColorMap(np.linspace(0, 255, 400).astype(np.uint8)[None], cv2.COLORMAP_TURBO)
    bar[8:22, 250:650] = ramp
    cv2.putText(bar, "distance: 0", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f">= {limit:.3g} reference units (2% of object diagonal {diagonal:.4g}); "
                "up = platform normal; views not mirrored", (660, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), np.vstack(rows + [bar])):
        raise OSError(f"could not write {path}")


def evaluate(mesh_triangles, reference_vertices, reference_faces, *, seed=215, samples=200000,
             random_starts=96, keep=0.8, platform_margin=0.006, above_margin=0.03,
             handedness="auto", polish_points=20000, overlay=None, platform=True):
    """Return the result dictionary; `overlay` is an optional PNG path. `platform=False` is for
    references without a support (complete object meshes such as YCB's Google scans): nothing is
    removed, and 'above_margin' then equals 'all'."""
    if samples < 4 or samples % 2:
        raise ValueError("samples must be an even number of at least 4")
    rng = np.random.default_rng(seed)
    if platform:
        keep_faces, normal, offset, platform = remove_platform(
            reference_vertices, reference_faces, rng, margin=platform_margin)
    else:
        keep_faces = np.ones(len(reference_faces), bool)
        normal, offset = np.array([0.0, 0.0, 1.0]), -1e12
        platform = {"method": "none: the reference is a complete object without a support", "margin_above_plane": 0.0}
    object_faces = reference_faces[keep_faces]
    reference_triangles = reference_vertices[object_faces]
    used = reference_vertices[np.unique(object_faces)]
    diagonal = float(np.linalg.norm(used.max(0) - used.min(0)))
    platform["object_boundary"] = boundary_report(object_faces)

    mesh_points, mesh_normals = sample_surface(mesh_triangles, samples, rng)
    reference_points, reference_normals = sample_surface(reference_triangles, samples, rng)
    half = samples // 2
    mesh_fit = np.zeros(samples, bool)
    mesh_fit[rng.permutation(samples)[:half]] = True
    reference_fit = np.zeros(samples, bool)
    reference_fit[rng.permutation(samples)[:half]] = True

    if handedness not in ("auto", "proper", "mirrored"):
        raise ValueError("handedness must be auto, proper or mirrored")
    mirror = np.array([-1.0, 1.0, 1.0])
    fits = {}
    for name, factor in (("proper", np.ones(3)), ("mirrored", mirror)):
        if handedness in ("auto", name):
            fits[name] = register(
                mesh_points[mesh_fit] * factor, reference_points[reference_fit], seed,
                random_starts=random_starts, keep=keep)
    chosen = min(fits, key=lambda name: fits[name][1]["best_trimmed_rms"])
    transform, alignment = fits[chosen]
    mirrored = chosen == "mirrored"
    if mirrored:
        mesh_triangles, mesh_points = mesh_triangles * mirror, mesh_points * mirror
        mesh_normals = mesh_normals * mirror
    mesh_tree = cKDTree(mesh_triangles.mean(1))
    reference_tree = cKDTree(reference_triangles.mean(1))
    polish_count = min(half, polish_points)
    before = transform
    transform, polished_residual, polish_iterations = polish(
        before, mesh_points[mesh_fit][:polish_count], reference_points[reference_fit][:polish_count],
        mesh_triangles, reference_triangles, mesh_tree, reference_tree, keep)
    scale, rotation, translation = transform
    alignment["polish"] = {
        "method": "same trimmed symmetric objective on fit-half samples, paired with closest "
                  "points on the other surface's triangles",
        "points_per_surface": polish_count, "iterations": polish_iterations,
        "trimmed_point_to_surface_rms": polished_residual,
        "rotation_change_degrees": rotation_angle(before[1], rotation),
        "scale_change_ratio": scale / before[0],
    }
    handedness_report = {
        "requested": handedness,
        "scored": chosen,
        "mesh_mirrored_before_transform": mirrored,
        "mirror_definition": "mesh x coordinate negated before the similarity transform",
        "best_trimmed_rms": {name: fit[1]["best_trimmed_rms"] for name, fit in fits.items()},
        "rejected_fit": next(({"handedness": name, "scale": fit[0][0], **{
            key: fit[1][key] for key in ("coarse_starts_at_best_optimum", "refined")}}
            for name, fit in fits.items() if name != chosen), None),
    }
    warnings = []
    rms = handedness_report["best_trimmed_rms"]
    if mirrored:
        warnings.append(
            "CHIRALITY: the mesh fits the scan only as its mirror image (trimmed rms proper "
            f"{rms.get('proper', float('nan')):.4g}, mirrored {rms.get('mirrored', float('nan')):.4g}); the scored "
            "transform includes a reflection. The reconstruction keeps the handedness of the photographed "
            "object (checked on a synthetic object of known handedness, crates/dense/README.md), so the "
            "scan is the mirror image of the object as photographed. Expected for the 3DLF Revopoint scans.")

    # All samples get a distance (the picture uses them); only the held-out half is scored.
    mesh_in_reference = apply(transform, mesh_points)
    mesh_distance = surface_distance(mesh_in_reference, reference_triangles, reference_tree)
    reference_distance = scale * surface_distance(
        apply_inverse(transform, reference_points), mesh_triangles, mesh_tree)
    mesh_height = mesh_in_reference @ normal - offset
    reference_height = reference_points @ normal - offset
    cut = platform["margin_above_plane"] + above_margin * diagonal
    mesh_normals_in_reference = mesh_normals @ rotation.T

    oriented = mesh_normals @ rotation.T  # already mirrored with the points when the fit is
    if "kept_height_range" in platform:
        underside = (mesh_height < 0.02 * platform["kept_height_range"][1]) & (oriented @ normal < -0.7)
    else:  # no platform: nothing is an underside resting on it
        underside = np.zeros(len(mesh_height), bool)

    def scored(mesh_mask, reference_mask):
        return score(mesh_distance[mesh_mask], reference_distance[reference_mask], diagonal)

    result = {
        "schema": "scan_evaluate_v1",
        "evaluation_only": True,
        "reference_used_only_posthoc": True,
        "warnings": warnings,
        "notes": [
            "The support platform was removed from the reference only; the mesh was not edited.",
            "The reconstruction is a closed solid with an invented underside; the scan is open "
            "where the object stood on the platform and loses the band below the removal margin. "
            "'all' includes those regions; 'above_margin' keeps only samples of both surfaces "
            "higher than the stated height over the platform plane.",
            "Shape-only similarity fit: absolute scale and pose are not evaluated, and the fit "
            "absorbs part of any systematic shape error.",
            "Distances are measured from held-out surface samples (not used by the fit) to the "
            "other surface's triangles.",
        ],
        "parameters": {"seed": seed, "samples_per_surface": samples, "fit_samples": half,
                       "held_out_samples": samples - half, "random_starts": random_starts,
                       "icp_keep_fraction": keep, "platform_margin": platform_margin,
                       "above_margin": above_margin, "handedness": handedness},
        "mesh": {"triangles": int(len(mesh_triangles)),
                 "area_mesh_units": float(triangle_areas(mesh_triangles).sum()),
                 "bbox_extent_mesh_units": np.ptp(mesh_triangles.reshape(-1, 3), axis=0).tolist()},
        "platform_removal": platform,
        "reference_object": {
            "bbox_min": used.min(0).tolist(), "bbox_max": used.max(0).tolist(),
            "bbox_diagonal": diagonal,
            "height_over_platform_plane": platform["kept_height_range"][1] if "kept_height_range" in platform else None,
            "diagonal_definition": "axis-aligned bounding box of the platform-free reference in "
                                   "the scanner frame",
        },
        "alignment": {
            "convention": "reference_point = scale * rotation @ (m * mesh_point) + translation, "
                          "m = (-1, 1, 1) if mesh_mirrored_before_transform else (1, 1, 1)",
            "handedness": handedness_report,
            "scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
            "rotation_determinant": float(np.linalg.det(rotation)),
            "linear_part_including_mirror": (
                scale * rotation * (mirror if mirrored else np.ones(3))).tolist(),
            "mesh_extent_in_reference_units": (scale * np.ptp(
                mesh_triangles.reshape(-1, 3), axis=0)).tolist(),
            "mesh_height_range_over_platform_plane": [float(mesh_height.min()),
                                                      float(mesh_height.max())],
            **alignment,
        },
        "metrics": {
            "all": scored(~mesh_fit, ~reference_fit),
            "above_margin": {
                "height_over_platform_plane": cut,
                "definition": "platform removal margin + above_margin * object diagonal",
                "mesh_sample_fraction_kept": float((mesh_height > cut)[~mesh_fit].mean()),
                "reference_sample_fraction_kept": float(
                    (reference_height > cut)[~reference_fit].mean()),
                **scored(~mesh_fit & (mesh_height > cut), ~reference_fit & (reference_height > cut)),
            },
            # The scan is open where the object stood; the reconstruction is closed there. This second
            # whole-surface score leaves the closed underside out of precision (mesh samples within 2 % of
            # the object's height over the platform plane whose normal points down); 'all' is unchanged.
            "all_without_underside": {
                "definition": "as 'all', without mesh samples below 2 % of the object height over the platform "
                              "plane whose normal points down (cosine with the plane normal below -0.7)",
                "mesh_sample_fraction_removed": float(underside[~mesh_fit].mean()),
                **scored(~mesh_fit & ~underside, ~reference_fit),
            },
            "fit_half_for_comparison_only": scored(mesh_fit, reference_fit),
        },
    }
    if overlay is not None:
        render_overlay(overlay, mesh_in_reference, mesh_normals_in_reference, mesh_distance,
                       reference_points, reference_normals, reference_distance, normal, diagonal)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, required=True, help="binary STL, any frame and scale")
    parser.add_argument("--reference", type=Path, required=True, help="binary PLY scan")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    parser.add_argument("--seed", type=int, default=215)
    parser.add_argument("--samples", type=int, default=200000, help="per surface; half fit, half held out")
    parser.add_argument("--random-starts", type=int, default=96)
    parser.add_argument("--icp-keep", type=float, default=0.8)
    parser.add_argument("--platform-margin", type=float, default=0.006,
                        help="removal height over the plane, fraction of the scan diagonal")
    parser.add_argument("--above-margin", type=float, default=0.03,
                        help="extra height for the restricted metrics, fraction of the object diagonal")
    parser.add_argument("--no-platform", action="store_true",
                        help="the reference has no support to remove (a complete object mesh)")
    parser.add_argument("--handedness", choices=("auto", "proper", "mirrored"), default="auto",
                        help="auto fits the mesh and its mirror image and scores the better fit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    mesh_triangles = read_binary_stl(args.mesh)
    vertices, faces = read_reference(args.reference)
    result = evaluate(
        mesh_triangles, vertices, faces, seed=args.seed, samples=args.samples,
        random_starts=args.random_starts, keep=args.icp_keep,
        platform_margin=args.platform_margin, above_margin=args.above_margin,
        handedness=args.handedness, overlay=args.output / "overlay.png", platform=not args.no_platform)
    result["inputs"] = {
        "mesh": {"path": str(args.mesh), "bytes": args.mesh.stat().st_size,
                 "sha256": sha256_file(args.mesh)},
        "reference": {"path": str(args.reference), "bytes": args.reference.stat().st_size,
                      "sha256": sha256_file(args.reference)},
    }
    result["seconds"] = time.time() - started
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for warning in result["warnings"]:
        print("WARNING:", warning)
    print("handedness residuals:", result["alignment"]["handedness"]["best_trimmed_rms"])
    for name in ("all", "above_margin"):
        block = result["metrics"][name]
        accuracy = block["accuracy_mesh_to_reference"]["fraction_of_diagonal"]
        completeness = block["completeness_reference_to_mesh"]["fraction_of_diagonal"]
        f1 = " ".join(f"F1@{key}={value['f1']:.3f}" for key, value in block["thresholds"].items())
        print(f"{name}: accuracy median {accuracy['median']:.4%} p90 {accuracy['p90']:.4%} | "
              f"completeness median {completeness['median']:.4%} p90 {completeness['p90']:.4%} | {f1}")


if __name__ == "__main__":
    main()
