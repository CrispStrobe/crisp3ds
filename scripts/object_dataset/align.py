#!/usr/bin/env python3
"""Reference-fitted Sim(3) diagnostic; never an independent accuracy calibration."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import evaluate
else:
    import evaluate

MAX_SAMPLES = 1500
MAX_SECONDS = 60
INITIAL_CANDIDATES = 24
REFINE_CANDIDATES = 4
REFINE_STEPS = 8
FINAL_STEPS = 20
TRIM_FRACTION = 0.80
SEED = 2026


def nearest(source, target, deadline):
    """Bounded exact nearest *sampled* points, no spatial-index approximation."""
    target_norm = np.einsum("ij,ij->i", target, target)
    indices = np.empty(len(source), dtype=np.int64)
    squared_min = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), 128):
        if time.monotonic() > deadline:
            raise TimeoutError("alignment exceeded 60-second deadline")
        block = source[start:start + 128]
        squared = (np.einsum("ij,ij->i", block, block)[:, None]
                   + target_norm[None, :] - 2 * block @ target.T)
        closest = np.argmin(squared, axis=1)
        indices[start:start + len(block)] = closest
        squared_min[start:start + len(block)] = np.maximum(0, squared[np.arange(len(block)), closest])
    return indices, squared_min


def umeyama(source, target):
    """Least-squares proper Sim(3), mapping paired source points into target."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= 1e-18:
        raise ValueError("source correspondence set is degenerate")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0 or np.linalg.det(rotation) <= 0:
        raise ValueError("fitted transform is not a proper Sim(3)")
    translation = target_mean - scale * rotation @ source_mean
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def pca_frame(points):
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.max() <= 1e-18:
        raise ValueError("surface samples have no 3D spread")
    return eigenvectors[:, np.argsort(eigenvalues)[::-1]]


def seed_matrices(reference, output):
    ref_center, out_center = reference.mean(axis=0), output.mean(axis=0)
    ref_radius = float(np.sqrt(np.mean(np.sum((reference - ref_center)**2, axis=1))))
    out_radius = float(np.sqrt(np.mean(np.sum((output - out_center)**2, axis=1))))
    if min(ref_radius, out_radius) <= 1e-9:
        raise ValueError("surface samples have degenerate radius")
    scale = ref_radius / out_radius
    ref_axes, out_axes = pca_frame(reference), pca_frame(output)
    matrices = []
    for permutation in itertools.permutations(range(3)):
        permute = np.eye(3)[:, permutation]
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = ref_axes @ permute @ np.diag(signs) @ out_axes.T
            if np.linalg.det(rotation) < 0:
                continue
            matrix = np.eye(4)
            matrix[:3, :3] = scale * rotation
            matrix[:3, 3] = ref_center - scale * rotation @ out_center
            matrices.append(matrix)
    if len(matrices) != INITIAL_CANDIDATES:
        raise AssertionError("expected 24 proper PCA axis/sign seeds")
    return matrices


def transformed(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def objective(reference, output, matrix, deadline):
    mapped = transformed(output, matrix)
    _, output_to_reference = nearest(mapped, reference, deadline)
    _, reference_to_output = nearest(reference, mapped, deadline)
    return float(np.sqrt((np.mean(output_to_reference) + np.mean(reference_to_output)) / 2))


def refine(reference, output, initial, steps, deadline):
    matrix = initial.copy()
    initial_value = objective(reference, output, matrix, deadline)
    history = [initial_value]
    best = (initial_value, matrix.copy())
    keep_output = max(3, int(len(output) * TRIM_FRACTION))
    keep_reference = max(3, int(len(reference) * TRIM_FRACTION))
    for _ in range(steps):
        mapped = transformed(output, matrix)
        forward_indices, forward_squared = nearest(mapped, reference, deadline)
        reverse_indices, reverse_squared = nearest(reference, mapped, deadline)
        selected_output = np.argsort(forward_squared, kind="stable")[:keep_output]
        selected_reference = np.argsort(reverse_squared, kind="stable")[:keep_reference]
        paired_output = np.concatenate((output[selected_output], output[reverse_indices[selected_reference]]))
        paired_reference = np.concatenate((reference[forward_indices[selected_output]],
                                           reference[selected_reference]))
        matrix = umeyama(paired_output, paired_reference)
        value = objective(reference, output, matrix, deadline)
        history.append(value)
        if value < best[0]:
            best = (value, matrix.copy())
    return best[1], best[0], history


def align_points(reference, output, *, deadline=None):
    reference = np.asarray(reference, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    if (reference.ndim != 2 or output.ndim != 2 or reference.shape[1] != 3
            or output.shape[1] != 3 or not np.isfinite(reference).all()
            or not np.isfinite(output).all() or min(len(reference), len(output)) < 10
            or max(len(reference), len(output)) > MAX_SAMPLES):
        raise ValueError("need 10–1500 finite 3D samples per mesh")
    deadline = deadline or time.monotonic() + MAX_SECONDS
    ranked = sorted(((objective(reference, output, matrix, deadline), rank, matrix)
                     for rank, matrix in enumerate(seed_matrices(reference, output))),
                    key=lambda item: (item[0], item[1]))
    finalists = []
    for initial_value, rank, matrix in ranked[:REFINE_CANDIDATES]:
        fitted, value, history = refine(reference, output, matrix, REFINE_STEPS, deadline)
        finalists.append((value, rank, initial_value, fitted, history))
    value, rank, initial_value, matrix, history = min(finalists, key=lambda item: (item[0], item[1]))
    fitted, final_value, later = refine(reference, output, matrix, FINAL_STEPS, deadline)
    if final_value < value:
        matrix, value = fitted, final_value
    all_history = history + later
    return matrix, {"initial_candidate_count": INITIAL_CANDIDATES,
                    "refined_candidate_count": REFINE_CANDIDATES,
                    "refine_steps_per_candidate": REFINE_STEPS,
                    "final_steps": FINAL_STEPS,
                    "deadline_seconds": MAX_SECONDS,
                    "objective": "symmetric RMS nearest sampled-point distance in reference units",
                    "trim_fraction": TRIM_FRACTION,
                    "selected_initial_rank": rank,
                    "selected_initial_symmetric_rms_nearest_sampled_point": initial_value,
                    "final_symmetric_rms_nearest_sampled_point": value,
                    "selected_history": all_history,
                    "converged_by_final_relative_change": (len(later) >= 2 and
                        abs(later[-1] - later[-2]) <= 1e-4 * max(1, later[-2]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--reference-units", required=True)
    parser.add_argument("--save-transform", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    if not 10 <= args.samples <= MAX_SAMPLES:
        parser.error("--samples must be between 10 and 1500")
    if not args.object_id.strip() or not args.reference_units.strip():
        parser.error("object ID and reference units must be nonempty")
    deadline = time.monotonic() + MAX_SECONDS
    reference_report, rv, rf = evaluate.inspect_ply(args.reference, geometry=True)
    output_report, ov, of = evaluate.inspect_ply(args.output, geometry=True)
    reference = evaluate.sample_surface(rv, rf, args.samples, SEED)
    output = evaluate.sample_surface(ov, of, args.samples, SEED)
    matrix, diagnostics = align_points(reference, output, deadline=deadline)
    scale = float(np.linalg.svd(matrix[:3, :3], compute_uv=False).mean())
    record = {"matrix": matrix.tolist(), "object_id": args.object_id,
              "source_frame": "reconstruction PLY coordinates",
              "target_frame": "independent reference PLY coordinates",
              "reference_units": args.reference_units,
              "provenance": "reference-fitted diagnostic: 24 proper PCA seeds, top 4 bidirectional trimmed nearest-sample Sim(3) refinements; fixed algorithm",
              "scale_provenance": "scale fitted to the same reference used for comparison; not independently recovered metric scale",
              "registration_basis": "reference-fit",
              "reference_sha256": evaluate.sha256_file(args.reference),
              "output_sha256": evaluate.sha256_file(args.output),
              "reference_geometry": {k: reference_report[k] for k in ("vertices", "faces", "zero_area_triangle_faces")},
              "output_geometry": {k: output_report[k] for k in ("vertices", "faces", "zero_area_triangle_faces")},
              "fitted_scale": scale,
              "sample_count_per_mesh": args.samples,
              "sample_seed": SEED,
              "alignment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "alignment_diagnostics": diagnostics,
              "interpretation": "reference-fitted descriptive comparison only; not independent scale or metric accuracy"}
    args.save_transform.parent.mkdir(parents=True, exist_ok=True)
    with args.save_transform.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"transform": str(args.save_transform), "fitted_scale": scale,
                      "initial_symmetric_rms": diagnostics["selected_initial_symmetric_rms_nearest_sampled_point"],
                      "final_symmetric_rms": diagnostics["final_symmetric_rms_nearest_sampled_point"]}, indent=2))


if __name__ == "__main__":
    main()
