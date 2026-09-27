#!/usr/bin/env python3
"""Score same-camera YCB meshes under the frozen 006 reference-fit gauge."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.object_dataset import evaluate, preview, surface_metrics

SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sparse_hashes(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("sparse model must be a real directory")
    hashes = {}
    for name in SPARSE_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or linked sparse model file: {name}")
        hashes[name] = digest(path)
    return hashes


def pca_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("need finite XYZ samples")
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def axis_angle_degrees(a, b):
    cosine = float(np.clip(abs(np.dot(a, b)), 0, 1))
    return math.degrees(math.acos(cosine))


def relative_rotation_degrees(parent, candidate):
    parent = np.asarray(parent, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    sp = np.linalg.svd(parent[:3, :3], compute_uv=False).mean()
    sc = np.linalg.svd(candidate[:3, :3], compute_uv=False).mean()
    rotation = (candidate[:3, :3] / sc) @ (parent[:3, :3] / sp).T
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--parent-output", required=True, type=Path)
    parser.add_argument("--parent-transform", required=True, type=Path)
    parser.add_argument("--parent-sparse", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--candidate-primary-transform", required=True, type=Path)
    parser.add_argument("--candidate-sparse", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float, action="append")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save-report", required=True, type=Path)
    parser.add_argument("--save-preview", type=Path)
    args = parser.parse_args()
    parent_sparse = sparse_hashes(args.parent_sparse)
    candidate_sparse = sparse_hashes(args.candidate_sparse)
    if parent_sparse != candidate_sparse:
        raise ValueError("candidate does not share the parent's exact sparse camera/world model")
    parent_provenance, parent_matrix, parent_scale = evaluate.load_sim3(
        args.parent_transform, args.reference, args.parent_output)
    candidate_provenance, candidate_matrix, candidate_scale = evaluate.load_sim3(
        args.candidate_primary_transform, args.reference, args.candidate_output)
    if parent_provenance["registration_basis"] != "reference-fit" or candidate_provenance["registration_basis"] != "reference-fit":
        raise ValueError("expected separately reference-fitted parent and candidate transforms")
    ref = evaluate.inspect_ply(args.reference, geometry=True)
    parent = evaluate.inspect_ply(args.parent_output, geometry=True)
    candidate = evaluate.inspect_ply(args.candidate_output, geometry=True)
    if any(mesh[0]["nontriangle_faces"] for mesh in (ref, parent, candidate)):
        raise ValueError("shared-gauge scoring requires triangle meshes")
    comparison = surface_metrics.compare(ref[1:], candidate[1:], parent_matrix,
                                         thresholds=args.threshold, count=args.samples, seed=args.seed)
    parent_samples = evaluate.sample_surface(*parent[1:], args.samples, args.seed)
    candidate_samples = evaluate.sample_surface(*candidate[1:], args.samples, args.seed)
    parent_eigenvalues, parent_axes = pca_points(parent_samples)
    candidate_eigenvalues, candidate_axes = pca_points(candidate_samples)
    report = {"schema": "same_sparse_model_shared_gauge_v1",
              "interpretation": "006 reference-fitted matrix reused unchanged in an exact shared COLMAP world gauge; shape sensitivity only, not independent metric accuracy",
              "reference_sha256": evaluate.sha256_file(args.reference),
              "parent_output_sha256": evaluate.sha256_file(args.parent_output),
              "candidate_output_sha256": evaluate.sha256_file(args.candidate_output),
              "parent_transform_sha256": digest(args.parent_transform),
              "candidate_primary_transform_sha256": digest(args.candidate_primary_transform),
              "parent_transform_registration_basis": parent_provenance["registration_basis"],
              "candidate_primary_registration_basis": candidate_provenance["registration_basis"],
              "parent_sparse_file_sha256": parent_sparse,
              "candidate_sparse_file_sha256": candidate_sparse,
              "matrix_used_output_to_reference": parent_matrix.tolist(),
              "parent_fit_scale": parent_scale,
              "candidate_independent_fit_scale": candidate_scale,
              "candidate_independent_fit_rotation_difference_degrees": relative_rotation_degrees(parent_matrix, candidate_matrix),
              "pca_native_frame": {"parent_eigenvalues": parent_eigenvalues.tolist(),
                                   "candidate_eigenvalues": candidate_eigenvalues.tolist(),
                                   "principal_axis_angle_degrees": axis_angle_degrees(parent_axes[:, 0], candidate_axes[:, 0]),
                                   "interpretation": "absolute-axis angle from independent area samples; unstable if leading eigenvalues nearly equal"},
              "comparison": comparison, "metric_accuracy_claim_allowed": False}
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    with args.save_report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if args.save_preview:
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)
        image = preview.make_preview(ref[1:], candidate[1:], parent_matrix, count=10000)
        with args.save_preview.open("xb") as stream:
            image.save(stream, format="PNG")
    print(json.dumps({"threshold_scores": comparison["threshold_scores"],
                      "pca_native_frame": report["pca_native_frame"],
                      "candidate_independent_fit_rotation_difference_degrees": report["candidate_independent_fit_rotation_difference_degrees"]}, indent=2))


if __name__ == "__main__":
    main()
