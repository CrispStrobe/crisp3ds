#!/usr/bin/env python3
"""Predeclared known-pose stress audit of the unchanged surface aligner."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

if __package__:
    from . import align, evaluate
else:
    import align
    import evaluate

REPO = Path(__file__).resolve().parents[2]
FULL_BUNNY = REPO / ".local-tools/test-data/3dlf-scan-bunny/revopoint/bunny/fuse_mesh_rgb.ply"
ROI_BUNNY = REPO / "build-opencv/bunny-scanner-roi-v1/reference-ygtminus24.ply"
FULL_SHA = "28ed4462b6ee84edcc87157b49aee9216ef170ddd1e176928c72c74233ef9520"
ROI_SHA = "113c1a230a88559760fb7dbedf3e9e09fda9b099ebb4bf11953d5c860d33accc"
REF_SEED = 100
OUT_SEED = 200
HOLDOUT_SEED = 400
SYNTHETIC_COUNT = 384


def known_sim3():
    ax, ay, az = 0.31, -0.47, 0.69
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = 1.7 * (rz @ ry @ rx)
    matrix[:3, 3] = [2.3, -1.1, 3.7]
    return matrix


def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0],
                         [x0, y1, z0], [x0, y0, z1], [x1, y0, z1],
                         [x1, y1, z1], [x0, y1, z1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    return vertices, faces


def concatenate_meshes(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def synthetic_meshes():
    body_v = np.array([[-1.4, -0.5, -0.8], [2.2, -0.3, -0.1],
                       [-0.4, 1.6, 0.0], [0.2, 0.1, 1.3]], dtype=float)
    body_f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    body = body_v, body_f
    base = box([-2.3, -0.75, -1.8], [2.9, -0.55, 2.1])
    marked = concatenate_meshes(box([-1.05, -0.55, -1.0], [1.05, 0.55, 1.0]),
                                box([0.8, 0.55, 0.61], [1.0, 0.82, 0.83]))
    return {"asymmetric_full": (body, body),
            "reference_base_missing_output": (concatenate_meshes(body, base), body),
            "output_outliers_20pct": (body, body),
            "near_square_with_marker": (marked, marked)}


def apply(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def synthesize_output(reference_frame_points, truth_output_to_reference):
    """Place reference-frame points into the known output frame."""
    return apply(reference_frame_points, np.linalg.inv(truth_output_to_reference))


def pose_errors(fitted, truth, heldout_native, reference_diagonal):
    fitted_scale = float(np.linalg.det(fitted[:3, :3]) ** (1 / 3))
    truth_scale = float(np.linalg.det(truth[:3, :3]) ** (1 / 3))
    fitted_r = fitted[:3, :3] / fitted_scale
    truth_r = truth[:3, :3] / truth_scale
    cosine = float(np.clip((np.trace(fitted_r @ truth_r.T) - 1) / 2, -1, 1))
    distance = np.linalg.norm(apply(heldout_native, fitted) - apply(heldout_native, truth), axis=1)
    return {"rotation_error_degrees": float(np.degrees(np.arccos(cosine))),
            "scale_relative_error": abs(fitted_scale / truth_scale - 1),
            "translation_error_normalized_by_reference_diagonal":
                float(np.linalg.norm(fitted[:3, 3] - truth[:3, 3]) / reference_diagonal),
            "heldout_corresponding_rms": float(np.sqrt(np.mean(distance ** 2))),
            "heldout_corresponding_rms_normalized_by_reference_diagonal":
                float(np.sqrt(np.mean(distance ** 2)) / reference_diagonal)}


def case(name, reference_mesh, output_mesh, count, truth, *, outliers=False):
    reference = evaluate.sample_surface(*reference_mesh, count, REF_SEED)
    output_reference_frame = evaluate.sample_surface(*output_mesh, count, OUT_SEED)
    heldout_reference_frame = evaluate.sample_surface(*output_mesh, count, HOLDOUT_SEED)
    output_native = synthesize_output(output_reference_frame, truth)
    heldout_native = synthesize_output(heldout_reference_frame, truth)
    if not np.allclose(apply(output_native, truth), output_reference_frame, atol=1e-12):
        raise AssertionError("synthetic truth round-trip failed before alignment")
    if outliers:
        rng = np.random.default_rng(300)
        extra = rng.uniform([4, 4, 4], [7, 7, 7], size=(round(0.2 * count), 3))
        output_native = np.concatenate((output_native, extra))
    reference_vertices = np.asarray(reference_mesh[0])
    diagonal = float(np.linalg.norm(reference_vertices.max(axis=0) - reference_vertices.min(axis=0)))
    start = time.monotonic()
    fitted, diagnostic = align.align_points(reference, output_native,
                                            deadline=start + align.MAX_SECONDS)
    truth_objective = align.objective(reference, output_native, truth,
                                      time.monotonic() + align.MAX_SECONDS)
    fitted_objective = diagnostic["final_symmetric_rms_nearest_sampled_point"]
    return {"name": name, "reference_samples": count,
            "output_samples": len(output_native), "elapsed_seconds": time.monotonic() - start,
            "reference_bbox_diagonal": diagonal,
            "fitted_matrix": fitted.tolist(), "errors": pose_errors(fitted, truth, heldout_native, diagonal),
            "objective_at_known_truth": truth_objective,
            "fitted_objective_over_truth_objective": fitted_objective / truth_objective,
            "alignment_diagnostics": diagnostic,
            "ambiguity_note": ("near-square footprint: discrete rotations may have similar nearest-point objective; corresponding-point errors still expose pose disagreement"
                               if name == "near_square_with_marker" else None)}


def paired_controls(truth, body):
    landmarks = body[0]
    native = apply(landmarks, np.linalg.inv(truth))
    fitted = align.umeyama(native, landmarks)
    diagonal = float(np.linalg.norm(np.ptp(landmarks, axis=0)))
    reflected = native.copy()
    reflected[:, 0] *= -1
    mirrored_fit = align.umeyama(reflected, landmarks)
    mirrored_rms = float(np.sqrt(np.mean(np.sum((apply(reflected, mirrored_fit) - landmarks) ** 2, axis=1))))
    line_native = np.array([[x, 0.4, -0.2] for x in (-2., -1., 0., 1., 2.)])
    line_reference = apply(line_native, truth)
    line_fit = align.umeyama(line_native, line_reference)
    line_singular = np.linalg.svd(line_native - line_native.mean(axis=0), compute_uv=False)
    line_rms = float(np.sqrt(np.mean(np.sum((apply(line_native, line_fit) - line_reference) ** 2, axis=1))))
    return {"exact_landmark_umeyama": pose_errors(fitted, truth, native, diagonal),
            "exact_landmark_source_singular_values":
                np.linalg.svd(native - native.mean(axis=0), compute_uv=False).tolist(),
            "mirrored_landmark_proper_fit_residual": mirrored_rms,
            "mirrored_landmark_fitted_rotation_determinant":
                float(np.linalg.det(mirrored_fit[:3, :3]) / (np.linalg.det(mirrored_fit[:3, :3]) ** (1 / 3)) ** 3),
            "collinear_rank1_ambiguity": {
                "source_singular_values": line_singular.tolist(),
                "rank": int(np.linalg.matrix_rank(line_native - line_native.mean(axis=0))),
                "point_fit_rms": line_rms,
                "heldout_body_pose_error": pose_errors(line_fit, truth, body[0], diagonal),
                "interpretation": "zero paired residual on a line does not constrain rotation about that line"}}


def identical_unknown_correspondence(truth, mesh):
    reference_native = evaluate.sample_surface(*mesh, SYNTHETIC_COUNT, REF_SEED)
    reference = apply(reference_native, truth)
    output = reference_native.copy()
    start = time.monotonic()
    fitted, diagnostic = align.align_points(reference, output,
                                            deadline=start + align.MAX_SECONDS)
    diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
    return {"elapsed_seconds": time.monotonic() - start,
            "errors": pose_errors(fitted, truth, output, diagonal),
            "alignment_diagnostics": diagnostic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-report", type=Path, required=True)
    args = parser.parse_args()
    if args.save_report.exists() or args.save_report.is_symlink():
        parser.error("report path must be fresh")
    writing = args.save_report.with_name(args.save_report.name + ".writing")
    if writing.exists() or writing.is_symlink():
        parser.error("report writing path must be fresh")
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.save_report.parent).free < 10 * 1024 ** 3:
        parser.error("less than 10 GiB free")
    if evaluate.sha256_file(FULL_BUNNY) != FULL_SHA or evaluate.sha256_file(ROI_BUNNY) != ROI_SHA:
        parser.error("saved bunny scanner or ROI hash changed")
    truth = known_sim3()
    meshes = synthetic_meshes()
    report = {"schema": "registration_audit_v2", "status": "running",
              "truth_output_to_reference": truth.tolist(),
              "reference_seed": REF_SEED, "output_seed": OUT_SEED,
              "heldout_seed": HOLDOUT_SEED, "synthetic_count_per_side": SYNTHETIC_COUNT,
              "algorithm": "unchanged scripts/object_dataset/align.py",
              "align_script_sha256": hashlib.sha256(Path(align.__file__).read_bytes()).hexdigest(),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "paired_controls": paired_controls(truth, meshes["asymmetric_full"][0]),
              "identical_unknown_correspondence_control":
                  identical_unknown_correspondence(truth, meshes["asymmetric_full"][0]),
              "cases": []}
    def checkpoint():
        with writing.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        os.replace(writing, args.save_report)

    checkpoint()
    for name, (reference_mesh, output_mesh) in meshes.items():
        try:
            result = case(name, reference_mesh, output_mesh, SYNTHETIC_COUNT, truth,
                          outliers=name == "output_outliers_20pct")
            result["status"] = "complete"
        except Exception as exc:
            result = {"name": name, "status": "failed", "error_type": type(exc).__name__,
                      "error": str(exc)}
        report["cases"].append(result)
        checkpoint()
        print(f"{name}: {result['status']}" +
              (f", heldout RMS {result['errors']['heldout_corresponding_rms']:.6g}, "
               f"rotation {result['errors']['rotation_error_degrees']:.3f} deg"
               if result["status"] == "complete" else f", {result['error_type']}: {result['error']}"),
              flush=True)
    _, fv, ff = evaluate.inspect_ply(FULL_BUNNY, geometry=True)
    _, rv, rf = evaluate.inspect_ply(ROI_BUNNY, geometry=True)
    full, roi = (np.asarray(fv), np.asarray(ff)), (np.asarray(rv), np.asarray(rf))
    report["real_control_sources"] = {"full_bunny_sha256": FULL_SHA, "roi_bunny_sha256": ROI_SHA,
                                      "scope": "self-transformed scanner samples only; no reconstruction candidates"}
    for count in (512, 1024):
        for name, reference_mesh, output_mesh in (("bunny_full_self", full, full),
                                                  ("bunny_full_reference_roi_output", full, roi)):
            try:
                result = case(name, reference_mesh, output_mesh, count, truth)
                result["status"] = "complete"
            except Exception as exc:
                result = {"name": name, "reference_samples": count, "status": "failed",
                          "error_type": type(exc).__name__, "error": str(exc)}
            report["cases"].append(result)
            checkpoint()
            print(f"{name}/{count}: {result['status']}" +
                  (f", heldout RMS {result['errors']['heldout_corresponding_rms']:.6g}, "
                   f"rotation {result['errors']['rotation_error_degrees']:.3f} deg"
                   if result["status"] == "complete" else f", {result['error_type']}: {result['error']}"),
                  flush=True)
    report["status"] = ("complete" if all(c["status"] == "complete" for c in report["cases"])
                        else "complete_with_case_failures")
    checkpoint()
    print(args.save_report)


if __name__ == "__main__":
    main()
