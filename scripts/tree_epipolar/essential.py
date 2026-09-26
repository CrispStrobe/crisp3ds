#!/usr/bin/env python3
"""One train-only calibrated essential pose check for the fixed tree pair 1/4."""

import argparse
import json
import math
import os
from pathlib import Path
import signal

import cv2
import numpy as np

from run import binary_path, digest, metrics, normalize_f, sampson, skew


def rotation_angle_deg(a, b):
    cosine = (np.trace(a @ b.T) - 1.) / 2.
    return math.degrees(math.acos(float(np.clip(cosine, -1., 1.))))


def direction_angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1., 1.))))


def ray_angles_deg(normalized_a, normalized_b, rotation):
    angles = []
    for pa, pb in zip(normalized_a, normalized_b):
        ray_a = np.array([pa[0], pa[1], 1.])
        ray_b = rotation.T @ np.array([pb[0], pb[1], 1.])
        angles.append(direction_angle_deg(ray_a, ray_b))
    return angles


def raw_triangulation(normalized_a, normalized_b, rotation, translation):
    """Bookkeeping only: all E-RANSAC training inliers, no pose/model refit."""
    projection_a = np.column_stack((np.eye(3), np.zeros(3)))
    projection_b = np.column_stack((rotation, translation.reshape(3)))
    homogeneous = cv2.triangulatePoints(projection_a, projection_b,
                                        normalized_a.T, normalized_b.T)
    valid = np.isfinite(homogeneous).all(axis=0) & (np.abs(homogeneous[3]) > 1e-12)
    points = homogeneous[:3, valid] / homogeneous[3, valid]
    in_b = (rotation @ points + translation.reshape(3, 1))
    depth_a, depth_b = points[2], in_b[2]
    positive = (depth_a > 0) & (depth_b > 0)
    rays = sorted(ray_angles_deg(normalized_a[valid][positive],
                                 normalized_b[valid][positive], rotation))
    return dict(total=int(len(normalized_a)), finite=int(valid.sum()),
                invalid=int(len(valid)-valid.sum()),
                positive_both=int(positive.sum()),
                nonpositive_either=int((~positive).sum()),
                positive_either_camera_depth_gt_50=int(np.sum(positive & ((depth_a > 50) | (depth_b > 50)))),
                positive_either_camera_distance_gt_50=int(np.sum(positive &
                    ((np.linalg.norm(points, axis=0) > 50) | (np.linalg.norm(in_b, axis=0) > 50)))),
                positive_ray_angle_min_deg=rays[0] if rays else None,
                positive_ray_angle_median_deg=float(np.median(rays)) if rays else None)


def assess(views, rows):
    a, b = (next(v for v in views if v["id"] == index) for index in (1, 4))
    ka, kb = np.asarray(a["K"]), np.asarray(b["K"])
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "holdout"]
    if len(train) < 16 or len(held) < 8:
        raise ValueError("frozen split fails sample gate")
    if [r["order"] for r in rows] != list(range(len(rows))) or any(
            r["split"] != ("holdout" if r["order"] % 5 == 4 else "train") for r in rows):
        raise ValueError("frozen descriptor-order partition is inconsistent")
    pa = np.asarray([r["a_xy"] for r in train], dtype=np.float64)
    pb = np.asarray([r["b_xy"] for r in train], dtype=np.float64)
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), ka, None).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), kb, None).reshape(-1, 2)
    mean_fx = float((ka[0, 0] + kb[0, 0]) / 2.)
    threshold = 1. / mean_fx
    cv2.setRNGSeed(3414)
    e, ransac_mask = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, .999, threshold, 10000)
    if e is None or e.shape != (3, 3) or not np.isfinite(e).all():
        raise RuntimeError("essential fit unavailable or returned multiple candidates")
    inliers = ransac_mask.ravel() != 0
    if inliers.sum() < 5:
        raise RuntimeError("fewer than five essential RANSAC training inliers")
    positive, r, t, positive_mask, _triangulated = cv2.recoverPose(
        e, na[inliers], nb[inliers], np.eye(3), distanceThresh=50.)
    positive_flags = positive_mask.ravel() != 0
    if int(positive) != int(positive_flags.sum()):
        raise RuntimeError("recoverPose positive-depth count mismatch")
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise RuntimeError("nonfinite recovered pose")
    calibrated_f = normalize_f(np.linalg.inv(kb).T @ skew(t.ravel()) @ r @ np.linalg.inv(ka))
    if calibrated_f is None:
        raise RuntimeError("degenerate calibrated F")
    supplied_r = np.asarray(b["R"]) @ np.asarray(a["R"]).T
    supplied_t = np.asarray(b["t"]) - supplied_r @ np.asarray(a["t"])
    rays = sorted(ray_angles_deg(na[inliers][positive_flags], nb[inliers][positive_flags], r))
    raw = raw_triangulation(na[inliers], nb[inliers], r, t)
    held_metrics = metrics([sampson(calibrated_f, row["a_xy"], row["b_xy"]) for row in held])
    return dict(pair=[1, 4], training_count=len(train), heldout_count=len(held),
                essential_ransac_training_inliers=int(inliers.sum()),
                recover_pose_positive_depth_training_count=int(positive),
                recover_pose_positive_depth_fraction_of_inliers=float(positive/inliers.sum()),
                normalized_threshold=threshold, mean_focal_x_px=mean_fx,
                essential_E=e.tolist(), recovered_R=r.tolist(), recovered_unit_t=t.ravel().tolist(),
                calibrated_F=calibrated_f.tolist(), calibrated_heldout=held_metrics,
                supplied_relative_R=supplied_r.tolist(), supplied_relative_t=supplied_t.tolist(),
                rotation_difference_deg=rotation_angle_deg(r, supplied_r),
                translation_direction_difference_deg=direction_angle_deg(t, supplied_t),
                training_positive_depth_ray_angle_median_deg=float(np.median(rays)) if rays else None,
                training_positive_depth_ray_angle_p90_deg=rays[math.ceil(.9*len(rays))-1] if rays else None,
                raw_triangulation_training_inliers=raw,
                essential_ransac_train_indices=[int(i) for i in np.flatnonzero(inliers)],
                recover_pose_positive_train_indices=[int(np.flatnonzero(inliers)[i]) for i in np.flatnonzero(positive_flags)])


def main():
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("60 second run limit")))
    signal.alarm(60)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path, output = args.report.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error("output exists; use a fresh report path")
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    stat = os.statvfs(output.parent)
    if stat.f_bavail * stat.f_frsize < 10*1024**3 + 1024**2:
        parser.error("less than 10 GiB plus 1 MiB free")
    source = json.loads(report_path.read_text())
    if source["status"] != "completed":
        raise ValueError("source report is not completed")
    pair = next(p for p in source["pairs"] if (p["a"], p["b"]) == (1, 4))
    if pair["status"] != "available":
        raise ValueError("pair 1/4 was unavailable")
    match_path = report_path.parent / pair["matches_file"]
    rows = json.loads(match_path.read_text())["matches"]
    inputs = [report_path, match_path, Path(__file__).resolve(), Path(__file__).with_name("run.py").resolve(), binary_path()]
    before = {str(path):digest(path) for path in inputs}
    if source["feature_counts"] != {str(i):3000 for i in range(10)}:
        raise ValueError("source is not the exact 3000-feature capped run")
    result = assess(source["views"], rows)
    after = {str(path):digest(path) for path in inputs}
    if before != after:
        raise RuntimeError("input, source, or binary changed during essential run")
    result.update(status="completed", source_report=str(report_path), opencv_version=cv2.__version__,
                  settings=dict(pair=[1,4], train_only=True, ransac_seed=3414, confidence=.999,
                                max_iterations=10000, threshold="1 / mean(fx_a, fx_b) normalized coordinates",
                                recover_pose_input="essential RANSAC training inliers only",
                                recover_pose_distance_threshold_unit_baselines=50.),
                  input_sha256_before=before, input_sha256_after=after)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 1024*1024:
        raise RuntimeError("essential report exceeds 1 MiB")
    output.write_text(payload)
    print(json.dumps({key:result[key] for key in ("training_count", "essential_ransac_training_inliers",
                                                  "recover_pose_positive_depth_training_count",
                                                  "rotation_difference_deg", "translation_direction_difference_deg",
                                                  "training_positive_depth_ray_angle_median_deg", "calibrated_heldout")}, indent=2))


if __name__ == "__main__":
    main()
