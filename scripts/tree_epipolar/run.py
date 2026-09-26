#!/usr/bin/env python3
"""Fixed, full-frame tree epipolar diagnostic. No camera filtering of matches."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys

import cv2
import numpy as np


MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def binary_path():
    package = Path(cv2.__file__).resolve().parent
    binaries = sorted(package.glob("cv2*.so")) + sorted(package.glob("cv2*.pyd"))
    if not binaries:
        raise RuntimeError("OpenCV extension binary not found")
    return binaries[0]


def read_views(path):
    views = []
    for line in path.read_text().splitlines():
        words = line.split()
        if len(words) != 20:
            raise ValueError("views.tsv requires 20 fields per row")
        view_id, width, height = map(int, words[:3])
        image = Path(words[3]).resolve(strict=True)
        values = list(map(float, words[4:]))
        fx, fy, cx, cy = values[:4]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        r = np.array(values[4:13], dtype=np.float64).reshape(3, 3)
        t = np.array(values[13:16], dtype=np.float64)
        if ((width, height) != (768, 512) or fx <= 0 or fy <= 0 or
                not np.isfinite(k).all() or not np.isfinite(r).all() or not np.isfinite(t).all() or
                not np.allclose(r @ r.T, np.eye(3), atol=1e-3) or
                not math.isclose(np.linalg.det(r), 1., abs_tol=1e-3)):
            raise ValueError("unexpected dimensions or nonfinite camera")
        views.append(dict(id=view_id, width=width, height=height, image=str(image),
                          K=k.tolist(), R=r.tolist(), t=t.tolist()))
    if len(views) != 10 or [v["id"] for v in views] != list(range(10)):
        raise ValueError("expected exactly original views 0 through 9")
    return views


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def normalize_f(f):
    f = np.asarray(f, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all():
        return None
    norm = np.linalg.norm(f)
    if not math.isfinite(norm) or norm <= 1e-15:
        return None
    return f / norm


def supplied_f(a, b):
    ka, kb = np.asarray(a["K"]), np.asarray(b["K"])
    ra, rb = np.asarray(a["R"]), np.asarray(b["R"])
    ta, tb = np.asarray(a["t"]), np.asarray(b["t"])
    relative_r = rb @ ra.T
    relative_t = tb - relative_r @ ta
    return normalize_f(np.linalg.inv(kb).T @ skew(relative_t) @ relative_r @ np.linalg.inv(ka))


def sampson(f, pa, pb):
    if f is None:
        return None
    xa = np.array([pa[0], pa[1], 1.0])
    xb = np.array([pb[0], pb[1], 1.0])
    fxa, ftxb = f @ xa, f.T @ xb
    denominator = fxa[0] ** 2 + fxa[1] ** 2 + ftxb[0] ** 2 + ftxb[1] ** 2
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None
    value = abs(xb @ fxa) / math.sqrt(denominator)
    return float(value) if math.isfinite(value) else None


def symmetric_transfer(h, pa, pb):
    if h is None:
        return None
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None
    xa = np.array([pa[0], pa[1], 1.0])
    xb = np.array([pb[0], pb[1], 1.0])
    forward, backward = h @ xa, inverse @ xb
    if abs(forward[2]) <= 1e-12 or abs(backward[2]) <= 1e-12:
        return None
    forward, backward = forward[:2] / forward[2], backward[:2] / backward[2]
    value = math.sqrt((np.sum((forward - xb[:2]) ** 2) + np.sum((backward - xa[:2]) ** 2)) / 2)
    return float(value) if math.isfinite(value) else None


def metrics(values):
    finite = sorted(x for x in values if x is not None and math.isfinite(x))
    total = len(values)
    return dict(total=total, finite=len(finite), invalid=total-len(finite),
                median_px=statistics.median(finite) if finite else None,
                p90_px=finite[math.ceil(.9 * len(finite))-1] if finite else None,
                **{f"within_{limit}px_count":sum(x <= limit for x in finite) for limit in (1, 2, 4)},
                **{f"within_{limit}px_fraction":sum(x <= limit for x in finite)/total if total else None
                   for limit in (1, 2, 4)})


def mutual_matches(descriptors_a, descriptors_b):
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    accepted_reverse = {(row[0].trainIdx, row[0].queryIdx) for row in reverse
                        if len(row) == 2 and row[0].distance < .8 * row[1].distance}
    accepted = [row[0] for row in forward if len(row) == 2 and
                row[0].distance < .8 * row[1].distance and
                (row[0].queryIdx, row[0].trainIdx) in accepted_reverse]
    return sorted(accepted, key=lambda m: (m.queryIdx, m.trainIdx))


def pair_result(a, b, features):
    keypoints_a, descriptors_a = features[a["id"]]
    keypoints_b, descriptors_b = features[b["id"]]
    matches = mutual_matches(descriptors_a, descriptors_b)
    rows = [dict(order=index, query_idx=m.queryIdx, train_idx=m.trainIdx,
                 a_xy=[float(x) for x in keypoints_a[m.queryIdx].pt],
                 b_xy=[float(x) for x in keypoints_b[m.trainIdx].pt],
                 split="holdout" if index % 5 == 4 else "train")
            for index, m in enumerate(matches)]
    train = [row for row in rows if row["split"] == "train"]
    held = [row for row in rows if row["split"] == "holdout"]
    outcome = dict(a=a["id"], b=b["id"], match_count=len(rows), train_count=len(train),
                   heldout_count=len(held), supplied_F=None, fitted_F=None,
                   homography_H=None, fitted_train_inliers=None, homography_train_inliers=None,
                   supplied_heldout=None, fitted_heldout=None, homography_heldout=None)
    sf = supplied_f(a, b)
    outcome["supplied_F"] = sf.tolist() if sf is not None else None
    if sf is not None and held:
        outcome["supplied_heldout"] = metrics([sampson(sf, r["a_xy"], r["b_xy"]) for r in held])
    if len(train) < 16 or len(held) < 8:
        outcome.update(status="unavailable", reason="insufficient_train_or_heldout")
        return outcome, rows
    if sf is None:
        outcome.update(status="unavailable", reason="degenerate_supplied_F")
        return outcome, rows
    pa = np.asarray([r["a_xy"] for r in train], dtype=np.float32)
    pb = np.asarray([r["b_xy"] for r in train], dtype=np.float32)
    cv2.setRNGSeed(1337 + 10*a["id"] + b["id"])
    fitted, inlier_mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, .999, 10000)
    ff = normalize_f(fitted)
    if ff is not None:
        outcome["fitted_F"] = ff.tolist()
        outcome["fitted_train_inliers"] = int(np.count_nonzero(inlier_mask))
    cv2.setRNGSeed(2337 + 10*a["id"] + b["id"])
    h, hmask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0, None, 10000, .999)
    if h is not None and h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-15:
        outcome["homography_H"] = h.tolist()
        outcome["homography_train_inliers"] = int(np.count_nonzero(hmask))
    if ff is None:
        outcome.update(status="unavailable", reason="fitted_F_failed")
        return outcome, rows
    outcome.update(status="available", reason=None)
    outcome["fitted_heldout"] = metrics([sampson(ff, r["a_xy"], r["b_xy"]) for r in held])
    if outcome["homography_H"] is not None:
        outcome["homography_heldout"] = metrics([symmetric_transfer(h, r["a_xy"], r["b_xy"]) for r in held])
    return outcome, rows


def main():
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("600 second run limit")))
    signal.alarm(600)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene, output = args.scene.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error("output exists; choose a fresh path")
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    if os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize < 10*1024**3:
        parser.error("less than 10 GiB free")
    views_path = scene / "views.tsv"
    views = read_views(views_path)
    code_path = Path(__file__).resolve(strict=True)
    input_paths = [views_path, code_path, binary_path()] + [Path(v["image"]) for v in views]
    before = {str(path): digest(path) for path in input_paths}
    sift = cv2.SIFT_create(nfeatures=3000)
    features = {}
    for view in views:
        image = cv2.imread(view["image"], cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (512, 768):
            raise ValueError(f"failed image or unexpected shape: {view['image']}")
        keypoints, descriptors = sift.detectAndCompute(image, None)
        # SIFT may return one extra keypoint on a tied response cutoff.
        features[view["id"]] = (keypoints[:3000], descriptors[:3000] if descriptors is not None else None)
    report = dict(status="running", scene=str(scene), views=views, opencv_version=cv2.__version__,
                  settings=dict(sift_features_per_view=3000, descriptor="SIFT_L2", mutual_ratio=.8,
                                holdout="zero-based descriptor-order index modulo 5 == 4",
                                fundamental=dict(method="FM_RANSAC", threshold_px=1, confidence=.999,
                                                 max_iterations=10000, seed="1337 + 10*a + b"),
                                homography=dict(method="RANSAC", threshold_px=3, confidence=.999,
                                                max_iterations=10000, seed="2337 + 10*a + b"),
                                p90="nearest rank ceil(0.9*n)-1; finite values only",
                                fraction_denominator="all heldout matches"),
                  feature_counts={str(k):len(v[0]) for k, v in features.items()},
                  input_sha256_before=before, pairs=[])
    output.mkdir(parents=True)
    (output / "matches").mkdir()
    match_bytes = 0
    for a, b in itertools.combinations(views, 2):
        result, matches = pair_result(a, b, features)
        match_path = output / "matches" / f"{a['id']:02d}_{b['id']:02d}.json"
        payload = json.dumps(dict(a=a["id"], b=b["id"], matches=matches), allow_nan=False) + "\n"
        match_bytes += len(payload.encode())
        if match_bytes > 90*1024**2:
            raise RuntimeError("match audit data exceeds reserved 90 MiB bound")
        match_path.write_text(payload)
        result["matches_file"] = str(match_path.relative_to(output))
        report["pairs"].append(result)
    after = {str(path): digest(path) for path in input_paths}
    if before != after:
        raise RuntimeError("inputs, source, or OpenCV binary changed during run")
    report["input_sha256_after"] = after
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"output exceeded 100 MiB: {size} bytes")
    print(json.dumps(dict(output=str(output), bytes=size, available=sum(p["status"] == "available" for p in report["pairs"]),
                          focus=next(p for p in report["pairs"] if (p["a"],p["b"]) == (1,4))),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
