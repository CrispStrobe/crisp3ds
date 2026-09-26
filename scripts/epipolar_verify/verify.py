#!/usr/bin/env python3
"""Independent arithmetic check for the fixed-camera SIFT epipolar report.

This module intentionally uses only the Python standard library and never
estimates a model. Its input is the producer's saved, ordered observations.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def matvec(a, v):
    return [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]


def determinant(a):
    return (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
            - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
            + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))


def inverse_k(fx, fy, cx, cy):
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx == 0 or fy == 0:
        raise ValueError("invalid camera intrinsics")
    return [[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]]


def matrix(values):
    if len(values) == 3 and all(isinstance(row, list) and len(row) == 3 for row in values):
        values = [v for row in values for v in row]
    if len(values) != 9:
        raise ValueError("expected nine matrix entries")
    result = [[float(values[3 * i + j]) for j in range(3)] for i in range(3)]
    if not all(math.isfinite(x) for row in result for x in row):
        raise ValueError("nonfinite matrix")
    return result


def flatten(a):
    return [x for row in a for x in row]


def supplied_f(a, b):
    """World-to-camera R,t; x_b^T F x_a = 0 in pixel coordinates."""
    ra, rb = matrix(a["R"]), matrix(b["R"])
    ta, tb = a["t"], b["t"]
    relative_r = matmul(rb, transpose(ra))
    transformed_t = matvec(relative_r, ta)
    x, y, z = (tb[i] - transformed_t[i] for i in range(3))
    cross_t = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    ka_inv = inverse_k(*a["K"])
    kb_inv = inverse_k(*b["K"])
    return matmul(transpose(kb_inv), matmul(cross_t, matmul(relative_r, ka_inv)))


def normalize_f(f):
    flat = flatten(matrix(f))
    norm = math.hypot(*flat)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("zero or nonfinite fundamental matrix")
    return [x / norm for x in flat]


def f_difference_up_to_scale(a, b):
    x, y = normalize_f(a), normalize_f(b)
    return min(math.hypot(*(u - v for u, v in zip(x, y))),
               math.hypot(*(u + v for u, v in zip(x, y))))


def sqrt_sampson(f, a_xy, b_xy):
    f = matrix(f)
    norm = math.hypot(*flatten(f))
    if norm == 0:
        return None
    f = [[v / norm for v in row] for row in f]
    xa = [float(a_xy[0]), float(a_xy[1]), 1.0]
    xb = [float(b_xy[0]), float(b_xy[1]), 1.0]
    if not all(math.isfinite(v) for v in xa + xb):
        raise ValueError("nonfinite observation")
    fxa = matvec(f, xa)
    ftxb = matvec(transpose(f), xb)
    numerator = sum(xb[i] * fxa[i] for i in range(3))
    denominator = sum(v * v for v in (fxa[0], fxa[1], ftxb[0], ftxb[1]))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("nonfinite Sampson expression")
    if denominator <= 1e-12:
        return None
    return abs(numerator) / math.sqrt(denominator)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if fraction == 0.5:
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summary(f, points):
    distances = [sqrt_sampson(f, p["a_xy"], p["b_xy"]) for p in points]
    finite = [d for d in distances if d is not None and math.isfinite(d)]
    return {
        "count": len(points), "finite_count": len(finite),
        "degenerate_count": len(points) - len(finite),
        "median_px": percentile(finite, 0.5), "p90_px": percentile(finite, 0.9),
        "within_1px_count": sum(d <= 1 for d in finite),
        "within_2px_count": sum(d <= 2 for d in finite),
        "within_4px_count": sum(d <= 4 for d in finite),
    }


def read_views(path):
    views = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        words = line.split()
        if not words:
            continue
        if len(words) != 20:
            raise ValueError(f"{path}:{line_number}: expected 20 TSV fields, got {len(words)}")
        identifier = int(words[0])
        if identifier in views:
            raise ValueError(f"duplicate view {identifier}")
        floats = [float(v) for v in words[4:]]
        if not all(math.isfinite(v) for v in floats):
            raise ValueError(f"nonfinite camera {identifier}")
        views[identifier] = {"K": floats[:4], "R": floats[4:13], "t": floats[13:16]}
    return views


def reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON token {token}")


def read_json(path):
    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def near(actual, expected, label, atol=1e-6, rtol=1e-6):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            raise ValueError(f"{label}: reported {actual!r}, computed {expected!r}")
    elif actual != expected:
        raise ValueError(f"{label}: reported {actual!r}, computed {expected!r}")


def verify_split(matches, train_count, heldout_count):
    if len({(m["query_idx"], m["train_idx"]) for m in matches}) != len(matches):
        raise ValueError("duplicate descriptor pair")
    for i, m in enumerate(matches):
        if m["order"] != i:
            raise ValueError(f"descriptor order changed at row {i}")
        wanted = "holdout" if (i + 1) % 5 == 0 else "train"
        if m["split"] != wanted:
            raise ValueError(f"split changed at row {i}")
        if i and m["query_idx"] <= matches[i - 1]["query_idx"]:
            raise ValueError(f"query descriptor order changed at row {i}")
    train = [m for m in matches if m["split"] == "train"]
    heldout = [m for m in matches if m["split"] == "holdout"]
    near(train_count, len(train), "train_count")
    near(heldout_count, len(heldout), "heldout_count")
    return train, heldout


def verify_report(report_path, views_path):
    report = read_json(report_path)
    views = read_views(views_path)
    failures = []
    checked = 0
    before = report.get("input_sha256_before", {})
    after = report.get("input_sha256_after", {})
    if not before or before != after or str(views_path.resolve()) not in before:
        failures.append("input hashes missing, changed during run, or omit views TSV")
    for path_text, expected_hash in before.items():
        try:
            if file_sha256(Path(path_text)) != expected_hash:
                failures.append(f"input hash mismatch: {path_text}")
        except OSError as exc:
            failures.append(f"input inaccessible: {path_text}: {exc}")
    if len(views) != 10 or set(views) != set(range(10)):
        failures.append("TSV must contain exactly views 0 through 9")
    if len(report["views"]) != 10 or {v["id"] for v in report["views"]} != set(range(10)):
        failures.append("report must contain exactly views 0 through 9")
    for reported_view in report["views"]:
        identifier = reported_view["id"]
        if identifier not in views:
            failures.append(f"view {identifier}: missing in TSV")
            continue
        for key in ("K", "R", "t"):
            try:
                actual = flatten(matrix(reported_view[key])) if key in ("K", "R") else reported_view[key]
                expected = ([views[identifier]["K"][0], 0, views[identifier]["K"][2],
                             0, views[identifier]["K"][1], views[identifier]["K"][3],
                             0, 0, 1] if key == "K" else views[identifier][key])
                if len(actual) != len(expected):
                    raise ValueError("wrong parameter count")
                for i, (got, expected_value) in enumerate(zip(actual, expected)):
                    near(got, expected_value, f"view {identifier}.{key}[{i}]", atol=1e-10, rtol=1e-10)
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"view {identifier}: {exc}")
    for pair in report["pairs"]:
        a, b = pair["a"], pair["b"]
        name = f"{a:02d}_{b:02d}"
        try:
            if not a < b or a not in views or b not in views:
                raise ValueError("invalid pair identifiers")
            relative = Path(pair["matches_file"])
            if relative.is_absolute() or ".." in relative.parts or relative != Path("matches") / f"{name}.json":
                raise ValueError("unexpected matches_file path")
            match_path = report_path.parent / relative
            payload = read_json(match_path)
            if (payload["a"], payload["b"]) != (a, b):
                raise ValueError("matches belong to a different pair")
            train, heldout = verify_split(payload["matches"], pair["train_count"], pair["heldout_count"])
            near(pair["match_count"], len(payload["matches"]), "match_count")
            if pair["status"] not in ("available", "unavailable"):
                raise ValueError(f"unexpected status {pair['status']!r}")
            if pair["status"] == "available" and (not train or not heldout):
                raise ValueError("available pair has empty train or heldout split")
            expected_f = supplied_f(views[a], views[b])
            if pair["supplied_F"] is None:
                raise ValueError("missing supplied F for nondegenerate TSV cameras")
            delta = f_difference_up_to_scale(matrix(pair["supplied_F"]), expected_f)
            if delta > 1e-6:
                raise ValueError(f"supplied F differs from TSV camera geometry: {delta:.3g}")
            if pair["status"] == "unavailable" and pair["fitted_F"] is not None:
                raise ValueError("unavailable pair contains fitted F")
            for label, f in (("supplied", pair["supplied_F"]), ("fitted", pair["fitted_F"])):
                if f is None:
                    if pair[f"{label}_heldout"] is not None:
                        raise ValueError(f"{label} metrics present without F")
                    continue
                normalize_f(matrix(f))
                if not heldout and pair[f"{label}_heldout"] is None:
                    continue
                values = summary(matrix(f), heldout)
                claimed = pair[f"{label}_heldout"]
                for claimed_key, computed_key in (("total", "count"), ("finite", "finite_count"),
                                                  ("invalid", "degenerate_count"),
                                                  ("median_px", "median_px"), ("p90_px", "p90_px")):
                    near(claimed[claimed_key], values[computed_key], f"{label}_heldout.{claimed_key}", atol=1e-5)
                for threshold in (1, 2, 4):
                    count_key = f"within_{threshold}px_count"
                    fraction_key = f"within_{threshold}px_fraction"
                    near(claimed[count_key], values[count_key], f"{label}_heldout.{count_key}")
                    near(claimed[fraction_key], values[count_key] / len(heldout),
                         f"{label}_heldout.{fraction_key}")
            checked += 1
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            failures.append(f"{name}: {exc}")
    expected_pairs = {(a, b) for a in range(10) for b in range(a + 1, 10)}
    if len(report["pairs"]) != 45 or {(p["a"], p["b"]) for p in report["pairs"]} != expected_pairs:
        failures.append("expected exactly the 45 unique pairs of ten views")
    return {"pairs_checked": checked, "pairs_failed": len(failures), "failures": failures}


def verify_essential(essential_path, report_path, views_path):
    """Check saved pair-1/4 pose and held-out score; no pose estimation."""
    essential = read_json(essential_path)
    source = read_json(report_path)
    views = read_views(views_path)
    failures = []
    try:
        if essential["pair"] != [1, 4] or Path(essential["source_report"]).resolve() != report_path.resolve():
            raise ValueError("essential source pair/report mismatch")
        pair = next(p for p in source["pairs"] if (p["a"], p["b"]) == (1, 4))
        rows = read_json(report_path.parent / pair["matches_file"])["matches"]
        _, heldout = verify_split(rows, essential["training_count"], essential["heldout_count"])
        if essential["input_sha256_before"] != essential["input_sha256_after"]:
            raise ValueError("essential inputs changed during run")
        for path_text, digest in essential["input_sha256_before"].items():
            if file_sha256(Path(path_text)) != digest:
                raise ValueError(f"essential input hash mismatch: {path_text}")
        relative_r = matmul(matrix(views[4]["R"]), transpose(matrix(views[1]["R"])))
        relative_t = [views[4]["t"][i] - matvec(relative_r, views[1]["t"])[i] for i in range(3)]
        for key, expected in (("supplied_relative_R", flatten(relative_r)),
                              ("supplied_relative_t", relative_t)):
            reported = flatten(matrix(essential[key])) if key.endswith("_R") else essential[key]
            for i, (got, want) in enumerate(zip(reported, expected)):
                near(got, want, f"{key}[{i}]", atol=1e-9)
        recovered_r = matrix(essential["recovered_R"])
        recovered_t = essential["recovered_unit_t"]
        orthogonal = matmul(recovered_r, transpose(recovered_r))
        if (abs(determinant(recovered_r) - 1) > 1e-6 or
                max(abs(orthogonal[i][j] - (i == j)) for i in range(3) for j in range(3)) > 1e-6):
            raise ValueError("recovered rotation is not SO(3)")
        if len(recovered_t) != 3 or not all(math.isfinite(v) for v in recovered_t):
            raise ValueError("invalid recovered translation")
        near(math.hypot(*recovered_t), 1, "recovered unit translation", atol=1e-6)
        cross_t = [[0, -recovered_t[2], recovered_t[1]],
                   [recovered_t[2], 0, -recovered_t[0]],
                   [-recovered_t[1], recovered_t[0], 0]]
        delta_e = f_difference_up_to_scale(essential["essential_E"], matmul(cross_t, recovered_r))
        if delta_e > 1e-5:
            raise ValueError(f"essential E differs from recovered pose: {delta_e:.3g}")
        rotation_product = matmul(recovered_r, transpose(relative_r))
        rotation_cosine = max(-1.0, min(1.0, (sum(rotation_product[i][i] for i in range(3)) - 1) / 2))
        near(essential["rotation_difference_deg"], math.degrees(math.acos(rotation_cosine)),
             "rotation_difference_deg", atol=1e-6)
        direction_cosine = sum(recovered_t[i] * relative_t[i] for i in range(3)) / (math.hypot(*recovered_t) * math.hypot(*relative_t))
        direction_cosine = max(-1.0, min(1.0, direction_cosine))
        near(essential["translation_direction_difference_deg"], math.degrees(math.acos(direction_cosine)),
             "translation_direction_difference_deg", atol=1e-6)
        inliers = essential["essential_ransac_train_indices"]
        positive = essential["recover_pose_positive_train_indices"]
        if (len(inliers) != essential["essential_ransac_training_inliers"] or
                len(positive) != essential["recover_pose_positive_depth_training_count"] or
                len(set(inliers)) != len(inliers) or len(set(positive)) != len(positive) or
                any(not isinstance(i, int) or i < 0 or i >= essential["training_count"] for i in inliers) or
                not set(positive).issubset(inliers)):
            raise ValueError("essential training index bookkeeping inconsistent")
        raw = essential.get("raw_triangulation_training_inliers")
        if raw is not None and (raw["total"] != len(inliers) or
                raw["finite"] + raw["invalid"] != raw["total"] or
                raw["positive_both"] + raw["nonpositive_either"] != raw["finite"] or
                raw["positive_either_camera_depth_gt_50"] > raw["positive_both"] or
                raw["positive_either_camera_distance_gt_50"] > raw["positive_both"]):
            raise ValueError("raw triangulation training counts inconsistent")
        pose_f = supplied_f({"K": views[1]["K"], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "t": [0, 0, 0]},
                            {"K": views[4]["K"], "R": flatten(recovered_r), "t": recovered_t})
        delta = f_difference_up_to_scale(essential["calibrated_F"], pose_f)
        if delta > 1e-6:
            raise ValueError(f"calibrated F differs from recovered pose: {delta:.3g}")
        values = summary(essential["calibrated_F"], heldout)
        claimed = essential["calibrated_heldout"]
        for claimed_key, computed_key in (("total", "count"), ("finite", "finite_count"),
                                          ("invalid", "degenerate_count"),
                                          ("median_px", "median_px"), ("p90_px", "p90_px")):
            near(claimed[claimed_key], values[computed_key], f"calibrated_heldout.{claimed_key}", atol=1e-5)
        for threshold in (1, 2, 4):
            key = f"within_{threshold}px"
            near(claimed[f"{key}_count"], values[f"{key}_count"], f"calibrated_heldout.{key}_count")
            near(claimed[f"{key}_fraction"], values[f"{key}_count"] / len(heldout),
                 f"calibrated_heldout.{key}_fraction")
    except (KeyError, IndexError, TypeError, ValueError, OSError, StopIteration) as exc:
        failures.append(str(exc))
    return {"essential_checked": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--essential", type=Path)
    args = parser.parse_args()
    result = verify_report(args.report, args.views)
    if args.essential is not None:
        result["essential"] = verify_essential(args.essential, args.report, args.views)
    print(json.dumps(result, indent=2))
    return bool(result["pairs_failed"] or (args.essential is not None and result["essential"]["failures"]))


if __name__ == "__main__":
    sys.exit(main())
