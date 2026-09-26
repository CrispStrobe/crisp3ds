#!/usr/bin/env python3
"""Evaluate frozen sparse-point retention using previously measured laser distances."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/PIPES-CONTEXT-PROTOCOL.md"
THRESHOLDS_M = (0.01, 0.02, 0.05, 0.10)
MAX_INPUT_BYTES = 8 * 1024**2
MAX_OUTPUT_BYTES = 4 * 1024**2
FROZEN_POINT_COUNT = 258
FROZEN_HASHES = {
    "points": "49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0",
    "report": "00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e",
    "protocol": "9aebdb6b3e1bac2ae7ba5e5fd759494347bd8bcd73b591058a92230847798bd4",
    "score": "9df4234128ca06e1ee10842e34d93d9fc66336ca37f81c598e28af6921523796",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"missing, linked, or oversized input: {path}")
    return json.loads(path.read_text())


def point_id(record):
    if not isinstance(record, dict):
        raise ValueError("point record is not an object")
    ident = record.get("id")
    if type(ident) is not int or ident < 1:
        raise ValueError("point ID must be a positive integer")
    return ident


def verdict_id(record):
    if not isinstance(record, dict):
        raise ValueError("verdict record is not an object")
    ident = record.get("original_point_id")
    if type(ident) is not int or ident < 1:
        raise ValueError("verdict point ID must be a positive integer")
    return ident


def indexed(records, label, identify=point_id):
    if not isinstance(records, list):
        raise ValueError(f"{label} must be a list")
    result = {}
    for record in records:
        ident = identify(record)
        if ident in result:
            raise ValueError(f"duplicate {label} ID: {ident}")
        result[ident] = record
    return result


def require_same_ids(expected, actual, label):
    if expected.keys() != actual.keys():
        raise ValueError(f"{label} ID universe differs from frozen sparse points")


def reject_geometry_output(value):
    if isinstance(value, dict):
        if any(key in value for key in ("xyz", "point_xyz", "coordinates", "points")):
            raise ValueError("verdict artifact must not emit alternate XYZ geometry")
        for nested in value.values():
            reject_geometry_output(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_geometry_output(nested)


def validate_witnesses(point, verdict):
    observations = point["observations"]
    expected = {}
    for i, left in enumerate(observations):
        for right in observations[i + 1:]:
            pair = sorted((left, right), key=lambda item: item["image"])
            names = tuple(item["image"] for item in pair)
            features = tuple(item.get("original_feature_id") for item in pair)
            expected[names] = features
    witnesses = verdict.get("pair_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != len(expected):
        raise ValueError("verdict witness count disagrees with observations")
    seen = set()
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("invalid pair witness")
        names, features = witness.get("images"), witness.get("original_feature_ids")
        if (not isinstance(names, list) or not isinstance(features, list)
                or len(names) != 2 or len(features) != 2
                or tuple(names) not in expected or expected[tuple(names)] != tuple(features)
                or tuple(names) in seen or type(witness.get("pass")) is not bool):
            raise ValueError("invalid pair witness")
        missing = witness.get("missing_descriptor_images")
        if (not isinstance(missing, list) or len(missing) != len(set(missing))
                or any(name not in names for name in missing)
                or (bool(missing) and witness["pass"])):
            raise ValueError("invalid missing-descriptor witness")
        seen.add(tuple(names))
    passed = sum(w["pass"] for w in witnesses)
    missing = sum(bool(w["missing_descriptor_images"]) for w in witnesses)
    total = len(witnesses)
    baseline_passed = verdict.get("baseline_pairs_passed")
    if (type(verdict.get("pairs_passed")) is not int or verdict["pairs_passed"] != passed
            or type(verdict.get("pairs_total")) is not int or verdict["pairs_total"] != total
            or type(verdict.get("pairs_missing_descriptor")) is not int
            or verdict["pairs_missing_descriptor"] != missing
            or type(baseline_passed) is not int or not 0 <= baseline_passed <= total
            or verdict["retained"] != (passed == total)
            or verdict["baseline_all_pairs_retained"] != (baseline_passed == total)):
        raise ValueError("verdict pair counts disagree with witnesses")
    reason = "all_pairs_match" if passed == total else (
        "missing_descriptor" if missing else "pair_match_failed")
    if verdict.get("reason") != reason:
        raise ValueError("verdict reason disagrees with witnesses")


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def distance_summary(ids, distances, original_count):
    values = [distances[ident] for ident in ids]
    count = len(values)
    within = {f"{threshold:.2f}": sum(value <= threshold for value in values)
              for threshold in THRESHOLDS_M}
    return {
        "denominator": count,
        "median_m": quantile(values, 0.5),
        "p90_m": quantile(values, 0.9),
        "p95_m": quantile(values, 0.95),
        "max_m": max(values) if values else None,
        "within_threshold_counts": within,
        "within_threshold_fractions": {key: value / count if count else None
                                       for key, value in within.items()},
        "within_threshold_original_fractions": {key: value / original_count
                                                for key, value in within.items()},
        "missing_inclusive_error_fractions": {key: (original_count - value) / original_count
                                              for key, value in within.items()},
    }


def support_summary(ids, points, images):
    histogram = {}
    by_image = {name: 0 for name in images}
    total = 0
    for ident in ids:
        observations = points[ident].get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"point {ident} lacks observations")
        names = [entry.get("image") if isinstance(entry, dict) else None
                 for entry in observations]
        if any(name not in by_image for name in names) or len(names) != len(set(names)):
            raise ValueError(f"point {ident} has invalid observation images")
        histogram[str(len(names))] = histogram.get(str(len(names)), 0) + 1
        for name in names:
            by_image[name] += 1
        total += len(names)
    return {"observations": total, "observations_by_image": by_image,
            "points_by_support_views": histogram}


def evaluate_data(points_data, report, score, verdicts, point_hash, report_hash,
                  points_path=None, report_path=None, protocol_path=None, protocol_hash=None):
    if (points_data.get("schema") != "fixed_camera_sparse_points_v1"
            or report.get("schema") != "eth3d_pipes_fixed_camera_sparse_v1"
            or report.get("reference_geometry_used") is not False
            or score.get("schema") != "eth3d_pipes_sparse_laser_proximity_v1"
            or score.get("status") != "pass"):
        raise ValueError("unrecognized or reference-fed frozen input")
    points = indexed(points_data.get("points"), "points")
    if not points or len(points) != report.get("accepted_tracks"):
        raise ValueError("sparse point count disagrees with report")
    ids = list(points)
    if ids != list(range(1, len(points) + 1)):
        raise ValueError("frozen sparse IDs must be ordered and contiguous")
    for ident, point in points.items():
        xyz = point.get("xyz")
        if (not isinstance(xyz, list) or len(xyz) != 3
                or any(type(value) not in (float, int) or not math.isfinite(value)
                       for value in xyz)):
            raise ValueError(f"invalid frozen XYZ for point {ident}")
    score_hashes = score.get("input_sha256")
    if not isinstance(score_hashes, dict):
        raise ValueError("score lacks input hashes")
    for name, digest in (("points.json", point_hash), ("report.json", report_hash)):
        matches = [value for key, value in score_hashes.items() if Path(key).name == name]
        if matches != [digest]:
            raise ValueError(f"score source {name} hash mismatch")
    scored = indexed(score.get("point_distances_m"), "score distances")
    require_same_ids(points, scored, "score distances")
    if score.get("summary", {}).get("denominator") != len(points):
        raise ValueError("score denominator disagrees with frozen points")
    distances = {}
    for ident, item in scored.items():
        value = item.get("distance_m")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid score distance for point {ident}")
        distances[ident] = value
    scored_summary = score["summary"]
    baseline_recomputed = distance_summary(ids, distances, len(ids))
    for key in ("within_threshold_counts", "median_m", "p90_m", "p95_m", "max_m"):
        given, expected = scored_summary.get(key), baseline_recomputed[key]
        mismatch = (given != expected if isinstance(expected, dict) else
                    type(given) not in (int, float) or
                    not math.isclose(given, expected, abs_tol=1e-12))
        if mismatch:
            raise ValueError(f"score summary {key} disagrees with per-point distances")
    # The verdict format is deliberately point keyed. An emitted coordinate would
    # create a second geometry artifact and must never be silently accepted.
    if verdicts.get("schema") != "pipes_context_frozen_point_verification_v1":
        raise ValueError("unrecognized verdict schema")
    if verdicts.get("reference_geometry_used") is not False or verdicts.get("geometry_recomputed") is not False:
        raise ValueError("verdict must declare unchanged image-only geometry")
    reject_geometry_output(verdicts)
    decisions = indexed(verdicts.get("verdicts"), "verdicts", verdict_id)
    require_same_ids(points, decisions, "verdicts")
    for ident, item in decisions.items():
        if type(item.get("retained")) is not bool or type(item.get("baseline_all_pairs_retained")) is not bool:
            raise ValueError(f"point {ident} lacks boolean candidate/control verdicts")
        validate_witnesses(points[ident], item)
    if (verdicts.get("frozen_points") != len(ids)
            or verdicts.get("retained_points") != sum(item["retained"] for item in decisions.values())
            or verdicts.get("baseline_all_pairs_retained_points") != sum(
                item["baseline_all_pairs_retained"] for item in decisions.values())
            or verdicts.get("rejected_points") != sum(
                not item["retained"] for item in decisions.values())):
        raise ValueError("verdict summary counts disagree with point verdicts")
    if verdicts.get("input_sha256_before") != verdicts.get("input_sha256_after"):
        raise ValueError("verdict input hashes changed during verification")
    for label in ("input_sha256_before", "input_sha256_after"):
        source = verdicts.get(label)
        if not isinstance(source, dict):
            raise ValueError(f"verdict lacks {label}")
        if source.get("points.json") not in (None, point_hash) or source.get("report.json") not in (None, report_hash):
            raise ValueError(f"verdict {label} source hash disagrees")
        for path, digest in ((points_path, point_hash), (report_path, report_hash),
                             (protocol_path, protocol_hash)):
            if path is not None and source.get(str(path.resolve())) != digest:
                raise ValueError(f"verdict {label} source hash disagrees: {path}")
    if "point_distances_m" in verdicts:
        raise ValueError("verdict artifact must not emit alternate geometry or distances")
    images = [view.get("name") for view in report.get("views", [])]
    if not images or any(type(name) is not str for name in images) or len(images) != len(set(images)):
        raise ValueError("invalid source image universe")
    all_ids = ids
    retained_ids = [ident for ident in ids if decisions[ident]["retained"]]
    control_ids = [ident for ident in ids if decisions[ident]["baseline_all_pairs_retained"]]
    three_ids = [ident for ident in ids if len(points[ident]["observations"]) >= 3]
    if report.get("accepted_tracks_three_or_more_views") != len(three_ids):
        raise ValueError("source three-view count disagrees with points")
    baseline_support = support_summary(all_ids, points, images)
    if (report.get("accepted_observations") != baseline_support["observations"]
            or report.get("accepted_observations_by_image") != baseline_support["observations_by_image"]
            or report.get("accepted_tracks_by_support_views") != baseline_support["points_by_support_views"]):
        raise ValueError("source observation support disagrees with points")
    def policy_summary(selected):
        selected_set = set(selected)
        discarded = [ident for ident in ids if ident not in selected_set]
        selected_three = [ident for ident in three_ids if ident in selected_set]
        discarded_by_threshold = {}
        for threshold in THRESHOLDS_M:
            key = f"{threshold:.2f}"
            near = sum(distances[ident] <= threshold for ident in discarded)
            discarded_by_threshold[key] = {"near_reference": near,
                                           "far_from_reference": len(discarded) - near}
        return {"retained_points": len(selected), "discarded_points": len(discarded),
                "distance": distance_summary(selected, distances, len(ids)),
                "support": support_summary(selected, points, images),
                "discarded_by_threshold": discarded_by_threshold,
                "three_or_more_views_descriptive": {
                    "retained_points": len(selected_three),
                    "retained_distance": distance_summary(selected_three, distances, len(ids))}}
    candidate = policy_summary(retained_ids)
    control = policy_summary(control_ids)
    return {
        "schema": "eth3d_pipes_context_rejection_evaluation_v1",
        "status": "pass",
        "interpretation": "fixed-population one-way laser proximity after sparse-point rejection; no completeness or geometric improvement claim",
        "scene_unit": "metre",
        "threshold_m": list(THRESHOLDS_M),
        "original_point_denominator": len(ids),
        "retained_points": len(retained_ids),
        "discarded_points": len(ids) - len(retained_ids),
        "baseline": {"distance": baseline_recomputed,
                     "support": baseline_support},
        "candidate": candidate,
        "unchanged_size_all_pairs_control": control,
        "candidate_minus_control": {
            "retained_points": candidate["retained_points"] - control["retained_points"],
            "retained_near_reference_counts": {
                key: candidate["distance"]["within_threshold_counts"][key]
                     - control["distance"]["within_threshold_counts"][key]
                for key in candidate["distance"]["within_threshold_counts"]}},
        "three_or_more_views_descriptive": {
            "baseline_points": len(three_ids),
            "baseline_distance": distance_summary(three_ids, distances, len(ids)),
        },
    }


def run(sparse, score_path, verdict_path, output, expected_verdict_sha256):
    paths = {"points": sparse / "points.json", "report": sparse / "report.json",
             "score": score_path, "verdicts": verdict_path, "protocol": PROTOCOL,
             "evaluator_source": Path(__file__).resolve()}
    for path in paths.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError(f"missing, linked, or oversized input: {path}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    if any(hashes[name] != digest for name, digest in FROZEN_HASHES.items()):
        raise ValueError("input differs from sealed frozen point/report/protocol/score artifact")
    if hashes["verdicts"] != expected_verdict_sha256:
        raise ValueError("verdict hash differs from expected sealed artifact")
    result = evaluate_data(read_json(paths["points"]), read_json(paths["report"]),
                           read_json(paths["score"]), read_json(paths["verdicts"]),
                           hashes["points"], hashes["report"], paths["points"],
                           paths["report"], paths["protocol"], hashes["protocol"])
    if result["original_point_denominator"] != FROZEN_POINT_COUNT:
        raise ValueError("input is not the frozen 258-point run")
    if {name: sha256(path) for name, path in paths.items()} != hashes:
        raise ValueError("evaluator input changed during execution")
    if output.is_symlink() or not output.is_dir() or (output / "evaluation.json").exists():
        raise ValueError("output directory missing or evaluation already exists")
    result["input_sha256"] = {str(path.resolve()): hashes[name] for name, path in paths.items()}
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("evaluation output exceeds 4 MiB")
    with (output / "evaluation.json").open("xb") as stream:
        stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-verdict-sha256", required=True)
    args = parser.parse_args()
    result = run(args.sparse.absolute(), args.score.absolute(), args.verdicts.absolute(),
                 args.output.absolute(), args.expected_verdict_sha256)
    print(json.dumps({"retained_points": result["retained_points"],
                      "evaluation_sha256": sha256(args.output / "evaluation.json")}))


if __name__ == "__main__":
    main()
