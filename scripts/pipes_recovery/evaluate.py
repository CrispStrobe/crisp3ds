#!/usr/bin/env python3
"""Score sealed additional-view candidate coordinates against the frozen pipes baseline."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

from scripts.pipes_score import run as score

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[2]
PROTOCOL = ROOT / "docs/PIPES-RECOVERY-PROTOCOL.md"
BASELINE_POINTS_SHA = "49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0"
BASELINE_REPORT_SHA = "00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e"
BASELINE_SCORE_SHA = "9df4234128ca06e1ee10842e34d93d9fc66336ca37f81c598e28af6921523796"
PROTOCOL_SHA = "3f6b838747b42f94d9d54dfdc346678f928330be126d42c2df10bf1c775172ed"
NEW_NAMES = {f"DSC_{n:04d}.JPG" for n in range(638, 648)}
ORIGINAL_NAMES = {f"DSC_{n:04d}.JPG" for n in range(634, 638)}
MAX_INPUT_BYTES = 128 * 1024**2


def _json(path):
    return json.loads(score.bounded_file(path, MAX_INPUT_BYTES).read_text())


def _raw_coordinates(path):
    # Preserve numeric JSON tokens; 1, 1.0, and 1e0 are distinct here.
    data = json.loads(score.bounded_file(path, MAX_INPUT_BYTES).read_text(),
                      parse_float=lambda value: ("float", value),
                      parse_int=lambda value: ("int", value))
    return [point["xyz"] for point in data["points"]]


def _finite_xyz(value):
    return (isinstance(value, list) and len(value) == 3 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value))


def _rotation(q):
    """Evaluator-local unit-quaternion rotation, independent of reconstruction."""
    norm = math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or abs(norm - 1.) > 1e-5:
        raise ValueError("invalid camera quaternion")
    w, x, y, z = (v / norm for v in q)
    return ((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)))


def project(image, xyz, source_camera, size):
    """Independent world-to-camera projection in resized pixel-edge coordinates."""
    if image.get("pose_convention") != "world_to_camera":
        raise ValueError("unsupported camera pose convention")
    if source_camera.get("model") != "PINHOLE" or len(source_camera.get("params", [])) != 4:
        raise ValueError("expected PINHOLE calibration")
    width, height = size
    if (not all(type(v) is int and v > 0 for v in size) or
            width > source_camera["width"] or height > source_camera["height"]):
        raise ValueError("invalid resized image dimensions")
    q, t = image["qvec"], image["tvec"]
    if len(q) != 4 or len(t) != 3 or not all(math.isfinite(v) for v in q + t):
        raise ValueError("invalid camera pose")
    r = _rotation(q)
    cam = [sum(row[k] * xyz[k] for k in range(3)) + t[i] for i, row in enumerate(r)]
    if not all(math.isfinite(v) for v in cam) or cam[2] <= 0:
        raise ValueError("nonpositive projection depth")
    fx, fy, cx, cy = source_camera["params"]
    sx, sy = width / source_camera["width"], height / source_camera["height"]
    return [sx * (fx * cam[0] / cam[2] + cx),
            sy * (fy * cam[1] / cam[2] + cy)]


def validate_candidates(candidate, original, original_raw, candidate_raw, metadata, size=(1024, 682)):
    if candidate.get("schema") != "pipes_anchor_recovery_candidates_v1":
        raise ValueError("unrecognized candidate schema")
    points = candidate.get("points")
    frozen = original.get("points")
    if not isinstance(points, list) or len(points) != 258 or len(frozen) != 258:
        raise ValueError("candidate must preserve all 258 original IDs")
    images = {image["name"]: image for image in metadata.get("images", [])}
    if len(images) != 14:
        raise ValueError("prepared metadata does not contain fourteen unique images")
    for image in images.values():
        common = metadata.get("cameras", {}).get(str(image.get("camera_id")))
        camera = image.get("camera", {})
        if (common is None or camera.get("model") != "PINHOLE" or
                any(camera.get(key) != common.get(key) for key in ("width", "height", "params"))):
            raise ValueError("image calibration differs from prepared camera")
    queries, recovered_ids, reasons = [], [], {}
    new_features = {}
    for i, (point, base) in enumerate(zip(points, frozen)):
        ident = i + 1
        if type(point.get("id")) is not int or point["id"] != ident or base.get("id") != ident:
            raise ValueError("candidate IDs must exactly match ordered original IDs")
        xyz = point.get("xyz")
        if not _finite_xyz(xyz):
            raise ValueError(f"nonfinite candidate XYZ for {ident}")
        anchor = min(base["observations"], key=lambda ob: (ob["image"], ob["original_feature_id"]))
        source_anchor = point.get("source_anchor")
        if source_anchor != {"image": anchor["image"], "original_feature_id": anchor["original_feature_id"],
                             "xy_edge_frame": anchor["xy_edge_frame"]}:
            raise ValueError(f"changed anchor for {ident}")
        fallback = point.get("baseline_fallback")
        if type(fallback) is not bool:
            raise ValueError(f"missing fallback decision for {ident}")
        if fallback:
            if point.get("source_track") != [[anchor["image"], anchor["original_feature_id"]]]:
                raise ValueError(f"fallback anchor track differs for {ident}")
            if candidate_raw[i] != original_raw[i] or xyz != base["xyz"]:
                raise ValueError(f"fallback XYZ tokens differ for {ident}")
            reason = point.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"fallback reason missing for {ident}")
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            observations = point.get("observations")
            if not isinstance(observations, list):
                raise ValueError(f"missing recovered observations for {ident}")
            names = [ob.get("image") for ob in observations]
            if (len(names) < 3 or len(set(names)) != len(names) or
                    names.count(anchor["image"]) != 1 or
                    len(set(names) & NEW_NAMES) < 2 or
                    any(name not in NEW_NAMES | {anchor["image"]} for name in names)):
                raise ValueError(f"recovered {ident} lacks anchor plus two distinct new views")
            if point.get("source_track") != [[ob["image"], ob["original_feature_id"]] for ob in observations]:
                raise ValueError(f"recovered source track differs for {ident}")
            for ob in observations:
                name = ob["image"]
                if name not in images or type(ob.get("original_feature_id")) is not int or ob["original_feature_id"] < 0:
                    raise ValueError(f"invalid observation identity for {ident}")
                if name == anchor["image"] and ob["original_feature_id"] != anchor["original_feature_id"]:
                    raise ValueError(f"changed anchor feature for {ident}")
                xy = ob.get("xy_edge_frame")
                if not isinstance(xy, list) or len(xy) != 2 or not all(type(v) in (float, int) and math.isfinite(v) for v in xy):
                    raise ValueError(f"invalid observation coordinate for {ident}")
                image = images[name]
                camera = image["camera"]
                uv = project(image, xyz, camera, size)
                err = math.dist(uv, xy)
                if err > 4.0 + 1e-9:
                    raise ValueError(f"reprojection above 4 px for {ident}: {name}")
                reported = ob.get("reprojection_px")
                if type(reported) not in (int, float) or not math.isfinite(reported) or abs(reported-err) > 1e-5:
                    raise ValueError(f"reported reprojection differs for {ident}: {name}")
                if name == anchor["image"] and xy != anchor["xy_edge_frame"]:
                    raise ValueError(f"changed anchor pixel for {ident}")
                if name in NEW_NAMES:
                    feature = name, ob["original_feature_id"]
                    if feature in new_features:
                        raise ValueError(f"shared new-view feature by IDs {new_features[feature]} and {ident}")
                    new_features[feature] = ident
            recovered_ids.append(ident)
        queries.append(xyz)
    return queries, recovered_ids, reasons


def paired_changes(baseline, candidate):
    result = {}
    for threshold in score.THRESHOLDS_M:
        near_before = baseline <= threshold
        near_after = candidate <= threshold
        result[f"{threshold:.2f}"] = {
            "near_to_far": int((near_before & ~near_after).sum()),
            "far_to_near": int((~near_before & near_after).sum())}
    return result


def validate_decisions(decisions, candidate, original):
    if decisions.get("schema") != "pipes_anchor_recovery_decisions_v1":
        raise ValueError("unrecognized decisions schema")
    records = decisions.get("decisions")
    if not isinstance(records, list) or len(records) != 258:
        raise ValueError("decisions must contain all 258 IDs")
    for ident, (decision, point, base) in enumerate(zip(records, candidate["points"], original["points"]), 1):
        if decision.get("original_point_id") != ident:
            raise ValueError("decision ID sequence differs")
        if decision.get("source_anchor") != point.get("source_anchor"):
            raise ValueError(f"decision anchor differs for {ident}")
        if decision.get("original_xyz") != base["xyz"]:
            raise ValueError(f"decision original XYZ differs for {ident}")
        if decision.get("original_source_track") != base["source_track"]:
            raise ValueError(f"decision original source track differs for {ident}")
        if decision.get("reason") != point.get("reason"):
            raise ValueError(f"decision reason differs for {ident}")
        if point["baseline_fallback"] and decision.get("status") != "unresolved":
            raise ValueError(f"decision status differs for {ident}")
        if not point["baseline_fallback"] and decision.get("status") != "reestimated":
            raise ValueError(f"decision status differs for {ident}")
        if not point["baseline_fallback"]:
            observed_new = [ob for ob in point["observations"] if ob["image"] in NEW_NAMES]
            if decision.get("new_view_support_count") != len(observed_new):
                raise ValueError(f"decision support count differs for {ident}")


def evaluate(prepared, sparse, baseline_path, candidate_path, decisions_path, report_path_new,
             expected_candidate_sha, expected_decisions_sha, expected_report_sha, output):
    import numpy as np
    deadline = time.monotonic() + score.TIME_LIMIT_S
    score.configure_threads()
    scan = prepared / "pipes/dslr_scan_eval/scan1.ply"
    alignment = prepared / "pipes/dslr_scan_eval/scan_alignment.mlp"
    metadata_path = prepared / "prepare-metadata.json"
    points_path, report_path = sparse / "points.json", sparse / "report.json"
    paths = [scan, alignment, metadata_path, points_path, report_path,
             baseline_path, candidate_path, decisions_path, report_path_new, SOURCE, PROTOCOL,
             Path(score.__file__),
             Path(score.cv2.__file__), Path(score.numpy_binary.__file__)]
    binaries = sorted(Path(score.cv2.__file__).resolve().parent.glob("cv2*.so"))
    if len(binaries) != 1:
        raise ValueError("cannot identify OpenCV binary")
    paths.append(binaries[0])
    hashes = {str(path.resolve()): score.sha(score.bounded_file(path)) for path in paths}
    if (hashes[str(points_path.resolve())] != BASELINE_POINTS_SHA or
            hashes[str(report_path.resolve())] != BASELINE_REPORT_SHA or
            hashes[str(baseline_path.resolve())] != BASELINE_SCORE_SHA):
        raise ValueError("frozen baseline hash differs")
    if hashes[str(PROTOCOL.resolve())] != PROTOCOL_SHA:
        raise ValueError("frozen recovery protocol hash differs")
    if len(expected_candidate_sha) != 64 or hashes[str(candidate_path.resolve())] != expected_candidate_sha.lower():
        raise ValueError("sealed candidate SHA-256 differs")
    if len(expected_decisions_sha) != 64 or hashes[str(decisions_path.resolve())] != expected_decisions_sha.lower():
        raise ValueError("sealed decisions SHA-256 differs")
    if len(expected_report_sha) != 64 or hashes[str(report_path_new.resolve())] != expected_report_sha.lower():
        raise ValueError("sealed recovery report SHA-256 differs")
    metadata, baseline, original, candidate, decisions, recovery_report = map(
        _json, (metadata_path, baseline_path, points_path, candidate_path, decisions_path, report_path_new))
    if (recovery_report.get("schema") != "pipes_anchor_recovery_report_v1" or
            recovery_report.get("reference_geometry_used") is not False or
            recovery_report.get("fixed_camera_oracle_lane") is not True or
            recovery_report.get("new_observations_role") != "training"):
        raise ValueError("unrecognized or reference-fed recovery report")
    before, after_inputs = recovery_report.get("input_sha256_before"), recovery_report.get("input_sha256_after")
    if not isinstance(before, dict) or before != after_inputs or not before:
        raise ValueError("reconstruction inputs were not sealed consistently")
    for path in (metadata_path, points_path, report_path, PROTOCOL):
        if before.get(str(path.resolve())) != hashes[str(path.resolve())]:
            raise ValueError("reconstruction input hash differs")
    if metadata.get("status") != "validated" or baseline.get("schema") != "eth3d_pipes_sparse_laser_proximity_v1":
        raise ValueError("unrecognized prepared or baseline score")
    offset, count, dtype = score.ply_layout(scan)
    declared = metadata["ply"]
    if (declared.get("header_bytes"), declared.get("vertex_count"), declared.get("vertex_stride_bytes")) != (offset, count, dtype.itemsize):
        raise ValueError("scan layout differs from prepared metadata")
    for path in (scan, alignment):
        if metadata["file_sha256"].get(path.relative_to(prepared).as_posix()) != hashes[str(path.resolve())]:
            raise ValueError("prepared reference hash differs")
    matrix = score.mlp_matrix(alignment, scan)
    if not np.allclose(metadata["alignment"]["matrix_row_major"], matrix, rtol=0, atol=1e-10):
        raise ValueError("alignment matrix differs")
    ids, _, source_report = score.sparse_points(points_path, report_path)
    if len(ids) != 258 or source_report.get("source_manifest_sha256") != hashes[str(metadata_path.resolve())]:
        raise ValueError("sparse provenance differs")
    recorded = baseline.get("input_sha256", {})
    for path in (scan, alignment, metadata_path, points_path, report_path):
        if recorded.get(str(path.resolve())) != hashes[str(path.resolve())]:
            raise ValueError("baseline scored different inputs")
    baseline_rows = baseline.get("point_distances_m", [])
    if len(baseline_rows) != 258 or [row.get("id") for row in baseline_rows] != ids:
        raise ValueError("baseline point IDs differ")
    baseline_dist = np.asarray([row["distance_m"] for row in baseline_rows], dtype=np.float64)
    if not np.isfinite(baseline_dist).all() or not np.all(baseline_dist >= 0):
        raise ValueError("invalid baseline distances")
    if score.summarise(baseline_dist) != baseline.get("summary"):
        raise ValueError("baseline summary differs")
    queries, recovered_ids, reasons = validate_candidates(
        candidate, original, _raw_coordinates(points_path), _raw_coordinates(candidate_path), metadata)
    validate_decisions(decisions, candidate, original)
    after = baseline_dist.copy()
    nearest = [row["nearest_scan_index"] for row in baseline_rows]
    if recovered_ids:
        world = score.scan_points(scan, matrix, deadline)
        changed, indices = score.exact_distances(world, np.asarray([queries[i-1] for i in recovered_ids], dtype=np.float64), deadline)
        for ident, dist, index in zip(recovered_ids, changed, indices):
            after[ident-1], nearest[ident-1] = dist, int(index)
    if {str(path.resolve()): score.sha(path) for path in paths} != hashes:
        raise ValueError("input changed during evaluation")
    if time.monotonic() > deadline:
        raise TimeoutError("recovery evaluation exceeded 300 seconds")
    selected = np.asarray([ident-1 for ident in recovered_ids], dtype=np.int32)
    result = {
        "schema": "eth3d_pipes_anchor_recovery_laser_proximity_v1", "status": "pass",
        "interpretation": "one-way nearest laser point proximity; neither surface truth nor completeness",
        "scene_unit": "metre", "laser_points": count, "original_denominator": 258,
        "recovered_count": len(recovered_ids), "unresolved_count": 258-len(recovered_ids),
        "unresolved_reasons": reasons, "changed_coordinate_count": sum(queries[i] != original["points"][i]["xyz"] for i in range(258)),
        "baseline_summary": baseline["summary"], "all_points_summary": score.summarise(after),
        "paired_threshold_changes": paired_changes(baseline_dist, after),
        "recovered_subset": {"denominator": len(selected),
                             "baseline_summary": score.summarise(baseline_dist[selected]) if len(selected) else None,
                             "candidate_summary": score.summarise(after[selected]) if len(selected) else None},
        "point_distances_m": [{"id": ident, "baseline_distance_m": float(baseline_dist[ident-1]),
                               "candidate_distance_m": float(after[ident-1]),
                               "nearest_scan_index": int(nearest[ident-1]),
                               "recovered": ident in set(recovered_ids)} for ident in ids],
        "candidate_sha256": expected_candidate_sha.lower(),
        "decisions_sha256": expected_decisions_sha.lower(),
        "recovery_report_sha256": expected_report_sha.lower(), "input_sha256": hashes,
        "scan_local_to_world_matrix_row_major": matrix.tolist(),
        "algorithm": "frozen fallback scores; changed points use chunked exhaustive float64 scan search",
        "opencv_effective_threads": score.cv2.getNumThreads(),
        "python_version": sys.version.split()[0], "opencv_version": score.cv2.__version__,
        "numpy_version": np.__version__, "cpu": platform.processor() or platform.machine(),
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_unit": "bytes" if sys.platform == "darwin" else "kilobytes"}
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > score.MAX_REPORT_BYTES:
        raise ValueError("report cap exceeded")
    if not output.is_dir() or output.is_symlink() or (output / "score.json").exists():
        raise ValueError("fresh output directory required")
    stat = os.statvfs(output)
    if stat.f_bavail * stat.f_frsize - len(encoded) < score.FREE_FLOOR_BYTES:
        raise ValueError("free-space reserve would be crossed")
    with (output / "score.json").open("xb") as stream:
        stream.write(encoded)
    result["score_sha256"] = score.sha(output / "score.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--decisions-sha256", required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--recovery-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    locations = [args.prepared, args.sparse, args.baseline_score, args.candidate,
                 args.decisions, args.recovery_report, args.output]
    prepared, sparse, baseline, candidate, decisions, recovery_report, output = [
        path.absolute() for path in locations]
    if not args.worker:
        from scripts.research_job.guard import run_child
        command = [sys.executable, "-m", "scripts.pipes_recovery.evaluate", "--worker",
                   "--prepared", str(prepared), "--sparse", str(sparse),
                   "--baseline-score", str(baseline), "--candidate", str(candidate),
                   "--candidate-sha256", args.candidate_sha256,
                   "--decisions", str(decisions), "--decisions-sha256", args.decisions_sha256,
                   "--recovery-report", str(recovery_report),
                   "--recovery-report-sha256", args.recovery_report_sha256,
                   "--output", str(output)]
        status = run_child(command, ROOT, output, timeout_seconds=score.TIME_LIMIT_S,
                           max_output_bytes=256 * 1024**2, reserve_bytes=score.FREE_FLOOR_BYTES)
        print(json.dumps(status))
        if status["status"] != "succeeded":
            raise SystemExit(1)
        return
    result = evaluate(prepared, sparse, baseline, candidate, decisions, recovery_report,
                      args.candidate_sha256, args.decisions_sha256,
                      args.recovery_report_sha256, output)
    print(json.dumps({"recovered_count": result["recovered_count"],
                      "median_m": result["all_points_summary"]["median_m"],
                      "score_sha256": result["score_sha256"]}))


if __name__ == "__main__":
    main()
