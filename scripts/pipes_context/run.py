#!/usr/bin/env python3
"""Verify larger SIFT support on the frozen four-view Pipes sparse points."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import cv2
import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.pipes_sparse import run as sparse

MAX_RESULT_BYTES = 128 * 1024**2
PROTOCOL = ROOT / "docs/PIPES-CONTEXT-PROTOCOL.md"
COORD_TOLERANCE = 1e-6
CONTEXT_MULTIPLIER = 2.0
FROZEN_POINTS_SHA256 = "49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0"
FROZEN_REPORT_SHA256 = "00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e"
PROTOCOL_SHA256 = "9aebdb6b3e1bac2ae7ba5e5fd759494347bd8bcd73b591058a92230847798bd4"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_json(path, expected_schema):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError(f"missing, linked, or oversized input: {path}")
    value = json.loads(path.read_text())
    if value.get("schema") != expected_schema:
        raise ValueError(f"unexpected input schema: {path}")
    return value


def source_digests(data, baseline, views):
    cv_binary = sorted(Path(cv2.__file__).resolve().parent.glob("cv2*.so"))
    if len(cv_binary) != 1:
        raise ValueError("expected one OpenCV binary")
    import numpy.core._multiarray_umath as np_binary
    from PIL import _imaging
    paths = [data / "prepare-metadata.json", baseline / "points.json",
             baseline / "report.json", PROTOCOL, Path(__file__),
             ROOT / "scripts/pipes_sparse/run.py", ROOT / "scripts/research_job/guard.py",
             Path(sys.executable).resolve(), cv_binary[0].resolve(),
             Path(np_binary.__file__).resolve(), Path(PIL.__file__).resolve(),
             Path(_imaging.__file__).resolve()]
    paths.extend(Path(v["path"]) for v in views)
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked source: {path}")
    return {str(p.resolve()): sha(p) for p in paths}


def assert_runtime_matches_baseline(report):
    expected = report["dependencies"]
    actual_versions = {"python": sys.version.split()[0], "opencv": cv2.__version__,
                       "numpy": np.__version__, "pillow": PIL.__version__}
    if actual_versions != expected["versions"]:
        raise ValueError("runtime versions differ from frozen baseline")
    for path_text, digest in expected["binary_and_source_sha256"].items():
        path = Path(path_text)
        if path.suffix in (".so", ".dylib") or path == Path(sys.executable).resolve():
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f"runtime binary differs from frozen baseline: {path}")


def assert_unchanged(before, after):
    if before != after:
        raise ValueError("source, image, baseline, protocol, or runtime changed during run")


def assert_frozen_observations(points, xy, counts):
    if len(points) != 258 or [p["id"] for p in points] != list(range(1, 259)):
        raise ValueError("expected frozen 258 sequential point IDs")
    for point in points:
        observations = point["observations"]
        if len(observations) < 2 or len({o["image"] for o in observations}) != len(observations):
            raise ValueError(f"invalid observations on point {point['id']}")
        if sorted((o["image"], o["original_feature_id"]) for o in observations) != sorted(
                (o[0], o[1]) for o in point["source_track"]):
            raise ValueError(f"source track changed on point {point['id']}")
        for obs in observations:
            name, idx = obs["image"], obs["original_feature_id"]
            if name not in xy or type(idx) is not int or idx < 0 or idx >= counts[name]:
                raise ValueError(f"invalid frozen feature on point {point['id']}")
            stored = np.asarray(obs["xy_edge_frame"], dtype=float)
            if stored.shape != (2,) or not np.isfinite(stored).all() or not np.allclose(
                    stored, xy[name][idx], atol=COORD_TOLERANCE, rtol=0):
                raise ValueError(f"frozen observation coordinate mismatch on point {point['id']}")


def enlarged_descriptors(detector, gray, keypoints):
    """Return a feature-indexed matrix with absent descriptors recorded as None."""
    originals = [(tuple(k.pt), k.size, k.angle, k.response, k.octave, k.class_id)
                 for k in keypoints]
    requested = [cv2.KeyPoint(k.pt[0], k.pt[1], k.size * CONTEXT_MULTIPLIER,
                              k.angle, k.response, k.octave, index)
                 for index, k in enumerate(keypoints)]
    computed, descriptors = detector.compute(gray, requested)
    if computed is None:
        computed = []
    if len(computed) != (0 if descriptors is None else len(descriptors)):
        raise ValueError("computed keypoint/descriptor count mismatch")
    if descriptors is not None and (descriptors.dtype != np.float32 or descriptors.shape[1:] != (128,)):
        raise ValueError("unexpected SIFT descriptor type or width")
    result = [None] * len(keypoints)
    for index, kp in enumerate(computed):
        feature_id = kp.class_id
        if feature_id < 0 or feature_id >= len(keypoints) or result[feature_id] is not None:
            raise ValueError("unknown or duplicate computed feature ID")
        if not np.allclose(kp.pt, keypoints[feature_id].pt,
                           atol=COORD_TOLERANCE, rtol=0):
            raise ValueError("computed feature center changed")
        if not np.isfinite(descriptors[index]).all():
            raise ValueError("nonfinite computed descriptor")
        result[feature_id] = descriptors[index]
    if originals != [(tuple(k.pt), k.size, k.angle, k.response, k.octave, k.class_id)
                     for k in keypoints]:
        raise ValueError("original keypoints mutated")
    return result


def context_matches(left, right):
    """Run strict mutual ratio with every available original feature competing."""
    left_ids = [i for i, descriptor in enumerate(left) if descriptor is not None]
    right_ids = [i for i, descriptor in enumerate(right) if descriptor is not None]
    if min(len(left_ids), len(right_ids)) < 2:
        return set()
    a = np.stack([left[i] for i in left_ids]).astype(np.float32)
    b = np.stack([right[i] for i in right_ids]).astype(np.float32)
    return {(left_ids[i], right_ids[j]) for i, j in sparse.mutual_ratio(a, b)}


def point_verdict(point, matches, descriptors, original_matches):
    witnesses = []
    baseline_passed = 0
    for a, b in itertools.combinations(point["observations"], 2):
        left, right = sorted((a, b), key=lambda o: o["image"])
        name_a, name_b = left["image"], right["image"]
        id_a, id_b = left["original_feature_id"], right["original_feature_id"]
        missing = [name for name, idx in ((name_a, id_a), (name_b, id_b))
                   if descriptors[name][idx] is None]
        passed = not missing and (id_a, id_b) in matches[(name_a, name_b)]
        baseline_passed += (id_a, id_b) in original_matches[(name_a, name_b)]
        witnesses.append({"images": [name_a, name_b], "original_feature_ids": [id_a, id_b],
                          "pass": passed, "missing_descriptor_images": missing})
    passed_count = sum(w["pass"] for w in witnesses)
    missing_count = sum(bool(w["missing_descriptor_images"]) for w in witnesses)
    retained = passed_count == len(witnesses)
    return {"original_point_id": point["id"], "retained": retained,
            "baseline_all_pairs_retained": baseline_passed == len(witnesses),
            "baseline_pairs_passed": baseline_passed,
            "reason": "all_pairs_match" if retained else
                ("missing_descriptor" if missing_count else "pair_match_failed"),
            "pairs_passed": passed_count, "pairs_total": len(witnesses),
            "pairs_missing_descriptor": missing_count, "pair_witnesses": witnesses}


def worker(data, baseline, output):
    started = time.monotonic()
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} - {"child.log"}:
            raise ValueError("output directory is not fresh")
    elif not output.parent.is_dir():
        raise ValueError("output parent does not exist")
    if shutil.disk_usage(output.parent).free < 10 * 1024**3 + MAX_RESULT_BYTES:
        raise ValueError("less than 10 GiB free plus output allowance")
    threads = sparse.configure_opencv_threads()
    views, manifest_sha = sparse.selected_views(data)
    report = checked_json(baseline / "report.json", "eth3d_pipes_fixed_camera_sparse_v1")
    frozen = checked_json(baseline / "points.json", "fixed_camera_sparse_points_v1")
    if (sha(baseline / "points.json") != FROZEN_POINTS_SHA256 or
            sha(baseline / "report.json") != FROZEN_REPORT_SHA256):
        raise ValueError("frozen baseline hash mismatch")
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen context protocol hash mismatch")
    if (report["accepted_tracks"] != 258 or report["source_manifest_sha256"] != manifest_sha or
            report["views"] != [{k: v[k] for k in ("id", "name", "path", "sha256",
                                                 "camera_id", "source_camera", "camera",
                                                 "resize_scale_xy", "qvec", "R", "t", "center")}
                                for v in views] or
            report["settings"]["max_long_edge"] != sparse.MAX_EDGE or
            report["settings"]["sift_features_per_view"] != sparse.FEATURES or
            report["settings"]["mutual_ratio_both_directions"] != sparse.RATIO):
        raise ValueError("baseline configuration mismatch")
    assert_runtime_matches_baseline(report)
    before = source_digests(data, baseline, views)
    detector = cv2.SIFT_create(nfeatures=sparse.FEATURES)
    originals, context, xy, counts = {}, {}, {}, {}
    for view in views:
        gray = cv2.imread(view["path"], cv2.IMREAD_GRAYSCALE)
        source = view["source_camera"]
        if gray is None or gray.shape != (source["height"], source["width"]):
            raise ValueError("source image does not match camera dimensions")
        size = (view["camera"]["width"], view["camera"]["height"])
        if (gray.shape[1], gray.shape[0]) != size:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        keypoints, desc = detector.detectAndCompute(gray, None)
        keypoints = keypoints[:sparse.FEATURES]
        desc = desc[:sparse.FEATURES] if desc is not None else None
        name = view["name"]
        counts[name] = len(keypoints)
        if counts[name] != report["feature_counts"][name] or desc is None or len(desc) != counts[name]:
            raise ValueError(f"baseline feature count mismatch: {name}")
        xy[name] = [sparse.edge_xy(k) for k in keypoints]
        originals[name] = desc
        context[name] = enlarged_descriptors(detector, gray, keypoints)
    assert_frozen_observations(frozen["points"], xy, counts)
    matches, original_match_sets, pair_counts = {}, {}, []
    reported_pairs = {tuple(p["images"]): p for p in report["pairs"]}
    for a, b in itertools.combinations([v["name"] for v in views], 2):
        original_matches = sparse.mutual_ratio(originals[a], originals[b])
        if len(original_matches) != reported_pairs[(a, b)]["mutual_ratio_matches"]:
            raise ValueError(f"baseline descriptor pair count mismatch: {a}, {b}")
        original_match_sets[(a, b)] = set(original_matches)
        matches[(a, b)] = context_matches(context[a], context[b])
        pair_counts.append({"images": [a, b], "baseline_mutual_ratio_matches": len(original_matches),
                            "context_mutual_ratio_matches": len(matches[(a, b)])})
    verdicts = [point_verdict(point, matches, context, original_match_sets)
                for point in frozen["points"]]
    after = source_digests(data, baseline, views)
    assert_unchanged(before, after)
    result = {"schema": "pipes_context_frozen_point_verification_v1",
              "purpose": "image_only_descriptor_context_check_on_frozen_points",
              "reference_geometry_used": False, "geometry_recomputed": False,
              "settings": {"selected_images": [v["name"] for v in views],
                           "max_long_edge": sparse.MAX_EDGE, "sift_features_per_view": sparse.FEATURES,
                           "original_feature_coordinate_tolerance_px": COORD_TOLERANCE,
                           "context_keypoint_size_multiplier": CONTEXT_MULTIPLIER,
                           "keypoint_center_angle_octave_preserved": True,
                           "boundary_handling": "OpenCV SIFT.compute default; missing descriptors reject affected points",
                           "ratio": sparse.RATIO, "matcher": "strict mutual L2 2NN ratio",
                           "retention": "all original observation pairs must match"},
              "opencv_threads": threads,
              "runtime_versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                                   "numpy": np.__version__, "pillow": PIL.__version__},
              "input_sha256_before": before, "input_sha256_after": after,
              "feature_counts": counts,
              "context_descriptor_counts": {name: sum(d is not None for d in desc)
                                             for name, desc in context.items()},
              "pair_counts": pair_counts, "frozen_points": len(verdicts),
              "retained_points": sum(v["retained"] for v in verdicts),
              "baseline_all_pairs_retained_points": sum(v["baseline_all_pairs_retained"]
                                                        for v in verdicts),
              "rejected_points": sum(not v["retained"] for v in verdicts),
              "rejection_reasons": dict(sorted(Counter(v["reason"] for v in verdicts
                                                        if not v["retained"]).items())),
              "verdicts": verdicts,
              "wall_seconds_at_report": round(time.monotonic() - started, 3),
              "peak_rss_bytes_at_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1024 if platform.system() == "Linux" else 1)}
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_RESULT_BYTES:
        raise ValueError("128 MiB result cap exceeded")
    output.mkdir(exist_ok=True)
    with (output / "verdicts.json").open("xb") as stream:
        stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "build-opencv/pipes-prepare/run-001")
    parser.add_argument("--baseline", type=Path, default=ROOT / "build-opencv/pipes-sparse/run-003")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, baseline, output = args.data.absolute(), args.baseline.absolute(), args.output.absolute()
    if args.worker:
        result = worker(data, baseline, output)
        print(json.dumps({"frozen_points": result["frozen_points"],
                          "retained_points": result["retained_points"]}))
    else:
        from scripts.research_job.guard import run_child
        status = run_child([sys.executable, "-m", "scripts.pipes_context.run", "--worker",
                            "--data", str(data), "--baseline", str(baseline),
                            "--output", str(output)], cwd=ROOT, output_dir=output,
                           timeout_seconds=300, max_output_bytes=MAX_RESULT_BYTES,
                           reserve_bytes=10 * 1024**3, max_log_bytes=10 * 1024**2)
        print(json.dumps(status, sort_keys=True))
        if status["status"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
