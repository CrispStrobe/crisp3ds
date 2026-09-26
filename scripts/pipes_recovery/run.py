#!/usr/bin/env python3
"""Re-estimate frozen Pipes points from one original anchor and new training views."""

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path, PurePosixPath
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fixed_camera.run import reconstruct
from scripts.pipes_context import run as context
from scripts.pipes_sparse import run as sparse
from scripts.sparse_verify.verify import rotation

ORIGINAL = tuple(f"DSC_{n:04d}.JPG" for n in range(634, 638))
NEW = tuple(f"DSC_{n:04d}.JPG" for n in range(638, 648))
ALL = ORIGINAL + NEW
PROTOCOL = ROOT / "docs/PIPES-RECOVERY-PROTOCOL.md"
MAX_RESULT_BYTES = 256 * 1024**2
PROTOCOL_SHA256 = "3f6b838747b42f94d9d54dfdc346678f928330be126d42c2df10bf1c775172ed"
VIEW_KEYS = ("id", "name", "path", "sha256", "camera_id", "source_camera",
             "camera", "resize_scale_xy", "qvec", "R", "t", "center")


def checked_new_view(data, record, manifest):
    """Validate a new image against its own explicit ten-image allowlist."""
    name = record.get("name")
    rel = record.get("path")
    if name not in NEW or not isinstance(rel, str):
        raise ValueError("image outside additional-view allowlist")
    pure = PurePosixPath(rel)
    if (pure.is_absolute() or str(pure) != rel or "\\" in rel or
            any(part in (".", "..") for part in pure.parts) or
            rel != f"pipes/images/dslr_images_undistorted/{name}"):
        raise ValueError("image outside additional-view allowlist")
    target = data.joinpath(*pure.parts)
    if data.is_symlink() or any(data.joinpath(*pure.parts[:i]).is_symlink()
                                for i in range(1, len(pure.parts) + 1)):
        raise ValueError("additional image path is linked")
    if not target.is_file() or target.stat().st_size != record.get("size_bytes"):
        raise ValueError("additional image missing or size changed")
    digest = sparse.sha(target)
    if digest != record.get("sha256") or digest != manifest["file_sha256"].get(rel):
        raise ValueError("additional image digest changed")
    return target


def all_views(data):
    original, manifest_sha = sparse.selected_views(data)
    manifest = json.loads((data / "prepare-metadata.json").read_text())
    if sparse.sha(data / "prepare-metadata.json") != manifest_sha:
        raise ValueError("prepared manifest changed while loading views")
    records = manifest.get("images")
    if (manifest.get("status") != "validated" or not isinstance(records, list) or
            len(records) != 14 or sorted(r.get("name") for r in records) != list(ALL) or
            len({r.get("id") for r in records}) != 14):
        raise ValueError("prepared fourteen-view inventory changed")
    by_name = {r["name"]: r for r in records}
    new_views = []
    for name in NEW:
        record = by_name[name]
        if record.get("pose_convention") != "world_to_camera":
            raise ValueError("unexpected supplied pose convention")
        if str(record.get("camera_id")) not in manifest.get("cameras", {}):
            raise ValueError("missing supplied camera calibration")
        source = record["camera"]
        common = manifest["cameras"][str(record["camera_id"])]
        if {k: source[k] for k in ("width", "height", "params")} != common:
            raise ValueError("supplied camera calibration changed")
        path = checked_new_view(data, record, manifest)
        camera, scales = sparse.resize_camera(source)
        qvec = np.asarray(record.get("qvec"), dtype=float)
        t = np.asarray(record.get("tvec"), dtype=float)
        if (qvec.shape != (4,) or t.shape != (3,) or
                not np.isfinite(qvec).all() or not np.isfinite(t).all() or
                abs(np.linalg.norm(qvec) - 1) > 1e-3):
            raise ValueError("invalid supplied fixed pose")
        r = np.asarray(rotation(qvec.tolist()), dtype=float)
        if r.shape != (3, 3) or not np.isfinite(r).all():
            raise ValueError("invalid supplied rotation")
        new_views.append({"id": record["id"], "name": name, "path": str(path),
                          "sha256": record["sha256"], "camera_id": record["camera_id"],
                          "source_camera": source, "camera": camera,
                          "resize_scale_xy": scales, "qvec": record["qvec"],
                          "R": r.tolist(), "t": t.tolist(),
                          "center": (-r.T @ t).tolist()})
    return original + new_views, manifest_sha


def source_digests(data, baseline, views):
    digests = context.source_digests(data, baseline, views)
    paths = [PROTOCOL, Path(__file__), ROOT / "scripts/pipes_recovery/__init__.py",
             ROOT / "scripts/pipes_context/run.py", ROOT / "scripts/fixed_camera/run.py",
             ROOT / "scripts/sparse_verify/verify.py"]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked source: {path}")
        digests[str(path.resolve())] = sparse.sha(path)
    return digests


def extract_features(views, expected_original_counts):
    detector = cv2.SIFT_create(nfeatures=sparse.FEATURES)
    features = {}
    for view in views:
        gray = cv2.imread(view["path"], cv2.IMREAD_GRAYSCALE)
        source = view["source_camera"]
        if gray is None or gray.shape != (source["height"], source["width"]):
            raise ValueError(f"source image dimensions changed: {view['name']}")
        size = (view["camera"]["width"], view["camera"]["height"])
        if (gray.shape[1], gray.shape[0]) != size:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints = keypoints[:sparse.FEATURES]
        descriptors = descriptors[:sparse.FEATURES] if descriptors is not None else None
        if descriptors is None or len(descriptors) != len(keypoints):
            raise ValueError(f"missing SIFT descriptors: {view['name']}")
        if (descriptors.dtype != np.float32 or descriptors.shape[1:] != (128,) or
                not np.isfinite(descriptors).all()):
            raise ValueError(f"invalid SIFT descriptors: {view['name']}")
        name = view["name"]
        if name in ORIGINAL and len(keypoints) != expected_original_counts[name]:
            raise ValueError(f"frozen feature count mismatch: {name}")
        features[name] = ([sparse.edge_xy(k) for k in keypoints], descriptors)
    return features


def match_pairs(views, features):
    """Global strict mutual ratio before fixed-pose epipolar gating."""
    verified = {}
    summaries = []
    for a, b in itertools.combinations(views, 2):
        left, right = a["name"], b["name"]
        if left in ORIGINAL and right in ORIGINAL:
            continue
        xy_a, desc_a = features[left]
        xy_b, desc_b = features[right]
        raw = sparse.mutual_ratio(desc_a, desc_b)
        f = sparse.fundamental(a, b)
        kept = {}
        for ia, ib in raw:
            error = sparse.sampson(f, xy_a[ia], xy_b[ib])
            if error <= sparse.SAMPSON_PX:
                kept[(ia, ib)] = error
        verified[(left, right)] = kept
        summaries.append({"images": [left, right], "mutual_ratio_matches": len(raw),
                          "fixed_pose_epipolar_matches": len(kept)})
    if len(summaries) != 4 * 10 + 45:
        raise ValueError("incomplete additional-view pair matrix")
    return verified, summaries


def replay_original_pairs(views, features, baseline_report):
    recorded = {tuple(pair["images"]): pair for pair in baseline_report["pairs"]}
    if len(recorded) != 6:
        raise ValueError("frozen original pair inventory changed")
    for a, b in itertools.combinations(views[:4], 2):
        left, right = a["name"], b["name"]
        xy_a, desc_a = features[left]
        xy_b, desc_b = features[right]
        matches = sparse.mutual_ratio(desc_a, desc_b)
        f = sparse.fundamental(a, b)
        kept = sum(sparse.sampson(f, xy_a[i], xy_b[j]) <= sparse.SAMPSON_PX
                   for i, j in matches)
        expected = recorded[(left, right)]
        if (len(matches) != expected["mutual_ratio_matches"] or
                kept != expected["fixed_pose_epipolar_matches"] or
                not np.allclose(f, expected["supplied_fundamental"], atol=1e-12, rtol=0)):
            raise ValueError(f"frozen original matching replay changed: {left}, {right}")


def anchor_for(point):
    observations = point["observations"]
    return min(((o["image"], o["original_feature_id"]) for o in observations),
               key=lambda item: (item[0], item[1]))


def decision_for(point, verified, images, cameras, xy, reconstructor=reconstruct):
    """Decide from the frozen anchor, all direct new matches, and all new pairs."""
    name, fid = anchor_for(point)
    anchor = {"image": name, "original_feature_id": fid,
              "xy_edge_frame": xy[name][fid]}
    supports = []
    anchor_proofs = []
    for target in NEW:
        pair = (name, target)
        candidates = [(j, error) for (i, j), error in verified[pair].items() if i == fid]
        if len(candidates) > 1:
            raise ValueError("mutual matching produced multiple targets for anchor")
        if candidates:
            j, error = candidates[0]
            supports.append((target, j))
            anchor_proofs.append({"images": [name, target], "feature_ids": [fid, j],
                                  "sqrt_sampson_px": error})
    pair_proofs = []
    conflicts = []
    for (left, ia), (right, ib) in itertools.combinations(supports, 2):
        error = verified[(left, right)].get((ia, ib))
        proof = {"images": [left, right], "feature_ids": [ia, ib],
                 "mutual_ratio_and_epipolar_pass": error is not None,
                 "sqrt_sampson_px": error}
        pair_proofs.append(proof)
        if error is None:
            conflicts.append(proof)
    reason = None
    recovered = None
    if len(supports) < 2:
        reason = "fewer_than_two_new_views"
    elif conflicts:
        reason = "new_view_pair_conflict"
    else:
        track = ((name, fid), *supports)
        recovered, reason = reconstructor(track, images, cameras, xy)
    if recovered is not None and reason is not None:
        raise ValueError("inconsistent reconstructor result")
    if recovered is not None:
        for observation in recovered["observations"]:
            observation["xy_edge_frame"] = xy[observation["image"]][observation["original_feature_id"]]
        xyz = recovered["xyz"]
    else:
        xyz = point["xyz"][:]
    candidate = {"id": point["id"], "xyz": xyz,
                 "baseline_fallback": recovered is None, "source_anchor": anchor,
                 "reason": "reestimated" if recovered else reason,
                 "source_track": recovered["source_track"] if recovered else [[name, fid]],
                 "observations": recovered["observations"] if recovered else []}
    decision = {"original_point_id": point["id"], "source_anchor": anchor,
                "original_xyz": point["xyz"][:], "status": "reestimated" if recovered else "unresolved",
                "original_source_track": point["source_track"],
                "reason": "reestimated" if recovered else reason,
                "new_view_support_count": len(supports),
                "new_view_observations": [{"image": target, "original_feature_id": j,
                                           "xy_edge_frame": xy[target][j]} for target, j in supports],
                "anchor_match_proofs": anchor_proofs, "new_view_pair_proofs": pair_proofs,
                "reconstruction": recovered}
    return candidate, decision


def reject_shared_new_features(candidates, decisions):
    """Reject every otherwise accepted point in any shared-new-feature group."""
    owners = {}
    for decision in decisions:
        if decision["status"] != "reestimated":
            continue
        for obs in decision["new_view_observations"]:
            key = (obs["image"], obs["original_feature_id"])
            owners.setdefault(key, set()).add(decision["original_point_id"])
    collisions = {key: ids for key, ids in owners.items() if len(ids) > 1}
    affected = set().union(*collisions.values()) if collisions else set()
    for candidate, decision in zip(candidates, decisions):
        if candidate["id"] not in affected:
            continue
        decision["status"] = "unresolved"
        decision["reason"] = "shared_new_view_feature"
        decision["collision_proofs"] = [
            {"image": name, "original_feature_id": fid, "original_point_ids": sorted(ids)}
            for (name, fid), ids in sorted(collisions.items()) if candidate["id"] in ids]
        candidate["xyz"] = decision["original_xyz"][:]
        candidate["baseline_fallback"] = True
        candidate["reason"] = "shared_new_view_feature"
        candidate["source_track"] = [[candidate["source_anchor"]["image"],
                                       candidate["source_anchor"]["original_feature_id"]]]
        candidate["observations"] = []
        decision["rejected_reconstruction"] = decision["reconstruction"]
        decision["reconstruction"] = None
    return len(affected)


def worker(data, baseline, output):
    started = time.monotonic()
    if sparse.sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen recovery protocol digest mismatch")
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} - {"child.log"}:
            raise ValueError("output directory is not fresh")
    elif not output.parent.is_dir():
        raise ValueError("output parent does not exist")
    if shutil.disk_usage(output.parent).free < 10 * 1024**3 + MAX_RESULT_BYTES:
        raise ValueError("less than 10 GiB free plus output allowance")
    threads = sparse.configure_opencv_threads()
    views, manifest_sha = all_views(data)
    report = context.checked_json(baseline / "report.json", "eth3d_pipes_fixed_camera_sparse_v1")
    frozen = context.checked_json(baseline / "points.json", "fixed_camera_sparse_points_v1")
    if (sparse.sha(baseline / "points.json") != context.FROZEN_POINTS_SHA256 or
            sparse.sha(baseline / "report.json") != context.FROZEN_REPORT_SHA256):
        raise ValueError("frozen baseline digest mismatch")
    if (report["accepted_tracks"] != 258 or report["source_manifest_sha256"] != manifest_sha or
            report["views"] != [{k: v[k] for k in VIEW_KEYS} for v in views[:4]] or
            report["settings"]["max_long_edge"] != sparse.MAX_EDGE or
            report["settings"]["sift_features_per_view"] != sparse.FEATURES or
            report["settings"]["mutual_ratio_both_directions"] != sparse.RATIO or
            report["settings"]["fixed_pose_sqrt_sampson_max_px"] != sparse.SAMPSON_PX):
        raise ValueError("frozen baseline setup mismatch")
    context.assert_runtime_matches_baseline(report)
    before = source_digests(data, baseline, views)
    features = extract_features(views, report["feature_counts"])
    xy = {name: item[0] for name, item in features.items()}
    counts = {name: len(item[0]) for name, item in features.items()}
    context.assert_frozen_observations(frozen["points"], xy, counts)
    replay_original_pairs(views, features, report)
    verified, pair_summaries = match_pairs(views, features)
    images = {v["name"]: v for v in views}
    cameras = {v["camera_id"]: v["camera"] for v in views}
    candidates, decisions = [], []
    for point in frozen["points"]:
        candidate, decision = decision_for(point, verified, images, cameras, xy)
        candidates.append(candidate)
        decisions.append(decision)
    shared_feature_rejections = reject_shared_new_features(candidates, decisions)
    after = source_digests(data, baseline, views)
    context.assert_unchanged(before, after)
    reasons = Counter(d["reason"] for d in decisions if d["status"] == "unresolved")
    result = {"schema": "pipes_anchor_recovery_report_v1",
              "purpose": "additional_training_view_anchor_reestimation",
              "reference_geometry_used": False, "fixed_camera_oracle_lane": True,
              "new_observations_role": "training", "heldout_observations_used": False,
              "camera_poses_refined": False, "camera_calibrations_refined": False,
              "source_manifest_sha256": manifest_sha,
              "input_sha256_before": before, "input_sha256_after": after,
              "settings": {"frozen_points": 258, "anchor": "lexical_first_original_observation",
                           "additional_images": list(NEW), "max_long_edge": sparse.MAX_EDGE,
                           "sift_features_per_view": sparse.FEATURES,
                           "descriptor": "original_size_SIFT_L2", "strict_mutual_ratio": sparse.RATIO,
                           "fixed_pose_sqrt_sampson_max_px": sparse.SAMPSON_PX,
                           "minimum_distinct_new_views": 2, "require_all_new_view_pairs": True,
                           "min_parallax_deg": 1., "max_reprojection_px": 4.,
                           "max_normal_matrix_condition": 1e8},
              "views": [{k: v[k] for k in VIEW_KEYS} for v in views],
              "feature_counts": counts, "pair_summaries": pair_summaries,
              "frozen_points": len(decisions),
              "reestimated_points": sum(d["status"] == "reestimated" for d in decisions),
              "unresolved_points": sum(d["status"] == "unresolved" for d in decisions),
              "shared_feature_rejections": shared_feature_rejections,
              "unresolved_reasons": dict(sorted(reasons.items())),
              "opencv_threads_requested": 0, "opencv_threads": threads,
              "wall_seconds_at_report": round(time.monotonic() - started, 3),
              "peak_rss_bytes_at_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1024 if platform.system() == "Linux" else 1)}
    payloads = {"report.json": result,
                "candidates.json": {"schema": "pipes_anchor_recovery_candidates_v1", "points": candidates},
                "decisions.json": {"schema": "pipes_anchor_recovery_decisions_v1", "decisions": decisions}}
    encoded = {name: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
               for name, value in payloads.items()}
    if sum(map(len, encoded.values())) > MAX_RESULT_BYTES:
        raise ValueError("256 MiB result cap exceeded")
    output.mkdir(exist_ok=True)
    for name, content in encoded.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
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
                          "reestimated_points": result["reestimated_points"]}))
    else:
        from scripts.research_job.guard import run_child
        status = run_child([sys.executable, "-m", "scripts.pipes_recovery.run", "--worker",
                            "--data", str(data), "--baseline", str(baseline),
                            "--output", str(output)], cwd=ROOT, output_dir=output,
                           timeout_seconds=300, max_output_bytes=256 * 1024**2,
                           reserve_bytes=10 * 1024**3, max_log_bytes=10 * 1024**2)
        print(json.dumps(status, sort_keys=True))
        if status["status"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
