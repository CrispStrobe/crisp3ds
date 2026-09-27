"""Read-only sealed TRAIN two-view RANSAC/USAC input-robustness comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

from scripts.object_motion import mustard_pair_stability as stable
from scripts.object_motion import mustard_two_view_pose as pose


METHODS = ("RANSAC", "USAC_DEFAULT", "USAC_FAST", "USAC_ACCURATE", "USAC_MAGSAC")
ORDERS = ("blob", "canonical", "reverse_canonical")
SELECTED = stable.SELECTED
REMOVALS_PER_PAIR = 3
MAX_SECONDS = 55
SAMPLE_TAG = b"mustard-usac-robustness-v1|"
EXPECTED_PYTHON = pose.ROOT / ".local-tools/colmap-sparse/venv/bin/python"
EXPECTED_OPENCV = "4.10.0"


def selected_removals(row: dict) -> list[int]:
    name = (row["stratum"] + "/" + row["left"] + "/" + row["right"]).encode()
    matches = np.asarray(row["feature_index_pairs"], dtype="<u4")
    ranked = [(hashlib.sha256(SAMPLE_TAG + name + b"|" + pair.tobytes()).digest(), i)
              for i, pair in enumerate(matches)]
    return [index for _, index in sorted(ranked)[:REMOVALS_PER_PAIR]]


def ordered_row(row: dict, order: str, removed: int | None = None) -> dict:
    if order not in ORDERS:
        raise ValueError("unknown correspondence order")
    indices = np.arange(len(row["feature_index_pairs"]))
    if removed is not None:
        if not 0 <= removed < len(indices):
            raise ValueError("removed row outside verified matches")
        indices = indices[indices != removed]
    if order in ("canonical", "reverse_canonical"):
        matches = row["feature_index_pairs"][indices]
        indices = indices[np.lexsort((matches[:, 1], matches[:, 0]))]
        if order == "reverse_canonical":
            indices = indices[::-1]
    changed = row.copy()
    changed["points_left"] = row["points_left"][indices]
    changed["points_right"] = row["points_right"][indices]
    changed["feature_index_pairs"] = row["feature_index_pairs"][indices]
    changed["verified_correspondences"] = len(indices)
    return changed


def fit_usac(row: dict, method: str) -> dict:
    """Mirror historical pose gates, changing only the essential estimator method."""
    if method not in METHODS[1:]:
        raise ValueError("USAC method outside frozen panel")
    left, right = row["points_left"], row["points_right"]
    output = {key: row[key] for key in ("stratum", "left", "right", "cyclic_separation",
                                       "verified_correspondences", "two_view_config")}
    fits = []
    for seed in pose.SEEDS:
        cv2.setRNGSeed(seed)
        try:
            essential, inliers = cv2.findEssentialMat(
                left, right, np.eye(3), method=getattr(cv2, method), prob=.999,
                threshold=pose.THRESHOLD_NORM, maxIters=2000)
            if essential is None or inliers is None or essential.shape != (3, 3):
                fits.append({"seed": seed, "status": "unavailable", "reason": "no unique 3x3 essential matrix"})
                continue
            e_mask = inliers.ravel().astype(bool)
            count = int(e_mask.sum())
            if count < 5:
                fits.append({"seed": seed, "status": "unavailable", "reason": "fewer than five E inliers",
                             "essential_inliers": count})
                continue
            cheirality, rotation, translation, pose_mask = cv2.recoverPose(
                essential, left, right, np.eye(3), mask=e_mask.astype("uint8")[:, None])
            positive = pose_mask.ravel().astype(bool)
            angle = pose.rotation_difference_degrees(rotation, np.eye(3))
            ray_angles = pose.ray_angles_degrees(left, right, rotation, positive)
            median_parallax = float(np.median(ray_angles)) if len(ray_angles) else None
            reliable = (len(left) >= 20 and count >= 15 and count / len(left) >= .65 and
                        cheirality >= 12 and cheirality / count >= .6 and
                        median_parallax is not None and median_parallax >= 1)
            fits.append({"seed": seed, "status": "passed" if reliable else "unavailable",
                         "essential_inliers": count, "essential_fraction": count / len(left),
                         "cheirality_positive": int(cheirality),
                         "cheirality_fraction_of_essential": int(cheirality) / count,
                         "rotation_degrees": angle, "rotation": rotation.tolist(),
                         "translation_unit": translation.ravel().tolist(),
                         "median_inter_ray_parallax_degrees": median_parallax,
                         "reason": None if reliable else "essential/cheirality/parallax gate failed"})
        except cv2.error as error:
            fits.append({"seed": seed, "status": "unavailable", "reason": str(error)[:300]})
    cv2.setRNGSeed(pose.SEEDS[0])
    try:
        homography, h_mask = cv2.findHomography(
            left, right, method=cv2.RANSAC, ransacReprojThreshold=pose.THRESHOLD_NORM,
            maxIters=2000, confidence=.999)
        h_count = int(h_mask.sum()) if homography is not None and h_mask is not None else 0
    except cv2.error:
        h_count = 0
    output["seeds"] = fits
    output["homography_inliers"] = h_count
    passed = [fit for fit in fits if fit["status"] == "passed"]
    if len(passed) != len(pose.SEEDS):
        output.update({"status": "unavailable", "reason": "one or more seed fits failed reliability gate"})
        return output
    rotations = [np.asarray(fit["rotation"]) for fit in fits]
    spread = max(pose.rotation_difference_degrees(a, b) for i, a in enumerate(rotations)
                 for b in rotations[i + 1:])
    output["max_interseed_rotation_difference_degrees"] = spread
    if spread > 5:
        output.update({"status": "unavailable", "reason": "rotation unstable across fixed seeds"})
    elif h_count >= min(fit["essential_inliers"] for fit in fits):
        output.update({"status": "unavailable", "reason": "homography matches or exceeds essential support"})
    else:
        angle = float(np.median([fit["rotation_degrees"] for fit in fits]))
        output["median_rotation_degrees"] = angle
        output["status"] = "near_identity" if angle < 5 else "distinct" if angle > 15 else "indeterminate"
    return output


def fit(row: dict, method: str) -> dict:
    if method == "RANSAC":
        return pose.fit_pair(row)
    return fit_usac(row, method)


def rotation_spread(first: dict, second: dict) -> float | None:
    differences = [pose.rotation_difference_degrees(np.asarray(a["rotation"]), np.asarray(b["rotation"]))
                   for a, b in zip(first["seeds"], second["seeds"])
                   if "rotation" in a and "rotation" in b]
    return max(differences) if differences else None


def compact(raw: dict) -> dict:
    return stable.compact_fit(raw)


def summarise_method(baselines: dict, removals: list[dict]) -> dict:
    original = baselines["blob"]
    order_differences = [rotation_spread(original, raw) for order, raw in baselines.items()
                         if order != "blob"]
    order_numeric = [value for value in order_differences if value is not None]
    same_order_differences = [rotation_spread(baselines[item["order"]], item["raw"])
                              for item in removals]
    numeric = [value for value in same_order_differences if value is not None]
    return {"max_baseline_order_rotation_spread_degrees": max(order_numeric) if order_numeric else None,
            "max_same_order_removal_rotation_spread_degrees": max(numeric) if numeric else None,
            "baseline_reliable_count": sum(raw["status"] in ("near_identity", "distinct")
                                           for raw in baselines.values()),
            "removal_reliable_count": sum(item["raw"]["status"] in ("near_identity", "distinct")
                                          for item in removals),
            "removal_abstention_count": sum(item["raw"]["status"] in ("unavailable", "indeterminate")
                                           for item in removals)}


def run(output: Path) -> dict:
    if Path(sys.executable) != EXPECTED_PYTHON or cv2.__version__ != EXPECTED_OPENCV:
        raise RuntimeError("use the pinned historical Python/OpenCV 4.10.0 environment")
    output = Path(output)
    if (output.exists() or output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir() or output.parent.resolve() != pose.DATA):
        raise ValueError("fresh output file required directly on backup data volume")
    free_before = pose.disk_floor()
    source, panels, inventories = {}, {}, {}
    for arm, (folder, db_hash, producer_hash) in pose.SOURCES.items():
        db, _, sidecars = pose.safe_database(folder, db_hash, producer_hash)
        rows, names = pose.read_panel(db, arm)
        rows = stable.attach_index_details(db, rows)
        panels[arm] = {stable.identity(row): row for row in rows}
        inventories[arm] = names
        source[arm] = {"database": str(db), "database_sha256": db_hash,
                       "producer_result_sha256": producer_hash, "sidecars": sidecars}
    if inventories["cached"] != inventories["sam"] or len(panels["cached"]) != 6 or \
            set(panels["cached"]) != set(panels["sam"]):
        raise ValueError("sealed panel/inventory differs")
    cv2.setNumThreads(1)
    started = time.monotonic()
    report = {"schema": "mustard_usac_robustness_v1", "status": "complete",
              "source": source,
              "runner_sha256": pose.sha(Path(__file__)),
              "pose_runner_sha256": pose.sha(Path(pose.__file__)),
              "index_runner_sha256": pose.sha(Path(stable.__file__)),
              "software": {"python_executable": sys.executable,
                           "opencv_version": cv2.__version__, "numpy_version": np.__version__},
              "protocol": {"methods": METHODS, "orders": ORDERS, "selected": SELECTED,
                           "removals_per_pair_arm": REMOVALS_PER_PAIR,
                           "sample_tag": SAMPLE_TAG.decode(), "seeds": pose.SEEDS,
                           "threshold_normalized": pose.THRESHOLD_NORM,
                           "confidence": .999, "max_iterations": 2000,
                           "elapsed_cap_seconds": MAX_SECONDS},
              "disk_free_before_bytes": free_before, "arms": {}}
    for arm in pose.SOURCES:
        report["arms"][arm] = []
        for key, row in panels[arm].items():
            removals = selected_removals(row) if key in SELECTED else []
            entry = {"pair": key, "feature_index_pairs_sha256": stable.index_hash(row),
                     "verified_correspondences": len(row["feature_index_pairs"]),
                     "sampled_removed_rows": [{"row": i,
                                               "feature_indices": row["feature_index_pairs"][i].tolist()}
                                              for i in removals], "methods": {}}
            for method in METHODS:
                baselines = {order: fit(ordered_row(row, order), method) for order in ORDERS}
                sampled = [{"removed_row": removed, "order": order,
                            "raw": fit(ordered_row(row, order, removed), method)}
                           for removed in removals for order in ORDERS]
                summary = summarise_method(baselines, sampled)
                entry["methods"][method] = {
                    "baselines": {order: compact(raw) for order, raw in baselines.items()},
                    "sampled_removals": [{"removed_row": item["removed_row"],
                                          "order": item["order"],
                                          "rotation_spread_from_same_order_full_degrees":
                                              rotation_spread(baselines[item["order"]], item["raw"]),
                                          "rotation_spread_from_blob_full_degrees":
                                              rotation_spread(baselines["blob"], item["raw"]),
                                          "fit": compact(item["raw"])} for item in sampled],
                    "summary": summary}
            report["arms"][arm].append(entry)
    report["elapsed_seconds"] = time.monotonic() - started
    for folder, db_hash, producer_hash in pose.SOURCES.values():
        pose.safe_database(folder, db_hash, producer_hash)
    if (pose.sha(Path(__file__)) != report["runner_sha256"] or
            pose.sha(Path(pose.__file__)) != report["pose_runner_sha256"] or
            pose.sha(Path(stable.__file__)) != report["index_runner_sha256"]):
        raise ValueError("runner changed during fit")
    report["disk_free_after_bytes"] = pose.disk_floor()
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) >= pose.MAX_REPORT:
        raise ValueError("report exceeds 20 MiB")
    output.write_bytes(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-reviewed-preflight", action="store_true")
    args = parser.parse_args()
    if not args.approve_reviewed_preflight:
        parser.error("live diagnostic requires root-reviewed preflight")
    def timeout(_signum, _frame):
        raise TimeoutError("USAC robustness diagnostic exceeded 55 seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        result = run(args.output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
