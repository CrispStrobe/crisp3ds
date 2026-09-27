"""Bounded TRAIN-only correspondence perturbations of the sealed mustard panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import time

import cv2
import numpy as np

from scripts.object_motion import mustard_two_view_pose as pose


MAX_SECONDS = 55
SELECTED = (("middle", "NP3_012.jpg", "NP3_336.jpg"),
            ("opposing", "NP3_030.jpg", "NP3_198.jpg"),
            ("opposing", "NP3_036.jpg", "NP3_222.jpg"))


def identity(row: dict) -> tuple[str, str, str]:
    return row["stratum"], row["left"], row["right"]


def index_hash(row: dict) -> str:
    return hashlib.sha256(np.asarray(row["feature_index_pairs"], dtype="<u4").tobytes()).hexdigest()


def attach_index_details(db: Path, rows: list[dict]) -> list[dict]:
    """Join index blobs to validated pose rows without changing the sealed runner."""
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        ids = {name: image_id for image_id, name in
               connection.execute("SELECT image_id,name FROM images")}
        output = []
        for original in rows:
            row = original.copy()
            left_id, right_id = ids[row["left"]], ids[row["right"]]
            first, second = sorted((left_id, right_id))
            pair_id = first * pose.MAX_IMAGE_ID + second
            record = connection.execute(
                "SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
            count = row["verified_correspondences"]
            if (record is None or record[0] != count or record[1] != 2 or
                    record[2] is None or len(record[2]) != count * 8):
                raise ValueError("index blob differs from validated pose row")
            matches = np.frombuffer(record[2], dtype="<u4").reshape(count, 2).copy()
            if left_id != first:
                matches = matches[:, ::-1].copy()
            all_points = {}
            for image_id in (left_id, right_id):
                key = connection.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?",
                                         (image_id,)).fetchone()
                if (key is None or not 0 < key[0] <= 10000 or key[1] != 6 or
                        key[2] is None or len(key[2]) != key[0] * key[1] * 4):
                    raise ValueError("keypoint blob differs from validated pose row")
                all_points[image_id] = np.frombuffer(key[2], dtype="<f4").reshape(key[0], 6)[:, :2].astype(float)
            left_points, right_points = all_points[left_id], all_points[right_id]
            if (np.any(matches[:, 0] >= len(left_points)) or
                    np.any(matches[:, 1] >= len(right_points)) or
                    not np.array_equal((left_points[matches[:, 0]] - [640, 512]) / 1536,
                                       row["points_left"]) or
                    not np.array_equal((right_points[matches[:, 1]] - [640, 512]) / 1536,
                                       row["points_right"])):
                raise ValueError("index-to-coordinate mapping differs from validated pose row")
            row["feature_index_pairs"] = matches
            row["all_keypoints_left"] = left_points
            row["all_keypoints_right"] = right_points
            output.append(row)
        return output
    finally:
        connection.close()


def subset(row: dict, indices: list[int]) -> dict:
    changed = row.copy()
    changed["points_left"] = row["points_left"][indices]
    changed["points_right"] = row["points_right"][indices]
    changed["verified_correspondences"] = len(indices)
    return changed


def addition_candidates(target: dict, other: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Use only verified pairs from the other arm at identical target keypoints."""
    own = {tuple(map(int, pair)) for pair in target["feature_index_pairs"]}
    candidate, incompatible = [], []
    for pair in other["feature_index_pairs"]:
        first, second = map(int, pair)
        if (first, second) in own:
            continue
        valid = (first < len(target["all_keypoints_left"]) and
                 second < len(target["all_keypoints_right"]) and
                 first < len(other["all_keypoints_left"]) and
                 second < len(other["all_keypoints_right"]) and
                 np.array_equal(target["all_keypoints_left"][first], other["all_keypoints_left"][first]) and
                 np.array_equal(target["all_keypoints_right"][second], other["all_keypoints_right"][second]))
        (candidate if valid else incompatible).append((first, second))
    return candidate, incompatible


def augmented(row: dict, pair: tuple[int, int]) -> dict:
    first, second = pair
    changed = row.copy()
    changed["points_left"] = np.vstack((row["points_left"],
                                        (row["all_keypoints_left"][first] - [640, 512]) / 1536))
    changed["points_right"] = np.vstack((row["points_right"],
                                         (row["all_keypoints_right"][second] - [640, 512]) / 1536))
    changed["verified_correspondences"] = len(changed["points_left"])
    return changed


def compact_fit(fit: dict, baseline: dict | None = None) -> dict:
    compact = {key: value for key, value in fit.items()
               if key not in ("seeds", "translation_unit", "rotation")}
    seeds = []
    differences = []
    for i, seed in enumerate(fit["seeds"]):
        item = {key: value for key, value in seed.items()
                if key not in ("rotation", "translation_unit")}
        if baseline is not None and "rotation" in seed and "rotation" in baseline["seeds"][i]:
            difference = pose.rotation_difference_degrees(
                np.asarray(seed["rotation"]), np.asarray(baseline["seeds"][i]["rotation"]))
            item["rotation_difference_from_full_input_degrees"] = difference
            differences.append(difference)
        seeds.append(item)
    compact["seeds"] = seeds
    rotations = [np.asarray(seed["rotation"]) for seed in fit["seeds"] if "rotation" in seed]
    compact["numeric_interseed_rotation_spread_degrees"] = (
        max(pose.rotation_difference_degrees(a, b) for i, a in enumerate(rotations)
            for b in rotations[i + 1:]) if len(rotations) >= 2 else None)
    compact["max_rotation_difference_from_full_input_degrees"] = max(differences) if differences else None
    return compact


def summarize_perturbations(removals: list[dict], additions: list[dict]) -> dict:
    fits = [item["fit"] for item in removals + additions]
    numeric = [fit["max_rotation_difference_from_full_input_degrees"] for fit in fits
               if fit["max_rotation_difference_from_full_input_degrees"] is not None]
    spread = [fit["numeric_interseed_rotation_spread_degrees"] for fit in fits
              if fit["numeric_interseed_rotation_spread_degrees"] is not None]
    return {"trial_count": len(fits),
            "abstention_count": sum(fit["status"] in ("unavailable", "indeterminate") for fit in fits),
            "unavailable_count": sum(fit["status"] == "unavailable" for fit in fits),
            "max_rotation_difference_from_full_input_degrees": max(numeric) if numeric else None,
            "max_numeric_interseed_rotation_spread_degrees": max(spread) if spread else None}


def run(output: Path) -> dict:
    output = Path(output)
    if (output.exists() or output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir() or output.parent.resolve() != pose.DATA):
        raise ValueError("fresh output file required directly on backup data volume")
    free_before = pose.disk_floor()
    source, panels, inventories = {}, {}, {}
    for arm, (folder, db_hash, producer_hash) in pose.SOURCES.items():
        db, _, sidecars = pose.safe_database(folder, db_hash, producer_hash)
        panel, names = pose.read_panel(db, arm)
        panel = attach_index_details(db, panel)
        source[arm] = {"database": str(db), "database_sha256": db_hash,
                       "producer_result_sha256": producer_hash, "sidecars": sidecars}
        panels[arm], inventories[arm] = {identity(row): row for row in panel}, names
    if inventories["cached"] != inventories["sam"]:
        raise ValueError("TRAIN name inventories differ")
    if set(panels["cached"]) != set(panels["sam"]) or len(panels["cached"]) != 6:
        raise ValueError("sealed six-pair panel differs")
    cv2.setNumThreads(1)
    start = time.monotonic()
    report = {"schema": "mustard_pair_stability_v1", "status": "complete",
              "scope": "sealed TRAIN verified pairs and fixed initial camera only",
              "source": source, "runner_sha256": pose.sha(Path(__file__)),
              "pose_runner_sha256": pose.sha(Path(pose.__file__)),
              "software": {"opencv_version": cv2.__version__, "numpy_version": np.__version__},
              "protocol": {"selected": SELECTED, "removals": "all leave-one-out verified pairs",
                           "additions": "each other-arm-only verified index pair, one at a time, only when both target keypoints equal the source keypoints exactly",
                           "baseline": "all six sealed pair-arm rows",
                           "pose_method": "same three seeds, RANSAC, homography, and preset reliability gates as sealed audit",
                           "elapsed_cap_seconds": MAX_SECONDS},
              "disk_free_before_bytes": free_before, "arms": {}}
    baselines = {}
    for arm in pose.SOURCES:
        report["arms"][arm] = []
        for key, row in panels[arm].items():
            fit = pose.fit_pair(row)
            baselines[(arm, key)] = fit
            report["arms"][arm].append({"pair": key, "feature_index_pairs_sha256": index_hash(row),
                                         "baseline": compact_fit(fit), "perturbations": None})
    for arm in pose.SOURCES:
        other_arm = "sam" if arm == "cached" else "cached"
        for entry in report["arms"][arm]:
            key = tuple(entry["pair"])
            if key not in SELECTED:
                continue
            row, other = panels[arm][key], panels[other_arm][key]
            baseline = baselines[(arm, key)]
            removals = []
            for removed in range(len(row["feature_index_pairs"])):
                indices = [i for i in range(len(row["feature_index_pairs"])) if i != removed]
                fit = pose.fit_pair(subset(row, indices))
                removals.append({"removed_row": removed,
                                 "removed_feature_indices": row["feature_index_pairs"][removed].tolist(),
                                 "fit": compact_fit(fit, baseline)})
            candidates, incompatible = addition_candidates(row, other)
            additions = [{"added_feature_indices": list(pair),
                          "fit": compact_fit(pose.fit_pair(augmented(row, pair)), baseline)}
                         for pair in candidates]
            entry["perturbations"] = {"removals": removals, "additions": additions,
                                      "incompatible_other_arm_feature_indices": incompatible,
                                      "summary": summarize_perturbations(removals, additions)}
    report["elapsed_seconds"] = time.monotonic() - start
    for folder, db_hash, producer_hash in pose.SOURCES.values():
        pose.safe_database(folder, db_hash, producer_hash)
    if pose.sha(Path(__file__)) != report["runner_sha256"] or \
            pose.sha(Path(pose.__file__)) != report["pose_runner_sha256"]:
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
        raise TimeoutError("pair stability diagnostic exceeded 55 seconds")
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
