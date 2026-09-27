#!/usr/bin/env python3
"""Optional image-estimated turntable camera plausibility gate.

Capture slots encode order and nominal equal spacing only. They are never poses.
This gate does not establish physical pose or shape accuracy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


MIN_FREE = 10 * 1024**3
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
THRESHOLDS = {
    "minimum_registered_fraction": 0.75,
    "minimum_registered": 12,
    "minimum_planar_axis_ratio": 0.25,
    "maximum_adjacent_center_step_per_slot_over_radius": 0.75,
    "maximum_adjacent_orientation_step_per_slot_degrees": 45.0,
    "collapsed_opposing_center_over_radius": 0.5,
    "collapsed_opposing_orientation_degrees": 30.0,
    "minimum_collapsed_opposing_pairs_to_fail": 2,
    "minimum_winding_degrees": 270.0,
    "maximum_winding_degrees": 450.0,
    "minimum_significant_step_degrees": 2.0,
    "maximum_reversed_step_fraction": 0.15,
    "minimum_inward_facing_fraction": 0.75,
    "minimum_inward_cosine": 0.5,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profile(profile):
    if (not isinstance(profile, dict) or profile.get("schema") != "turntable_full_turn_profile_v1" or
            profile.get("full_turn") is not True or profile.get("uniform_slots") is not True):
        raise ValueError("explicit uniformly sampled full-turn capture profile required")
    slots = profile.get("slots")
    if (not isinstance(slots, list) or len(slots) < 12 or len(slots) % 2 or
            any(not isinstance(name, str) or not name or len(name) > 255 for name in slots) or
            len(set(slots)) != len(slots)):
        raise ValueError("profile needs at least 12 distinct, even-numbered ordered slots")
    return slots


def rotation_angle(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


def evaluate(rows, profile):
    """Return pass/fail/unavailable from image-estimated centers and camera-to-world rotations."""
    slots = validate_profile(profile)
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    by_name = {}
    for row in rows:
        name = row.get("name")
        if name not in slots or name in by_name:
            raise ValueError("unlisted or duplicate camera name")
        center = np.asarray(row.get("center"), dtype=float)
        rotation = np.asarray(row.get("camera_to_world_rotation"), dtype=float)
        if (center.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(center).all() or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
                not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)):
            raise ValueError("invalid center or proper camera rotation")
        by_name[name] = (center, rotation)
    indices = [i for i, name in enumerate(slots) if name in by_name]
    coverage = len(indices) / len(slots)
    base = {"schema": "turntable_orbit_plausibility_v1", "status": "unavailable",
            "registered": len(indices), "slots": len(slots), "coverage": coverage,
            "thresholds": THRESHOLDS.copy(),
            "meaning": "necessary trajectory plausibility checks, not physical pose or shape accuracy"}
    if len(indices) < THRESHOLDS["minimum_registered"] or coverage < THRESHOLDS["minimum_registered_fraction"]:
        return {**base, "reason": "insufficient_full_turn_coverage"}
    centers = np.array([by_name[slots[i]][0] for i in indices])
    mean = centers.mean(axis=0)
    _, singular, basis = np.linalg.svd(centers - mean, full_matrices=False)
    axis_ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    if axis_ratio < THRESHOLDS["minimum_planar_axis_ratio"]:
        return {**base, "reason": "degenerate_or_non_orbit_centers", "planar_axis_ratio": axis_ratio}
    plane = (centers - mean) @ basis[:2].T
    radius = float(np.median(np.linalg.norm(plane, axis=1)))
    if not math.isfinite(radius) or radius <= 1e-12 * max(1.0, float(np.max(np.abs(centers)))):
        return {**base, "reason": "unresolved_orbit_radius", "planar_axis_ratio": axis_ratio}
    locations = {index: position for position, index in enumerate(indices)}
    adjacent = []
    for index in indices:
        following = (index + 1) % len(slots)
        if following not in locations:
            continue
        a, b = by_name[slots[index]], by_name[slots[following]]
        adjacent.append({"from": slots[index], "to": slots[following],
                         "center_step_over_radius": float(np.linalg.norm(a[0] - b[0]) / radius),
                         "orientation_step_degrees": rotation_angle(a[1], b[1])})
    opposing = []
    for index in indices:
        other = (index + len(slots) // 2) % len(slots)
        if index >= other or other not in locations:
            continue
        a, b = by_name[slots[index]], by_name[slots[other]]
        opposing.append({"a": slots[index], "b": slots[other],
                         "center_distance_over_radius": float(np.linalg.norm(a[0] - b[0]) / radius),
                         "orientation_angle_degrees": rotation_angle(a[1], b[1])})
    if len(adjacent) < 6 or len(opposing) < 4:
        return {**base, "reason": "insufficient_adjacent_or_opposing_pairs",
                "adjacent_pairs": len(adjacent), "opposing_pairs": len(opposing)}
    # Eigenvector sign can flip; absolute winding and reversal fraction are invariant.
    phases = np.arctan2(plane[:, 1], plane[:, 0])
    cyclic_indices = indices + [indices[0] + len(slots)]
    cyclic_phases = np.r_[phases, phases[0]]
    steps = np.rad2deg(np.angle(np.exp(1j * np.diff(cyclic_phases))))
    winding = abs(float(np.sum(steps)))
    significant = steps[np.abs(steps) >= THRESHOLDS["minimum_significant_step_degrees"]]
    direction = 1 if np.sum(steps) >= 0 else -1
    reversal_fraction = (float(np.mean(significant * direction < 0)) if len(significant) else 1.0)
    # Both camera +Z and center-to-centroid are in the arbitrary reconstructed frame.
    if any(np.linalg.norm(mean - by_name[slots[index]][0]) < 1e-9 * radius for index in indices):
        return {**base, "reason": "camera_at_orbit_center", "planar_axis_ratio": axis_ratio}
    inward = [float(np.dot(by_name[slots[index]][1][:, 2],
                           (mean - by_name[slots[index]][0]) /
                           np.linalg.norm(mean - by_name[slots[index]][0])))
              for index in indices]
    inward_fraction = float(np.mean(np.asarray(inward) >= THRESHOLDS["minimum_inward_cosine"]))
    collapsed = [pair for pair in opposing
                 if pair["center_distance_over_radius"] < THRESHOLDS["collapsed_opposing_center_over_radius"]
                 and pair["orientation_angle_degrees"] < THRESHOLDS["collapsed_opposing_orientation_degrees"]]
    failures = []
    if any(pair["center_step_over_radius"] > THRESHOLDS["maximum_adjacent_center_step_per_slot_over_radius"]
           or pair["orientation_step_degrees"] > THRESHOLDS["maximum_adjacent_orientation_step_per_slot_degrees"]
           for pair in adjacent):
        failures.append("adjacent_jump")
    if len(collapsed) >= THRESHOLDS["minimum_collapsed_opposing_pairs_to_fail"]:
        failures.append("opposing_view_collapse")
    if (not THRESHOLDS["minimum_winding_degrees"] <= winding <= THRESHOLDS["maximum_winding_degrees"]
            or reversal_fraction > THRESHOLDS["maximum_reversed_step_fraction"]):
        failures.append("angular_winding_or_reversal")
    if inward_fraction < THRESHOLDS["minimum_inward_facing_fraction"]:
        failures.append("camera_orientation")
    return {**base, "status": "fail" if failures else "pass", "failures": failures,
            "planar_axis_ratio": axis_ratio, "winding_degrees": winding,
            "reversed_significant_step_fraction": reversal_fraction,
            "inward_facing_fraction": inward_fraction,
            "maximum_adjacent_center_step_over_radius": max(p["center_step_over_radius"] for p in adjacent),
            "maximum_adjacent_orientation_step_degrees": max(p["orientation_step_degrees"] for p in adjacent),
            "collapsed_opposing_pairs": collapsed, "adjacent_pairs": len(adjacent),
            "opposing_pairs": len(opposing), "registered_names": [slots[i] for i in indices]}


def preflight(model_dir, profile_path, external_root):
    """Read-only binding and capacity check; never opens a reconstruction."""
    model_dir, profile_path, external_root = map(Path, (model_dir, profile_path, external_root))
    if any(p.is_symlink() for p in (model_dir, profile_path, external_root)):
        raise ValueError("linked input or disk root")
    if not profile_path.is_file() or not 0 < profile_path.stat().st_size <= 64 * 1024:
        raise ValueError("missing or oversized profile")
    validate_profile(json.loads(profile_path.read_text()))
    if not model_dir.is_dir() or {p.name for p in model_dir.iterdir()} != set(MODEL_FILES):
        raise ValueError("model directory must contain exactly three COLMAP binaries")
    hashes = {}
    for name in MODEL_FILES:
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024**2:
            raise ValueError("missing, linked, empty or oversized model file")
        hashes[name] = sha256(path)
    free_internal = shutil.disk_usage(Path(__file__).resolve()).free
    free_external = shutil.disk_usage(external_root).free
    if min(free_internal, free_external) < MIN_FREE:
        raise RuntimeError("internal and external disks each need 10 GiB free")
    return {"schema": "turntable_orbit_preflight_v1", "status": "ready",
            "model_files_sha256": hashes, "profile_sha256": sha256(profile_path),
            "internal_free_bytes": free_internal, "external_free_bytes": free_external,
            "writes": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    ready = preflight(args.model, args.profile, args.external_root)
    if not args.evaluate:
        print(json.dumps(ready, sort_keys=True))
        return
    import pycolmap
    model = pycolmap.Reconstruction(str(args.model))
    rows = []
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        pose = np.asarray(image.cam_from_world.matrix(), dtype=float)
        rotation, translation = pose[:, :3], pose[:, 3]
        rows.append({"name": image.name, "center": (-rotation.T @ translation).tolist(),
                     "camera_to_world_rotation": rotation.T.tolist()})
    result = evaluate(rows, json.loads(args.profile.read_text()))
    if (ready["model_files_sha256"] != {name: sha256(args.model / name) for name in MODEL_FILES}
            or ready["profile_sha256"] != sha256(args.profile)):
        raise ValueError("input changed during read-only evaluation")
    keys = ("schema", "status", "reason", "registered", "slots", "coverage", "failures",
            "planar_axis_ratio", "winding_degrees", "reversed_significant_step_fraction",
            "inward_facing_fraction", "maximum_adjacent_center_step_over_radius",
            "maximum_adjacent_orientation_step_degrees", "adjacent_pairs", "opposing_pairs")
    summary = {key: result[key] for key in keys if key in result}
    summary["collapsed_opposing_pair_count"] = len(result.get("collapsed_opposing_pairs", ()))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
