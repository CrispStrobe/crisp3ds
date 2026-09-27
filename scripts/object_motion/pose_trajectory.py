"""Read-only, hash-bound diagnostics of an image-estimated camera trajectory.

NP3 numeric suffixes are acquisition-order labels, not supplied camera poses.
No result here is a reconstruction constraint or a physical-accuracy gate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


FILES = ("cameras.bin", "images.bin", "points3D.bin")
NAME = re.compile(r"^NP3_(\d{3})\.jpg$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


def rotation_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float((np.trace(relative) - 1) / 2)
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


def trajectory(rows: list[dict]) -> dict:
    """Analyze center and orientation geometry without assuming label angles true."""
    if len(rows) < 4:
        raise ValueError("need at least four registered named cameras")
    parsed = []
    for row in rows:
        match = NAME.fullmatch(row["name"])
        center = np.asarray(row["center"], dtype=float)
        rotation = np.asarray(row["camera_to_world_rotation"], dtype=float)
        if (match is None or not 0 <= int(match.group(1)) < 360 or
                center.shape != (3,) or rotation.shape != (3, 3) or
                not np.isfinite(center).all() or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)):
            raise ValueError("invalid named camera center or proper rotation")
        parsed.append((int(match.group(1)), row["name"], center, rotation))
    parsed.sort(key=lambda item: item[0])
    labels = [item[0] for item in parsed]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate acquisition-order suffix")
    centers = np.asarray([item[2] for item in parsed])
    center_mean = centers.mean(axis=0)
    _, singular, vh = np.linalg.svd(centers - center_mean, full_matrices=False)
    normal = vh[-1]
    offsets = centers - center_mean
    plane_signed = offsets @ normal
    planar = offsets - np.outer(plane_signed, normal)
    radii = np.linalg.norm(planar, axis=1)
    radius = float(np.median(radii))
    if not math.isfinite(radius) or radius <= 1e-9:
        raise ValueError("camera centers have no measurable median orbit radius")
    by_label = {item[0]: (index, item) for index, item in enumerate(parsed)}
    views = []
    for index, (label, name, center, rotation) in enumerate(parsed):
        views.append({"name": name, "order_label": label, "center": center.tolist(),
                      "view_direction_world": rotation[:, 2].tolist(),
                      "plane_residual_over_median_radius": abs(float(plane_signed[index])) / radius,
                      "planar_radius_over_median_radius": float(radii[index]) / radius})
    adjacent = []
    for left, right in zip(parsed, parsed[1:]):
        a, b = left[2], right[2]
        adjacent.append({"from": left[1], "to": right[1],
                         "order_label_step": right[0] - left[0],
                         "center_step_over_median_radius": float(np.linalg.norm(b - a)) / radius,
                         "view_direction_step_degrees": angle_degrees(left[3][:, 2], right[3][:, 2]),
                         "full_orientation_step_degrees": rotation_angle_degrees(left[3], right[3]),
                         "orbit_phase_step_degrees": angle_degrees(planar[by_label[left[0]][0]],
                                                                    planar[by_label[right[0]][0]])
                         if min(np.linalg.norm(planar[by_label[left[0]][0]]),
                                np.linalg.norm(planar[by_label[right[0]][0]])) > 1e-9 else None})
    first, last = parsed[0], parsed[-1]
    closing = {"from": last[1], "to": first[1],
               "wrapped_order_label_step": 360 - last[0] + first[0],
               "center_step_over_median_radius": float(np.linalg.norm(last[2] - first[2])) / radius,
               "view_direction_step_degrees": angle_degrees(last[3][:, 2], first[3][:, 2]),
               "full_orientation_step_degrees": rotation_angle_degrees(last[3], first[3])}
    opposing = []
    for label, name, center, rotation in parsed:
        other = (label + 180) % 360
        if label >= other or other not in by_label:
            continue
        _, (_, other_name, other_center, other_rotation) = by_label[other]
        opposing.append({"a": name, "b": other_name,
                         "center_distance_over_median_radius":
                             float(np.linalg.norm(center - other_center)) / radius,
                         "view_direction_angle_degrees":
                             angle_degrees(rotation[:, 2], other_rotation[:, 2]),
                         "full_orientation_angle_degrees":
                             rotation_angle_degrees(rotation, other_rotation)})
    def stats(values):
        return {"count": len(values), "median": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "max": max(values) if values else None}
    return {"registered": len(parsed),
            "normalization_note": "descriptive PCA-plane radius about mean center, not fitted/calibrated orbit",
            "median_planar_radius_model_units": radius,
            "mean_center_model_units": center_mean.tolist(), "orbit_plane_normal_world": normal.tolist(),
            "center_singular_values_model_units": singular.tolist(),
            "plane_residual_over_median_radius":
                stats([row["plane_residual_over_median_radius"] for row in views]),
            "adjacent_center_step_over_median_radius":
                stats([row["center_step_over_median_radius"] for row in adjacent]),
            "adjacent_full_orientation_step_degrees":
                stats([row["full_orientation_step_degrees"] for row in adjacent]),
            "opposing_center_distance_over_median_radius":
                stats([row["center_distance_over_median_radius"] for row in opposing]),
            "opposing_full_orientation_angle_degrees":
                stats([row["full_orientation_angle_degrees"] for row in opposing]),
            "largest_plane_residuals": sorted(views, key=lambda row: row["plane_residual_over_median_radius"],
                                               reverse=True)[:8],
            "largest_adjacent_center_steps": sorted(adjacent,
                key=lambda row: row["center_step_over_median_radius"], reverse=True)[:8],
            "largest_adjacent_orientation_steps": sorted(adjacent,
                key=lambda row: row["full_orientation_step_degrees"], reverse=True)[:8],
            "smallest_opposing_center_distances": sorted(opposing,
                key=lambda row: row["center_distance_over_median_radius"])[:8],
            "views": views, "adjacent_pairs": adjacent, "closing_pair": closing,
            "opposing_pairs": opposing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--repair-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pycolmap

    source = args.repair_report
    output = args.output
    if (source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= 1 << 20 or
            digest(source) != args.repair_report_sha256):
        raise ValueError("repair report is missing or differs from sealed hash")
    report = json.loads(source.read_text())
    if report.get("schema") != "mustard_sparse_track_repair_v1" or report.get("status") != "candidate_unreviewed":
        raise ValueError("not a sealed repair candidate")
    model_dir = source.parent / "model"
    if (model_dir.is_symlink() or not model_dir.is_dir() or
            str(model_dir.resolve()) != report["output"]["model_dir"] or
            {item.name for item in model_dir.iterdir()} != set(FILES)):
        raise ValueError("repair model directory inventory differs")
    hashes = {}
    for name in FILES:
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 256 << 20:
            raise ValueError("repair model file missing, linked, or oversized")
        hashes[name] = digest(path)
    if hashes != report["output"]["model_files_sha256"]:
        raise ValueError("repair model hash differs")
    model = pycolmap.Reconstruction(str(model_dir))
    model.check()
    if (model.num_reg_images() != report["output"]["registered_names"].__len__() or
            model.num_points3D() != report["output"]["points"]):
        raise ValueError("repair model counts differ")
    rows = []
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        pose = np.asarray(image.cam_from_world.matrix(), dtype=float)
        if pose.shape != (3, 4) or not np.isfinite(pose).all():
            raise ValueError("nonfinite model camera pose")
        rotation_world_to_cam, translation = pose[:, :3], pose[:, 3]
        rows.append({"name": image.name,
                     "center": (-rotation_world_to_cam.T @ translation).tolist(),
                     "camera_to_world_rotation": rotation_world_to_cam.T.tolist()})
    if sorted(row["name"] for row in rows) != sorted(report["output"]["registered_names"]):
        raise ValueError("repair registered camera names differ")
    result = {"schema": "mustard_pose_trajectory_diagnostic_v1", "status": "diagnostic_only",
              "scope": "TRAIN-only image-estimated poses; no GT/heldout or reconstruction constraints",
              "acquisition_order_caveat": "NP3 suffixes order the photographs; they are not supplied camera poses",
              "repair_report_sha256": args.repair_report_sha256,
              "model_files_sha256": hashes, "runner_sha256": digest(Path(__file__)),
              "trajectory": trajectory(rows)}
    if (digest(source) != args.repair_report_sha256 or
            any(digest(model_dir / name) != sha for name, sha in hashes.items()) or
            digest(Path(__file__)) != result["runner_sha256"]):
        raise ValueError("source or diagnostic code changed during audit")
    if output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise FileExistsError("diagnostic output must be a fresh path")
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > 256 << 10:
        raise ValueError("diagnostic report exceeds 256 KiB")
    with output.open("xb") as stream:
        stream.write(payload)


if __name__ == "__main__":
    main()
