#!/usr/bin/env python3
"""Quantify pose plausibility and photo-mask track support in a sparse model."""

import json
import math
from pathlib import Path


def in_support(image_record, xy):
    x, y = map(float, xy)
    row = int(math.floor(y))
    lines = image_record["pose_support_scanlines"]
    if not lines or row < lines[0][0] or row > lines[-1][0]:
        return False
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi) // 2
        if lines[mid][0] < row:
            lo = mid + 1
        else:
            hi = mid
    index = lo
    if index >= len(lines) or lines[index][0] != row:
        return False
    return lines[index][1] <= x <= lines[index][2]


def diagnose(model, manifest):
    import numpy as np
    records = {r["name"]: r for r in manifest["images"]}
    registered = [model.images[i] for i in model.reg_image_ids()]
    ordered = sorted(registered, key=lambda image: records[image.name]["angle_degrees"])
    per_image = []
    centers = []
    for image in ordered:
        camera = model.cameras[image.camera_id]
        count = image.num_points3D
        inside = sum(in_support(records[image.name], p.xy) for p in image.points2D if p.has_point3D())
        center = image.cam_from_world.inverse().translation
        centers.append(center)
        per_image.append({"name": image.name, "angle_degrees": records[image.name]["angle_degrees"],
                          "camera_id": image.camera_id, "center": [float(x) for x in center],
                          "triangulated_observations": count, "pose_support_observations": inside,
                          "outside_support_observations": count - inside,
                          "support_fraction": inside / count if count else None})
    cameras = []
    for cid, camera in model.cameras.items():
        params = [float(v) for v in camera.params]
        f = params[0] if params else float("nan")
        cx, cy = params[1:3] if len(params) >= 3 else (float("nan"), float("nan"))
        cameras.append({"camera_id": cid, "model": camera.model.name,
                        "width": camera.width, "height": camera.height, "params": params,
                        "focal_over_width": f / camera.width,
                        "focal_plausible_0_3_to_3_widths": bool(0.3 <= f / camera.width <= 3),
                        "principal_inside_image": bool(0 <= cx < camera.width and 0 <= cy < camera.height),
                        "finite_params": bool(all(math.isfinite(v) for v in params))})
    track_counts = {"all_inside": 0, "mixed": 0, "all_outside": 0, "shorter_than_2": 0}
    observations = {"inside": 0, "outside": 0}
    lengths = []
    for point in model.points3D.values():
        elements = point.track.elements
        flags = []
        for item in elements:
            image = model.images[item.image_id]
            if image.name not in records:
                continue
            flag = in_support(records[image.name], image.points2D[item.point2D_idx].xy)
            flags.append(flag)
            observations["inside" if flag else "outside"] += 1
        lengths.append(len(flags))
        if len(flags) < 2:
            track_counts["shorter_than_2"] += 1
        elif all(flags):
            track_counts["all_inside"] += 1
        elif any(flags):
            track_counts["mixed"] += 1
        else:
            track_counts["all_outside"] += 1
    centers_arr = np.asarray(centers)
    spread = float(np.linalg.norm(centers_arr.std(axis=0))) if len(centers) > 1 else None
    adjacent = [float(np.linalg.norm(centers_arr[i+1] - centers_arr[i]))
                for i in range(len(centers)-1)
                if per_image[i+1]["angle_degrees"] - per_image[i]["angle_degrees"] == 6]
    orbit = None
    if len(centers) >= 3:
        centered = centers_arr - centers_arr.mean(axis=0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        plane = centered @ vt[:2].T
        angles = np.unwrap(np.arctan2(plane[:, 1], plane[:, 0]))
        steps = np.diff(angles)
        radii = np.linalg.norm(plane, axis=1)
        dominant_sign = 1 if np.sum(steps > 0) >= np.sum(steps < 0) else -1
        orbit = {"pca_plane_variance_fraction": float(sum(singular[:2]**2) / sum(singular**2)) if sum(singular**2) else None,
                 "angular_span_degrees": float(np.degrees(angles[-1] - angles[0])),
                 "median_step_degrees": float(np.degrees(np.median(steps))),
                 "steps_in_dominant_direction": int(np.sum(steps * dominant_sign > 0)),
                 "steps_total": len(steps),
                 "radius_coefficient_of_variation": float(np.std(radii) / np.mean(radii)) if np.mean(radii) else None,
                 "first_last_center_distance_over_median_neighbor_step": float(np.linalg.norm(centers_arr[-1]-centers_arr[0]) / np.median(adjacent)) if adjacent and np.median(adjacent) else None,
                 "note": "PCA path consistency only; axes/sign and scale arbitrary, no supplied pose comparison"}
    return {"registered": len(registered), "total_images": len(records),
            "registered_angle_degrees": [r["angle_degrees"] for r in per_image],
            "points3D": len(model.points3D), "cameras": cameras, "images": per_image,
            "tracks": track_counts, "track_observations": observations,
            "median_track_length": float(np.median(lengths)) if lengths else None,
            "camera_center_spread_arbitrary_units": spread,
            "adjacent_6deg_center_step_median_arbitrary_units": float(np.median(adjacent)) if adjacent else None,
            "camera_center_orbit_diagnostic": orbit,
            "camera_plausibility_note": "focal/principal checks and center movement are diagnostics, not pose truth; object frame has arbitrary Sim(3) scale",
            "support_note": "photo-derived coarse pose-support mask; all-inside tracks may still be non-object and mask coverage is not mesh quality"}


def main():
    import argparse
    import pycolmap
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    model = pycolmap.Reconstruction(str(a.model))
    report = diagnose(model, json.loads(a.manifest.read_text()))
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("registered", "points3D", "tracks")}))


if __name__ == "__main__":
    main()
