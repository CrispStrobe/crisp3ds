#!/usr/bin/env python3
"""Read-only coarse-mask membership of a sealed mustard sparse model's tracks."""

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from scripts.object_motion import mustard_sparse_support as support
from scripts.object_motion import ycb_object_masks as common


FILES = ("cameras.bin", "images.bin", "points3D.bin")


def camera_diagnostics(model, registered):
    cameras = list(model.cameras.values())
    if len(cameras) != 1:
        raise ValueError("expected one shared camera for this paired image-only experiment")
    camera = cameras[0]
    params = [float(value) for value in camera.params]
    if (camera.model.name != "SIMPLE_RADIAL" or camera.width != 1280 or camera.height != 1024 or
            len(params) != 4 or not all(math.isfinite(value) for value in params)):
        raise ValueError("shared camera is not finite 1280x1024 SIMPLE_RADIAL")
    centers = np.asarray([registered[name].projection_center() for name in sorted(registered)], dtype=float)
    if centers.shape != (len(registered), 3) or not np.isfinite(centers).all():
        raise ValueError("nonfinite camera center")
    centered = centers - centers.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not 1 <= len(singular) <= 3:
        raise ValueError("invalid camera center covariance")
    singular = np.pad(singular, (0, 3 - len(singular)))
    focal, cx, cy, radial = params
    return {"model": camera.model.name, "width": camera.width, "height": camera.height,
            "params": params, "focal_over_width": focal / camera.width,
            "principal_over_size": [cx / camera.width, cy / camera.height],
            "abs_radial_within_mapper_max_extra_param_1": abs(radial) <= 1.0,
            "center_singular_values_arbitrary_scale": singular.tolist(),
            "center_extent_arbitrary_scale": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
            "interpretation": "finite and mapper-guard checks only; no calibration or pose accuracy proof"}


def model_observation_support(model, masks, names):
    registered = {model.images[image_id].name: model.images[image_id]
                  for image_id in model.reg_image_ids()}
    if (len(registered) != model.num_reg_images() or len(registered) != len(set(registered)) or
            not set(registered).issubset(names)):
        raise ValueError("registered images do not match TRAIN names")
    by_id = {int(image.image_id): image for image in registered.values()}
    per_image = {name: {"observations": 0, "inside_coarse_pose_mask": 0,
                        "outside_coarse_pose_mask": 0, "outside_image_grid": 0}
                 for name in sorted(registered)}
    tracks = {"point_denominator": 0, "all_observations_inside": 0,
              "at_least_80_percent_observations_inside": 0,
              "no_observations_inside": 0, "duplicate_same_image_track_count": 0,
              "duplicate_same_image_observations": 0, "at_least_3_distinct_views": 0}
    for point_id, point in model.points3D.items():
        xyz = [float(value) for value in point.xyz]
        if len(xyz) != 3 or not all(map(math.isfinite, xyz)):
            raise ValueError("nonfinite sparse point")
        elements = point.track.elements
        if len(elements) < 2:
            raise ValueError("track shorter than two observations")
        tracks["point_denominator"] += 1
        seen, seen_observations, inside = set(), set(), 0
        for element in elements:
            image = by_id.get(int(element.image_id))
            if image is None:
                raise ValueError("track references unregistered image")
            index = int(element.point2D_idx)
            if not 0 <= index < len(image.points2D):
                raise ValueError("track point2D index out of range")
            if (int(element.image_id), index) in seen_observations:
                raise ValueError("duplicate exact track observation")
            seen_observations.add((int(element.image_id), index))
            observed = image.points2D[index]
            if int(observed.point3D_id) != int(point_id):
                raise ValueError("track-to-point2D backlink differs")
            xy = np.asarray(observed.xy, dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("nonfinite observed point2D")
            x, y = math.floor(xy[0]), math.floor(xy[1])
            mask = masks[image.name]
            row = per_image[image.name]
            row["observations"] += 1
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                row["outside_image_grid"] += 1
            elif mask[y, x]:
                row["inside_coarse_pose_mask"] += 1
                inside += 1
            else:
                row["outside_coarse_pose_mask"] += 1
            if image.name in seen:
                tracks["duplicate_same_image_observations"] += 1
            seen.add(image.name)
        if len(seen) < len(elements):
            tracks["duplicate_same_image_track_count"] += 1
        tracks["at_least_3_distinct_views"] += int(len(seen) >= 3)
        tracks["all_observations_inside"] += int(inside == len(elements))
        tracks["at_least_80_percent_observations_inside"] += int(inside / len(elements) >= 0.8)
        tracks["no_observations_inside"] += int(inside == 0)
    total = {key: sum(row[key] for row in per_image.values()) for key in
             ("observations", "inside_coarse_pose_mask", "outside_coarse_pose_mask", "outside_image_grid")}
    if total["observations"] != sum(total[key] for key in
                                    ("inside_coarse_pose_mask", "outside_coarse_pose_mask", "outside_image_grid")):
        raise ValueError("observation denominator mismatch")
    return {"registered_names": sorted(registered), "registered_count": len(registered),
            "missing_names": [name for name in names if name not in registered],
            "sparse_points": len(model.points3D), "observations": total,
            "observations_by_image": per_image, "tracks": tracks,
            "camera": camera_diagnostics(model, registered)}


def evaluate(run, result_sha, stage_root, stage_sha):
    import pycolmap
    run, stage_root = Path(run), Path(stage_root)
    runner_sha = common.digest(__file__)
    helper_sha = common.digest(support.__file__)
    result = support.bound_json(run / "result.json", result_sha)
    stage = support.bound_json(stage_root / "stage-report.json", stage_sha)
    if (result.get("schema") != "classical_backend_v1" or
            result.get("status") != "sparse_complete" or
            result.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap"):
        raise ValueError("not a sealed image-only sparse-complete run")
    names, masks = support.load_masks(stage_root, stage)
    inputs = result.get("inputs", [])
    if ([row.get("name") for row in inputs] != names or
            {row.get("name"): row.get("sha256") for row in inputs} != stage["train_photo_sha256"]):
        raise ValueError("run source names/hashes differ from staged TRAIN images")
    for row in inputs:
        photo = run / "images" / row["name"]
        if photo.is_symlink() or not photo.is_file() or common.digest(photo) != row["sha256"]:
            raise ValueError("run source photo changed")
    sfm_path = run / "sfm.json"
    sfm_sha = common.digest(sfm_path)
    sfm = json.loads(sfm_path.read_text())
    index = sfm.get("selected_model_index")
    if (not isinstance(index, int) or sfm.get("registered_images", 0) < 3 or
            not isinstance(sfm.get("registered_names"), list)):
        raise ValueError("saved sparse selection invalid")
    if index not in [item.get("index") for item in sfm.get("candidate_models", [])]:
        raise ValueError("selected candidate model missing from saved SfM summary")
    # Classical backend exports the selected candidate as sparse/0, regardless
    # of its original candidate-model index.
    model_dir = run / "sparse" / "0"
    if (model_dir.is_symlink() or not model_dir.is_dir() or
            {path.name for path in model_dir.iterdir()} != set(FILES)):
        raise ValueError("accepted sparse export file inventory differs")
    for name in FILES:
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024**2:
            raise ValueError("accepted sparse binary missing, linked or oversized")
    binaries = {name: common.digest(model_dir / name) for name in FILES}
    model = pycolmap.Reconstruction(str(model_dir))
    audit = model_observation_support(model, masks, names)
    if (audit["registered_count"] != sfm["registered_images"] or
            audit["sparse_points"] != sfm["sparse_points"] or
            audit["registered_names"] != sorted(sfm["registered_names"])):
        raise ValueError("saved model differs from sparse summary")
    if (common.digest(run / "result.json") != result_sha or common.digest(sfm_path) != sfm_sha or
            any(common.digest(model_dir / name) != sha for name, sha in binaries.items()) or
            common.digest(stage_root / "stage-report.json") != stage_sha or
            common.digest(__file__) != runner_sha or common.digest(support.__file__) != helper_sha or
            any(common.digest(run / "images" / row["name"]) != row["sha256"] for row in inputs)):
        raise ValueError("model/source changed during observation audit")
    support.load_masks(stage_root, stage)  # rehash every staged source image and exact PNG
    return {"schema": "mustard_track_support_audit_v1", "status": "diagnostic_only",
            "run_result_sha256": result_sha, "sfm_sha256": sfm_sha,
            "model_files_sha256": binaries, "stage_report_sha256": stage_sha,
            "runner_sha256": runner_sha, "helper_sha256": helper_sha,
            "mask_role": "accepted coarse pose support, not exact silhouette or geometric truth",
            "pixel_mapping": "floor each measured COLMAP point2D XY into full-resolution PNG",
            "model": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "result-sha256", "stage-root", "stage-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if (output.exists() or output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir() or shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES):
        raise ValueError("fresh output on >=10 GiB volume required")
    report = evaluate(args.run, args.result_sha256, args.stage_root, args.stage_sha256)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > 1024**2:
        raise ValueError("track-support report >1 MiB")
    with output.open("xb") as file:
        file.write(payload)


if __name__ == "__main__":
    main()
