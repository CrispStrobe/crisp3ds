#!/usr/bin/env python3
"""Fixed truth-grid world-geometry comparison for changed MVE camera poses."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.quality_gate.compare_depth import (
    read_camera, read_mvei, regions_from_truth, truth_at, percentile,
    validate_fixture_camera,
)


THRESHOLDS = (1, 2, 5)


def ray(camera, x, y, shape, full_size):
    width, height = shape
    full_width, full_height = full_size
    sx, sy = width / full_width, height / full_height
    focal = camera["focal_length"][0] * full_width
    ux = ((x + .5) - camera["principal_point"][0] * full_width * sx) / (focal * sx)
    uy = ((y + .5) - camera["principal_point"][1] * full_height * sy) / (
        focal * camera["pixel_aspect"][0] * sy)
    rotation, translation = camera["rotation"], camera["translation"]
    origin = tuple(-sum(rotation[3 * row + col] * translation[row]
                        for row in range(3)) for col in range(3))
    direction = tuple(rotation[col] * ux + rotation[3 + col] * uy + rotation[6 + col]
                      for col in range(3))
    norm = math.sqrt(ux * ux + uy * uy + 1)
    return origin, direction, norm


def world_point(camera, x, y, shape, full_size, radial_depth):
    origin, direction, norm = ray(camera, x, y, shape, full_size)
    return tuple(origin[i] + radial_depth * direction[i] / norm for i in range(3))


def rectangle_distance(point, plane):
    x, y, z = point
    dx = max(plane["x"][0] - x, 0, x - plane["x"][1])
    dy = max(plane["y"][0] - y, 0, y - plane["y"][1])
    return math.sqrt(dx * dx + dy * dy + (z - plane["z"]) ** 2)


def errors_on_fixed_truth(shape, depths, camera, full_size, truth_plane_ids, planes):
    """None means no valid depth; the visible truth plane is independent of pose."""
    width, _ = shape
    errors, plane_errors = [None] * len(depths), [None] * len(depths)
    outside_valid = 0
    for i, depth in enumerate(depths):
        valid = math.isfinite(depth) and depth > 0
        plane_id = truth_plane_ids[i]
        if plane_id is None:
            outside_valid += valid
        elif valid:
            point = world_point(camera, i % width, i // width, shape, full_size, depth)
            errors[i] = rectangle_distance(point, planes[plane_id])
            plane_errors[i] = abs(point[2] - planes[plane_id]["z"])
    return errors, plane_errors, outside_valid


def score(indices, errors, plane_errors):
    values = sorted(errors[i] for i in indices if errors[i] is not None)
    z_values = sorted(plane_errors[i] for i in indices if plane_errors[i] is not None)
    population, matched = len(indices), len(values)
    if population == 0:
        return None
    return {
        "truth_pixels": population, "matched_pixels": matched,
        "missing_pixels": population - matched, "coverage": matched / population,
        "matched_rectangle_mae_mm": sum(values) / matched if matched else None,
        "matched_rectangle_p95_mm": percentile(values, .95),
        "matched_plane_z_mae_mm": sum(z_values) / matched if matched else None,
        "all_valid_rectangle_bad_fraction": {
            str(threshold): (population - matched +
                             sum(value > threshold for value in values)) / population
            for threshold in THRESHOLDS},
        "all_valid_plane_z_bad_fraction": {
            str(threshold): (population - matched +
                             sum(value > threshold for value in z_values)) / population
            for threshold in THRESHOLDS},
    }


def compare_errors(shape, truth_plane_ids, planes, full_size,
                   baseline_depths, candidate_depths, baseline_camera,
                   candidate_camera, band_pixels=2):
    if len(truth_plane_ids) != shape[0] * shape[1] or len(baseline_depths) != len(truth_plane_ids) or len(candidate_depths) != len(truth_plane_ids):
        raise ValueError("depths/truth length differs from resolution")
    regions = regions_from_truth([0 if label is not None else None for label in truth_plane_ids],
                                 shape, band_pixels, truth_plane_ids)
    if not regions["full_object"] or not regions["boundary_band"] or not regions["interior"]:
        raise ValueError("fixed truth has empty full-object, boundary, or interior population")
    baseline_error, baseline_z, baseline_outside = errors_on_fixed_truth(
        shape, baseline_depths, baseline_camera, full_size, truth_plane_ids, planes)
    candidate_error, candidate_z, candidate_outside = errors_on_fixed_truth(
        shape, candidate_depths, candidate_camera, full_size, truth_plane_ids, planes)
    result = {"baseline": {}, "candidate": {}, "shared_support_selection_biased": {},
              "outside_truth_valid": {"baseline": baseline_outside,
                                      "candidate": candidate_outside}}
    for name, indices in regions.items():
        result["baseline"][name] = score(indices, baseline_error, baseline_z)
        result["candidate"][name] = score(indices, candidate_error, candidate_z)
        shared = {i for i in indices if baseline_error[i] is not None
                  and candidate_error[i] is not None}
        if shared:
            result["shared_support_selection_biased"][name] = {
                "pixels": len(shared), "fraction_of_truth": len(shared) / len(indices),
                "baseline_rectangle_mae_mm": sum(baseline_error[i] for i in shared) / len(shared),
                "candidate_rectangle_mae_mm": sum(candidate_error[i] for i in shared) / len(shared),
            }
        else:
            result["shared_support_selection_biased"][name] = None
    base = result["baseline"]["full_object"]
    cand = result["candidate"]["full_object"]
    base_bad2 = base["all_valid_rectangle_bad_fraction"]["2"]
    cand_bad2 = cand["all_valid_rectangle_bad_fraction"]["2"]
    base_edge = result["baseline"]["boundary_band"]["all_valid_rectangle_bad_fraction"]["2"]
    cand_edge = result["candidate"]["boundary_band"]["all_valid_rectangle_bad_fraction"]["2"]
    result["provisional_checks"] = {
        "full_object_bad2_relative_reduction_at_least_25pct":
            base_bad2 > 0 and cand_bad2 <= .75 * base_bad2,
        "coverage_loss_at_most_2_percentage_points":
            cand["coverage"] >= base["coverage"] - .02,
        "boundary_bad2_no_regression": cand_edge <= base_edge,
    }
    result["provisional_relative_gate_pass"] = all(result["provisional_checks"].values())
    return result


def digest(path):
    hash_ = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hash_.update(block)
    return hash_.hexdigest()


def evaluate(baseline_depth, candidate_depth, baseline_scene, candidate_scene,
             fixture, view, band_pixels=2):
    bw, bh, baseline = read_mvei(baseline_depth)
    cw, ch, candidate = read_mvei(candidate_depth)
    if (bw, bh) != (cw, ch):
        raise ValueError(f"same-grid world comparison requires identical resolution: {(bw,bh)} vs {(cw,ch)}")
    project = json.loads((fixture / "project.json").read_text())
    sparse = json.loads((fixture / "sparse-report.json").read_text())
    truth = json.loads((fixture / "truth.json").read_text())
    if project["units"] != "mm" or sparse["units"] != "mm":
        raise ValueError("fixture must use millimetres")
    if not 0 <= view < len(sparse["views"]):
        raise ValueError("view index out of bounds")
    image_id = sparse["views"][view]["imageId"]
    truth_views = {entry["imageId"]: entry for entry in truth["views"]}
    truth_view = truth_views[image_id]
    calibration = project["calibration"]
    if any(abs(item) > 1e-12 for item in calibration["distortion"]):
        raise ValueError("distorted fixture unsupported")
    full_size = (calibration["width"], calibration["height"])
    base_camera = read_camera(baseline_scene, view)
    cand_camera = read_camera(candidate_scene, view)
    validate_fixture_camera(base_camera, project, sparse, view)
    expected = ((calibration["fx"] / full_size[0],),
                (calibration["fy"] / calibration["fx"],),
                ((calibration["cx"] + .5) / full_size[0],
                 (calibration["cy"] + .5) / full_size[1]))
    for camera in (base_camera, cand_camera):
        for key, values in zip(("focal_length", "pixel_aspect", "principal_point"), expected):
            if any(not math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
                   for a, b in zip(camera[key], values)):
                raise ValueError(f"scene intrinsics differ from fixed fixture: {key}")
    truth_camera = {
        "focal_length": expected[0], "pixel_aspect": expected[1],
        "principal_point": expected[2],
        "rotation": tuple(truth_view["rotation"]),
        "translation": tuple(truth_view["translationMm"]),
    }
    shape = (bw, bh)
    planes = truth["objectPlanes"]
    hits = [truth_at(x, y, shape, truth_camera, full_size, planes, with_plane=True)
            for y in range(bh) for x in range(bw)]
    labels = [plane_id for _, plane_id in hits]
    result = compare_errors(shape, labels, planes, full_size,
                            baseline, candidate, base_camera, cand_camera,
                            band_pixels)
    files = {"baseline_depth": baseline_depth, "candidate_depth": candidate_depth,
             "baseline_meta": baseline_scene / "views" / f"view_{view:04d}.mve" / "meta.ini",
             "candidate_meta": candidate_scene / "views" / f"view_{view:04d}.mve" / "meta.ini",
             "project": fixture / "project.json", "sparse_report": fixture / "sparse-report.json",
             "truth": fixture / "truth.json"}
    scorer_sources = {"compare_world.py": Path(__file__),
                      "compare_depth.py": Path(__file__).with_name("compare_depth.py")}
    result.update({"status": "same_grid_changed_pose_allowed", "view": image_id,
                   "shape": [bw, bh], "band_pixels": band_pixels,
                   "metric": "3D Euclidean distance to finite visible truth rectangle in object frame; plane-Z diagnostic also reported",
                   "truth_eligibility": "true camera and visible synthetic planes, fixed before scoring either output",
                   "input_sha256": {name: digest(path) for name, path in files.items()},
                   "input_paths": {name: str(path) for name, path in files.items()},
                   "scorer_sha256": {name: digest(path) for name, path in scorer_sources.items()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-depth", "candidate-depth", "baseline-scene",
                 "candidate-scene", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--view", type=int, required=True)
    parser.add_argument("--band-pixels", type=int, default=2)
    args = parser.parse_args()
    if args.band_pixels < 1:
        parser.error("--band-pixels must be positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    try:
        result = evaluate(args.baseline_depth, args.candidate_depth,
                          args.baseline_scene, args.candidate_scene,
                          args.fixture, args.view, args.band_pixels)
    except ValueError as error:
        parser.error(str(error))
    with args.output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline": result["baseline"], "candidate": result["candidate"],
                      "provisional_checks": result["provisional_checks"]}, indent=2))


if __name__ == "__main__":
    main()
