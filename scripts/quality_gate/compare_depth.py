#!/usr/bin/env python3
"""Compare paired MVE radial depths to synthetic planes, in millimetres."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct


THRESHOLDS_MM = (1, 2, 5)


def read_mvei(path):
    with path.open("rb") as source:
        header = source.read(27)
        if len(header) != 27 or header[:11] != b"\x89MVE_IMAGE\n":
            raise ValueError(f"invalid MVEI signature: {path}")
        width, height, channels, kind = struct.unpack("=4i", header[11:])
        if not (0 < width <= 10000 and 0 < height <= 10000 and
                width * height <= 10_000_000 and channels == 1 and kind == 9):
            raise ValueError(f"expected bounded one-channel float MVEI: {path}")
        payload = source.read(width * height * 4 + 1)
    if len(payload) != width * height * 4:
        raise ValueError(f"MVEI payload length mismatch: {path}")
    values = array("f")
    values.frombytes(payload)
    return width, height, values


def read_camera(scene, view):
    path = scene / "views" / f"view_{view:04d}.mve" / "meta.ini"
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    camera = {}
    for key, length in (("rotation", 9), ("translation", 3),
                        ("principal_point", 2), ("radial_distortion", 2),
                        ("focal_length", 1), ("pixel_aspect", 1)):
        numbers = tuple(float(item) for item in fields[key].split())
        if len(numbers) != length or not all(math.isfinite(item) for item in numbers):
            raise ValueError(f"invalid camera {key}: {path}")
        camera[key] = numbers
    if camera["focal_length"][0] <= 0 or camera["pixel_aspect"][0] <= 0:
        raise ValueError(f"invalid camera focal/aspect: {path}")
    rotation = camera["rotation"]
    rows = [rotation[i:i + 3] for i in (0, 3, 6)]
    for i in range(3):
        for j in range(3):
            dot = sum(rows[i][k] * rows[j][k] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-3:
                raise ValueError(f"nonorthogonal MVE camera rotation: {path}")
    cross = (rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1],
             rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2],
             rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0])
    if sum(cross[k] * rows[2][k] for k in range(3)) < 0.999:
        raise ValueError(f"reflected MVE camera rotation: {path}")
    if any(abs(value) > 1e-12 for value in camera["radial_distortion"]):
        raise ValueError("distorted MVE images are unsupported by plane scorer")
    return camera


def require_paired(base_shape, candidate_shape, base_camera, candidate_camera):
    if base_shape != candidate_shape:
        raise ValueError(f"unpaired depth resolutions: {base_shape} versus {candidate_shape}")
    for key in base_camera:
        if key not in candidate_camera or len(base_camera[key]) != len(candidate_camera[key]):
            raise ValueError(f"unpaired MVE camera metadata: {key} missing or malformed")
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
               for a, b in zip(base_camera[key], candidate_camera[key])):
            raise ValueError(f"unpaired MVE camera metadata: {key} differs")


def validate_fixture_camera(camera, project, sparse, view):
    calibration = project["calibration"]
    if project["units"] != "mm" or sparse["units"] != "mm":
        raise ValueError("fixture and sparse poses must use millimetres")
    if any(abs(value) > 1e-12 for value in calibration["distortion"]):
        raise ValueError("distorted fixture unsupported")
    width, height = calibration["width"], calibration["height"]
    expected = {
        "focal_length": (calibration["fx"] / width,),
        "pixel_aspect": (calibration["fy"] / calibration["fx"],),
        "principal_point": ((calibration["cx"] + 0.5) / width,
                            (calibration["cy"] + 0.5) / height),
        "rotation": tuple(sparse["views"][view]["rotation"]),
        "translation": tuple(sparse["views"][view]["translationMm"]),
    }
    for key, values in expected.items():
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-7)
               for a, b in zip(camera[key], values)):
            raise ValueError(f"MVE camera disagrees with estimated sparse camera: {key}")
    return width, height


def truth_at(x, y, shape, camera, original_size, planes, with_plane=False):
    """Nearest positive ray/rectangle hit; convert camera Z to radial range."""
    width, height = shape
    full_width, full_height = original_size
    scale_x, scale_y = width / full_width, height / full_height
    focal = camera["focal_length"][0] * full_width
    fx = focal * scale_x
    fy = focal * camera["pixel_aspect"][0] * scale_y
    cx = camera["principal_point"][0] * full_width * scale_x
    cy = camera["principal_point"][1] * full_height * scale_y
    ux, uy = ((x + 0.5) - cx) / fx, ((y + 0.5) - cy) / fy
    rotation, translation = camera["rotation"], camera["translation"]
    # MVE rotation is object-to-camera. Transpose it to obtain an object ray.
    origin = tuple(-sum(rotation[3 * row + col] * translation[row]
                        for row in range(3)) for col in range(3))
    ray = tuple(rotation[col] * ux + rotation[3 + col] * uy + rotation[6 + col]
                for col in range(3))
    if abs(ray[2]) < 1e-12:
        return None
    nearest, nearest_plane = None, None
    for plane_index, plane in enumerate(planes):
        distance_z = (plane["z"] - origin[2]) / ray[2]
        px = origin[0] + distance_z * ray[0]
        py = origin[1] + distance_z * ray[1]
        if (distance_z > 0 and plane["x"][0] <= px <= plane["x"][1] and
                plane["y"][0] <= py <= plane["y"][1]):
            radial = distance_z * math.sqrt(ux * ux + uy * uy + 1)
            if nearest is None or radial < nearest:
                nearest, nearest_plane = radial, plane_index
    return (nearest, nearest_plane) if with_plane else nearest


def expand_band(seed, object_indices, shape, band_pixels):
    width, height = shape
    band = set(seed)
    frontier = seed
    for _ in range(1, band_pixels):
        next_frontier = set()
        for i in frontier:
            x, y = i % width, i // width
            for neighbor in ((i - 1 if x else None),
                             (i + 1 if x < width - 1 else None),
                             (i - width if y else None),
                             (i + width if y < height - 1 else None)):
                if neighbor in object_indices and neighbor not in band:
                    next_frontier.add(neighbor)
        band.update(next_frontier)
        frontier = next_frontier
    return band


def regions_from_truth(truth, shape, band_pixels=2, plane_ids=None):
    """Fixed inside-object silhouette and surface-step bands."""
    width, height = shape
    object_indices = {i for i, value in enumerate(truth) if value is not None}
    silhouette = set()
    depth_step = set()
    for i in object_indices:
        x, y = i % width, i // width
        neighbors = ((i - 1 if x else None),
                     (i + 1 if x < width - 1 else None),
                     (i - width if y else None),
                     (i + width if y < height - 1 else None))
        if any(n not in object_indices for n in neighbors):
            silhouette.add(i)
        if plane_ids is not None and any(n in object_indices and
                                         plane_ids[n] != plane_ids[i] for n in neighbors):
            depth_step.add(i)
    silhouette_band = expand_band(silhouette, object_indices, shape, band_pixels)
    depth_step_band = expand_band(depth_step, object_indices, shape, band_pixels)
    band = silhouette_band | depth_step_band
    return {"full_object": object_indices, "boundary_band": band,
            "silhouette_band": silhouette_band, "depth_step_band": depth_step_band,
            "interior": object_indices - band}


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    return sorted_values[math.ceil(fraction * len(sorted_values)) - 1]


def score_region(indices, truth, depths):
    errors = []
    missing = 0
    for i in indices:
        prediction = depths[i]
        if math.isfinite(prediction) and prediction > 0:
            errors.append(abs(prediction - truth[i]))
        else:
            missing += 1
    errors.sort()
    count = len(indices)
    matched = len(errors)
    if not count:
        raise ValueError("empty truth region; cannot score")
    return {
        "truth_pixels": count, "valid_pixels": matched, "missing_pixels": missing,
        "coverage": matched / count,
        "matched_mae_mm": sum(errors) / matched if matched else None,
        "matched_p95_mm": percentile(errors, 0.95),
        "all_valid_bad_fraction": {
            str(t): (missing + sum(error > t for error in errors)) / count
            for t in THRESHOLDS_MM},
    }


def compare_arrays(shape, truth, baseline, candidate, band_pixels=2, plane_ids=None):
    if len(truth) != shape[0] * shape[1] or len(baseline) != len(truth) or len(candidate) != len(truth):
        raise ValueError("array size and image resolution mismatch")
    regions = regions_from_truth(truth, shape, band_pixels, plane_ids)
    base_scores, candidate_scores, shared_scores = {}, {}, {}
    for name, indices in regions.items():
        if not indices and name == "depth_step_band":
            base_scores[name] = candidate_scores[name] = shared_scores[name] = None
            continue
        if not indices:
            raise ValueError(f"empty {name} region; lower band width or use another view")
        base_scores[name] = score_region(indices, truth, baseline)
        candidate_scores[name] = score_region(indices, truth, candidate)
        shared = {i for i in indices if math.isfinite(baseline[i]) and baseline[i] > 0
                  and math.isfinite(candidate[i]) and candidate[i] > 0}
        if shared:
            shared_scores[name] = {
                "pixels": len(shared), "fraction_of_truth": len(shared) / len(indices),
                "baseline_matched_mae_mm": score_region(shared, truth, baseline)["matched_mae_mm"],
                "candidate_matched_mae_mm": score_region(shared, truth, candidate)["matched_mae_mm"],
            }
        else:
            shared_scores[name] = {"pixels": 0, "fraction_of_truth": 0,
                                   "baseline_matched_mae_mm": None,
                                   "candidate_matched_mae_mm": None}
    base_full = base_scores["full_object"]
    cand_full = candidate_scores["full_object"]
    base_bad2 = base_full["all_valid_bad_fraction"]["2"]
    cand_bad2 = cand_full["all_valid_bad_fraction"]["2"]
    edge_base = base_scores["boundary_band"]["all_valid_bad_fraction"]["2"]
    edge_cand = candidate_scores["boundary_band"]["all_valid_bad_fraction"]["2"]
    checks = {
        "all_valid_bad2_relative_reduction_at_least_25pct":
            base_bad2 > 0 and cand_bad2 <= 0.75 * base_bad2,
        "coverage_loss_at_most_2_percentage_points":
            cand_full["coverage"] >= base_full["coverage"] - 0.02,
        "boundary_all_valid_bad2_no_regression": edge_cand <= edge_base,
    }
    outside = set(range(len(truth))) - regions["full_object"]
    outside_counts = {
        "baseline_valid": sum(math.isfinite(baseline[i]) and baseline[i] > 0 for i in outside),
        "candidate_valid": sum(math.isfinite(candidate[i]) and candidate[i] > 0 for i in outside),
        "outside_truth_pixels": len(outside),
    }
    return {"baseline": base_scores, "candidate": candidate_scores,
            "shared_support_selection_biased": shared_scores,
            "outside_object": outside_counts, "checks": checks,
            "provisional_relative_gate_pass": all(checks.values())}


def evaluate(baseline_depth, candidate_depth, baseline_scene, candidate_scene,
             fixture, view, band_pixels=2, binaries=None):
    baseline_width, baseline_height, baseline = read_mvei(baseline_depth)
    candidate_width, candidate_height, candidate = read_mvei(candidate_depth)
    base_camera = read_camera(baseline_scene, view)
    candidate_camera = read_camera(candidate_scene, view)
    shape = (baseline_width, baseline_height)
    require_paired(shape, (candidate_width, candidate_height), base_camera, candidate_camera)
    project = json.loads((fixture / "project.json").read_text())
    sparse = json.loads((fixture / "sparse-report.json").read_text())
    ground_truth = json.loads((fixture / "truth.json").read_text())
    if not 0 <= view < len(sparse["views"]):
        raise ValueError("view index out of bounds")
    original_size = validate_fixture_camera(base_camera, project, sparse, view)
    planes = ground_truth["objectPlanes"]
    hits = [truth_at(x, y, shape, base_camera, original_size, planes, with_plane=True)
            for y in range(baseline_height) for x in range(baseline_width)]
    truth = [hit[0] for hit in hits]
    plane_ids = [hit[1] for hit in hits]
    result = compare_arrays(shape, truth, baseline, candidate, band_pixels, plane_ids)
    sources = {"baseline_depth": baseline_depth, "candidate_depth": candidate_depth,
               "baseline_meta": baseline_scene / "views" / f"view_{view:04d}.mve" / "meta.ini",
               "candidate_meta": candidate_scene / "views" / f"view_{view:04d}.mve" / "meta.ini",
               "project": fixture / "project.json", "sparse_report": fixture / "sparse-report.json",
               "truth": fixture / "truth.json"}
    for name, scene in (("baseline_conversion", baseline_scene),
                        ("candidate_conversion", candidate_scene)):
        conversion = scene / "conversion.json"
        if conversion.exists():
            sources[name] = conversion
    sources.update(binaries or {})
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in sources.items()}
    result["input_paths"] = {name: str(path) for name, path in sources.items()}
    result.update({"status": "paired", "view": sparse["views"][view]["imageId"],
                   "shape": [baseline_width, baseline_height],
                   "band_pixels": band_pixels,
                   "depth_convention": "Euclidean radial range along normalized camera ray; mm",
                   "truth": "nearest intersection with synthetic object rectangles, using estimated MVE camera",
                   "baseline_depth": str(baseline_depth), "candidate_depth": str(candidate_depth),
                   "baseline_camera": str(baseline_scene / "views" / f"view_{view:04d}.mve" / "meta.ini"),
                   "candidate_camera": str(candidate_scene / "views" / f"view_{view:04d}.mve" / "meta.ini")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-depth", "candidate-depth", "baseline-scene",
                 "candidate-scene", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--view", type=int, required=True)
    parser.add_argument("--band-pixels", type=int, default=2)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--candidate-binary", type=Path)
    args = parser.parse_args()
    if args.band_pixels < 1:
        parser.error("--band-pixels must be positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    try:
        result = evaluate(args.baseline_depth, args.candidate_depth,
                          args.baseline_scene, args.candidate_scene,
                          args.fixture, args.view, args.band_pixels,
                          {name: path for name, path in
                           (("baseline_binary", args.baseline_binary),
                            ("candidate_binary", args.candidate_binary)) if path})
    except ValueError as error:
        parser.error(str(error))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
