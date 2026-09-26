"""Geometry scores for the immutable synthetic MVE fixture (millimetres)."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mve_spike"))
from evaluate_depth import read_mvei_float


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo = int(index)
    return values[lo] * (1 - (index - lo)) + values[min(lo + 1, len(values) - 1)] * (index - lo)


def ray(camera, x, y, width, height, full_width, full_height, calibration):
    # MVE K has a +0.5 principal-point offset; pixel centres also have +0.5.
    sx, sy = width / full_width, height / full_height
    rx = ((x + .5) - sx * (calibration["cx"] + .5)) / (sx * calibration["fx"])
    ry = ((y + .5) - sy * (calibration["cy"] + .5)) / (sy * calibration["fy"])
    R, t = camera["rotation"], camera["translationMm"]
    origin = tuple(-sum(R[3*j+i] * t[j] for j in range(3)) for i in range(3))
    direction = tuple(R[i] * rx + R[3+i] * ry + R[6+i] for i in range(3))
    norm = math.sqrt(rx*rx + ry*ry + 1)
    return origin, direction, norm


def hit(camera, x, y, width, height, calibration, planes):
    origin, direction, norm = ray(camera, x, y, width, height,
                                  calibration["width"], calibration["height"], calibration)
    candidates = []
    for plane in planes:
        if abs(direction[2]) < 1e-12:
            continue
        t = (plane["z"] - origin[2]) / direction[2]
        px, py = origin[0] + t*direction[0], origin[1] + t*direction[1]
        if t > 0 and plane["x"][0] <= px <= plane["x"][1] and plane["y"][0] <= py <= plane["y"][1]:
            # Nearest visible plane and distance to its physical rectangle boundary.
            edge = min(px-plane["x"][0], plane["x"][1]-px,
                       py-plane["y"][0], plane["y"][1]-py)
            candidates.append((t, plane, edge, norm))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def score_values(errors, truth_count, valid_outside, edge_truth, edge_valid):
    count = len(errors)
    return {"truth_pixels": truth_count, "matched_pixels": count,
            "missing_pixels": truth_count-count, "outside_truth_valid_pixels": valid_outside,
            "coverage": count/truth_count if truth_count else None,
            "mae_mm": sum(errors)/count if count else None,
            "p50_mm": percentile(errors, .5), "p95_mm": percentile(errors, .95),
            "bad5_matched": sum(e > 5 for e in errors)/count if count else None,
            "bad5_missing_inclusive": (truth_count-count+sum(e > 5 for e in errors))/truth_count if truth_count else None,
            "edge_truth_pixels": edge_truth, "edge_matched_pixels": edge_valid,
            "edge_coverage": edge_valid/edge_truth if edge_truth else None}


def score_depth(path, camera, truth_camera, calibration, planes, edge_mm=3):
    width, height, depths = read_mvei_float(path)
    errors, rectangle_errors, edge_errors = [], [], []
    truth_count, outside, edge_truth, edge_valid = 0, 0, 0, 0
    per_pixel = {}
    for y in range(height):
        for x in range(width):
            ref = hit(truth_camera, x, y, width, height, calibration, planes)
            d = depths[y*width+x]
            valid = math.isfinite(d) and d > 0
            if ref is None:
                outside += valid
                continue
            truth_count += 1
            edge = ref[2] <= edge_mm
            edge_truth += edge
            if not valid:
                continue
            origin, direction, norm = ray(camera, x, y, width, height,
                                          calibration["width"], calibration["height"], calibration)
            # The MVE depth is Euclidean along the inverse-K ray. A world-space
            # point-to-plane residual remains comparable across pose choices.
            xyz = [origin[i] + d * direction[i] / norm for i in range(3)]
            z = xyz[2]
            error = abs(z-ref[1]["z"])
            errors.append(error)
            plane = ref[1]
            dx = max(plane["x"][0]-xyz[0], 0, xyz[0]-plane["x"][1])
            dy = max(plane["y"][0]-xyz[1], 0, xyz[1]-plane["y"][1])
            rectangle_errors.append(math.sqrt(dx*dx+dy*dy+error*error))
            per_pixel[(x,y)] = error
            edge_valid += edge
            if edge: edge_errors.append(error)
    result = score_values(errors, truth_count, outside, edge_truth, edge_valid)
    result.update({"rectangle_mae_mm":sum(rectangle_errors)/len(rectangle_errors) if rectangle_errors else None,
                   "rectangle_p50_mm":percentile(rectangle_errors,.5),
                   "rectangle_p95_mm":percentile(rectangle_errors,.95),
                   "rectangle_bad5_missing_inclusive":
                   (truth_count-len(rectangle_errors)+sum(e>5 for e in rectangle_errors))/truth_count if truth_count else None,
                   "edge_mae_mm":sum(edge_errors)/len(edge_errors) if edge_errors else None,
                   "edge_bad5_matched":sum(e>5 for e in edge_errors)/len(edge_errors) if edge_errors else None})
    result.update({"width": width, "height": height, "edge_band_mm": edge_mm,
                   "error_definition": "world-Z point-to-visible-truth-plane; also finite-rectangle Euclidean distance"})
    return result, per_pixel


def paired_grid(low_pixels, high_pixels, low_width, low_height):
    """Same L2 footprint: compare L2 sample to mean of its valid L1 2x2 children."""
    low_errors, high_errors = [], []
    both = 0
    for y in range(low_height):
        for x in range(low_width):
            a = low_pixels.get((x,y))
            children = [high_pixels.get((2*x+dx, 2*y+dy)) for dy in range(2) for dx in range(2)]
            valid = [e for e in children if e is not None]
            if a is not None and valid:
                both += 1
                low_errors.append(a)
                high_errors.append(sum(valid)/len(valid))
    return {"paired_l2_cells": both, "l2_mae_mm": sum(low_errors)/both if both else None,
            "l1_2x2_mean_mae_mm": sum(high_errors)/both if both else None,
            "definition": "L2 truth cells valid in L2 and at least one L1 child; L1 error is mean of valid 2x2 child errors"}
