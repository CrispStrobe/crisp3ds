#!/usr/bin/env python3
"""Read-only patch agreement diagnostic for measured tree seed tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO / "build-opencv/tree-dense/scene"
NEIGHBORS = {1: (0, 4, 8), 4: (0, 1, 3)}  # Frozen dmrecon-{1,4}.log selections.
THRESHOLDS = (0.3, 0.6, 0.8)  # Diagnostic bins, not acceptance thresholds.
PATCH_RADIUS = 3


def project(point: tuple[float, float, float], view: dict) -> tuple[float, float] | None:
    rotation = view["rotation"]
    translation = view["translation"]
    camera = view["camera"]
    x, y, z = (sum(rotation[3*axis + column] * point[column] for column in range(3))
               + translation[axis] for axis in range(3))
    if not all(math.isfinite(value) for value in (x, y, z)) or z <= 0:
        return None
    return camera["fx"] * x / z + camera["cx"], camera["fy"] * y / z + camera["cy"]


def bilinear(image: Image.Image, x: float, y: float) -> float | None:
    width, height = image.size
    if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return None
    ix, iy = math.floor(x), math.floor(y)
    dx, dy = x - ix, y - iy
    p = image.load()
    return ((1-dx)*(1-dy)*p[ix, iy] + dx*(1-dy)*p[ix+1, iy]
            + (1-dx)*dy*p[ix, iy+1] + dx*dy*p[ix+1, iy+1])


def patch(image: Image.Image, x: float, y: float, radius: int = PATCH_RADIUS) -> list[float] | None:
    samples = [bilinear(image, x + dx, y + dy)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    return None if any(value is None for value in samples) else samples


def zncc(first: list[float] | None, second: list[float] | None) -> float | None:
    if first is None or second is None or len(first) != len(second) or not first:
        return None
    ma, mb = statistics.fmean(first), statistics.fmean(second)
    aa = [value - ma for value in first]
    bb = [value - mb for value in second]
    va = sum(value*value for value in aa)
    vb = sum(value*value for value in bb)
    if va <= 1e-8 or vb <= 1e-8:
        return None
    return sum(a*b for a, b in zip(aa, bb)) / math.sqrt(va*vb)


def read_seeds(path: Path) -> list[dict]:
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(f"seed line {number}: expected 11 fields")
        point = tuple(map(float, fields[:3]))
        first, second = int(fields[3]), int(fields[6])
        measured = {first: (float(fields[4]), float(fields[5])),
                    second: (float(fields[7]), float(fields[8]))}
        if first == second or not all(math.isfinite(value) for value in point + measured[first] + measured[second]):
            raise ValueError(f"seed line {number}: invalid observation")
        result.append({"line": number, "point": point, "pair": (first, second), "measured": measured})
    return result


def tally(values: list[float | None]) -> dict:
    valid = [value for value in values if value is not None]
    return {"valid": len(valid), "invalid": len(values)-len(valid),
            "median": statistics.median(valid) if valid else None,
            "at_least_0_3": sum(value >= .3 for value in valid),
            "at_least_0_6": sum(value >= .6 for value in valid),
            "at_least_0_8": sum(value >= .8 for value in valid)}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(scene: Path) -> dict:
    source_paths = [Path(__file__), scene / "conversion.json", scene / "measured-seeds.txt"]
    source_paths += [scene / "views" / f"view_{index:04d}.mve" / "undistorted.png" for index in range(10)]
    before = {str(path): file_hash(path) for path in source_paths}
    conversion = json.loads((scene / "conversion.json").read_text())
    views = conversion["views"]
    images = []
    for index, view in enumerate(views):
        # Keep reads inside the frozen scene even if conversion.json has paths.
        image_path = scene / "views" / f"view_{index:04d}.mve" / "undistorted.png"
        with Image.open(image_path) as source:
            image = source.convert("L")
        if image.size != (view["camera"]["width"], view["camera"]["height"]):
            raise ValueError(f"view {index} image and camera dimensions differ")
        images.append(image)
    seeds = read_seeds(scene / "measured-seeds.txt")
    report = {"scene": str(scene), "source_sha256": before,
              "total_pair_seed_records": len(seeds),
              "note": "Pair-derived seeds are correlated observations, not independent 3D ground truth.",
              "patch": "7x7 bilinear grayscale axis-aligned ZNCC; constant or clipped patches invalid. MVE uses warped patches and different scale/normal handling.",
              "references": {}}
    for ref, neighbors in NEIGHBORS.items():
        ref_seeds = [seed for seed in seeds if ref in seed["pair"]]
        measured_all_scores = []
        measured_selected_scores = []
        matched_pose_scores = []
        pose_scores = {neighbor: [] for neighbor in neighbors}
        support = {str(threshold): [0] * (len(neighbors)+1) for threshold in THRESHOLDS}
        valid_neighbor_counts = []
        measured_pair_counts = {}
        for seed in ref_seeds:
            measured_ref = seed["measured"][ref]
            reference_measured_patch = patch(images[ref], *measured_ref)
            paired = next(view for view in seed["pair"] if view != ref)
            measured_pair_counts[str(paired)] = measured_pair_counts.get(str(paired), 0) + 1
            measured_score = zncc(reference_measured_patch, patch(images[paired], *seed["measured"][paired]))
            measured_all_scores.append(measured_score)
            projected_ref = project(seed["point"], views[ref])
            projected_ref_patch = patch(images[ref], *projected_ref) if projected_ref else None
            row = []
            for neighbor in neighbors:
                projected = project(seed["point"], views[neighbor])
                score = zncc(projected_ref_patch, patch(images[neighbor], *projected)) if projected else None
                pose_scores[neighbor].append(score)
                row.append(score)
            if paired in neighbors:
                measured_selected_scores.append(measured_score)
                matched_pose_scores.append(row[neighbors.index(paired)])
            valid_neighbor_counts.append(sum(value is not None for value in row))
            for threshold in THRESHOLDS:
                support[str(threshold)][sum(value is not None and value >= threshold for value in row)] += 1
        report["references"][str(ref)] = {
            "global_neighbors": list(neighbors),
            "seed_records_seen_in_reference": len(ref_seeds),
            "measured_pair_neighbor_counts": measured_pair_counts,
            "measured_pair_zncc_all": tally(measured_all_scores),
            "measured_pair_selected_neighbor_records": len(measured_selected_scores),
            "measured_pair_excluded_non_selected_neighbor_records": len(ref_seeds)-len(measured_selected_scores),
            "measured_pair_zncc_selected": tally(measured_selected_scores),
            "same_seed_same_neighbor_pose_projection_zncc": tally(matched_pose_scores),
            "pose_projection_zncc_by_neighbor": {str(neighbor): tally(pose_scores[neighbor]) for neighbor in neighbors},
            "valid_projected_neighbor_count_distribution": [valid_neighbor_counts.count(count) for count in range(len(neighbors)+1)],
            "pose_support_count_distribution_by_threshold": support,
        }
    after = {str(path): file_hash(path) for path in source_paths}
    if before != after:
        raise RuntimeError("diagnostic input changed during analysis")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    print(json.dumps(analyze(args.scene), indent=2))
