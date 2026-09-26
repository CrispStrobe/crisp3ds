#!/usr/bin/env python3
"""Score a single MVE depth image against synthetic object planes only."""

from array import array
import argparse
import json
import math
from pathlib import Path
import struct


def read_mvei_float(path):
    with path.open("rb") as source:
        header = source.read(27)
        if len(header) != 27 or header[:11] != b"\x89MVE_IMAGE\n":
            raise ValueError("invalid MVEI signature")
        width, height, channels, image_type = struct.unpack("=4i", header[11:])
        if width <= 0 or height <= 0 or width * height > 10_000_000 or channels != 1 or image_type != 9:
            raise ValueError("expected bounded single-channel float MVEI")
        payload = source.read(width * height * 4 + 1)
    if len(payload) != width * height * 4:
        raise ValueError("MVEI payload length mismatch")
    values = array("f")
    values.frombytes(payload)
    return width, height, values


def truth_depth(x, y, width, height, camera, planes):
    k, rotation, translation, (full_width, full_height) = camera
    # MVE fills K at original resolution; scale its rows to depth resolution.
    sx, sy = width / full_width, height / full_height
    rx = ((x + 0.5) - sx * k[2]) / (sx * k[0])
    ry = ((y + 0.5) - sy * k[5]) / (sy * k[4])
    origin = [-sum(rotation[3 * j + i] * translation[j] for j in range(3)) for i in range(3)]
    ray = [rotation[i] * rx + rotation[3 + i] * ry + rotation[6 + i] for i in range(3)]
    if abs(ray[2]) < 1e-12:
        return None
    candidates = []
    for plane in planes:
        depth = (plane["z"] - origin[2]) / ray[2]
        px, py = origin[0] + depth * ray[0], origin[1] + depth * ray[1]
        if depth > 0 and plane["x"][0] <= px <= plane["x"][1] and plane["y"][0] <= py <= plane["y"][1]:
            candidates.append(depth)
    # MVE's pixel_3dpos() normalizes the inverse-K ray, and DMRecon seeds
    # depth from Euclidean camera-to-feature distance (not camera Z).
    return min(candidates) * math.sqrt(rx * rx + ry * ry + 1) if candidates else None


def evaluate(depth_path, scene, fixture, view_index):
    width, height, depths = read_mvei_float(depth_path)
    project = json.loads((fixture / "project.json").read_text())
    sparse = json.loads((fixture / "sparse-report.json").read_text())
    truth = json.loads((fixture / "truth.json").read_text())
    calibration = project["calibration"]
    if view_index < 0 or view_index >= len(sparse["views"]):
        raise ValueError("view index out of bounds")
    view = sparse["views"][view_index]
    # Use the checked MVE scene metadata, rather than truth camera poses.
    meta = (scene / "views" / f"view_{view_index:04d}.mve" / "meta.ini").read_text()
    fields = {}
    for line in meta.splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            fields[key] = value
    rotation = list(map(float, fields["rotation"].split()))
    translation = list(map(float, fields["translation"].split()))
    flen = float(fields["focal_length"])
    paspect = float(fields["pixel_aspect"])
    ppx, ppy = map(float, fields["principal_point"].split())
    full_width, full_height = calibration["width"], calibration["height"]
    if (full_width / full_height) * paspect < 1:
        raise ValueError("unsupported portrait calibration")
    k = [flen * full_width, 0, ppx * full_width, 0,
         flen * full_width * paspect, ppy * full_height, 0, 0, 1]
    from collections import namedtuple
    Camera = namedtuple("Camera", "k rotation translation full_size")
    camera = Camera(k, rotation, translation, (full_width, full_height))
    counts = {"object_truth_pixels": 0, "valid_on_object": 0, "valid_outside_object": 0,
              "invalid_on_object": 0, "nonfinite_depths": 0}
    absolute_errors = []
    for y in range(height):
        for x in range(width):
            target = truth_depth(x, y, width, height, camera, truth["objectPlanes"])
            prediction = depths[y * width + x]
            valid = math.isfinite(prediction) and prediction > 0
            if not math.isfinite(prediction):
                counts["nonfinite_depths"] += 1
            if target is not None:
                counts["object_truth_pixels"] += 1
                if valid:
                    counts["valid_on_object"] += 1
                    absolute_errors.append(abs(prediction - target))
                else:
                    counts["invalid_on_object"] += 1
            elif valid:
                counts["valid_outside_object"] += 1
    absolute_errors.sort()
    matched = len(absolute_errors)
    return {"view": view["imageId"], "width": width, "height": height, "counts": counts,
            "object_coverage": counts["valid_on_object"] / counts["object_truth_pixels"]
            if counts["object_truth_pixels"] else None,
            "mae_on_object_mm": sum(absolute_errors) / matched if matched else None,
            "median_abs_error_on_object_mm": absolute_errors[matched // 2] if matched else None,
            "bad5_on_object": sum(error > 5 for error in absolute_errors) / matched if matched else None,
            "false_positive_scope": "valid depth outside the two rendered object planes; mask not enforced",
            "depth_convention": "Euclidean camera-to-point distance along normalized inverse-K ray (mm)",
            "camera_source": str(scene / "views" / f"view_{view_index:04d}.mve" / "meta.ini")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--view", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    result = evaluate(args.depth, args.scene, args.fixture, args.view)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
