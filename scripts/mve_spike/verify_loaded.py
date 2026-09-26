#!/usr/bin/env python3
"""Verify selected MVE's loaded calibration and seed reprojections."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def verify(inspector, scene, fixture):
    lines = subprocess.check_output([str(inspector.resolve()), str(scene.resolve())], text=True).splitlines()
    sparse = json.loads((fixture / "sparse-report.json").read_text())
    project = json.loads((fixture / "project.json").read_text())
    calibration = project["calibration"]
    bundle = [line for line in lines if line.startswith("BUNDLE ")]
    if len(bundle) != 1 or list(map(int, bundle[0].split()[1:])) != [len(sparse["views"]), len(sparse["points"])]:
        raise ValueError("MVE bundle camera/seed count differs")
    loaded = {}
    for line in lines:
        if not line.startswith("VIEW "):
            continue
        values = [float(token) for token in line.split()[1:]]
        if len(values) != 24:
            raise ValueError("unexpected loaded view record")
        index = int(values[0])
        width, height = values[1:3]
        k, rotation, translation = values[3:12], values[12:21], values[21:24]
        if (width, height) != (calibration["width"], calibration["height"]):
            raise ValueError("loaded image size differs")
        expected_k = [calibration["fx"], 0, calibration["cx"] + 0.5,
                      0, calibration["fy"], calibration["cy"] + 0.5, 0, 0, 1]
        if max(abs(a - b) for a, b in zip(k, expected_k)) > 0.001:
            raise ValueError("MVE loaded K differs from expected pixel-center convention")
        view = sparse["views"][index]
        if max(abs(a - b) for a, b in zip(rotation, view["rotation"])) > 0.001 or \
                max(abs(a - b) for a, b in zip(translation, view["translationMm"])) > 0.001:
            raise ValueError("MVE loaded object-to-camera pose differs")
        loaded[view["imageId"]] = (k, rotation, translation)
    if len(loaded) != len(sparse["views"]):
        raise ValueError("MVE loaded view count differs")
    errors = []
    for point in sparse["points"]:
        for observation in point["observations"]:
            k, rotation, translation = loaded[observation["imageId"]]
            x, y, z = [sum(rotation[3 * row + col] * point["positionMm"][col] for col in range(3))
                       + translation[row] for row in range(3)]
            if z <= 0:
                raise ValueError("seed behind loaded camera")
            u, v = k[0] * x / z + k[2] - 0.5, k[4] * y / z + k[5] - 0.5
            errors.append(math.hypot(u - observation["pixel"][0], v - observation["pixel"][1]))
    result = {"loaded_views": len(loaded), "loaded_seeds": len(sparse["points"]),
              "observations": len(errors),
              "reprojection_rms_px": math.sqrt(sum(error * error for error in errors) / len(errors)),
              "reprojection_max_px": max(errors),
              "pixel_convention": "MVE K principal point includes +0.5; subtract 0.5 after projection"}
    if result["reprojection_max_px"] > 2:
        raise ValueError("loaded MVE camera reprojection exceeds 2 px")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.inspector, args.scene, args.fixture)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
