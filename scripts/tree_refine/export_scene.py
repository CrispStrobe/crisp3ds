#!/usr/bin/env python3
"""Export the one fixed tree BA candidate to a fresh selected-MVE scene."""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.tree_dense.import_scene import mve_intrinsics, project


def export(baseline, refined, scene):
    if scene.exists():
        raise FileExistsError(scene)
    baseline = baseline.resolve(strict=True)
    data = json.loads((baseline / "conversion.json").read_text())
    lines = refined.read_text().splitlines()
    summary = lines[0].split()
    lengths = lines[1].split()
    if summary[0] != "SUMMARY" or lengths[0] != "LENGTHS":
        raise ValueError("malformed refinement output")
    views = data["views"]
    source_poses = [(v["rotation"][:], v["translation"][:]) for v in views]
    cameras = lines[2:12]
    if len(cameras) != len(views) or len(views) != 10:
        raise ValueError("wrong camera count")
    for i, line in enumerate(cameras):
        words = line.split()
        if words[:2] != ["CAMERA", str(i)] or len(words) != 14:
            raise ValueError("invalid camera row")
        numbers = list(map(float, words[2:]))
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("nonfinite camera")
        views[i]["rotation"], views[i]["translation"] = numbers[:9], numbers[9:]
    points = []
    for line in lines[12:]:
        words = line.split()
        if not words or words[0] != "POINT":
            raise ValueError("invalid point row")
        x, y, z = map(float, words[1:4])
        count = int(words[4])
        if len(words) != 5 + 4 * count or not 2 <= count <= 10:
            raise ValueError("invalid observations")
        observations = []
        for k in range(count):
            start = 5 + 4 * k
            i, key = int(words[start]), int(words[start + 1])
            u, v = float(words[start + 2]), float(words[start + 3])
            if not 0 <= i < 10 or not all(map(math.isfinite, (x, y, z, u, v))):
                raise ValueError("nonfinite/out of range point")
            observations.append((i, key, u, v))
        if len({o[0] for o in observations}) != count:
            raise ValueError("same-view track conflict")
        points.append(((x, y, z), observations))
    if len(points) != int(summary[5]):
        raise ValueError("point count mismatch")
    if any(not math.isfinite(float(value)) for value in summary[8:12]):
        raise ValueError("nonfinite reprojection RMS")
    if int(summary[13]) != int(summary[14]):
        raise ValueError("training graph does not connect to anchor")
    def center(rotation, translation):
        return [-sum(rotation[3*r+c]*translation[r] for r in range(3)) for c in range(3)]
    initial_centers = [center(*pose) for pose in source_poses]
    final_centers = [center(v["rotation"], v["translation"]) for v in views]
    initial_baseline = math.dist(initial_centers[0], initial_centers[1])
    final_baseline = math.dist(final_centers[0], final_centers[1])
    anchor_delta = max(abs(a-b) for i in (0, 1) for original, final in
                       ((source_poses[i][0], views[i]["rotation"]),
                        (source_poses[i][1], views[i]["translation"]))
                       for a, b in zip(original, final))
    if anchor_delta > 1e-8 or abs(initial_baseline-final_baseline) > 1e-8:
        raise ValueError("anchored poses changed")
    scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline, scene, ignore=shutil.ignore_patterns("depth-L0.mvei"))
    bundle = ["drews 1.0", f"{len(views)} {len(points)}"]
    for i, view in enumerate(views):
        camera = view["camera"]
        focal, aspect, ppx, ppy = mve_intrinsics(camera)
        rotation, translation = view["rotation"], view["translation"]
        bundle.append(f"{focal:.12g} 0 0")
        for r in range(3):
            bundle.append(" ".join(f"{rotation[3*r+c]:.12g}" for c in range(3)))
        bundle.append(" ".join(f"{t:.12g}" for t in translation))
        meta = ["[view]", f"id = {i}", f"name = {view['id']}", "[camera]",
                f"focal_length = {focal:.12g}", "radial_distortion = 0 0",
                f"pixel_aspect = {aspect:.12g}",
                f"principal_point = {ppx:.12g} {ppy:.12g}",
                "rotation = " + " ".join(f"{x:.12g}" for x in rotation),
                "translation = " + " ".join(f"{x:.12g}" for x in translation)]
        (scene / "views" / f"view_{i:04d}.mve" / "meta.ini").write_text("\n".join(meta) + "\n")
    seeds = []
    all_errors = []
    behind = 0
    for number, (point, observations) in enumerate(points):
        bundle.extend((" ".join(f"{x:.12g}" for x in point), "255 255 255",
                       str(len(observations)) + " " + " ".join(
                           f"{i} {key} 0" for i, key, _, _ in observations)))
        errors = []
        for i, _, u, v in observations:
            try:
                p = project(views[i]["camera"], views[i]["rotation"],
                            views[i]["translation"], point)
                errors.append(math.dist(p, (u, v)))
            except ValueError:
                behind += 1
        a, b = observations[:2]
        seeds.append(" ".join(map(str, [*point, a[0], a[2], a[3], b[0], b[2], b[3],
                                        errors[0] if errors else 0, errors[1] if len(errors)>1 else 0])))
        all_errors.extend(errors)
    (scene / "synth_0.out").write_text("\n".join(bundle) + "\n")
    (scene / "measured-seeds.txt").write_text("\n".join(seeds) + "\n")
    data.update(measured_seed_tracks=len(points), measured_observations=sum(len(o) for _, o in points),
                reprojection_rms_px=math.sqrt(sum(x*x for x in all_errors)/len(all_errors)),
                reprojection_max_px=max(all_errors),
                seed_method="conflict-free multiview ORB; fixed two-camera gauge; Huber 1 px, 30 Ceres iterations",
                source_scale="arbitrary; fixed source baseline between views 0 and 1")
    (scene / "conversion.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    report = dict(pair_seeds=int(summary[1]), conflicts=int(summary[3]),
                  rejected_components=int(summary[4]), tracks=len(points),
                  track_lengths={str(i): int(n) for i, n in enumerate(lengths[1:], 2)},
                  training_observations=int(summary[6]), heldout_observations=int(summary[7]),
                  train_rms_before_px=float(summary[8]), train_rms_after_px=float(summary[9]),
                  heldout_rms_before_px=float(summary[10]), heldout_rms_after_px=float(summary[11]),
                  solver_iterations=int(summary[12]),
                  supported_cameras=int(summary[13]), connected_cameras=int(summary[14]),
                  bundle_adjustment_enabled=not bool(int(summary[15])),
                  fixed_anchor_max_parameter_delta=anchor_delta,
                  source_baseline=initial_baseline, final_baseline=final_baseline,
                  camera_center_displacements=[math.dist(a, b) for a, b in zip(initial_centers, final_centers)],
                  postfit_all_observation_rms_px=data["reprojection_rms_px"],
                  postfit_max_px=data["reprojection_max_px"], behind_camera_observations=behind,
                  gauge="entire world-to-camera poses 0 and 1 fixed; source baseline retained")
    (scene / "refinement-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.baseline, args.refined, args.scene), indent=2))
