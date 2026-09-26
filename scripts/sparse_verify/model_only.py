#!/usr/bin/env python3
"""Read-only COLMAP text-model audit when no valid observation holdout exists."""

import argparse
import json
import math
from pathlib import Path

try:
    from scripts.sparse_verify.verify import cameras, images, mv, points, project, sha256, stats
except ModuleNotFoundError:
    from verify import cameras, images, mv, points, project, sha256, stats


def audit(model_dir):
    root = Path(model_dir)
    paths = {name: root/name for name in ("cameras.txt", "images.txt", "points3D.txt")}
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    cs = cameras(paths["cameras.txt"])
    ims = images(paths["images.txt"])
    pts = points(paths["points3D.txt"])
    errors = []
    for iid, im in ims.items():
        if im["camera_id"] not in cs:
            errors.append(f"image {iid}: missing camera")
    seen = set(); residuals = []; positive = 0; nonpositive = 0
    multi_image_observation_tracks = []
    for pid, point in pts.items():
        track = point["track"]
        if len(track) < 2 or len(set(track)) != len(track):
            errors.append(f"point {pid}: short or duplicate track")
        if len({iid for iid, _ in track}) != len(track):
            multi_image_observation_tracks.append(pid)
        for iid, idx in track:
            if iid not in ims or idx < 0 or idx >= len(ims[iid]["xy"]):
                errors.append(f"point {pid}: invalid observation {iid}:{idx}")
                continue
            if (iid, idx) in seen:
                errors.append(f"observation {iid}:{idx} reused by another point")
            seen.add((iid, idx))
            x, y, linked = ims[iid]["xy"][idx]
            if linked != pid:
                errors.append(f"point {pid}: asymmetric link {iid}:{idx}")
            camxyz = [u+v for u, v in zip(mv(ims[iid]["R"], point["xyz"]), ims[iid]["t"])]
            if camxyz[2] <= 0:
                nonpositive += 1
            elif ims[iid]["camera_id"] in cs:
                positive += 1
                predicted = project(cs[ims[iid]["camera_id"]], camxyz)
                residuals.append(math.hypot(predicted[0]-x, predicted[1]-y))
    for iid, im in ims.items():
        for idx, (_, _, pid) in enumerate(im["xy"]):
            if pid != -1 and (iid, idx) not in seen:
                errors.append(f"image {iid}: dangling point observation {idx}:{pid}")
    if nonpositive:
        errors.append(f"{nonpositive} track observations have nonpositive depth")
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if hashes_before != hashes_after:
        errors.append("model files changed during audit")
    return dict(integrity_status="pass" if not errors else "fail", errors=errors,
                registered_images=len(ims), camera_models={str(k): v for k, v in cs.items()},
                camera_centers={im["name"]: im["center"] for im in ims.values()},
                points=len(pts), track_observations=len(seen),
                multi_observation_same_image_tracks=multi_image_observation_tracks,
                strict_one_observation_per_image_points=len(pts)-len(multi_image_observation_tracks),
                positive_depth_observations=positive,
                nonpositive_depth_observations=nonpositive,
                reprojection=stats(residuals),
                model_hashes_before=hashes_before, model_hashes_after=hashes_after,
                verifier_hashes={"model_only.py": sha256(__file__),
                                 "verify.py": sha256(project.__code__.co_filename)},
                quality_accepted=False,
                quality_reason="model consistency only; no valid heldout observations")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    result = audit(args.model)
    output = json.dumps(result, indent=2, sort_keys=True)+"\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output, end="")
    if result["integrity_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
