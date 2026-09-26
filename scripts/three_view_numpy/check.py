#!/usr/bin/env python3
"""Recompute frozen held-out three-view errors with NumPy least squares."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "build-opencv/colmap-sparse/heldout-v2-001"
DEFAULT_OUTPUT = ROOT / "build-opencv/colmap-sparse/three-view-numpy-001"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(saved, manifest, model_dir):
    found = {"manifest": sha256(manifest),
             "model": {name: sha256(model_dir / name) for name in saved["model"]},
             "inputs": {path: sha256(path) for path in saved["inputs"]},
             "verifier": sha256(ROOT / "scripts/sparse_verify/verify.py")}
    if found != saved:
        raise ValueError("frozen verifier input hashes differ")
    return found


def closed_cycles(rows):
    edges = defaultdict(lambda: defaultdict(set))
    names = set()
    for row in rows:
        a = (row["image1"], row["feature1"])
        b = (row["image2"], row["feature2"])
        if a[0] == b[0]:
            continue
        if a[0] > b[0]:
            a, b = b, a
        edges[(a[0], b[0])][a[1]].add(b[1])
        names.update((a[0], b[0]))
    ordered = sorted(names)
    cycles = set()
    for i, na in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            nb = ordered[j]
            for nc in ordered[j + 1:]:
                ab, ac, bc = (edges.get((x, y), {}) for x, y in ((na, nb), (na, nc), (nb, nc)))
                for ia, ids_b in ab.items():
                    for ib in ids_b:
                        for ic in ac.get(ia, set()) & bc.get(ib, set()):
                            cycles.add(((na, ia), (nb, ib), (nc, ic)))
    return sorted(cycles)


def evaluate_cycle(cycle, poses, features, camera):
    if any(name not in poses for name, _ in cycle):
        return {"status": "unregistered"}
    f, cx, cy = camera
    rays = []
    for name, index in cycle:
        pose = poses[name]
        x, y = features[name][index][:2]
        direction = pose["R"].T @ np.array([(x - cx) / f, (y - cy) / f, 1.0])
        direction /= np.linalg.norm(direction)
        rays.append((pose["C"], direction, pose["R"], pose["t"]))
    (ca, da, _, _), (cb, db, _, _), (_, _, rc, tc) = rays
    angle = math.degrees(math.acos(float(np.clip(da @ db, -1.0, 1.0))))
    if angle < 1.0:
        return {"status": "low_parallax", "parallax_deg": angle}
    coefficients, _, rank, _ = np.linalg.lstsq(np.column_stack((da, -db)), cb - ca,
                                               rcond=None)
    if rank < 2:
        return {"status": "invalid_triangulation", "parallax_deg": angle}
    point = ((ca + coefficients[0] * da) + (cb + coefficients[1] * db)) / 2
    depths = [float((r @ point + t)[2]) for _, _, r, t in rays]
    result = {"parallax_deg": angle, "point_xyz": point.tolist(),
              "ray_parameters": coefficients.tolist(), "depths": depths}
    if min(depths) <= 0:
        return {**result, "status": "nonpositive_depth", "error_px": None}
    z = rc @ point + tc
    predicted = np.array([f * z[0] / z[2] + cx, f * z[1] / z[2] + cy])
    observed = np.asarray(features[cycle[2][0]][cycle[2][1]][:2])
    return {**result, "status": "finite", "error_px": float(np.linalg.norm(predicted - observed))}


def stats(values):
    good = sorted(x for x in values if x is not None and math.isfinite(x))
    n = len(good)
    return {"count": len(values), "finite_count": n, "invalid_count": len(values) - n,
            "median_px": (good[(n - 1) // 2] + good[n // 2]) / 2 if n else None,
            "p90_px": good[math.ceil(.9 * n) - 1] if n else None,
            "within_1px": sum(x <= 1 for x in good),
            "within_2px": sum(x <= 2 for x in good),
            "within_4px": sum(x <= 4 for x in good)}


def check_aggregate(actual, saved):
    for key in ("closed_cycles", "registered_cycles", "low_parallax",
                "invalid_triangulation", "invalid_distortion", "nonpositive_depth"):
        if actual[key] != saved[key]:
            raise ValueError(f"frozen verifier population differs: {key}")
    for key, value in actual["third_view_reprojection"].items():
        expected = saved["third_view_reprojection"][key]
        if (abs(value - expected) > 1e-8 if isinstance(value, float) else value != expected):
            raise ValueError(f"frozen verifier aggregate differs: {key}")


def run(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 10 << 30:
        raise OSError("less than 10 GiB free")
    manifest_path = LANE / "holdout.json"
    model_dir = LANE / "model_text"
    verification_path = LANE / "verification-supervisor.json"
    software_before = {"checker": sha256(Path(__file__)),
                       "pycolmap_binary": sha256(Path(pycolmap._core.__file__)),
                       "saved_verification": sha256(verification_path)}
    frozen = json.loads(verification_path.read_text())
    before = input_hashes(frozen["hashes"], manifest_path, model_dir)
    manifest = json.loads(manifest_path.read_text())
    model = pycolmap.Reconstruction(str(model_dir))
    if len(model.cameras) != 1:
        raise ValueError("expected one shared camera")
    camera = next(iter(model.cameras.values()))
    if str(camera.model) != "CameraModelId.SIMPLE_PINHOLE":
        raise ValueError("frozen model is not SIMPLE_PINHOLE")
    poses = {im.name: {"R": np.asarray(im.cam_from_world.rotation.matrix()),
                       "t": np.asarray(im.cam_from_world.translation),
                       "C": np.asarray(im.projection_center())} for im in model.images.values()}
    features = {im["name"]: im["features"] for im in manifest["images"]}
    cycles = closed_cycles(manifest["heldout_pairs"])
    records = []
    for cycle in cycles:
        value = evaluate_cycle(cycle, poses, features, camera.params)
        records.append({"cycle": [[name, index] for name, index in cycle], **value})
    states = [r["status"] for r in records]
    values = [r.get("error_px") for r in records
              if r["status"] not in ("unregistered", "low_parallax", "invalid_triangulation")]
    aggregate = {"closed_cycles": len(cycles), "registered_cycles": len(cycles) - states.count("unregistered"),
                 "low_parallax": states.count("low_parallax"),
                 "invalid_triangulation": states.count("invalid_triangulation"),
                 "invalid_distortion": 0, "nonpositive_depth": states.count("nonpositive_depth"),
                 "third_view_reprojection": stats(values)}
    check_aggregate(aggregate, frozen["heldout"]["three_view"])
    after = input_hashes(frozen["hashes"], manifest_path, model_dir)
    software_after = {"checker": sha256(Path(__file__)),
                      "pycolmap_binary": sha256(Path(pycolmap._core.__file__)),
                      "saved_verification": sha256(verification_path)}
    if before != after or software_before != software_after:
        raise ValueError("frozen inputs or software changed during check")
    output.mkdir()
    (output / "report.json").write_text(json.dumps({"aggregate": aggregate, "cycles": records,
        "input_hashes_before": before, "input_hashes_after": after,
        "software_hashes_before": software_before, "software_hashes_after": software_after,
        "numpy_version": np.__version__,
        "pycolmap_version": pycolmap.__version__}, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.absolute().parent.mkdir(parents=True, exist_ok=True)
    run(args.output.absolute())


if __name__ == "__main__":
    main()
