#!/usr/bin/env python3
"""Post hoc Sceaux camera-center alignment and software-oracle mesh comparison."""

import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.object_dataset import align, evaluate, preview, surface_metrics


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def camera_center(rotation, translation):
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float)
    if (translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError("camera pose is not a finite proper world-to-camera rigid transform")
    return -rotation.T @ translation


def quaternion_rotation(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(q @ q, 1, atol=1e-3):
        raise ValueError("invalid COLMAP quaternion")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def read_mve(bundle_path, views_dir):
    lines = Path(bundle_path).read_text().splitlines()
    if len(lines) < 2 or lines[0].strip() != "drews 1.0":
        raise ValueError("expected MVE drews 1.0 bundle")
    count = int(lines[1].split()[0])
    if not 3 <= count <= 200 or len(lines) < 2 + 5*count:
        raise ValueError("MVE bundle camera count outside limits")
    centers, hashes = {}, {}
    for index in range(count):
        at = 2 + 5*index
        focal = float(lines[at].split()[0])
        rotation = np.asarray([[float(x) for x in lines[at+1+j].split()] for j in range(3)])
        translation = np.asarray([float(x) for x in lines[at+4].split()])
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("malformed MVE bundle camera")
        metadata = Path(views_dir) / f"view_{index:04}.mve" / "meta.ini"
        if metadata.is_symlink():
            raise ValueError("MVE view metadata symlink")
        config = configparser.ConfigParser()
        if not config.read(metadata):
            raise ValueError("missing MVE view metadata")
        if config.getint("view", "id") != index:
            raise ValueError("MVE bundle/view ID mismatch")
        name = config.get("view", "name") + ".jpg"
        if name in centers:
            raise ValueError("duplicate MVE image name")
        meta_rotation = np.asarray([float(x) for x in config.get("camera", "rotation").split()]).reshape(3, 3)
        meta_translation = np.asarray([float(x) for x in config.get("camera", "translation").split()])
        if not np.allclose(rotation, meta_rotation, atol=1e-4) or not np.allclose(translation, meta_translation, atol=1e-4):
            raise ValueError("MVE bundle and view poses differ")
        if focal > 0:
            centers[name] = camera_center(rotation, translation)
        hashes[name] = sha256(metadata)
    return centers, hashes


def read_colmap_images(path):
    centers = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            continue  # POINTS2D line, or a blank line for an empty observation list.
        try:
            int(fields[0]); int(fields[8])
            pose = [float(value) for value in fields[1:8]]
        except ValueError:
            continue
        name = fields[9]
        if Path(name).name != name or not name.endswith(".jpg") or name in centers:
            raise ValueError("invalid or duplicate COLMAP image name")
        centers[name] = camera_center(quaternion_rotation(pose[:4]), pose[4:])
    if not 3 <= len(centers) <= 200:
        raise ValueError("COLMAP image count outside limits")
    return centers


def fit_centers(source, target):
    if set(source) != set(target) or len(source) < 4:
        raise ValueError("camera names must match with at least four pairs")
    names = sorted(source)
    src = np.asarray([source[name] for name in names])
    dst = np.asarray([target[name] for name in names])
    matrix = align.umeyama(src, dst)
    mapped = src @ matrix[:3, :3].T + matrix[:3, 3]
    residual = np.linalg.norm(mapped - dst, axis=1)
    leave_one_out = []
    for i in range(len(names)):
        keep = np.arange(len(names)) != i
        held_out = align.umeyama(src[keep], dst[keep])
        predicted = held_out[:3, :3] @ src[i] + held_out[:3, 3]
        leave_one_out.append(float(np.linalg.norm(predicted - dst[i])))
    return matrix, {"camera_names": names, "fit_residual_by_name": dict(zip(names, map(float, residual))),
                    "fit_rms": float(np.sqrt(np.mean(residual**2))),
                    "fit_p95": float(np.quantile(residual, .95)),
                    "leave_one_out_rms": float(np.sqrt(np.mean(np.square(leave_one_out)))),
                    "leave_one_out_p95": float(np.quantile(leave_one_out, .95)),
                    "fitted_scale": float(np.linalg.svd(matrix[:3, :3], compute_uv=False).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mve-bundle", required=True, type=Path)
    parser.add_argument("--mve-views", required=True, type=Path)
    parser.add_argument("--upstream-images", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", action="append", required=True, type=float)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save-report", required=True, type=Path)
    parser.add_argument("--save-preview", type=Path)
    args = parser.parse_args()
    source, meta_hashes = read_mve(args.mve_bundle, args.mve_views)
    target = read_colmap_images(args.upstream_images)
    matrix, cameras = fit_centers(source, target)
    ref = evaluate.inspect_ply(args.reference, geometry=True)
    out = evaluate.inspect_ply(args.output, geometry=True)
    if ref[0]["nontriangle_faces"] or out[0]["nontriangle_faces"]:
        raise ValueError("camera-aligned scoring requires triangle meshes")
    surface = surface_metrics.compare(ref[1:], out[1:], matrix,
                                      thresholds=args.threshold, count=args.samples, seed=args.seed)
    report = {"schema": "sceaux_camera_aligned_software_oracle_v1",
              "interpretation": "post hoc named-camera-center fit to supplied upstream poses; software agreement only, not ground truth or metric scale",
              "matrix_output_to_upstream_scene": matrix.tolist(),
              "camera_fit": cameras, "surface": surface,
              "mve_bundle_sha256": sha256(args.mve_bundle), "mve_view_meta_sha256": meta_hashes,
              "upstream_images_txt_sha256": sha256(args.upstream_images),
              "reference_sha256": evaluate.sha256_file(args.reference),
              "output_sha256": evaluate.sha256_file(args.output),
              "registration_basis": "named-camera-center-fit", "metric_accuracy_claim_allowed": False}
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    with args.save_report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if args.save_preview:
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)
        image = preview.make_preview(ref[1:], out[1:], matrix, count=10000)
        with args.save_preview.open("xb") as stream:
            image.save(stream, format="PNG")
    print(json.dumps({"camera_fit": cameras, "threshold_scores": surface["threshold_scores"]}, indent=2))


if __name__ == "__main__":
    main()
