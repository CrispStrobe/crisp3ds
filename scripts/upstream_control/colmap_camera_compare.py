#!/usr/bin/env python3
"""Post hoc COLMAP/Sceaux camera agreement; the upstream cameras are not ground truth."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.upstream_control.camera_compare import camera_center, quaternion_rotation

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "build-opencv/upstream-control/sceaux-v23-camera-export-001/export/sparse"
SCENE = ROOT / ".local-tools/upstream-control/openmvs-sceaux-v23/scene.mvs"
EXPECTED = {
    "images.txt": "e210acf8eb4c99107b8d8bec0f4a29c35703c82062a461f665dbcc0d01c77b06",
    "cameras.txt": "85a5ae76af0712f50e34b845be1c50fc0510582485f46d15a3f27c52dc8507ab",
    "scene.mvs": "b69f87384284d7716d700ce61296e9f2a4cd8973cb4527d549126b0968edafde",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def basename(name):
    if not name or "\\" in name or Path(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"invalid image name: {name!r}")
    return name.rsplit("/", 1)[-1]


def add_pose(poses, name, rotation, translation):
    key = basename(name)
    if key in poses:
        raise ValueError(f"image basename collision: {key}")
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    center = camera_center(rotation, translation)
    poses[key] = (center, rotation)


def read_reference(path):
    """Parse COLMAP's two-line text image records, including empty observations."""
    lines = Path(path).read_text().splitlines()
    poses = {}
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    # Empty POINTS2D lines are omitted by the filter above; parse pose rows by
    # their exact 10-field layout, rejecting malformed pose-like records.
    for line in data:
        fields = line.split()
        if len(fields) == 10:
            try:
                int(fields[0]); int(fields[8])
                values = [float(value) for value in fields[1:8]]
            except ValueError as exc:
                raise ValueError("malformed COLMAP image pose") from exc
            add_pose(poses, fields[9], quaternion_rotation(values[:4]), values[4:])
    if not poses:
        raise ValueError("reference has no camera poses")
    return poses


def read_candidate(model_dir):
    import pycolmap

    path = Path(model_dir)
    binary = ("cameras.bin", "images.bin", "points3D.bin")
    textual = ("cameras.txt", "images.txt", "points3D.txt")
    names = binary if all((path / name).is_file() for name in binary) else textual
    if not all((path / name).is_file() for name in names):
        raise ValueError("candidate model needs a complete COLMAP binary or text file set")
    model_hashes = {name: sha256(path / name) for name in names}
    model = pycolmap.Reconstruction(str(path))
    poses = {}
    for image in model.images.values():
        if image.has_pose:
            add_pose(poses, image.name, image.cam_from_world.rotation.matrix(),
                     image.cam_from_world.translation)
    if model_hashes != {name: sha256(path / name) for name in names}:
        raise ValueError("candidate model changed while loading")
    return poses, model_hashes, {
        "pycolmap_version": pycolmap.__version__,
        "pycolmap_binary_sha256": sha256(pycolmap._core.__file__),
        "numpy_version": np.__version__,
    }


def proper_sim3(source, target):
    """Least-squares positive-scale, proper-rotation similarity for paired centers."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 4:
        raise ValueError("at least four paired 3D camera centers are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite camera centers")
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src, dst = source - src_mean, target - dst_mean
    src_singular = np.linalg.svd(src, compute_uv=False)
    dst_singular = np.linalg.svd(dst, compute_uv=False)
    if src_singular[1] <= src_singular[0] * 1e-6 or dst_singular[1] <= dst_singular[0] * 1e-6:
        raise ValueError("camera centers are collinear or degenerate")
    u, singular, vt = np.linalg.svd(dst.T @ src / len(source))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src * src, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0 or np.linalg.det(rotation) < 1 - 1e-8:
        raise ValueError("fitted similarity is improper")
    translation = dst_mean - scale * rotation @ src_mean
    residual = np.linalg.norm((scale * src @ rotation.T + dst_mean) - target, axis=1)
    # A perfect reflected fit on genuinely 3D centers is not a valid Sim(3).
    if min(src_singular[2] / src_singular[0], dst_singular[2] / dst_singular[0]) > 1e-4:
        reflected = u @ vt
        reflected_scale = float(np.sum(singular) / variance)
        reflected_residual = np.linalg.norm((reflected_scale * src @ reflected.T + dst_mean) - target, axis=1)
        if np.linalg.det(reflected) < 0 and np.sqrt(np.mean(reflected_residual ** 2)) < 1e-8 * np.sqrt(np.mean(np.sum(dst * dst, axis=1))) and np.sqrt(np.mean(residual ** 2)) > 1e-6 * np.sqrt(np.mean(np.sum(dst * dst, axis=1))):
            raise ValueError("camera centers require a reflection")
    return scale, rotation, translation


def compare(source, target):
    names = sorted(source.keys() & target.keys())
    missing = sorted(target.keys() - source.keys())
    extra = sorted(source.keys() - target.keys())
    if len(names) < 4:
        raise ValueError("fewer than four named camera pairs")
    src = np.array([source[name][0] for name in names])
    dst = np.array([target[name][0] for name in names])
    scale, world_rotation, translation = proper_sim3(src, dst)
    mapped = scale * src @ world_rotation.T + translation
    center_error = np.linalg.norm(mapped - dst, axis=1)
    radius = float(np.sqrt(np.mean(np.sum((dst - dst.mean(axis=0)) ** 2, axis=1))))
    if radius <= 0:
        raise ValueError("reference camera radius is zero")
    angles = []
    for name in names:
        source_rotation = source[name][1]
        reference_rotation = target[name][1]
        predicted_rotation = source_rotation @ world_rotation.T
        cosine = np.clip((np.trace(predicted_rotation @ reference_rotation.T) - 1) / 2, -1, 1)
        angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "source_camera_count": len(source), "reference_camera_count": len(target),
        "paired_camera_count": len(names), "paired_names": names,
        "missing_reference_names": missing, "extra_candidate_names": extra,
        "all_reference_cameras_paired": not missing,
        "similarity_source_to_reference": {"scale": scale, "rotation": world_rotation.tolist(),
                                           "translation": translation.tolist()},
        "reference_camera_radius": radius,
        "reference_camera_radius_definition": "RMS distance of paired reference camera centers from their centroid, in upstream scene units",
        "center_rms": float(np.sqrt(np.mean(center_error ** 2))),
        "center_rms_over_reference_radius": float(np.sqrt(np.mean(center_error ** 2)) / radius),
        "center_p95_over_reference_radius": float(np.quantile(center_error, .95) / radius),
        "orientation_median_degrees": float(np.median(angles)),
        "orientation_p95_degrees": float(np.quantile(angles, .95)),
        "per_name": {name: {"center_error_over_reference_radius": float(center_error[i] / radius),
                             "orientation_error_degrees": angles[i]} for i, name in enumerate(names)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="image-only COLMAP sparse model directory")
    parser.add_argument("--report", type=Path, required=True, help="fresh report JSON path")
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report already exists")
    files = {"images.txt": REFERENCE / "images.txt", "cameras.txt": REFERENCE / "cameras.txt",
             "scene.mvs": SCENE}
    reference_hashes = {name: sha256(path) for name, path in files.items()}
    if reference_hashes != EXPECTED:
        raise ValueError("pinned upstream Sceaux scene/export hashes differ")
    software_files = {"script_sha256": Path(__file__),
                      "camera_compare_helper_sha256": Path(camera_center.__code__.co_filename)}
    software_hashes = {name: sha256(path) for name, path in software_files.items()}
    source, model_hashes, software = read_candidate(args.model)
    comparison = compare(source, read_reference(files["images.txt"]))
    if model_hashes != {name: sha256(args.model / name) for name in model_hashes}:
        raise ValueError("candidate model changed during comparison")
    if reference_hashes != {name: sha256(path) for name, path in files.items()}:
        raise ValueError("upstream reference changed during comparison")
    if software_hashes != {name: sha256(path) for name, path in software_files.items()}:
        raise ValueError("comparison software changed during comparison")
    report = {"schema": "sceaux_colmap_camera_software_agreement_v1",
              "interpretation": "evaluation-only named-camera agreement with upstream software poses; not ground truth or metric accuracy",
              "candidate_model_dir": str(args.model.resolve()),
              "candidate_model_sha256": model_hashes,
              "upstream_reference_sha256": reference_hashes,
              "software": {**software, **software_hashes},
              "comparison": comparison}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
