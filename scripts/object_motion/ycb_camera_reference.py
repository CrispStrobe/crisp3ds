#!/usr/bin/env python3
"""Post hoc YCB Berkeley camera-oracle diagnostic; never a mesh alignment.

Only calibration and pose HDF5 members of the previously verified official
Berkeley RGB-D archive are extracted. No depth, masks, or Google mesh is read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import numpy as np

from scripts.upstream_control.camera_compare import fit_centers


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / ".local-tools/test-data/ycb-cracker-box/003_cracker_box_berkeley_rgbd.tgz"
PHOTO_MANIFEST = ROOT / ".local-tools/test-data/ycb-cracker-box/manifest.json"
MODEL = ROOT / "build-opencv/object-motion/foreground/models/0"
PRODUCER_SUMMARY = ROOT / "build-opencv/object-motion/foreground/summary.json"
PRODUCER_PROVENANCE = ROOT / "build-opencv/object-motion/foreground/provenance.json"
OUTPUT = ROOT / "build-opencv/object-motion/ycb-camera-reference-003"
ARCHIVE_SHA256 = "15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5"
PREFIX = "003_cracker_box/"
ANGLES = tuple(range(0, 360, 6))
MEMBERS = {PREFIX + "calibration.h5"} | {PREFIX + f"poses/NP5_{a}_pose.h5" for a in ANGLES}
MAX_METADATA_BYTES = 5_000_000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_h5dump(output, key, shape):
    """Parse the finite numeric payload of a single `h5dump -d` dataset."""
    if f'DATASET "{key}"' not in output or "H5T_IEEE_F64LE" not in output:
        raise ValueError(f"unexpected HDF5 dataset/type: {key}")
    match = re.search(r"DATASPACE\s+SIMPLE\s*\{\s*\(\s*([\d,\s]+)\s*\)", output)
    if not match or tuple(map(int, match.group(1).split(","))) != shape:
        raise ValueError(f"unexpected HDF5 shape: {key}")
    match = re.search(r"\bDATA\s*\{(.*?)\n\s*\}", output, re.S)
    if not match:
        raise ValueError(f"missing HDF5 data: {key}")
    values = []
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        line_match = re.fullmatch(r"\s*\([\d, ]+\):\s*([^()]*)", line)
        if not line_match:
            raise ValueError(f"malformed HDF5 data line: {key}")
        values.extend(float(x.strip()) for x in line_match.group(1).split(",") if x.strip())
    if len(values) != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"nonfinite or incomplete HDF5 data: {key}")
    return np.asarray(values, dtype=np.float64).reshape(shape)


def dataset(path, key, shape):
    result = subprocess.run(["h5dump", "-m", "%.17g", "-d", key, str(path)], capture_output=True,
                            text=True, timeout=10, check=True)
    return parse_h5dump(result.stdout, key, shape)


def rigid(matrix, label):
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or (
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)):
        raise ValueError(f"{label} is not a proper rigid transform")
    return matrix


def np3_from_table(h_np3_from_np5, h_table_from_np5):
    """Compose documented NP5-to-NP3 and NP5-to-turntable frame maps."""
    return rigid(h_np3_from_np5, "NP3 calibration") @ np.linalg.inv(
        rigid(h_table_from_np5, "turntable pose"))


def extract_metadata(archive, folder):
    archive, folder = Path(archive), Path(folder)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Berkeley archive does not match the pinned SHA-256")
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    found, total = {}, 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name not in MEMBERS:
                continue
            if member.name in found or not member.isfile() or member.size <= 0:
                raise ValueError(f"duplicate or non-file metadata member: {member.name}")
            total += member.size
            if total > MAX_METADATA_BYTES:
                raise ValueError("metadata extraction exceeds 5 MB")
            dest = folder / member.name.removeprefix(PREFIX)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target, 1 << 20)
            if dest.stat().st_size != member.size:
                raise ValueError(f"truncated metadata member: {member.name}")
            found[member.name] = {"bytes": member.size, "sha256": sha256(dest)}
    if set(found) != MEMBERS:
        raise ValueError(f"metadata members absent: {sorted(MEMBERS - set(found))}")
    return found, total


def reference_cameras(folder):
    folder = Path(folder)
    calib = folder / "calibration.h5"
    h_np3_from_np5 = rigid(dataset(calib, "/H_NP3_from_NP5", (4, 4)), "NP3 calibration")
    k = dataset(calib, "/NP3_rgb_K", (3, 3))
    d = dataset(calib, "/NP3_rgb_d", (5,))
    if not np.allclose(k[2], [0, 0, 1], atol=1e-10) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("invalid NP3 RGB intrinsics")
    centers, rotations = {}, {}
    for angle in ANGLES:
        name = f"NP3_{angle:03}.jpg"
        pose = rigid(dataset(folder / "poses" / f"NP5_{angle}_pose.h5",
                             "/H_table_from_reference_camera", (4, 4)), name)
        # The camera was physically fixed while the object rotated. Expressing
        # each exposure in its own turntable/object frame creates a virtual orbit.
        h_np3_from_table = np3_from_table(h_np3_from_np5, pose)
        centers[name] = -h_np3_from_table[:3, :3].T @ h_np3_from_table[:3, 3]
        rotations[name] = h_np3_from_table[:3, :3]
    return centers, rotations, k, d


def estimated_cameras(model):
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(model))
    centers, rotations = {}, {}
    if len(reconstruction.images) != len(ANGLES) or len(reconstruction.cameras) != 1:
        raise ValueError("expected 60 registered images with shared intrinsics")
    for image in reconstruction.images.values():
        name = image.name
        if not re.fullmatch(r"NP3_\d{3}\.jpg", name) or name in centers:
            raise ValueError("unexpected or duplicate reconstructed image name")
        centers[name] = np.asarray(image.cam_from_world.inverse().translation, dtype=float)
        rotations[name] = np.asarray(image.cam_from_world.rotation.matrix(), dtype=float)
    expected = {f"NP3_{angle:03}.jpg" for angle in ANGLES}
    if set(centers) != expected:
        raise ValueError("registered image names do not match selected photos")
    return centers, rotations, next(iter(reconstruction.cameras.values()))


def rotation_errors(source_rotations, target_rotations, matrix):
    scale = float(np.linalg.svd(matrix[:3, :3], compute_uv=False).mean())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("invalid camera-center alignment scale")
    world_rotation = matrix[:3, :3] / scale
    errors = {}
    for name in sorted(source_rotations):
        predicted = source_rotations[name] @ world_rotation.T
        delta = predicted @ target_rotations[name].T
        cosine = np.clip((np.trace(delta) - 1) / 2, -1, 1)
        errors[name] = float(np.degrees(np.arccos(cosine)))
    return errors


def evaluate(archive=ARCHIVE, model=MODEL, output=OUTPUT,
             photo_manifest=PHOTO_MANIFEST, producer_summary=PRODUCER_SUMMARY,
             producer_provenance=PRODUCER_PROVENANCE):
    archive, model, output = map(Path, (archive, model, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    photo_manifest, producer_summary, producer_provenance = map(
        Path, (photo_manifest, producer_summary, producer_provenance))
    selected = json.loads(photo_manifest.read_text())
    producer = json.loads(producer_summary.read_text())
    provenance = json.loads(producer_provenance.read_text())
    if selected["sources"]["berkeley_rgbd"]["sha256"] != ARCHIVE_SHA256 or (
            producer.get("arm") != "foreground" or producer.get("registered") != 60):
        raise ValueError("unexpected source photo or producer provenance")
    photo_hashes = {Path(item["path"]).name: item["sha256"] for item in selected["photos"]}
    if (provenance.get("image_hashes") != photo_hashes or
            producer.get("producer_provenance_sha256") != sha256(producer_provenance)):
        raise ValueError("reconstructed photos do not match the official image selection")
    expected_hashes = producer.get("model_files_sha256", {})
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        if sha256(model / filename) != expected_hashes.get(filename):
            raise ValueError(f"reconstructed model hash mismatch: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    found, total = extract_metadata(archive, output / "metadata")
    target_centers, target_rotations, k, d = reference_cameras(output / "metadata")
    source_centers, source_rotations, camera = estimated_cameras(model)
    matrix, diagnostics = fit_centers(source_centers, target_centers)
    angles = rotation_errors(source_rotations, target_rotations, matrix)
    source_params = list(map(float, camera.params))
    if camera.model.name != "SIMPLE_RADIAL":
        raise ValueError("unexpected reconstructed camera model")
    report = {"schema": "ycb_berkeley_camera_oracle_v1",
              "interpretation": "post hoc Berkeley rig camera diagnostic only; not Google mesh alignment or image-only reconstruction input",
              "source_archive_sha256": ARCHIVE_SHA256,
              "photo_manifest_sha256": sha256(photo_manifest),
              "producer_summary_sha256": sha256(producer_summary),
              "producer_provenance_sha256": sha256(producer_provenance),
              "producer_model_files_sha256": expected_hashes,
              "metadata": {"members": found, "total_uncompressed_bytes": total},
              "transform_convention": "H_NP3_from_table = H_NP3_from_NP5 @ inverse(H_table_from_reference_camera); centers in per-angle Berkeley table frame",
              "matrix_estimated_world_to_berkeley_table": matrix.tolist(),
              "camera_centers": diagnostics,
              "rotation_error_degrees_by_name": angles,
              "rotation_error_median_degrees": float(np.median(list(angles.values()))),
              "rotation_error_p95_degrees": float(np.quantile(list(angles.values()), .95)),
              "intrinsics": {"berkeley_np3_rgb_K": k.tolist(), "berkeley_np3_rgb_d": d.tolist(),
                             "estimated_model": camera.model.name, "estimated_params": source_params,
                             "estimated_minus_supplied_focal_px": [source_params[0] - k[0, 0], source_params[0] - k[1, 1]],
                             "estimated_minus_supplied_principal_px": [source_params[1] - k[0, 2], source_params[2] - k[1, 2]],
                             "distortion_models_directly_comparable": False},
              "google_mesh_cross_frame_transform_known": False,
              "metric_google_mesh_accuracy_claim_allowed": False}
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate(archive=args.archive, model=args.model, output=args.output)
    print(json.dumps({"fit_rms": result["camera_centers"]["fit_rms"],
                      "rotation_p95_degrees": result["rotation_error_p95_degrees"]}))


if __name__ == "__main__":
    main()
