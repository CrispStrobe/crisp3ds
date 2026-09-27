#!/usr/bin/env python3
"""Bounded, candidate-independent sensor-depth ray comparison primitives.

The live Berkeley comparison is intentionally gated until the independent
RGB/depth projection check and each candidate's camera Sim(3) are approved.
These functions do not fit a mesh or choose support from its predictions.
"""

import hashlib
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

if __package__:
    from . import evaluate
    from .surface_metrics import TriangleBVH, positive_triangles
else:
    import evaluate
    from surface_metrics import TriangleBVH, positive_triangles

MAX_RAYS_PER_VIEW = 2048
MAX_SECONDS = 300
MIN_FREE_BYTES = 10 * 1024**3
NEAR_METRES = 0.2
FAR_METRES = 1.5
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_003 = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/report.json"
PROJECTION_002 = ROOT / "build-opencv/object-motion/berkeley-projection-check-002/report.json"
CALIBRATION = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/metadata/calibration.h5"
DEPTH_FOLDER = ROOT / "build-opencv/ycb-depth-feasibility-001/003_cracker_box"
POSE_FOLDER = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/metadata/poses"
MASK_FOLDER = ROOT / "build-opencv/object-motion/prepare-001/masks"
ANGLES = (0, 120, 240)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_disk_floor():
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise OSError("sensor-depth run requires at least 10 GiB free disk")


def proper_similarity(matrix):
    """Validate and return a finite, positive-scale, orientation-preserving Sim(3)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("similarity must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("invalid homogeneous row")
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular))
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-8, atol=1e-12):
        raise ValueError("linear part is not uniform positive scale")
    if np.linalg.det(linear) <= 0:
        raise ValueError("similarity must preserve orientation")
    return matrix


def proper_rigid(matrix):
    """Validate a supplied Berkeley camera/turntable extrinsic, not fit one."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3, rtol=0)):
        raise ValueError("extrinsic is not a proper rigid transform")
    return matrix


def transform_vertices(vertices, matrix):
    matrix = proper_similarity(matrix)
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite XYZ rows")
    transformed = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(transformed).all():
        raise ValueError("vertex transform overflowed")
    return transformed


def depth_directions(u, v, intrinsic):
    """Integer-address depth rays; parameter along each ray equals camera Z."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid depth intrinsic matrix")
    if not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("invalid intrinsic homogeneous row")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("pixel addresses must be finite")
    return np.stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)), axis=-1)


def _box_entry(direction, node, near, far):
    lower, upper = node[:2]
    entry, leave = near, far
    for axis in range(3):
        d = float(direction[axis])
        if d == 0:
            if lower[axis] > 0 or upper[axis] < 0:
                return math.inf
            continue
        a, b = float(lower[axis] / d), float(upper[axis] / d)
        entry = max(entry, min(a, b))
        leave = min(leave, max(a, b))
        if leave < entry:
            return math.inf
    return entry


def _triangle_hits(direction, triangles, near, far):
    """Vectorized double-sided Möller–Trumbore hits for positive-area leaves."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b - a, c - a
    p = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, p)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(direction)
    valid = np.abs(det) > 1e-14 * scale
    inverse = np.zeros_like(det)
    inverse[valid] = 1 / det[valid]
    offset = -a
    u = np.einsum("ij,ij->i", offset, p) * inverse
    q = np.cross(offset, e1)
    v = q @ direction * inverse
    t = np.einsum("ij,ij->i", e2, q) * inverse
    valid &= (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
    valid &= (t >= near) & (t <= far) & np.isfinite(t)
    return np.where(valid, t, np.inf)


def first_hit_depths(vertices, faces, directions, *, near=NEAR_METRES,
                     far=FAR_METRES, deadline=None):
    """First triangle hit Z per ray, NaN for miss; returns degenerate-face count."""
    if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
        raise ValueError("invalid ray interval")
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) > MAX_RAYS_PER_VIEW:
        raise ValueError("ray array outside view limit")
    if not np.isfinite(directions).all() or not np.allclose(directions[:, 2], 1, atol=1e-12, rtol=0):
        raise ValueError("directions must be finite camera-Z rays")
    triangles, degenerates = positive_triangles(vertices, faces)
    tree = TriangleBVH(triangles, deadline=deadline)
    result = np.full(len(directions), np.nan)
    for i, direction in enumerate(directions):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("sensor-depth comparison exceeded total time limit")
        best = math.inf
        stack = [0]
        traversed = 0
        while stack:
            traversed += 1
            if deadline is not None and traversed % 256 == 0 and time.monotonic() > deadline:
                raise TimeoutError("sensor-depth ray traversal exceeded total time limit")
            node_index = stack.pop()
            node = tree.nodes[node_index]
            if _box_entry(direction, node, near, min(far, best)) == math.inf:
                continue
            if node[2] < 0:
                ids = tree.order[node[4]:node[5]]
                best = min(best, float(np.min(_triangle_hits(direction, tree.triangles[ids], near, min(far, best)))))
            else:
                children = [(child, _box_entry(direction, tree.nodes[child], near, min(far, best)))
                            for child in (node[2], node[3])]
                for child, entry in sorted(children, key=lambda pair: pair[1], reverse=True):
                    if entry != math.inf:
                        stack.append(child)
        if math.isfinite(best):
            result[i] = best
    return result, degenerates


def calibrated_depth_metres(raw, ir_scale, ir_bias):
    """Pinned Berkeley convention; nonzero bias is rejected until units verified."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint16 or not math.isfinite(ir_scale) or ir_scale <= 0 or ir_bias != 0:
        raise ValueError("require uint16 raw depth, positive IR scale, and verified zero IR bias")
    metres = raw.astype(np.float64) * (0.0001 * ir_scale)
    metres[raw == 0] = np.nan
    return metres


def project_depth_to_rgb(directions, observed_depth, rgb_from_ir, rgb_intrinsic, rgb_distortion):
    """Measured depth points to distorted RGB coordinates; never uses a candidate mesh."""
    directions = np.asarray(directions, dtype=np.float64)
    observed_depth = np.asarray(observed_depth, dtype=np.float64)
    h = proper_rigid(rgb_from_ir)
    k = np.asarray(rgb_intrinsic, dtype=np.float64)
    d = np.asarray(rgb_distortion, dtype=np.float64).ravel()
    if (directions.ndim != 2 or directions.shape[1] != 3 or observed_depth.shape != (len(directions),)
            or k.shape != (3, 3) or len(d) != 5
            or not np.isfinite(directions).all() or not np.isfinite(observed_depth).all()
            or not np.isfinite(h).all() or not np.isfinite(k).all() or not np.isfinite(d).all()
            or np.any(observed_depth <= 0)):
        raise ValueError("invalid measured-depth RGB projection inputs")
    points = directions * observed_depth[:, None]
    rgb = points @ h[:3, :3].T + h[:3, 3]
    valid = rgb[:, 2] > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x, y = rgb[:, 0] / rgb[:, 2], rgb[:, 1] / rgb[:, 2]
        r2 = x * x + y * y
        radial = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2
        xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x)
        yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y
        u = k[0, 0] * xd + k[0, 2]
        v = k[1, 1] * yd + k[1, 2]
    valid &= np.isfinite(u) & np.isfinite(v)
    return np.stack((u, v), axis=1), valid


def raster_support(rgb_coordinates, valid_projection, mask):
    """Nearest integer RGB address, with no clipping of out-of-frame points."""
    xy = np.asarray(rgb_coordinates, dtype=np.float64)
    valid = np.asarray(valid_projection, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2 or valid.shape != (len(xy),) or mask.ndim != 2:
        raise ValueError("invalid RGB raster inputs")
    out = np.zeros(len(xy), dtype=bool)
    safe = valid & np.isfinite(xy).all(axis=1)
    ids = np.flatnonzero(safe)
    x = np.floor(xy[ids, 0] + 0.5).astype(np.int64)
    y = np.floor(xy[ids, 1] + 0.5).astype(np.int64)
    inside = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    out[ids[inside]] = mask[y[inside], x[inside]]
    return out


def erode_square(mask, radius):
    """Boolean full-square erosion with outside-image pixels treated as false."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not isinstance(radius, int) or radius < 0 or radius > 64:
        raise ValueError("invalid mask or erosion radius")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask.astype(np.uint8), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    width = 2 * radius + 1
    area = (integral[width:, width:] - integral[:-width, width:]
            - integral[width:, :-width] + integral[:-width, :-width])
    return area == width * width


def selected_depth_indices(depth, support, angle, *, seed_base=20260927):
    """Select reproducible rays from all valid observed pixels, independent of mesh."""
    depth = np.asarray(depth, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if depth.shape != (480, 640) or support.shape != depth.shape:
        raise ValueError("expected 480x640 depth and support")
    if angle not in (0, 120, 240):
        raise ValueError("unfrozen angle")
    if np.isinf(depth).any():
        raise ValueError("infinite observed depth")
    candidate = np.arange(depth.size, dtype=np.uint32)
    flat_depth = depth.ravel()
    valid_depth = np.isfinite(flat_depth) & (flat_depth >= NEAR_METRES) & (flat_depth <= FAR_METRES)
    eligible = candidate[valid_depth & support.ravel()]
    if len(eligible) > MAX_RAYS_PER_VIEW:
        selected = np.random.default_rng(seed_base + angle).choice(eligible, MAX_RAYS_PER_VIEW,
                                                                    replace=False)
    else:
        selected = eligible
    selected = np.sort(selected.astype("<u4", copy=False))
    return selected, {"frame_pixels": int(len(candidate)),
                      "raw_missing_pixels": int(np.count_nonzero(np.isnan(depth))),
                      "valid_interval_depth_pixels": int(np.count_nonzero(valid_depth)),
                      "eligible_count": int(len(eligible)),
                      "selected_count": int(len(selected)),
                      "sampled_fraction_of_eligible": float(len(selected) / len(eligible)) if len(eligible) else None,
                      "selected_sha256": hashlib.sha256(selected.tobytes()).hexdigest()}


def residual_summary(observed, predicted, support=None):
    """Coverage and signed/absolute residuals; no-hit rays never become zeroes."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("observed and predicted depths must be paired vectors")
    if support is None:
        support = np.ones(len(observed), dtype=bool)
    support = np.asarray(support, dtype=bool)
    if support.shape != observed.shape or not np.isfinite(observed[support]).all():
        raise ValueError("invalid support or observed depths")
    if np.any(observed[support] <= 0) or np.isinf(predicted).any():
        raise ValueError("observed depths must be positive; misses use NaN")
    hit = support & np.isfinite(predicted)
    n, h = int(np.count_nonzero(support)), int(np.count_nonzero(hit))
    out = {"supported_rays": n, "hits": h, "missing_predictions": n - h,
           "hit_fraction": h / n if n else None}
    if not h:
        out.update({"signed_m": None, "absolute_m": None, "within_m": None})
        return out
    residual = predicted[hit] - observed[hit]
    absolute = np.abs(residual)
    out["signed_m"] = {"mean": float(np.mean(residual)),
                       "p05": float(np.percentile(residual, 5)),
                       "median": float(np.median(residual)),
                       "p95": float(np.percentile(residual, 95))}
    out["absolute_m"] = {"mean": float(np.mean(absolute)),
                         "median": float(np.median(absolute)),
                         "p90": float(np.percentile(absolute, 90)),
                         "p95": float(np.percentile(absolute, 95))}
    out["within_m"] = {str(threshold): {"among_hits": float(np.mean(absolute <= threshold)),
                                        "among_supported": float(np.count_nonzero(absolute <= threshold) / n)}
                       for threshold in (0.005, 0.01, 0.02)}
    return out


def nullable_depths(values):
    """Serialize no-hit NaNs as JSON null while preserving ray order."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("expected finite/NaN depth vector")
    return [float(value) if np.isfinite(value) else None for value in values]


def _metadata_dataset(path, key, shape):
    # Reuse the independently tested bounded HDF5 numeric parser; no h5py dependency.
    from scripts.object_motion.ycb_camera_reference import dataset
    return dataset(path, key, shape)


def _depth_h5(path):
    """Read only the pinned-size uint16 array via h5dump into workspace temp."""
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("depth H5 exceeds pinned size bound")
    temp_root = ROOT / ".local-tools/tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as folder:
        binary = Path(folder) / "depth.bin"
        subprocess.run(["h5dump", "-d", "/depth", "-b", "LE", "-o", str(binary), str(path)],
                       capture_output=True, text=True, timeout=15, check=True)
        if binary.stat().st_size != 480 * 640 * 2:
            raise ValueError("depth payload size differs from 480x640 uint16")
        return np.fromfile(binary, dtype="<u2").reshape(480, 640)


def _load_mask(path):
    from PIL import Image
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != (1024, 1280):
        raise ValueError("unexpected photo-derived mask shape")
    return mask


def _verified_camera_report(path, model, reference_report):
    report = json.loads(Path(path).read_text())
    if report.get("schema") not in ("ycb_berkeley_camera_oracle_v1",
                                    "ycb_calibrated_berkeley_camera_oracle_v1",
                                    "ycb_recovered_berkeley_camera_oracle_v1"):
        raise ValueError("candidate needs Berkeley named-camera report")
    if report.get("source_archive_sha256") != reference_report["source_archive_sha256"]:
        raise ValueError("camera report uses different Berkeley reference poses or calibration")
    if report["schema"] == "ycb_berkeley_camera_oracle_v1":
        if report.get("metadata", {}).get("members") != reference_report["metadata"]["members"]:
            raise ValueError("original camera report metadata members differ")
    elif report["schema"] == "ycb_calibrated_berkeley_camera_oracle_v1":
        expected_members = {name: item["sha256"] for name, item in reference_report["metadata"]["members"].items()}
        producer_path = ROOT / "build-opencv/object-motion/calibration-ablation-002/calibrated_shifted/report.json"
        if (report.get("metadata_members_sha256") != expected_members
                or report.get("oracle_metadata_report_sha256") != sha256(REFERENCE_003)
                or report.get("calibration_h5_sha256") != sha256(CALIBRATION)
                or report.get("producer_report_sha256") != sha256(producer_path)):
            raise ValueError("calibrated camera report lineage differs from sealed sources")
    else:
        from scripts.object_motion import recovered_camera_reference as recovered
        expected_members = {name: item["sha256"] for name, item in reference_report["metadata"]["members"].items()}
        five_path, four_path = recovered.PRODUCER_005, recovered.PRODUCER_004
        if (sha256(five_path) != recovered.PRODUCER_005_SHA256 or
                sha256(four_path) != recovered.PRODUCER_004_SHA256):
            raise ValueError("recovered source 005/004 changed")
        five, four = json.loads(five_path.read_text()), json.loads(four_path.read_text())
        prepared = json.loads(recovered.PREPARED.read_text())
        selected = json.loads(recovered.base.PHOTO_MANIFEST.read_text())
        photos = {Path(item["path"]).name: item["sha256"] for item in selected["photos"]}
        prepared_photos = {item["name"]: item["sha256"] for item in prepared["images"]}
        if (report.get("metadata_members_sha256") != expected_members or
                report.get("oracle_metadata_report_sha256") != sha256(REFERENCE_003) or
                report.get("calibration_h5_sha256") != sha256(CALIBRATION) or
                report.get("producer_005_report_sha256") != recovered.PRODUCER_005_SHA256 or
                report.get("producer_004_report_sha256") != recovered.PRODUCER_004_SHA256 or
                report.get("photo_manifest_sha256") != sha256(recovered.base.PHOTO_MANIFEST) or
                report.get("prepared_manifest_sha256") != sha256(recovered.PREPARED) or
                report.get("source_image_hashes") != photos or prepared_photos != photos or
                four.get("source_image_hashes") != photos or
                four.get("source_manifest_sha256") != report["prepared_manifest_sha256"] or
                five.get("source_producer_report_sha256") != recovered.PRODUCER_004_SHA256 or
                five.get("source_model_files_sha256") != four.get("model_files_sha256") or
                five.get("input_model_copy_sha256") != four.get("model_files_sha256") or
                five.get("refined_model_files_sha256") != report.get("producer_model_files_sha256") or
                Path(five.get("refined_model_dir", "")).resolve() != Path(model).resolve() or
                report.get("reference_mesh_used") is not False or
                report.get("supplied_poses_used_for_mapping") is not False):
            raise ValueError("recovered camera report lineage differs from sealed 005/004 and photos")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        expected = report["producer_model_files_sha256"][name]
        if sha256(Path(model) / name) != expected:
            raise ValueError(f"camera report does not bind exact candidate {name}")
    return report, proper_similarity(report["matrix_estimated_world_to_berkeley_table"])


def verify_dense_camera_gauge(dense_sparse, source_model):
    """Undistorted dense input must retain each named source image's world pose."""
    import pycolmap  # Live-only dependency; analytic tests do not require it.
    source = pycolmap.Reconstruction(str(source_model))
    dense = pycolmap.Reconstruction(str(dense_sparse))
    source_images = {image.name: image for image in source.images.values()}
    dense_images = {image.name: image for image in dense.images.values()}
    if len(source_images) != 60 or source_images.keys() != dense_images.keys():
        raise ValueError("dense sparse image names differ from fitted source camera set")
    max_delta = max(float(np.max(np.abs(source_images[name].cam_from_world.matrix()
                                    - dense_images[name].cam_from_world.matrix())))
                    for name in source_images)
    if max_delta > 1e-8:
        raise ValueError(f"dense sparse image poses differ from source model: {max_delta}")
    return {"named_registered_images": len(source_images), "max_world_to_camera_matrix_delta": max_delta,
            "dense_sparse_images_sha256": sha256(Path(dense_sparse) / "images.bin")}


def verify_recovered_dense_lineage(producer_report, source_model, dense_sparse, camera_report):
    """Bind recovered rough mesh's undistortion to exact 005 source binaries."""
    from scripts.object_motion import recovered_camera_reference as recovered
    producer = json.loads(Path(producer_report).read_text())
    source_hashes = camera_report["producer_model_files_sha256"]
    stages = [stage for stage in producer.get("stages", []) if stage.get("name") == "undistort"]
    if (producer.get("schema") != "classical_recovered_dense_v1" or
            producer.get("status") != "complete" or len(stages) != 1 or
            stages[0].get("status") != "complete" or
            Path(producer.get("source_model", "")).resolve() != Path(source_model).resolve() or
            producer.get("source_model_sha256") != source_hashes or
            producer.get("source_reports", {}).get("producer", {}).get("sha256") != recovered.PRODUCER_005_SHA256 or
            producer.get("source_reports", {}).get("source", {}).get("sha256") != recovered.PRODUCER_004_SHA256 or
            producer.get("source_reports", {}).get("manifest", {}).get("sha256") != camera_report.get("prepared_manifest_sha256")):
        raise ValueError("recovered dense producer does not bind camera source 005/004")
    expected_dense = stages[0].get("artifact", {}).get("model_sha256")
    if set(expected_dense or {}) != {"cameras.bin", "images.bin", "points3D.bin"}:
        raise ValueError("recovered undistortion omitted sparse binary hashes")
    for name in expected_dense:
        if sha256(Path(dense_sparse) / name) != expected_dense[name]:
            raise ValueError(f"recovered dense sparse model changed: {name}")
    return {"source_model_sha256": source_hashes, "undistorted_model_sha256": expected_dense}


def _prepared_frames(reference_report, projection_report, deadline):
    if (projection_report.get("schema") != "berkeley_rgb_depth_projection_check_v1"
            or projection_report.get("status") != "assumption_qualified_projection_plausible"
            or projection_report.get("calibration_sha256") != sha256(CALIBRATION)
            or projection_report.get("source_archive_sha256") != reference_report["source_archive_sha256"]):
        raise ValueError("independent projection gate/provenance not satisfied")
    metadata_members = reference_report["metadata"]["members"]
    if sha256(CALIBRATION) != metadata_members["003_cracker_box/calibration.h5"]["sha256"]:
        raise ValueError("calibration changed since camera reference")
    scale = float(_metadata_dataset(CALIBRATION, "/NP3_ir_depth_scale", (1,))[0])
    bias = float(_metadata_dataset(CALIBRATION, "/NP3_ir_depth_bias", (1,))[0])
    k_depth = _metadata_dataset(CALIBRATION, "/NP3_depth_K", (3, 3))
    k_rgb = _metadata_dataset(CALIBRATION, "/NP3_rgb_K", (3, 3))
    d_rgb = _metadata_dataset(CALIBRATION, "/NP3_rgb_d", (5,))
    rgb_from_np5 = proper_rigid(_metadata_dataset(CALIBRATION, "/H_NP3_from_NP5", (4, 4)))
    ir_from_np5 = proper_rigid(_metadata_dataset(CALIBRATION, "/H_NP3_ir_from_NP5", (4, 4)))
    rgb_from_ir = proper_rigid(rgb_from_np5 @ np.linalg.inv(ir_from_np5))
    if (not np.allclose(k_depth, projection_report["depth_K"], atol=1e-12, rtol=0)
            or not np.allclose(k_rgb, projection_report["rgb_K"], atol=1e-12, rtol=0)
            or not np.allclose(d_rgb, projection_report["rgb_d"], atol=1e-12, rtol=0)
            or not np.allclose(rgb_from_ir, projection_report["rgb_from_ir"], atol=1e-8, rtol=0)
            or scale != projection_report["ir_depth_scale"] or bias != projection_report["ir_depth_bias"]):
        raise ValueError("projection report and current calibration differ")
    rows = []
    for angle in ANGLES:
        if time.monotonic() > deadline:
            raise TimeoutError("comparison exceeded total 300-second limit during input preparation")
        depth_path = DEPTH_FOLDER / f"NP3_{angle}.h5"
        mask_path = MASK_FOLDER / f"NP3_{angle:03}.jpg.png"
        pose_path = POSE_FOLDER / f"NP5_{angle}_pose.h5"
        frame_gate = next((row for row in projection_report["frames"] if row["angle"] == angle), None)
        if frame_gate is None or sha256(depth_path) != frame_gate["depth_sha256"] or sha256(mask_path) != frame_gate["pose_mask_sha256"]:
            raise ValueError("depth/mask differs from inspected projection inputs")
        pose_key = f"003_cracker_box/poses/NP5_{angle}_pose.h5"
        if sha256(pose_path) != metadata_members[pose_key]["sha256"]:
            raise ValueError("Berkeley per-view table pose changed")
        raw = _depth_h5(depth_path)
        observed = calibrated_depth_metres(raw, scale, bias)
        mask = _load_mask(mask_path)
        interior = erode_square(mask, 16)
        valid = np.isfinite(observed) & (observed >= NEAR_METRES) & (observed <= FAR_METRES)
        indices = np.flatnonzero(valid.ravel())
        v, u = np.divmod(indices, 640)
        rays = depth_directions(u, v, k_depth)
        xy, projected = project_depth_to_rgb(rays, observed.ravel()[indices], rgb_from_ir, k_rgb, d_rgb)
        coarse_support = np.zeros(observed.size, dtype=bool)
        coarse_support[indices] = raster_support(xy, projected, mask)
        coarse_support = coarse_support.reshape(observed.shape)
        selected, selection = selected_depth_indices(observed, coarse_support, angle)
        selected_v, selected_u = np.divmod(selected, 640)
        selected_rays = depth_directions(selected_u, selected_v, k_depth)
        selected_xy, selected_projected = project_depth_to_rgb(selected_rays,
                    observed.ravel()[selected], rgb_from_ir, k_rgb, d_rgb)
        selected_interior = raster_support(selected_xy, selected_projected, interior)
        table_from_np5 = proper_rigid(_metadata_dataset(pose_path,
                    "/H_table_from_reference_camera", (4, 4)))
        ir_from_table = proper_rigid(ir_from_np5 @ np.linalg.inv(table_from_np5))
        rows.append({"angle": angle, "depth_sha256": sha256(depth_path),
                     "pose_mask_sha256": sha256(mask_path), "pose_sha256": sha256(pose_path),
                     "selection": selection, "selected_indices": selected.tolist(),
                     "selected_interior_count": int(np.count_nonzero(selected_interior)),
                     "observed": observed.ravel()[selected], "rays": selected_rays,
                     "interior": selected_interior, "ir_from_table": ir_from_table})
    return rows


def _write_checkpoint(path, report):
    if path is None:
        return
    require_disk_floor()
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 20_000_000:
        raise ValueError("result exceeds 20 MB cap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def compare_candidates(candidates, projection_path=PROJECTION_002, deadline=None,
                       checkpoint_path=None):
    """One shared-ray batch; caller must explicitly authorize qualified gate."""
    if not 1 <= len(candidates) <= 4:
        raise ValueError("comparison requires 1–4 candidates")
    require_disk_floor()
    deadline = time.monotonic() + MAX_SECONDS if deadline is None else deadline
    reference_report = json.loads(REFERENCE_003.read_text())
    projection_report = json.loads(Path(projection_path).read_text())
    frames = _prepared_frames(reference_report, projection_report, deadline)
    output = {"schema": "berkeley_sensor_depth_v1", "interpretation": "assumption-qualified sensor-depth diagnostic, not certified shape accuracy",
              "protocol": {"angles": list(ANGLES), "max_shared_rays_per_view": MAX_RAYS_PER_VIEW,
                           "seed_base": 20260927, "interval_m": [NEAR_METRES, FAR_METRES],
                           "interior_erosion_rgb_pixels": 16, "total_deadline_seconds": MAX_SECONDS},
              "source": {"reference_camera_report_sha256": sha256(REFERENCE_003),
                         "projection_report_sha256": sha256(projection_path),
                         "calibration_sha256": sha256(CALIBRATION),
                         "evaluator_sha256": sha256(Path(__file__))},
              "status": "in_progress", "frames": [{key: value for key, value in frame.items()
                          if key not in ("observed", "rays", "interior", "ir_from_table")}
                         for frame in frames], "candidates": []}
    _write_checkpoint(checkpoint_path, output)
    try:
        for label, mesh, model, camera_report_path in candidates:
            if time.monotonic() > deadline:
                raise TimeoutError("comparison exceeded total 300-second limit")
            if not label.replace("-", "").replace("_", "").isalnum():
                raise ValueError("candidate label must be simple alphanumeric text")
            mesh, model, camera_report_path = map(Path, (mesh, model, camera_report_path))
            producer_report = mesh.parent / "result.json"
            dense_sparse = mesh.parent / "dense/sparse"
            critical = [mesh, producer_report, camera_report_path, projection_path, REFERENCE_003]
            critical += [model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
            critical += [dense_sparse / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
            before = {str(path): sha256(path) for path in critical}
            camera_report, world_to_table = _verified_camera_report(camera_report_path, model, reference_report)
            gauge = verify_dense_camera_gauge(dense_sparse, model)
            recovered_lineage = (verify_recovered_dense_lineage(producer_report, model, dense_sparse, camera_report)
                                 if camera_report["schema"] == "ycb_recovered_berkeley_camera_oracle_v1"
                                 else None)
            from scripts.classical_backend.geometry import export_geometry
            if checkpoint_path is None:
                raise ValueError("live comparison requires checkpoint path for normalized geometry")
            normalized_path = Path(checkpoint_path).parent / f"{label}-normalized-geometry.ply"
            conversion = export_geometry(mesh, normalized_path)
            info, vertices, faces = evaluate.inspect_ply(normalized_path, geometry=True)
            if info["nontriangle_faces"]:
                raise ValueError("sensor-depth surface must be triangular")
            table_vertices = transform_vertices(vertices, world_to_table)
            row = {"label": label, "stage": "rough_mesh", "mesh_sha256": before[str(mesh)],
                   "native_mesh_path": str(mesh), "producer_result_sha256": before[str(producer_report)],
                   "normalization": conversion, "dense_camera_gauge": gauge,
                   "recovered_dense_lineage": recovered_lineage,
                   "model_files_sha256": camera_report["producer_model_files_sha256"],
                   "camera_report_sha256": before[str(camera_report_path)], "frames": []}
            pooled_observed, pooled_predicted, pooled_interior = [], [], []
            for frame in frames:
                if time.monotonic() > deadline:
                    raise TimeoutError("comparison exceeded total 300-second limit")
                ir_vertices = transform_vertices(table_vertices, frame["ir_from_table"])
                predicted, degenerate = first_hit_depths(ir_vertices, faces, frame["rays"], deadline=deadline)
                observed = frame["observed"]
                pooled_observed.append(observed)
                pooled_predicted.append(predicted)
                pooled_interior.append(frame["interior"])
                row["frames"].append({"angle": frame["angle"], "excluded_zero_area_faces": degenerate,
                                      "observed_m": observed.tolist(),
                                      "predicted_m": nullable_depths(predicted),
                                      "interior_member": frame["interior"].tolist(),
                                      "coarse": residual_summary(observed, predicted),
                                      "interior": residual_summary(observed, predicted, frame["interior"])})
            all_observed = np.concatenate(pooled_observed)
            all_predicted = np.concatenate(pooled_predicted)
            all_interior = np.concatenate(pooled_interior)
            row["pooled"] = {"coarse": residual_summary(all_observed, all_predicted),
                             "interior": residual_summary(all_observed, all_predicted, all_interior)}
            after = {str(path): sha256(path) for path in critical}
            if after != before:
                raise ValueError("mesh/model/report/projection provenance changed during comparison")
            row["verified_input_sha256"] = before
            output["candidates"].append(row)
            _write_checkpoint(checkpoint_path, output)
        output["status"] = "complete"
    except Exception as error:
        output["status"] = "failed"
        output["failure"] = {"type": type(error).__name__, "message": str(error),
                             "completed_candidate_count": len(output["candidates"])}
        _write_checkpoint(checkpoint_path, output)
        raise
    _write_checkpoint(checkpoint_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs=4, metavar=("LABEL", "MESH", "MODEL", "CAMERA_REPORT"),
                        action="append", required=True)
    parser.add_argument("--projection-report", type=Path, default=PROJECTION_002)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-assumption-qualified-gate", action="store_true")
    args = parser.parse_args()
    if not args.approve_assumption_qualified_gate:
        parser.error("real scoring requires explicit root-approved qualified mapping gate")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = compare_candidates(args.candidate, args.projection_report,
                                checkpoint_path=args.output)
    print(json.dumps({"output": str(args.output), "candidates": len(result["candidates"])}))


if __name__ == "__main__":
    main()
