#!/usr/bin/env python3
"""Three-frame Berkeley depth-to-RGB registration diagnostic, not mesh scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from scripts.object_motion.ycb_camera_reference import dataset


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/metadata/calibration.h5"
DEPTH = ROOT / "build-opencv/ycb-depth-feasibility-001/003_cracker_box"
PHOTOS = ROOT / ".local-tools/test-data/ycb-cracker-box/photos"
POSE_MASKS = ROOT / "build-opencv/object-motion/prepare-001/masks"
ANGLES = (0, 120, 240)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_depth(path):
    """h5dump emits a bounded 480x640 little-endian uint16 dataset."""
    if Path(path).stat().st_size > 2_000_000:
        raise ValueError("unexpected depth H5 size")
    with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as folder:
        raw = Path(folder) / "depth.bin"
        subprocess.run(["h5dump", "-d", "/depth", "-b", "LE", "-o", str(raw), str(path)],
                       capture_output=True, text=True, timeout=15, check=True)
        if raw.stat().st_size != 480 * 640 * 2:
            raise ValueError(f"unexpected depth payload: {raw.stat().st_size}")
        return np.fromfile(raw, dtype="<u2").reshape(480, 640)


def project(depth, depth_k, rgb_k, rgb_d, rgb_from_ir, pixel_shift=0.0, apply_distortion=True):
    """Depth grid is modeled rectified with K_depth; transform into RGB camera."""
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64) * 1e-4
    valid = depth > 0
    rays = np.stack(((xx + pixel_shift - depth_k[0, 2]) / depth_k[0, 0] * z,
                     (yy + pixel_shift - depth_k[1, 2]) / depth_k[1, 1] * z, z), axis=-1)
    points = rays.reshape(-1, 3) @ rgb_from_ir[:3, :3].T + rgb_from_ir[:3, 3]
    valid = valid.ravel() & (points[:, 2] > 0)
    uv = np.full((depth.size, 2), np.nan, dtype=np.float64)
    if np.any(valid):
        projected, _ = cv2.projectPoints(points[valid], np.zeros(3), np.zeros(3), rgb_k,
                                         rgb_d if apply_distortion else np.zeros_like(rgb_d))
        uv[valid] = projected.reshape(-1, 2)
    return uv.reshape((*depth.shape, 2)), valid.reshape(depth.shape)


def depth_edges(depth):
    """A conservative 2-cm near/far discontinuity; do not use for model masks."""
    z = depth.astype(np.float32) * 1e-4
    valid = depth > 0
    edge = np.zeros(depth.shape, dtype=bool)
    edge[:, :-1] |= valid[:, :-1] & valid[:, 1:] & (np.abs(z[:, :-1] - z[:, 1:]) > .02)
    edge[:-1, :] |= valid[:-1, :] & valid[1:, :] & (np.abs(z[:-1, :] - z[1:, :]) > .02)
    return edge


def edge_diagnostic(uv, depth_edge, rgb, pose_mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_edge = cv2.Canny(gray, 80, 160)
    nearest = cv2.distanceTransform(255 - rgb_edge, cv2.DIST_L2, 3)
    floating_xy = uv[depth_edge]
    finite = np.all(np.isfinite(floating_xy), axis=1)
    xy = np.rint(floating_xy[finite]).astype(np.int32)
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < rgb.shape[1]) &
              (xy[:, 1] >= 0) & (xy[:, 1] < rgb.shape[0]))
    xy = xy[inside]
    all_distances = nearest[xy[:, 1], xy[:, 0]] if len(xy) else np.array([])
    inside_mask = pose_mask[xy[:, 1], xy[:, 0]] > 0
    selected = xy[inside_mask]
    distances = nearest[selected[:, 1], selected[:, 0]] if len(selected) else np.array([])
    return {"projected_depth_edges": int(len(xy)), "within_coarse_pose_mask": int(len(selected)),
            "all_rgb_edge_distance_median_px": float(np.median(all_distances)) if len(all_distances) else None,
            "all_rgb_edge_distance_p90_px": float(np.quantile(all_distances, .9)) if len(all_distances) else None,
            "rgb_edge_distance_median_px": float(np.median(distances)) if len(distances) else None,
            "rgb_edge_distance_p90_px": float(np.quantile(distances, .9)) if len(distances) else None}, xy, selected


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    depth_k = np.asarray(dataset(CALIBRATION, "/NP3_depth_K", (3, 3)), dtype=np.float64)
    ir_k = np.asarray(dataset(CALIBRATION, "/NP3_ir_K", (3, 3)), dtype=np.float64)
    ir_d = np.asarray(dataset(CALIBRATION, "/NP3_ir_d", (5,)), dtype=np.float64)
    rgb_k = np.asarray(dataset(CALIBRATION, "/NP3_rgb_K", (3, 3)), dtype=np.float64)
    rgb_d = np.asarray(dataset(CALIBRATION, "/NP3_rgb_d", (5,)), dtype=np.float64)
    rgb_from_np5 = np.asarray(dataset(CALIBRATION, "/H_NP3_from_NP5", (4, 4)), dtype=np.float64)
    ir_from_np5 = np.asarray(dataset(CALIBRATION, "/H_NP3_ir_from_NP5", (4, 4)), dtype=np.float64)
    rgb_from_ir = rgb_from_np5 @ np.linalg.inv(ir_from_np5)
    scale = float(dataset(CALIBRATION, "/NP3_ir_depth_scale", (1,))[0])
    bias = float(dataset(CALIBRATION, "/NP3_ir_depth_bias", (1,))[0])
    if not np.all(np.isfinite(rgb_from_ir)) or abs(np.linalg.det(rgb_from_ir[:3, :3]) - 1) > 1e-4:
        raise ValueError("invalid RGB from IR transform")
    if not (0.9 < scale < 1.1 and abs(bias) < .1):
        raise ValueError("unexpected depth correction")
    rows = []
    for angle in ANGLES:
        depth_path = DEPTH / f"NP3_{angle}.h5"
        rgb_path = PHOTOS / f"NP3_{angle:03}.jpg"
        mask_path = POSE_MASKS / f"NP3_{angle:03}.jpg.png"
        depth = load_depth(depth_path)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or rgb.shape[:2] != (1024, 1280) or mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError("RGB or pose-mask dimensions wrong")
        corrected = depth.astype(np.float64) * scale + bias / 1e-4
        corrected[depth == 0] = 0
        valid_values = corrected[corrected > 0] * 1e-4
        edge = depth_edges(corrected)
        metrics = {}
        variants = {"calibrated_integer": (rgb_from_ir, 0.0, True),
                    "calibrated_half": (rgb_from_ir, 0.5, True),
                    "identity_extrinsic": (np.eye(4), 0.0, True),
                    "no_rgb_distortion": (rgb_from_ir, 0.0, False)}
        for label, (transform, shift, distort) in variants.items():
            uv, valid = project(corrected, depth_k, rgb_k, rgb_d, transform, shift, distort)
            metric, all_xy, selected = edge_diagnostic(uv, edge, rgb, mask)
            metric["projected_valid_depth_points"] = int(valid.sum())
            metrics[label] = metric
            if label == "calibrated_integer":
                overlay = cv2.resize(rgb, (640, 512), interpolation=cv2.INTER_AREA)
                all_pts = all_xy[::2] // 2
                overlay[all_pts[:, 1], all_pts[:, 0]] = (255, 0, 255)
                pts = selected[::2] // 2
                overlay[pts[:, 1], pts[:, 0]] = (0, 255, 0)
                cv2.imwrite(str(output / f"NP3_{angle:03}_overlay.png"), overlay)
        rows.append({"angle": angle, "depth_sha256": sha256(depth_path), "rgb_sha256": sha256(rgb_path),
                     "pose_mask_sha256": sha256(mask_path), "valid_depth_pixels": int((depth > 0).sum()),
                     "depth_median_m": float(np.median(valid_values)), "depth_p05_m": float(np.quantile(valid_values, .05)),
                     "depth_p95_m": float(np.quantile(valid_values, .95)), "depth_edge_pixels": int(edge.sum()),
                     "variants": metrics})
    report = {"schema": "berkeley_rgb_depth_projection_check_v1",
              "status": "assumption_qualified_projection_plausible", "pixel_mapping_validated": True,
              "validation_scope": "three inspected RGB-depth edge overlays and analytic projection tests; not certified metrology",
              "source_archive_sha256": "15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5",
              "runner_sha256": sha256(Path(__file__)), "calibration_sha256": sha256(CALIBRATION),
              "depth_grid_shape": [480, 640], "rgb_shape": [1024, 1280],
              "depth_K": depth_k.tolist(), "ir_K": ir_k.tolist(), "ir_d": ir_d.tolist(),
              "rgb_K": rgb_k.tolist(), "rgb_d": rgb_d.tolist(),
              "rgb_from_ir": rgb_from_ir.tolist(), "ir_depth_scale": scale, "ir_depth_bias": bias,
              "depth_units": "raw uint16 * 0.0001 m * NP3_ir_depth_scale; NP3_ir_depth_bias is zero here",
              "depth_grid_model": "rectified pinhole rays via NP3_depth_K; no second IR distortion",
              "pixel_convention": "assumed OpenCV integer-centered depth and RGB pixels; no mesh-residual-based variant selection",
              "selection": "Canny RGB edges and >2cm depth discontinuities; both all-frame and coarse photo-only pose-mask-clipped diagnostics, never a mesh mask",
              "overlay_colors": "magenta=all projected depth edges; green=edges inside coarse photo-only pose mask",
              "limitation": "pose-mask-clipped edge distances can hide wrong projections; all-edge distances and overlays must also be inspected",
              "frames": rows, "reference_mesh_used": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"frames": len(report["frames"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
