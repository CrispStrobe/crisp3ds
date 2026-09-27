"""Paired original-RGB-pixel depth-support audit of sealed masked YCB runs 008/015.

No reference geometry, surface fit, SfM, native reconstruction, or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from scripts.classical_backend import calibrated_control as camera_audit
from scripts.classical_backend.dense_masks import cv2, digest, model_hashes, native_mask_name
from scripts.classical_backend.run import RESERVE
from scripts.classical_backend.verify_dmap_masks import load_depth


ROOT = Path(__file__).resolve().parents[2]
RUNS = {"008": ROOT / "build-opencv/classical-ycb-native-masked-008",
        "015": ROOT / "build-opencv/classical-ycb-recovered-015"}
RUN_HASHES = {"008": "8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9",
              "015": "4a856ad150fd2817a99404bd3fe7e49bbb306e593786abe89384ff27923aed77"}
MANIFEST = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
SOURCE_MASKS = MANIFEST.parent / "masks"
MANIFEST_SHA256 = "0b78469039d11516d0ac96b642553099267cd97c9a6439a1ec4b2fcb133615bd"
STEP = 4
OFFSET = 2
MAX_OUTPUT_BYTES = 20 << 20
MAX_SECONDS = 120


def radial_inverse(pixel_centers: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Independent SIMPLE_RADIAL inverse golden, cross-checked against PyCOLMAP."""
    points = np.asarray(pixel_centers, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(params) != 4 or
            not np.isfinite(points).all() or not np.isfinite(params).all() or params[0] <= 0):
        raise ValueError("invalid SIMPLE_RADIAL pixel centers or camera")
    f, cx, cy, k = params
    distorted = (points - [cx, cy]) / f
    rd = np.linalg.norm(distorted, axis=1)
    radius = rd.copy()
    for _ in range(15):
        derivative = 1 + 3 * k * radius * radius
        if np.any(derivative <= 0):
            raise ValueError("SIMPLE_RADIAL inverse is nonmonotonic on probe domain")
        radius -= (radius + k * radius**3 - rd) / derivative
    if np.any(radius < 0) or not np.isfinite(radius).all():
        raise ValueError("SIMPLE_RADIAL inverse diverged")
    factors = np.divide(radius, rd, out=np.ones_like(radius), where=rd > 0)
    return distorted * factors[:, None]


def depth_indices(rays: np.ndarray, k: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map camera-normalized rays to nearest DMAP index (pixel-center convention)."""
    rays = np.asarray(rays, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    height, width = shape
    if (rays.ndim != 2 or rays.shape[1] != 2 or k.shape != (3, 3) or
            not np.isfinite(rays).all() or not np.isfinite(k).all() or
            not 0 < width <= 8192 or not 0 < height <= 8192 or min(k[0, 0], k[1, 1]) <= 0 or
            not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("invalid rays, DMAP K, or dimensions")
    coords = rays @ k[:2, :2].T + k[:2, 2]
    # Native K includes importer/resize -0.5; zero is the first depth sample.
    integer = np.floor(coords + 0.5)
    inside = ((integer[:, 0] >= 0) & (integer[:, 1] >= 0) &
              (integer[:, 0] < width) & (integer[:, 1] < height))
    safe = np.where(inside[:, None], integer, 0).astype(np.int64)
    return safe[:, 0], safe[:, 1], inside


def original_probe_pixels(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (1024, 1280) or mask.dtype != np.bool_:
        raise ValueError("expected original 1280x1024 boolean support mask")
    yy, xx = np.mgrid[OFFSET:1024:STEP, OFFSET:1280:STEP]
    selected = mask[yy, xx]
    return np.column_stack((xx[selected], yy[selected])).astype(np.int32)


def validate_run(label: str, run: Path, manifest: dict) -> dict:
    if run.is_symlink() or digest(run / "result.json") != RUN_HASHES[label]:
        raise ValueError(f"{label} run result differs from sealed bytes")
    report = json.loads((run / "result.json").read_text())
    expected_schema = "classical_masked_dense_v1" if label == "008" else "classical_recovered_dense_v1"
    stages = {stage["name"]: stage["status"] for stage in report.get("stages", [])}
    if (report.get("schema") != expected_schema or report.get("status") != "complete" or
            stages.get("densify") != "complete" or
            stages.get("mesh" if label == "008" else "rough_mesh") != "complete"):
        raise ValueError(f"{label} lacks a completed dense/rough chain")
    mask_report_path = run / "masks" / "report.json"
    mask_report = json.loads(mask_report_path.read_text())
    if (mask_report.get("schema") != "classical_dense_masks_v1" or
            mask_report.get("status") != "complete" or
            mask_report.get("manifest_sha256") != MANIFEST_SHA256 or
            mask_report.get("ignore_mask_label") != 0 or
            mask_report.get("max_pose_matrix_difference", float("inf")) > 1e-8 or
            report.get("mask_report_sha256", digest(mask_report_path)) != digest(mask_report_path) or
            mask_report.get("undistorted_model_sha256") != model_hashes(run / "dense" / "sparse") or
            mask_report.get("source_model_sha256") != model_hashes(Path(mask_report["source_model"]))):
        raise ValueError(f"{label} model/mask report chain differs")
    if label == "015":
        undistort = next((row for row in report["stages"] if row["name"] == "undistort"), None)
        if (mask_report["source_model_sha256"] != report.get("source_model_sha256") or
                not undistort or
                mask_report["undistorted_model_sha256"] != undistort.get("artifact", {}).get("model_sha256")):
            raise ValueError("015 mask/model bytes differ from sealed source/undistort stage")
    photo_hashes = {row["name"]: row["sha256"] for row in manifest["images"]}
    mask_hashes = {row["name"]: row["mask_sha256"] for row in manifest["images"]}
    if label == "008":
        source = Path(report["source_run"])
        if (digest(source / "result.json") != report["source_result_sha256"] or
                {row["name"]: row["sha256"] for row in json.loads((source / "result.json").read_text())["inputs"]} != photo_hashes):
            raise ValueError("008 source photo lineage differs")
    else:
        if (report.get("source_photo_sha256") != photo_hashes or
                report.get("source_pose_mask_sha256") != mask_hashes or
                report.get("source_reports", {}).get("manifest", {}).get("sha256") != MANIFEST_SHA256):
            raise ValueError("015 source photo/mask lineage differs")
    rows = {row["name"]: row for row in mask_report["images"]}
    if len(rows) != 60 or rows.keys() != photo_hashes.keys():
        raise ValueError(f"{label} mask names incomplete")
    for name, row in rows.items():
        if (row.get("source_mask_sha256") != mask_hashes[name] or
                row.get("native_mask_name") != native_mask_name(name) or
                digest(run / "masks" / row["native_mask_name"]) != row["mask_sha256"]):
            raise ValueError(f"{label} native mask differs for {name}")
    checked = camera_audit.validate_dmap_cameras(run, run / "dense" / "sparse")
    if checked["checked_views"] != 60 or checked["positive_depth_outside_mask"] != 0:
        raise ValueError(f"{label} DMAP camera or native mask check incomplete")
    return {"result_sha256": RUN_HASHES[label], "mask_report_sha256": digest(mask_report_path),
            "source_model_path": mask_report["source_model"],
            "source_model_sha256": mask_report["source_model_sha256"],
            "undistorted_model_sha256": mask_report["undistorted_model_sha256"],
            "dmap_camera_check": {key: checked[key] for key in
                                  ("checked_views", "max_abs_difference", "positive_depth_outside_mask")},
            "dmap_by_name": {row["name"]: row for row in checked["images"]}}


def status_for_probes(run: Path, name: str, probes: np.ndarray, source_camera,
                      dmap_row: dict) -> tuple[np.ndarray, dict]:
    dmap_path = next(path for path in run.glob("depth[0-9][0-9][0-9][0-9].dmap")
                     if camera_audit.read_dmap_camera(path)["name"] == name)
    entry = camera_audit.read_dmap_camera(dmap_path)
    dmap_name, depth = load_depth(dmap_path)
    if (entry["name"] != Path(dmap_name).name or digest(dmap_path) != dmap_row["dmap_sha256"]):
        raise ValueError("DMAP changed after camera check")
    # COLMAP image coordinates are pixel centers at x+0.5/y+0.5.
    centers = probes.astype(np.float64) + 0.5
    rays = np.asarray(source_camera.cam_from_img(centers), dtype=np.float64)
    independent = radial_inverse(centers, source_camera.params)
    inverse_error = float(np.max(np.abs(rays - independent))) if len(rays) else 0.0
    if inverse_error > 1e-7:
        raise ValueError(f"PyCOLMAP SIMPLE_RADIAL inverse differs from independent golden: {inverse_error}")
    x, y, inside = depth_indices(rays, entry["K"], depth.shape)
    with Image.open(run / "masks" / native_mask_name(name)) as source:
        native_mask = np.asarray(source.convert("L"))
    resized = cv2.resize(native_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    allowed = inside & (resized[y, x] != 0)
    valid = allowed & np.isfinite(depth[y, x]) & (depth[y, x] > 0)
    if np.any(valid & ~allowed):
        raise AssertionError("depth support outside native mask")
    return valid, {"probe_count": len(probes), "outside_native_frame": int(np.count_nonzero(~inside)),
                   "native_mask_excluded": int(np.count_nonzero(inside & ~allowed)),
                   "mask_allowed_but_missing_depth": int(np.count_nonzero(allowed & ~valid)),
                   "valid_depth": int(np.count_nonzero(valid)),
                   "max_inverse_difference_normalized": inverse_error,
                   "dmap_depth_size": list(entry["depth_size"]),
                   "dmap_sha256": dmap_row["dmap_sha256"]}


def run(output: Path) -> dict:
    started = time.monotonic()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a fresh file under an existing parent")
    if shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT_BYTES:
        raise ValueError("10 GiB free-space floor plus 20 MiB output allowance unavailable")
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("frozen common photo/mask manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    entries = {row["name"]: row for row in manifest["images"]}
    if len(entries) != 60 or set(entries) != {f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)}:
        raise ValueError("common original-photo inventory incomplete")
    for name, row in entries.items():
        if digest(SOURCE_MASKS / (name + ".png")) != row["mask_sha256"]:
            raise ValueError(f"common original pose mask changed: {name}")
    checks = {label: validate_run(label, path, manifest) for label, path in RUNS.items()}
    consumed = {MANIFEST, Path(__file__)}
    consumed.update(SOURCE_MASKS / (name + ".png") for name in entries)
    for label, run_path in RUNS.items():
        consumed.update((run_path / "result.json", run_path / "masks/report.json"))
        if label == "008":
            consumed.add(Path(json.loads((run_path / "result.json").read_text())["source_run"]) / "result.json")
        consumed.update(run_path / "masks" / native_mask_name(name) for name in entries)
        consumed.update(Path(checks[label]["source_model_path"]) / name
                        for name in ("cameras.bin", "images.bin", "points3D.bin"))
        consumed.update(run_path / "dense/sparse" / name
                        for name in ("cameras.bin", "images.bin", "points3D.bin"))
        consumed.update(path for path in run_path.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    consumed_before = {str(path): digest(path) for path in consumed}
    import pycolmap
    models = {label: pycolmap.Reconstruction(checks[label]["source_model_path"])
              for label in RUNS}
    cameras = {}
    for label, model in models.items():
        images = {image.name: image for image in model.images.values() if image.has_pose}
        if set(images) != set(entries):
            raise ValueError(f"{label} original camera names incomplete")
        dense_model = pycolmap.Reconstruction(str(RUNS[label] / "dense/sparse"))
        dense_images = {image.name: image for image in dense_model.images.values() if image.has_pose}
        if set(dense_images) != set(images) or max(float(np.max(np.abs(
                images[name].cam_from_world.matrix() - dense_images[name].cam_from_world.matrix())))
                for name in images) > 1e-8:
            raise ValueError(f"{label} original/undistorted camera poses differ")
        cameras[label] = {name: model.cameras[image.camera_id] for name, image in images.items()}
        if any(camera.model.name != "SIMPLE_RADIAL" or camera.width != 1280 or
               camera.height != 1024 for camera in cameras[label].values()):
            raise ValueError(f"{label} unexpected original camera model")
    result = {"schema": "classical_paired_original_rgb_depth_support_v1", "status": "complete",
              "scope": "same original photo-derived mask RGB probes; each candidate uses its own estimated camera and native DMAP",
              "not_geometry_truth": True, "reference_mesh_used": False,
              "probe_rule": {"image_size": [1280, 1024], "x_mod_4": OFFSET, "y_mod_4": OFFSET,
                             "step": STEP, "original_pixel_center": "(x+0.5,y+0.5)",
                             "native_nearest": "floor(projected_index+0.5)"},
              "manifest_sha256": MANIFEST_SHA256,
              "analyzer_sha256": digest(Path(__file__)),
              "runs": {label: {key: value for key, value in check.items() if key != "dmap_by_name"}
                       for label, check in checks.items()},
              "views": []}
    totals = {key: 0 for key in ("probe_count", "both", "only_008", "only_015", "neither")}
    for name in sorted(entries):
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("depth support audit exceeded 120 seconds")
        with Image.open(SOURCE_MASKS / (name + ".png")) as image:
            mask = np.asarray(image.convert("L")) != 0
        probes = original_probe_pixels(mask)
        probe_ids = np.asarray(probes[:, 1] * 1280 + probes[:, 0], dtype="<u4")
        flags, details = {}, {}
        for label, path in RUNS.items():
            flags[label], details[label] = status_for_probes(
                path, name, probes, cameras[label][name], checks[label]["dmap_by_name"][name])
        pairs = {"probe_count": len(probes),
                 "both": int(np.count_nonzero(flags["008"] & flags["015"])),
                 "only_008": int(np.count_nonzero(flags["008"] & ~flags["015"])),
                 "only_015": int(np.count_nonzero(~flags["008"] & flags["015"])),
                 "neither": int(np.count_nonzero(~flags["008"] & ~flags["015"]))}
        if sum(pairs[key] for key in ("both", "only_008", "only_015", "neither")) != len(probes):
            raise AssertionError("paired original probe partition differs")
        result["views"].append({"name": name, "original_mask_sha256": entries[name]["mask_sha256"],
                                "original_probe_ids_sha256": hashlib.sha256(probe_ids.tobytes()).hexdigest(),
                                "paired": pairs, "arms": details})
        for key, value in pairs.items():
            totals[key] += value
    result["pooled_paired"] = totals
    result["pooled_support_fraction"] = {
        "008": (totals["both"] + totals["only_008"]) / totals["probe_count"],
        "015": (totals["both"] + totals["only_015"]) / totals["probe_count"]}
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("depth support audit exceeded 120 seconds")
    if {str(path): digest(path) for path in consumed} != consumed_before:
        raise ValueError("a consumed camera, DMAP, mask, report or analyzer changed during audit")
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("depth support audit exceeded 120 seconds")
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_OUTPUT_BYTES or shutil.disk_usage(output.parent).free < RESERVE + len(encoded):
        raise ValueError("20 MiB output cap or 10 GiB free-space floor reached")
    with output.open("xb") as stream:
        stream.write(encoded)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "pooled": result["pooled_paired"],
                      "fractions": result["pooled_support_fraction"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
