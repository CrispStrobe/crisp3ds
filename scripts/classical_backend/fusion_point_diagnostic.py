"""Frozen point-cloud-only diagnostics for the two sealed cached-fusion arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.classical_backend import geometry
from scripts.object_dataset import evaluate, sensor_depth, surface_metrics


ROOT = sensor_depth.ROOT
PAIR = ROOT / "build-opencv/classical-ycb-fusion-control-016/fusion-pair.json"
SENSOR = ROOT / "build-opencv/sensor-depth-003/report.json"
FIT = ROOT / "build-opencv/classical-ycb-native-masked-008/reference-fit-v1.json"
REFERENCE = ROOT / ".local-tools/test-data/ycb-cracker-box/reference/google_64k_geometry_f64.ply"
EXPECTED_PAIR_SHA = "7bca23a2f64e3f3ee3af53228a38bac6f9bdb5d82e83806233c064444e052a2d"
EXPECTED_SENSOR_SHA = "40b1d821b9e7c8a9eaddf6946815922af93c3d72bacee25cfff0006dd3b0e8a7"
EXPECTED_FIT_SHA = "b51a2687faca9465e96aad795f9d16c966a3993e0ae2353b3a79eca80c44c49d"
EXPECTED_REFERENCE_SHA = "6e0187aa961aef4fa21dfc753023a4be7e82924e6398dc0875a2b0d609363ed4"
MAX_SECONDS = 120
MAX_OUTPUT = 20 << 20
SAMPLE_COUNT = 4096
SAMPLE_SEED = 2027
PIXEL_RADIUS = 2


def load_openmvs_cloud(path: Path, deadline: float | None = None) -> np.ndarray:
    """Validate and read bounded OpenMVS vertex XYZ, including variable-length lists."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= geometry.MAX_BYTES:
        raise ValueError("invalid point cloud path/size")
    with path.open("rb") as stream:
        vertex, face = geometry._header(stream)
        if face["count"] != 0:
            raise ValueError("diagnostic requires a point cloud, not a mesh")
        xyz = np.empty((vertex["count"], 3), dtype=np.float64)
        for i in range(vertex["count"]):
            if deadline is not None and i % 2048 == 0 and time.monotonic() >= deadline:
                raise TimeoutError("point-cloud parsing exceeded deadline")
            record = {}
            for name, fmt, item_fmt in vertex["props"]:
                if item_fmt is None:
                    value = geometry._read(stream, fmt)[0]
                else:
                    length = geometry._read(stream, fmt)[0]
                    value = geometry._read(stream, item_fmt, length)
                if name in ("x", "y", "z"):
                    record[name] = value
            xyz[i] = (record["x"], record["y"], record["z"])
        if stream.read(1) or not np.isfinite(xyz).all():
            raise ValueError("point cloud has trailing data or nonfinite XYZ")
    return xyz


def projected_cloud_depth(points_ir, intrinsic, selected_indices, *, radius=PIXEL_RADIUS,
                          width=640, height=480):
    """Nearest integer point z-buffer; fixed square neighborhood, not ray-surface hit."""
    points = np.asarray(points_ir, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or
            k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
            selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= width * height) or
            not isinstance(radius, int) or radius < 0 or radius > 4):
        raise ValueError("invalid point projection inputs")
    grid = np.full((height, width), np.inf, dtype=np.float64)
    z = points[:, 2]
    valid = (z >= sensor_depth.NEAR_METRES) & (z <= sensor_depth.FAR_METRES)
    x = np.floor(k[0, 0] * points[valid, 0] / z[valid] + k[0, 2] + 0.5).astype(np.int64)
    y = np.floor(k[1, 1] * points[valid, 1] / z[valid] + k[1, 2] + 0.5).astype(np.int64)
    zz = z[valid]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.minimum.at(grid, (y[inside], x[inside]), zz[inside])
    rows, cols = np.divmod(selected, width)
    result = np.full(len(selected), np.inf, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = rows + dy, cols + dx
            good = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            result[good] = np.minimum(result[good], grid[yy[good], xx[good]])
    result[~np.isfinite(result)] = np.nan
    return result


def paired_summary(observed, control, ablation, support):
    observed = np.asarray(observed, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    ablation = np.asarray(ablation, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if any(value.shape != observed.shape for value in (control, ablation, support)):
        raise ValueError("paired arrays disagree")
    a, b = support & np.isfinite(control), support & np.isfinite(ablation)
    both, lost, gained, neither = a & b, a & ~b, ~a & b, support & ~a & ~b
    def accuracy(values, mask):
        residual = values[mask] - observed[mask]
        return {"count": int(mask.sum()),
                "mean_absolute_m": float(np.mean(np.abs(residual))) if len(residual) else None,
                "within_5mm": float(np.mean(np.abs(residual) <= .005)) if len(residual) else None}
    return {"supported": int(support.sum()), "both_hit": int(both.sum()),
            "control_only": int(lost.sum()), "ablation_only": int(gained.sum()),
            "both_missing": int(neither.sum()),
            "shared_control": accuracy(control, both), "shared_ablation": accuracy(ablation, both),
            "gained_ablation": accuracy(ablation, gained), "lost_control": accuracy(control, lost)}


def _bounds(points):
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist(),
            "extent": np.ptp(points, axis=0).tolist()}


def _reference_distances(points, matrix, bvh, diagonal, deadline):
    chosen = np.random.default_rng(SAMPLE_SEED).choice(len(points), min(SAMPLE_COUNT, len(points)), replace=False)
    mapped = sensor_depth.transform_vertices(points[chosen], matrix)
    distances = bvh.distances(mapped, deadline=deadline)
    return {"sample_count": len(chosen), "seed": SAMPLE_SEED,
            "mean": float(np.mean(distances)), "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "within_reference_bbox_fraction": {str(t): float(np.mean(distances <= diagonal * t))
                                               for t in (.005, .01, .02)}}


def run(output: Path) -> dict:
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < sensor_depth.MIN_FREE_BYTES:
        raise ValueError("less than 10 GiB free")
    output.mkdir()
    report = {"schema": "openmvs_fusion_point_diagnostic_v1", "status": "running",
              "limitations": "point-cloud splat support, not ray-triangle depth or physical metrology"}
    try:
        for path, expected in ((PAIR, EXPECTED_PAIR_SHA), (SENSOR, EXPECTED_SENSOR_SHA),
                               (FIT, EXPECTED_FIT_SHA), (REFERENCE, EXPECTED_REFERENCE_SHA)):
            if sensor_depth.sha256(path) != expected:
                raise ValueError(f"sealed input changed: {path}")
        pair, sensor = json.loads(PAIR.read_text()), json.loads(SENSOR.read_text())
        fit = json.loads(FIT.read_text())
        sealed = {PAIR: EXPECTED_PAIR_SHA, SENSOR: EXPECTED_SENSOR_SHA,
                  FIT: EXPECTED_FIT_SHA, REFERENCE: EXPECTED_REFERENCE_SHA,
                  sensor_depth.REFERENCE_003: sensor["source"]["reference_camera_report_sha256"],
                  sensor_depth.PROJECTION_002: sensor["source"]["projection_report_sha256"],
                  sensor_depth.CALIBRATION: sensor["source"]["calibration_sha256"]}
        for label, arm_name in (("control", "control_dense_fuse"), ("ablation", "ablation_simple_fuse")):
            path = ROOT / ("build-opencv/classical-ycb-fusion-control-016" if label == "control"
                           else "build-opencv/classical-ycb-fusion-simple-017") / "dense.ply"
            sealed[path] = pair["arms"][arm_name]["stages"][0]["artifact"]["dense_ply_sha256"]
        for path, expected in sealed.items():
            if sensor_depth.sha256(path) != expected:
                raise ValueError(f"frozen input differs before diagnostic: {path}")
        reference_report = json.loads(sensor_depth.REFERENCE_003.read_text())
        projection_report = json.loads(sensor_depth.PROJECTION_002.read_text())
        if (pair["status"] != "complete" or sensor["status"] != "complete" or
                fit["reference_sha256"] != EXPECTED_REFERENCE_SHA or
                fit["output_sha256"] != "f883c5dc6a0983470b8971168620e796729d8380a341a6367e27c604640fb4ad"):
            raise ValueError("source result/frozen alignment differs")
        matrix_table = sensor_depth.proper_similarity(reference_report["matrix_estimated_world_to_berkeley_table"])
        matrix_ref = sensor_depth.proper_similarity(fit["matrix"])
        frames = sensor_depth._prepared_frames(reference_report, projection_report, deadline)
        baseline = next(c for c in sensor["candidates"] if c["label"] == "rough008")
        if len(frames) != len(sensor["frames"]) or len(frames) != len(baseline["frames"]):
            raise ValueError("frozen frame count differs")
        for index, (fresh, old) in enumerate(zip(frames, sensor["frames"])):
            if (fresh["angle"] != old["angle"] or fresh["selected_indices"] != old["selected_indices"] or
                    not np.array_equal(fresh["observed"], baseline["frames"][index]["observed_m"])):
                raise ValueError("selected rays/observations differ from frozen sensor report")
        k = np.asarray(projection_report["depth_K"], dtype=np.float64)
        reference_geometry, vertices, faces = evaluate.inspect_ply(REFERENCE, geometry=True)
        triangles, excluded = surface_metrics.positive_triangles(vertices, faces)
        if excluded or reference_geometry["nontriangle_faces"]:
            raise ValueError("reference mesh is not fully positive triangles")
        bvh = surface_metrics.TriangleBVH(triangles, deadline=deadline)
        diagonal = float(np.linalg.norm(reference_geometry["bbox_extent"]))
        predictions = {}
        candidates = {}
        for label, arm_name in (("control", "control_dense_fuse"), ("ablation", "ablation_simple_fuse")):
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic deadline exceeded")
            arm = pair["arms"][arm_name]
            path = ROOT / ("build-opencv/classical-ycb-fusion-control-016" if label == "control"
                           else "build-opencv/classical-ycb-fusion-simple-017") / "dense.ply"
            if (arm["status"] != "complete" or not arm["source_inputs_unchanged"] or
                    not arm["copied_inputs_unchanged"] or
                    sensor_depth.sha256(path) != arm["stages"][0]["artifact"]["dense_ply_sha256"]):
                raise ValueError("fusion arm unsealed")
            points = load_openmvs_cloud(path, deadline)
            if len(points) != arm["stages"][0]["artifact"]["points"]:
                raise ValueError("point count differs from sealed stage")
            table = sensor_depth.transform_vertices(points, matrix_table)
            predictions[label] = []
            frame_rows = []
            for frame in frames:
                ir = table @ frame["ir_from_table"][:3, :3].T + frame["ir_from_table"][:3, 3]
                predicted = projected_cloud_depth(ir, k, frame["selected_indices"])
                predictions[label].append(predicted)
                frame_rows.append({"angle": frame["angle"], "predicted_m": sensor_depth.nullable_depths(predicted),
                                   "coarse": sensor_depth.residual_summary(frame["observed"], predicted),
                                   "interior": sensor_depth.residual_summary(frame["observed"], predicted, frame["interior"])})
            candidates[label] = {"cloud_sha256": sensor_depth.sha256(path), "points": len(points),
                                 "world_bbox": _bounds(points), "table_bbox_m": _bounds(table),
                                 "reference_fit_point_to_triangle": _reference_distances(points, matrix_ref, bvh, diagonal, deadline),
                                 "frames": frame_rows}
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic deadline exceeded after candidate")
        paired = []
        for i, frame in enumerate(frames):
            paired.append({"angle": frame["angle"],
                           "coarse": paired_summary(frame["observed"], predictions["control"][i],
                                                    predictions["ablation"][i], np.ones(len(frame["observed"]), bool)),
                           "interior": paired_summary(frame["observed"], predictions["control"][i],
                                                      predictions["ablation"][i], frame["interior"])})
        for path, expected in sealed.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic deadline exceeded during final hash checks")
            if sensor_depth.sha256(path) != expected:
                raise ValueError(f"frozen input differs after diagnostic: {path}")
        report.update(status="complete", source_sha256={"pair": EXPECTED_PAIR_SHA, "sensor": EXPECTED_SENSOR_SHA,
                                                       "camera": sensor_depth.sha256(sensor_depth.REFERENCE_003),
                                                       "projection": sensor_depth.sha256(sensor_depth.PROJECTION_002),
                                                       "reference": EXPECTED_REFERENCE_SHA, "fit": sensor_depth.sha256(FIT)},
                      protocol={"angles": list(sensor_depth.ANGLES), "selected_rays_per_view": 2048,
                                "projection": "nearest integer depth pixel, first point Z, Chebyshev radius 2 pixels",
                                "sensor_thresholds_m": [.005, .01, .02], "scanner_sample_seed": SAMPLE_SEED,
                                "scanner_sample_max": SAMPLE_COUNT, "scanner_bbox_fractions": [.005, .01, .02],
                                "deadline_seconds": MAX_SECONDS}, sealed_inputs_before_after=True,
                      candidates=candidates, paired=paired)
    except Exception as exc:
        report.update(status="failed", failure=repr(exc))
    report["elapsed_seconds"] = time.monotonic() - started
    if report["elapsed_seconds"] > MAX_SECONDS:
        report.update(status="failed", failure="120-second total diagnostic deadline exceeded")
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) >= MAX_OUTPUT:
        raise ValueError("diagnostic report exceeds 20 MiB cap")
    (output / "report.json").write_bytes(payload)
    if report["status"] != "complete":
        raise RuntimeError(report["failure"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
