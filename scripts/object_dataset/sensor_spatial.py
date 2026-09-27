#!/usr/bin/env python3
"""Read-only fixed-grid partition of the frozen three-view depth report."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "build-opencv/sensor-depth-003/report.json"
INPUT_SHA256 = "40b1d821b9e7c8a9eaddf6946815922af93c3d72bacee25cfff0006dd3b0e8a7"
MIN_FREE_BYTES = 10 * 1024**3
THRESHOLDS_M = (0.005, 0.01)
ANGLES = (0, 120, 240)
LABELS = ("rough008", "rough014")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_ids(pixel_ids):
    """Fixed 4x4 grid in 640x480 depth-image coordinates, row major."""
    pixel_ids = np.asarray(pixel_ids)
    if pixel_ids.ndim != 1 or not np.issubdtype(pixel_ids.dtype, np.integer):
        raise ValueError("pixel IDs must be an integer vector")
    if len(pixel_ids) and (pixel_ids.min() < 0 or pixel_ids.max() >= 640 * 480):
        raise ValueError("pixel ID outside 640x480 frame")
    y, x = np.divmod(pixel_ids, 640)
    return (y // 120 * 4 + x // 160).astype(np.uint8)


def summarize(observed, predicted, membership):
    """Missing-inclusive coverage/rates and separate hit-only residuals."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    membership = np.asarray(membership, dtype=bool)
    if (observed.ndim != 1 or predicted.shape != observed.shape
            or membership.shape != observed.shape or not np.isfinite(observed).all()
            or np.any(observed <= 0) or np.isinf(predicted).any()
            or np.any(predicted[np.isfinite(predicted)] <= 0)):
        raise ValueError("invalid paired depths or support")
    support_count = int(np.count_nonzero(membership))
    hit = membership & np.isfinite(predicted)
    hit_count = int(np.count_nonzero(hit))
    out = {"supported": support_count, "first_hits": hit_count,
           "no_hit": support_count - hit_count,
           "first_hit_fraction": hit_count / support_count if support_count else None}
    absolute = np.abs(predicted[hit] - observed[hit])
    out["within_5mm_all_supported_fraction"] = (
        float(np.count_nonzero(absolute <= THRESHOLDS_M[0]) / support_count)
        if support_count else None)
    out["within_10mm_all_supported_fraction"] = (
        float(np.count_nonzero(absolute <= THRESHOLDS_M[1]) / support_count)
        if support_count else None)
    out["hit_only_absolute_m"] = (
        {"mean": float(np.mean(absolute)), "median": float(np.median(absolute)),
         "p95": float(np.percentile(absolute, 95))} if hit_count else None)
    return out


def partition(pixel_ids, observed, predicted, interior):
    pixel_ids = np.asarray(pixel_ids)
    cells = cell_ids(pixel_ids)
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    interior = np.asarray(interior, dtype=bool)
    if observed.shape != pixel_ids.shape or predicted.shape != pixel_ids.shape or interior.shape != pixel_ids.shape:
        raise ValueError("ray vectors must have identical lengths")
    rows = []
    for cell in range(16):
        member = cells == cell
        rows.append({"row": cell // 4, "column": cell % 4,
                     "pixel_bounds_xy_exclusive": [cell % 4 * 160, cell // 4 * 120,
                                                   (cell % 4 + 1) * 160, (cell // 4 + 1) * 120],
                     "coarse": summarize(observed, predicted, member),
                     "eroded_interior": summarize(observed, predicted, member & interior)})
    return rows


def _verified_input(report):
    if (report.get("schema") != "berkeley_sensor_depth_v1" or report.get("status") != "complete"
            or tuple(frame.get("angle") for frame in report.get("frames", [])) != ANGLES
            or tuple(candidate.get("label") for candidate in report.get("candidates", [])) != LABELS):
        raise ValueError("unexpected frozen sensor report schema, status, angles or candidates")
    for index, frame in enumerate(report["frames"]):
        ids = frame.get("selected_indices")
        if (not isinstance(ids, list) or len(ids) != 2048
                or any(type(value) is not int or value < 0 or value >= 640 * 480 for value in ids)
                or any(a >= b for a, b in zip(ids, ids[1:]))):
            raise ValueError("selected IDs differ from frozen unique ordered rays")
        digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
        if digest != frame.get("selection", {}).get("selected_sha256"):
            raise ValueError("selected IDs do not match recorded hash")
        baseline = report["candidates"][0]["frames"][index]
        if baseline.get("angle") != frame["angle"]:
            raise ValueError("candidate frame order differs")
        if (len(baseline.get("observed_m", [])) != len(ids)
                or len(baseline.get("predicted_m", [])) != len(ids)
                or len(baseline.get("interior_member", [])) != len(ids)
                or any(type(flag) is not bool for flag in baseline["interior_member"])):
            raise ValueError("missing or invalid per-ray arrays")
        for candidate in report["candidates"]:
            current = candidate["frames"][index]
            if (current.get("angle") != frame["angle"]
                    or current.get("observed_m") != baseline["observed_m"]
                    or current.get("interior_member") != baseline["interior_member"]
                    or len(current.get("predicted_m", [])) != len(ids)
                    or any(value is not None and (type(value) not in (float, int)
                                or not math.isfinite(value) or value <= 0) for value in current["predicted_m"])):
                raise ValueError("candidate per-ray support or prediction invalid")
        summarize(baseline["observed_m"], baseline["predicted_m"],
                  np.ones(len(ids), dtype=bool))


def analyze(report, parent_sha256):
    _verified_input(report)
    result = {"schema": "berkeley_sensor_spatial_v1", "status": "complete",
              "interpretation": "fixed image-space partition of existing sampled rays; not physical ground truth",
              "parent_report_sha256": parent_sha256,
              "cell_grid": {"columns": 4, "rows": 4, "cell_width_px": 160, "cell_height_px": 120},
              "thresholds_m": list(THRESHOLDS_M), "views": []}
    for frame_index, frame in enumerate(report["frames"]):
        view = {"angle": frame["angle"], "selected_sha256": frame["selection"]["selected_sha256"],
                "selected_count": len(frame["selected_indices"]), "candidates": []}
        for candidate in report["candidates"]:
            ray_frame = candidate["frames"][frame_index]
            view["candidates"].append({"label": candidate["label"],
                "native_mesh_sha256": candidate["mesh_sha256"],
                "camera_report_sha256": candidate["camera_report_sha256"],
                "cells": partition(frame["selected_indices"], ray_frame["observed_m"],
                                   ray_frame["predicted_m"], ray_frame["interior_member"])})
        result["views"].append(view)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise OSError("spatial audit requires at least 10 GiB free disk")
    if args.input.stat().st_size > 20_000_000 or sha256(args.input) != INPUT_SHA256:
        raise ValueError("input is not the pinned sensor-depth-003 report")
    report = json.loads(args.input.read_text())
    result = analyze(report, INPUT_SHA256)
    result["analyzer_sha256"] = sha256(Path(__file__))
    if sha256(args.input) != INPUT_SHA256:
        raise ValueError("input changed during audit")
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 1_000_000:
        raise ValueError("spatial report exceeds 1 MB cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"output": str(args.output), "bytes": len(encoded)}))


if __name__ == "__main__":
    main()
