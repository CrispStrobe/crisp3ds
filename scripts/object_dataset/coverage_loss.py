#!/usr/bin/env python3
"""Paired hit/miss and shared-hit residual audit of sealed sensor-depth004."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.object_dataset.sensor_spatial import cell_ids


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "build-opencv/sensor-depth-004/report.json"
INPUT_SHA256 = "9027600ccac5488ec422a6adc3736979eb126a7b33d9d8d19d851b107327cc5c"
LABELS = ("rough008", "rough015")
ANGLES = (0, 120, 240)
THRESHOLDS_M = (0.005, 0.01)
MIN_FREE_BYTES = 10 * 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values):
    return ({"mean": float(np.mean(values)), "median": float(np.median(values)),
             "p95": float(np.percentile(values, 95))} if len(values) else None)


def paired_summary(observed, baseline, recovered, membership):
    """Loss/gain counts; residual changes only where both candidates hit."""
    observed = np.asarray(observed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    membership = np.asarray(membership, dtype=bool)
    if (observed.ndim != 1 or baseline.shape != observed.shape or
            recovered.shape != observed.shape or membership.shape != observed.shape or
            not np.isfinite(observed).all() or np.any(observed <= 0) or
            np.isinf(baseline).any() or np.isinf(recovered).any() or
            np.any(baseline[np.isfinite(baseline)] <= 0) or
            np.any(recovered[np.isfinite(recovered)] <= 0)):
        raise ValueError("invalid paired observed/predicted depth arrays")
    left_hit = np.isfinite(baseline)
    right_hit = np.isfinite(recovered)
    both = membership & left_hit & right_hit
    lost = membership & left_hit & ~right_hit
    gained = membership & ~left_hit & right_hit
    missing = membership & ~left_hit & ~right_hit
    n = int(np.count_nonzero(membership))
    counts = {"supported": n, "both_hit": int(np.count_nonzero(both)),
              "lost_008_hit_015_missing": int(np.count_nonzero(lost)),
              "gained_008_missing_015_hit": int(np.count_nonzero(gained)),
              "both_missing": int(np.count_nonzero(missing))}
    if sum(value for key, value in counts.items() if key != "supported") != n:
        raise AssertionError("paired hit-state partition failed")
    left_signed = baseline[both] - observed[both]
    right_signed = recovered[both] - observed[both]
    left_abs, right_abs = np.abs(left_signed), np.abs(right_signed)
    delta_abs = right_abs - left_abs
    out = {"transitions": counts,
           "hit_fraction_all_supported": {"rough008": (int(np.count_nonzero(membership & left_hit)) / n if n else None),
                                          "rough015": (int(np.count_nonzero(membership & right_hit)) / n if n else None)},
           "shared_hit_residual_m": {
               "rough008_signed": _distribution(left_signed), "rough015_signed": _distribution(right_signed),
               "rough008_absolute": _distribution(left_abs), "rough015_absolute": _distribution(right_abs),
               "015_minus_008_absolute": _distribution(delta_abs),
               "015_lower_absolute_count": int(np.count_nonzero(delta_abs < 0)),
               "015_higher_absolute_count": int(np.count_nonzero(delta_abs > 0)),
               "equal_absolute_count": int(np.count_nonzero(delta_abs == 0))}}
    for threshold in THRESHOLDS_M:
        name = f"{threshold:.3f}"
        out.setdefault("missing_inclusive_within_threshold", {})[name] = {
            "rough008_count": int(np.count_nonzero(membership & left_hit &
                                                 (np.abs(baseline - observed) <= threshold))),
            "rough015_count": int(np.count_nonzero(membership & right_hit &
                                                 (np.abs(recovered - observed) <= threshold)))}
        record = out["missing_inclusive_within_threshold"][name]
        record["rough008_fraction"] = record["rough008_count"] / n if n else None
        record["rough015_fraction"] = record["rough015_count"] / n if n else None
        out.setdefault("shared_hit_within_threshold", {})[name] = {
            "rough008_count": int(np.count_nonzero(left_abs <= threshold)),
            "rough015_count": int(np.count_nonzero(right_abs <= threshold)),
            "denominator_both_hit": counts["both_hit"]}
    return out


def _verified_input(report):
    if (report.get("schema") != "berkeley_sensor_depth_v1" or report.get("status") != "complete" or
            tuple(frame.get("angle") for frame in report.get("frames", [])) != ANGLES or
            report.get("protocol", {}).get("max_shared_rays_per_view") != 2048):
        raise ValueError("not the frozen three-view sensor-depth protocol")
    candidates = {row.get("label"): row for row in report.get("candidates", [])}
    if set(candidates) != {"rough008", "rough014", "rough015"} or len(report["candidates"]) != 3:
        raise ValueError("unexpected frozen sensor candidates")
    for view, frame in enumerate(report["frames"]):
        ids = frame.get("selected_indices", [])
        if (len(ids) != 2048 or any(type(x) is not int or not 0 <= x < 640 * 480 for x in ids)
                or any(a >= b for a, b in zip(ids, ids[1:])) or
                hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest() !=
                frame.get("selection", {}).get("selected_sha256")):
            raise ValueError("selected ray IDs/hash differ from sealed protocol")
        left = candidates[LABELS[0]]["frames"][view]
        right = candidates[LABELS[1]]["frames"][view]
        if (left.get("angle") != frame["angle"] or right.get("angle") != frame["angle"] or
                left.get("observed_m") != right.get("observed_m") or
                left.get("interior_member") != right.get("interior_member") or
                any(len(row.get(key, [])) != len(ids) for row in (left, right)
                    for key in ("observed_m", "predicted_m", "interior_member")) or
                any(type(flag) is not bool for flag in left["interior_member"])):
            raise ValueError("candidate rays, observations or support differ")
        paired_summary(left["observed_m"], left["predicted_m"], right["predicted_m"],
                       np.ones(len(ids), dtype=bool))
    return candidates


def analyze(report, parent_sha256):
    candidates = _verified_input(report)
    views, aggregate = [], []
    for view, frame in enumerate(report["frames"]):
        left, right = (candidates[label]["frames"][view] for label in LABELS)
        ids = np.asarray(frame["selected_indices"])
        observed = np.asarray(left["observed_m"], dtype=float)
        baseline = np.asarray(left["predicted_m"], dtype=float)
        recovered = np.asarray(right["predicted_m"], dtype=float)
        interior = np.asarray(left["interior_member"], dtype=bool)
        cells = cell_ids(ids)
        regions = {"coarse": np.ones(len(ids), bool), "interior": interior,
                   "boundary_band": ~interior}
        row = {"angle": frame["angle"], "selected_sha256": frame["selection"]["selected_sha256"],
               "regions": {name: paired_summary(observed, baseline, recovered, mask)
                           for name, mask in regions.items()}, "cells": []}
        for cell in range(16):
            row["cells"].append({"row": cell // 4, "column": cell % 4,
                 "pixel_bounds_xy_exclusive": [cell % 4 * 160, cell // 4 * 120,
                                               (cell % 4 + 1) * 160, (cell // 4 + 1) * 120],
                 "regions": {name: paired_summary(observed, baseline, recovered, mask & (cells == cell))
                             for name, mask in regions.items()}})
        views.append(row)
        aggregate.append((ids, observed, baseline, recovered, interior))
    ids = np.concatenate([part[0] for part in aggregate])
    observed = np.concatenate([part[1] for part in aggregate])
    baseline = np.concatenate([part[2] for part in aggregate])
    recovered = np.concatenate([part[3] for part in aggregate])
    interior = np.concatenate([part[4] for part in aggregate])
    pooled_regions = {"coarse": np.ones(len(ids), bool), "interior": interior,
                      "boundary_band": ~interior}
    pooled = {name: paired_summary(observed, baseline, recovered, mask)
              for name, mask in pooled_regions.items()}
    for name in pooled_regions:
        for key in ("supported", "both_hit", "lost_008_hit_015_missing",
                    "gained_008_missing_015_hit", "both_missing"):
            expected = sum(view["regions"][name]["transitions"][key] for view in views)
            if pooled[name]["transitions"][key] != expected:
                raise AssertionError("pooled transition count does not reconcile with views")
    return {"schema": "berkeley_coverage_loss_v1", "status": "complete",
            "parent_report_sha256": parent_sha256,
            "candidate_mesh_sha256": {label: candidates[label]["mesh_sha256"] for label in LABELS},
            "interpretation": "paired selected rays only; fixed image-space cells and photo-mask partitions; no causal or metrology claim",
            "grid": {"rows": 4, "columns": 4, "depth_frame_pixels": [640, 480]},
            "thresholds_m": list(THRESHOLDS_M), "views": views, "pooled_regions": pooled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise OSError("coverage audit requires at least 10 GiB free disk")
    if args.input.stat().st_size > 20_000_000 or sha256(args.input) != INPUT_SHA256:
        raise ValueError("input is not pinned sensor-depth-004")
    report = analyze(json.loads(args.input.read_text()), INPUT_SHA256)
    report["analyzer_sha256"] = sha256(Path(__file__))
    if sha256(args.input) != INPUT_SHA256:
        raise ValueError("input changed during coverage audit")
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 1_000_000:
        raise ValueError("coverage report exceeds 1 MB cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"output": str(args.output), "bytes": len(encoded)}))


if __name__ == "__main__":
    main()
