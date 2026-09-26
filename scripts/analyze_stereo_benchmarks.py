#!/usr/bin/env python3
"""Dependency-free quality diagnostics for an existing stereo benchmark matrix."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys


RESERVE_BYTES = 10 * 1024**3
MAX_PIXELS = 10_000_000
BAD_THRESHOLD = 2.0
COLORS = {
    "truth_invalid": (48, 48, 48),
    "missing": (44, 100, 220),
    "correct": (34, 172, 83),
    "incorrect": (225, 55, 55),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pfm(path):
    with path.open("rb") as source:
        magic = source.readline(80)
        dimensions = source.readline(80)
        scale_line = source.readline(80)
        if magic != b"Pf\n":
            raise ValueError(f"{path}: expected grayscale Pf")
        try:
            width, height = (int(part) for part in dimensions.split())
            scale = float(scale_line)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}: invalid PFM header") from error
        if (width <= 0 or height <= 0 or width * height > MAX_PIXELS or
                not math.isfinite(scale) or scale == 0):
            raise ValueError(f"{path}: invalid PFM dimensions or scale")
        payload = source.read(width * height * 4 + 1)
    if len(payload) != width * height * 4:
        raise ValueError(f"{path}: PFM payload length mismatch")
    values = array("f")
    values.frombytes(payload)
    if (scale < 0) != (sys.byteorder == "little"):
        values.byteswap()
    # PFM rows are stored bottom first. As in the evaluator, scale magnitude
    # does not multiply disparity values; only its sign selects byte order.
    rows = [values[y * width:(y + 1) * width] for y in range(height - 1, -1, -1)]
    return width, height, rows


def read_mask(path, width, height):
    with path.open("rb") as source:
        if source.readline(80) != b"P5\n":
            raise ValueError(f"{path}: expected binary P5 mask")
        try:
            mask_width, mask_height = (int(part) for part in source.readline(80).split())
            maximum = int(source.readline(80))
        except ValueError as error:
            raise ValueError(f"{path}: invalid PGM mask header") from error
        data = source.read()
    if (mask_width, mask_height) != (width, height) or maximum != 255 or len(data) != width * height:
        raise ValueError(f"{path}: mask dimensions, range, or payload mismatch")
    return data


def empty_counts():
    return {key: 0 for key in COLORS}


def summarize(counts, absolute_error_sum, squared_error_sum):
    valid = counts["missing"] + counts["correct"] + counts["incorrect"]
    matched = counts["correct"] + counts["incorrect"]
    return {
        "pixels": sum(counts.values()), "categories": counts,
        "truth_valid": valid, "matched": matched,
        "coverage": matched / valid if valid else None,
        "missing_rate": counts["missing"] / valid if valid else None,
        "bad2_all_valid": (counts["missing"] + counts["incorrect"]) / valid if valid else None,
        "bad2_matched": counts["incorrect"] / matched if matched else None,
        "mae_matched_px": absolute_error_sum / matched if matched else None,
        "rmse_matched_px": math.sqrt(squared_error_sum / matched) if matched else None,
    }


def analyze_scene(truth, predictions, mask=None, search_count=0):
    width, height, truth_rows = truth
    if mask is not None and len(mask) != width * height:
        raise ValueError("evaluation mask dimensions differ from truth")
    if not predictions or not isinstance(search_count, int) or search_count < 0:
        raise ValueError("predictions and nonnegative search count required")
    for name, (pred_width, pred_height, _) in predictions.items():
        if (pred_width, pred_height) != (width, height):
            raise ValueError(f"{name}: prediction dimensions differ from truth")
    names = sorted(predictions)
    # Ceil of one tenth on each side, fixed by geometry before seeing predictions.
    left_bound = max((width + 9) // 10, search_count)
    right_margin = (width + 9) // 10
    margin_y = (height + 9) // 10
    if left_bound + right_margin >= width or 2 * margin_y >= height:
        raise ValueError("image too small for 10% interior ROI")
    regions = {region: {name: [empty_counts(), 0.0, 0.0] for name in names}
               for region in ("full", "interior", "left_search_strip", "common_full", "common_interior")}
    maps = {name: bytearray(width * height * 3) for name in names}
    for y in range(height):
        for x in range(width):
            offset = y * width + x
            if mask is not None and mask[offset] == 0:
                # Black means excluded by the prepared evaluation mask.
                continue
            gt = truth_rows[y][x]
            valid = math.isfinite(gt) and gt >= 0
            common = valid and all(math.isfinite(predictions[name][2][y][x]) and
                                   predictions[name][2][y][x] >= 0 for name in names)
            interior = left_bound <= x < width - right_margin and margin_y <= y < height - margin_y
            for name in names:
                pred = predictions[name][2][y][x]
                category = ("truth_invalid" if not valid else
                            "missing" if not math.isfinite(pred) or pred < 0 else
                            "incorrect" if abs(pred - gt) > BAD_THRESHOLD else "correct")
                maps[name][offset * 3:offset * 3 + 3] = bytes(COLORS[category])
                error = abs(pred - gt) if category in ("correct", "incorrect") else 0.0
                pixel_regions = ["full"]
                if interior:
                    pixel_regions.append("interior")
                if x < search_count:
                    pixel_regions.append("left_search_strip")
                for region in pixel_regions:
                    record = regions[region][name]
                    record[0][category] += 1
                    record[1] += error
                    record[2] += error * error
                if common:
                    for region in (("common_full", "common_interior") if interior else ("common_full",)):
                        record = regions[region][name]
                        record[0][category] += 1
                        record[1] += error
                        record[2] += error * error
    results = {region: {name: summarize(*record) for name, record in engines.items()}
               for region, engines in regions.items()}
    return {"width": width, "height": height, "search_count": search_count,
            "interior_rule": "x >= max(ceil(width/10), search_count), x < width-ceil(width/10); y >= ceil(height/10), y < height-ceil(height/10)",
            "interior_bounds_xyxy": [left_bound, margin_y, width - right_margin, height - margin_y],
            "regions": results}, maps


def analyze_benchmark(report_path, output_dir):
    report_path = report_path.resolve(strict=True)
    report = json.loads(report_path.read_text())
    if not report.get("ok") or not report.get("scenes"):
        raise ValueError("benchmark must be a completed successful matrix")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output already exists: {output_dir}")
    if shutil.disk_usage(output_dir.parent).free < RESERVE_BYTES:
        raise RuntimeError("10 GiB free-space reserve reached")
    result = {"source_benchmark": str(report_path), "source_benchmark_sha256": sha256(report_path),
              "bad_threshold_px": BAD_THRESHOLD,
              "common_population_warning": "Selection biased: only truth-valid pixels where every engine predicts.",
              "scenes": {}}
    all_maps = {}
    expected_output_bytes = 10 * 1024 * 1024
    for scene_name, scene in sorted(report["scenes"].items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_name):
            raise ValueError(f"unsafe scene name: {scene_name}")
        prepared = Path(scene["prepared"])
        if sha256(prepared / "inputs.json") != scene["inputs_manifest_sha256"]:
            raise ValueError(f"{scene_name}: prepared manifest hash differs from benchmark")
        truth_path = prepared / "truth.pfm"
        if sha256(truth_path) != scene["inputs_sha256"]["truth"]:
            raise ValueError(f"{scene_name}: truth hash differs from benchmark")
        truth = read_pfm(truth_path)
        mask = None
        manifest = json.loads((prepared / "inputs.json").read_text())
        if manifest.get("mask"):
            mask_path = prepared / manifest["mask"]
            if sha256(mask_path) != scene["inputs_sha256"]["mask"]:
                raise ValueError(f"{scene_name}: mask hash differs from benchmark")
            mask = read_mask(mask_path, truth[0], truth[1])
        predictions = {}
        for engine_name, engine in sorted(scene["engines"].items()):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", engine_name):
                raise ValueError(f"unsafe engine name: {engine_name}")
            if engine.get("status") != "ok":
                raise ValueError(f"{scene_name}/{engine_name}: engine did not succeed")
            relative = "score/disparity.pfm" if engine_name == "sgbm" else "disparity.pfm"
            prediction_path = report_path.parent / scene_name / engine_name / relative
            if sha256(prediction_path) != engine["outputs_sha256"][relative]:
                raise ValueError(f"{scene_name}/{engine_name}: prediction hash differs from benchmark")
            predictions[engine_name] = read_pfm(prediction_path)
        scene_result, maps = analyze_scene(truth, predictions, mask, scene["common_search_count"])
        for name, metrics in scene_result["regions"]["full"].items():
            expected = scene["engines"][name]["metrics"]
            if (metrics["truth_valid"], metrics["matched"], metrics["categories"]["incorrect"]) != (
                    expected["gt_valid"], expected["matched"], expected["bad2_matched_count"]):
                raise ValueError(f"{scene_name}/{name}: counts disagree with benchmark evaluator")
        result["scenes"][scene_name] = scene_result
        all_maps[scene_name] = maps
        expected_output_bytes += sum(3 * truth[0] * truth[1] + 32 for _ in maps)
    if shutil.disk_usage(output_dir.parent).free < RESERVE_BYTES + expected_output_bytes:
        raise RuntimeError("10 GiB free-space reserve reached")
    output_dir.mkdir()
    for scene_name, maps in all_maps.items():
        scene_dir = output_dir / scene_name
        scene_dir.mkdir()
        width = result["scenes"][scene_name]["width"]
        height = result["scenes"][scene_name]["height"]
        for engine_name, pixels in maps.items():
            (scene_dir / f"{engine_name}-errors.ppm").write_bytes(
                f"P6\n{width} {height}\n255\n".encode() + pixels)
    (output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="existing benchmark.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh directory for JSON and PPM maps")
    args = parser.parse_args()
    try:
        analyze_benchmark(args.benchmark, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(args.output_dir / "diagnostics.json")


if __name__ == "__main__":
    main()
