#!/usr/bin/env python3
"""Run fixed SGBM profiles on prepared scenes. Profiles are never selected from truth."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import uuid

try:
    from .run_stereo_benchmarks import (DEFAULT_EVALUATOR, RESERVE_BYTES, check_unchanged,
                                        digest, execute, parse_prepared, read_prepared, source_pins)
except ImportError:
    from run_stereo_benchmarks import (DEFAULT_EVALUATOR, RESERVE_BYTES, check_unchanged,
                                       digest, execute, parse_prepared, read_prepared, source_pins)

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ("baseline", "quality", "fast", "full")


def summarize(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("No measured samples")
    fields = ("coverage", "bad2_all_valid", "bad2_matched", "mae_matched_px", "elapsed_stereo_ms")
    result = {}
    for field in fields:
        values = [sample[field] for sample in samples if sample[field] is not None]
        result[field + "_median"] = statistics.median(values) if values else None
    for field in ("decode", "preprocess", "match", "filter", "score"):
        values = [sample["stage_ms"][field] for sample in samples]
        result[field + "_ms_median"] = statistics.median(values)
        result[field + "_ms_min"] = min(values)
        result[field + "_ms_max"] = max(values)
    times = [sample["elapsed_stereo_ms"] for sample in samples]
    result["elapsed_stereo_ms_min"] = min(times)
    result["elapsed_stereo_ms_max"] = max(times)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="append", required=True, type=parse_prepared)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output-root", type=Path, default=ROOT / "build-opencv/stereo-profiles")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="OpenCV threads per evaluator process (1–hardware CPUs, max 64)")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--allow-unpinned", action="store_true", help="synthetic fixtures only")
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 5 or not 1 <= args.repeats <= 20 or not 1 <= args.timeout_seconds <= 3600 or not 1 <= args.threads <= min(64, os.cpu_count() or 1):
        parser.error("Warmup, repeats or timeout outside bounds")
    if len({name for name, _ in args.prepared}) != len(args.prepared):
        parser.error("Prepared names must be unique")
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        parser.error(f"Missing evaluator: {evaluator}")
    prepared = [(name, read_prepared(folder, source_pins(), args.allow_unpinned)) for name, folder in args.prepared]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < RESERVE_BYTES + 512 * 1024**2:
        parser.error("Benchmark would breach 10 GiB free-space reserve")
    run_dir = output_root / (datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "role": "development scene profile comparison; no held-out winner",
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "processor": platform.processor(), "cpu_count": os.cpu_count(),
                       "opencv_num_threads_env": os.environ.get("OPENCV_FOR_THREADS_NUM")},
              "evaluator": {"path": str(evaluator), "sha256": digest(evaluator),
                            "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                                              for path in sorted((ROOT / "core/evaluation").glob("*.cpp")) +
                                              sorted((ROOT / "core/evaluation").glob("*.hpp"))}},
              "settings": {"warmup": args.warmup, "repeats": args.repeats,
                           "threads": args.threads, "timeout_seconds": args.timeout_seconds},
              "scenes": {}, "ok": False, "failures": []}
    summary_path = run_dir / "summary.json"
    try:
      for scene_name, scene in prepared:
        manifest = scene["manifest"]
        record = {"prepared_manifest_sha256": scene["manifest_sha256"],
                  "prepared_sha256": scene["prepared_sha256"], "sources": scene["sources"],
                  "ndisp": scene["ndisp"],
                  "width": manifest["width"], "height": manifest["height"], "profiles": {}}
        result["scenes"][scene_name] = record
        for profile in PROFILES:
            metrics = []
            trials = []
            record["profiles"][profile] = {"trials": trials}
            for index in range(args.warmup + args.repeats):
                if shutil.disk_usage(output_root).free < RESERVE_BYTES + 128 * 1024**2:
                    raise RuntimeError("10 GiB free-space reserve plus trial allowance reached")
                check_unchanged(scene)
                # Each process writes into its own exclusive directory; runs are serial.
                destination = run_dir / scene_name / profile / f"repeat-{index:02d}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                command = [str(evaluator), "--left", str(scene["files"]["left"]),
                           "--right", str(scene["files"]["right"]), "--gt", str(scene["files"]["truth"]),
                           "--ndisp", str(scene["ndisp"]), "--profile", profile,
                           "--threads", str(args.threads),
                           "--output-dir", str(destination)]
                calibration = manifest.get("calibration")
                if calibration:
                    command += ["--fx", str(calibration["fx"]), "--baseline", str(calibration["baseline"]),
                                "--doffs", str(calibration["doffs"])]
                if "mask" in scene["files"]:
                    command += ["--mask", str(scene["files"]["mask"])]
                trial = execute(command, destination.parent / f"repeat-{index:02d}.log", args.timeout_seconds)
                trial["warmup"] = index < args.warmup
                trials.append(trial)
                check_unchanged(scene)
                if trial["status"] != "ok":
                    raise RuntimeError(f"{scene_name}/{profile}/{index}: {trial['status']}; see {trial['log']}")
                sample = json.loads((destination / "metrics.json").read_text())
                if sample["profile"] != profile or sample["sgbm"]["num_disparities"] != scene["common_search_count"]:
                    raise RuntimeError("Profile or search range mismatch")
                if sample["opencv_requested_threads"] != args.threads or (args.threads == 1 and not sample["opencv_parallel_for_serial"]):
                    raise RuntimeError("OpenCV thread setting mismatch")
                trial["outputs_sha256"] = {path.name: digest(path) for path in destination.iterdir() if path.is_file()}
                if index >= args.warmup:
                    metrics.append(sample)
                if shutil.disk_usage(output_root).free < RESERVE_BYTES:
                    raise RuntimeError("10 GiB free-space reserve reached")
            record["profiles"][profile].update({"summary": summarize(metrics), "samples": metrics})
      result["ok"] = True
    except Exception as error:
      result["failures"].append(str(error))
    finally:
      summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(summary_path)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
