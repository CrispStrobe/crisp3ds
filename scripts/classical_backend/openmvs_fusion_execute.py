"""Bounded execution of an approved paired cached-depth OpenMVS fusion plan.

Copies, rather than links, every consumed input into two fresh working folders.
Both arms share a ten-minute global deadline and differ only in fusion filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.classical_backend.openmvs_fusion_plan import (
    ARM_CAP, BIN, SOURCE, build_plan,
)
from scripts.classical_backend.run import RESERVE, StageError, checked_ply, digest, folder_bytes, stage

TOTAL_SECONDS = 600
MAX_LOG_BYTES = 16 << 20
MAX_RSS_BYTES = 4 << 30


def _save(output: Path, report: dict) -> None:
    (output / "fusion-result.json").write_text(json.dumps(report, indent=2) + "\n")


def _verify_inputs(directory: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"copied fusion input differs: {name}")


def _copy_inputs(source: Path, output: Path, hashes: dict[str, str], deadline: float) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError("fusion output must be fresh")
    output.mkdir()
    copied = 0
    for name, expected in hashes.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("global fusion deadline reached during copy")
        src, dst = source / name, output / name
        if src.is_symlink() or not src.is_file() or digest(src) != expected:
            raise ValueError(f"source fusion input changed before copy: {name}")
        size = src.stat().st_size
        if copied + size >= ARM_CAP or shutil.disk_usage(output).free < RESERVE + size:
            raise ValueError("copy would exceed per-arm cap or 10 GiB reserve")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)  # independent bytes, never hardlinks or symlinks
        if dst.is_symlink() or digest(dst) != expected:
            raise ValueError(f"copied fusion input changed: {name}")
        copied += size
    _verify_inputs(output, hashes)


def _run_arm(label: str, output: Path, command: list[str], plan: dict,
             deadline: float) -> dict:
    source = Path(plan["source"])
    hashes = plan["input_sha256"]
    report = {"schema": "openmvs_fusion_arm_v1", "status": "preparing",
              "label": label, "source": str(source),
              "source_result_sha256": plan["source_result_sha256"],
              "binary_sha256": plan["binary_sha256"],
              "input_sha256": hashes, "command": command,
              "limits": {"max_output_bytes": ARM_CAP, "max_log_bytes": MAX_LOG_BYTES,
                         "max_child_rss_bytes": MAX_RSS_BYTES, "global_seconds": TOTAL_SECONDS,
                         "min_free_bytes": RESERVE}, "stages": []}
    try:
        if digest(source / "result.json") != plan["source_result_sha256"]:
            raise ValueError("source result.json changed before native stage")
        if digest(Path(command[0])) != plan["binary_sha256"]:
            raise ValueError("OpenMVS binary changed before native stage")
        _copy_inputs(source, output, hashes, deadline)
        report["status"] = "running"
        _save(output, report)
        if any(output.glob("*.cfg")):
            raise ValueError("unexpected working-folder OpenMVS config override")
        report["working_folder_config_override"] = False
        if folder_bytes(output) >= ARM_CAP or shutil.disk_usage(output).free < RESERVE:
            raise ValueError("post-copy resource limit reached")
        if digest(source / "result.json") != plan["source_result_sha256"] or \
                digest(Path(command[0])) != plan["binary_sha256"]:
            raise ValueError("source result or native binary changed before launch")
        result = stage(output, "fuse", command, deadline, ARM_CAP, MAX_LOG_BYTES, MAX_RSS_BYTES)
        report["stages"].append(result)
        # The scene stores relative dense/images paths. Keeping that exact
        # layout under cwd=output ensures native reads the copied images.
        if not (output / "dense.mvs").is_file() or (output / "dense.mvs").stat().st_size == 0:
            raise ValueError("fusion did not produce dense.mvs")
        point_count = checked_ply(output / "dense.ply", "vertex")
        result["artifact"] = {"points": point_count,
                              "dense_ply_sha256": digest(output / "dense.ply"),
                              "dense_mvs_sha256": digest(output / "dense.mvs")}
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        report.update(status="failed", failure=str(error))
    # Keep the evidence even if the native process fails or changes a map.
    try:
        _verify_inputs(source, hashes)
        report["source_inputs_unchanged"] = True
    except Exception as error:
        report.update(status="failed", source_inputs_unchanged=False,
                      source_postcheck_failure=str(error))
    try:
        _verify_inputs(output, hashes)
        report["copied_inputs_unchanged"] = True
    except Exception as error:
        report.update(status="failed", copied_inputs_unchanged=False,
                      copied_postcheck_failure=str(error))
    try:
        if digest(source / "result.json") != plan["source_result_sha256"]:
            raise ValueError("source result.json changed")
        if digest(Path(command[0])) != plan["binary_sha256"]:
            raise ValueError("OpenMVS binary changed")
        report["source_result_and_binary_unchanged"] = True
    except Exception as error:
        report.update(status="failed", source_result_and_binary_unchanged=False,
                      toolchain_postcheck_failure=str(error))
    report["output_bytes"] = folder_bytes(output) if output.is_dir() else 0
    report["free_bytes_after"] = shutil.disk_usage(output.parent).free
    if report["output_bytes"] > ARM_CAP or report["free_bytes_after"] < RESERVE:
        report.update(status="failed", resource_postcheck_failure="cap or reserve exceeded")
    if output.is_dir():
        _save(output, report)
    return report


def execute_pair(source: Path, control: Path, ablation: Path,
                 binary_dir: Path = BIN) -> dict:
    """Re-preflight and execute serially; never overwrite an existing output."""
    started = time.monotonic()
    plan = build_plan(source, control, ablation, binary_dir)
    source = Path(plan["source"])
    deadline = started + TOTAL_SECONDS
    reports = {}
    for label, output, command in (
        ("control_dense_fuse", Path(plan["control"]), plan["commands"]["control_dense_fuse"]),
        ("ablation_simple_fuse", Path(plan["ablation"]), plan["commands"]["ablation_simple_fuse"]),
    ):
        if time.monotonic() >= deadline:
            reports[label] = {"status": "not_started", "failure": "global deadline reached"}
            break
        if shutil.disk_usage(output.parent).free < RESERVE + ARM_CAP:
            reports[label] = {"status": "not_started", "failure": "reserve plus arm cap unavailable"}
            break
        reports[label] = _run_arm(label, output, command, plan, deadline)
        if reports[label]["status"] != "complete":
            break
    source_seal_unchanged = (digest(source / "result.json") == plan["source_result_sha256"] and
                             digest(Path(plan["commands"]["control_dense_fuse"][0])) ==
                             plan["binary_sha256"])
    pair = {"schema": "openmvs_fusion_pair_v1", "status": "complete" if
            len(reports) == 2 and all(r["status"] == "complete" for r in reports.values())
            and source_seal_unchanged and time.monotonic() <= deadline
            else "failed_or_incomplete", "source": str(source),
            "source_result_sha256": plan["source_result_sha256"],
            "source_result_and_binary_unchanged": source_seal_unchanged,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "sole_option_difference": plan["sole_option_difference"],
            "limits": {"global_seconds": TOTAL_SECONDS, "max_bytes_per_arm": ARM_CAP,
                       "min_free_bytes": RESERVE}, "arms": reports}
    control_path = Path(plan["control"])
    if control_path.is_dir():
        pair_path = control_path / "fusion-pair.json"
        pair_path.write_text(json.dumps(pair, indent=2) + "\n")
        if folder_bytes(control_path) > ARM_CAP or shutil.disk_usage(control_path).free < RESERVE:
            pair["status"] = "failed_or_incomplete"
            pair["pair_report_resource_failure"] = "cap or reserve exceeded after pair report"
            pair_path.write_text(json.dumps(pair, indent=2) + "\n")
    return pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    result = execute_pair(args.source, args.control, args.ablation, args.binary_dir)
    print(json.dumps({"status": result["status"], "arms": {key: {"status": value["status"],
          "points": value.get("stages", [{}])[-1].get("artifact", {}).get("points")}
          for key, value in result["arms"].items()}}, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
