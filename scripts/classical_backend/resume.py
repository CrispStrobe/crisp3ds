"""Copy a failed OpenMVS densification and continue from preserved depth maps.

The result is explicitly a continuation; original failure evidence is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, obj_counts, stage, tool_path)


def continue_run(args: argparse.Namespace) -> dict:
    if args.source.is_symlink() or args.output.is_symlink() or args.output.exists():
        raise ValueError("source must be a real prior run and output must be fresh")
    source = args.source.resolve()
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    if not source.is_dir() or source.is_symlink() or output.exists():
        raise ValueError("source must be a real prior run and output must be fresh")
    if (not 0 < args.max_threads <= 2 or not math.isfinite(args.max_gib) or
            not 0 < args.max_gib <= 2 or not math.isfinite(args.timeout_minutes) or
            not 0 < args.timeout_minutes <= 15 or not math.isfinite(args.max_rss_gib) or
            not 0 < args.max_rss_gib <= 10 or not 0 < args.max_log_mib <= 32):
        raise ValueError("invalid bounded continuation settings")
    source_report_path = source / "result.json"
    source_report = json.loads(source_report_path.read_text())
    if (source_report.get("schema") != "classical_backend_v1" or source_report.get("status") != "failed" or
            not source_report.get("stages") or source_report["stages"][-1]["name"] != "densify"):
        raise ValueError("source must be a failed densify-stage classical run")
    source_bytes = folder_bytes(source)
    if source_bytes > args.max_gib * (1 << 30) or shutil.disk_usage(source).free < RESERVE + source_bytes + (256 << 20):
        raise ValueError("copy would exceed output cap or 10 GiB free-space reserve")
    for item in source_report["inputs"]:
        if digest(source / "images" / item["name"]) != item["sha256"]:
            raise ValueError(f"source copied image changed: {item['name']}")
    cached_base = sorted(source.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    cached_geo = sorted(source.glob("depth[0-9][0-9][0-9][0-9].geo.dmap"))
    if len(cached_base) < 60 or not cached_geo:
        raise ValueError("source has insufficient saved OpenMVS depth maps")
    bin_hashes = source_report["software"]["openmvs_binaries"]
    for name in TOOLS:
        if digest(tool_path(binary_dir, name)) != bin_hashes[name]:
            raise ValueError(f"OpenMVS executable differs from source run: {name}")
    source_hashes = {name: digest(source / name) for name in
                     ("result.json", "scene.mvs", "inputs.json", "sfm.json")}
    shutil.copytree(source, output, symlinks=False,
                    ignore=shutil.ignore_patterns("*.geo.dmap", "*.tmp"))
    (output / "result.json").rename(output / "source-result.json")
    (output / "densify.log").rename(output / "source-densify.log")
    # The copied scene stores input image paths from the original run. A fresh
    # import below rewrites the scene against this continuation's dense images.
    report = {"schema": "classical_backend_continuation_v1", "status": "running",
              "source_run": str(source), "source_hashes": source_hashes,
              "source_failure": source_report["failure"],
              "source_runner_sha256": source_report["software"]["runner_sha256"],
              "source_stages": source_report["stages"],
              "cached_depth_maps_before": {"base": len(cached_base), "geometric": len(cached_geo)},
              "cache_strategy": "copy 60 completed base DMAP files; omit partial geometric maps; fuse cached base maps with geometric-iters=0",
              "binary_hashes": bin_hashes, "stages": [],
              "limits": {"max_output_bytes": args.max_gib * (1 << 30),
                         "max_child_rss_bytes": args.max_rss_gib * (1 << 30),
                         "timeout_minutes": args.timeout_minutes, "min_free_bytes": RESERVE},
              "shipping_approved": False, "metric_scale_verified": False, "quality_accepted": False}
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    deadline = time.monotonic() + args.timeout_minutes * 60
    common = ["--max-threads", str(args.max_threads), "--working-folder", str(output)]
    def execute(name, command, validate):
        result = stage(output, name, [str(x) for x in command], deadline,
                       int(args.max_gib * (1 << 30)), args.max_log_mib << 20,
                       int(args.max_rss_gib * (1 << 30)))
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()
    try:
        dense = output / "dense"
        execute("reimport", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                             "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", [tool_path(binary_dir, "DensifyPointCloud"), "-i", output / "scene.mvs",
                            "-o", output / "dense.mvs", "--resolution-level", "2",
                            "--geometric-iters", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("mesh", [tool_path(binary_dir, "ReconstructMesh"), "-i", output / "dense.mvs",
                         "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        execute("refine", [tool_path(binary_dir, "RefineMesh"), "-i", output / "dense.mvs",
                           "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                           "--resolution-level", "1", "--scales", "1", *common],
                lambda: {"faces": checked_ply(output / "refined.ply", "face")})
        execute("texture", [tool_path(binary_dir, "TextureMesh"), "-i", output / "dense.mvs",
                            "-m", output / "refined.ply", "-o", output / "textured.mvs",
                            "--export-type", "obj", *common],
                lambda: dict(zip(("vertices", "faces"), obj_counts(output / "textured.obj"))))
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = shutil.disk_usage(output).free
    report["source_result_unchanged"] = digest(source_report_path) == source_hashes["result.json"]
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary-dir", default=BIN, type=Path)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=2.0)
    parser.add_argument("--max-rss-gib", type=float, default=10.0)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--timeout-minutes", type=float, default=15.0)
    args = parser.parse_args()
    report = continue_run(args)
    print(json.dumps({"status": report["status"], "result": str(args.output.resolve()),
                      "failure": report.get("failure")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
