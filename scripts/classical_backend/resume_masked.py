"""Resume a failed full-resolution masked OpenMVS run from 60 complete base DMAPs.

This is a new cache continuation, not a fresh full-resolution timing. Partial
geometric maps are excluded; the native command requests both geometric passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from scripts.classical_backend.dense_masks import cv2, native_mask_name
from scripts.classical_backend.masked_dense import require_real_tree, validate_inputs
from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, obj_counts,
                                           stage, tool_path)
from scripts.classical_backend.verify_dmap_masks import depth_files


def complete_base_maps(source: Path, masks: Path, mask_report: dict) -> dict[str, str]:
    if cv2 is None:
        raise RuntimeError("OpenCV required for native DMAP mask verification")
    maps = depth_files(source)
    names = {item["name"] for item in mask_report["images"]}
    if maps.keys() != names:
        raise ValueError("source has fewer than all expected complete base depth maps")
    hashes = {}
    for item in mask_report["images"]:
        name = item["name"]
        path, depth = maps[name]
        if depth.shape != tuple(reversed(item["undistorted_size"])):
            raise ValueError(f"full-resolution base DMAP size differs from image: {name}")
        with Image.open(masks / native_mask_name(name)) as image:
            mask = np.asarray(image.convert("L"))
        if not np.count_nonzero(depth > 0) or np.any((depth > 0) & (mask == 0)):
            raise ValueError(f"base DMAP is empty or has positive depth outside native mask: {name}")
        hashes[path.name] = digest(path)
    return hashes


def resume(args: argparse.Namespace) -> dict:
    if args.source.is_symlink() or args.output.is_symlink() or args.output.exists():
        raise ValueError("real failed source and fresh output required")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    source = args.source.resolve()
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    if not source.is_dir() or source.is_symlink() or output.exists():
        raise ValueError("real failed source and fresh output required")
    source_result_path = source / "result.json"
    if source_result_path.is_symlink():
        raise ValueError("failed source report must not be a symlink")
    prior = json.loads(source_result_path.read_text())
    if (prior.get("schema") != "classical_masked_dense_v1" or prior.get("status") != "failed" or
            prior.get("native_options") != {"resolution_level": 0, "max_resolution": 1600,
                                            "geometric_iters": 2, "tower_mode": 4,
                                            "ignore_mask_label": 0,
                                            "refine_resolution_level": 1, "refine_scales": 1} or
            not prior.get("stages") or prior["stages"][-1].get("name") != "densify" or
            prior["stages"][-1].get("failure") != "output byte limit reached"):
        raise ValueError("source is not the bounded full-resolution masked depth failure")
    masks = source / "masks"
    require_real_tree(source / "dense", "cached dense tree")
    require_real_tree(masks, "cached mask tree")
    if (source / "scene.mvs").is_symlink() or not (source / "scene.mvs").is_file():
        raise ValueError("cached scene must be a real file")
    mask_report_path = masks / "report.json"
    mask_report = json.loads(mask_report_path.read_text())
    upstream = Path(prior["source_run"])
    if upstream.is_symlink() or not upstream.is_dir():
        raise ValueError("upstream source must be a real directory")
    validate_inputs(upstream, masks, mask_report)
    if digest(mask_report_path) != prior["mask_report_sha256"]:
        raise ValueError("mask report differs from failed source run")
    binary_hashes = {name: digest(tool_path(binary_dir, name)) for name in TOOLS}
    if binary_hashes != prior["binary_hashes"]:
        raise ValueError("native binary changed since failed full-resolution run")
    if (args.max_threads != 2 or args.max_gib != 3 or
            not math.isfinite(args.timeout_minutes) or not 0 < args.timeout_minutes <= 10 or
            not math.isfinite(args.max_rss_gib) or not 0 < args.max_rss_gib <= 10 or
            not 0 < args.max_log_mib <= 32):
        raise ValueError("continuation is restricted to 2 threads, 3 GiB and at most 10 minutes")
    dmaps = list(source.glob("*.dmap"))
    if len(dmaps) > 500 or any(path.is_symlink() or not path.is_file() or
                               path.stat().st_size > (128 << 20) for path in dmaps):
        raise ValueError("cached DMAP set contains links, special files, or oversized maps")
    base_hashes = complete_base_maps(source, masks, mask_report)
    partial_geo = len(list(source.glob("depth[0-9][0-9][0-9][0-9].geo.dmap")))
    if partial_geo < 1:
        raise ValueError("failed source has no partial geometric pass to exclude")
    if not output.parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    free = shutil.disk_usage(output.parent).free
    if free < RESERVE + (3 << 30) + (1 << 30):
        raise ValueError("requires 10 GiB reserve plus 3 GiB cap plus 1 GiB active-job allowance")
    initial_bytes = folder_bytes(source / "dense") + folder_bytes(masks) + (source / "scene.mvs").stat().st_size
    initial_bytes += sum((source / name).stat().st_size for name in base_hashes)
    if initial_bytes > args.max_gib * (1 << 30):
        raise ValueError("cached initial inputs exceed output cap")
    output.mkdir(parents=True)
    shutil.copytree(source / "dense", output / "dense", symlinks=False)
    shutil.copytree(masks, output / "masks", symlinks=False)
    shutil.copyfile(source / "scene.mvs", output / "source-scene.mvs")
    for name, expected in sorted(base_hashes.items()):
        if shutil.disk_usage(output).free < RESERVE + (source / name).stat().st_size:
            raise ValueError("disk reserve reached during base-map copy")
        shutil.copyfile(source / name, output / name)
        if digest(output / name) != expected:
            raise ValueError(f"copied base depth map differs: {name}")
    report = {"schema": "classical_masked_cache_continuation_v1", "status": "running",
              "runner_sha256": digest(Path(__file__)), "source_run": str(source),
              "source_result_sha256": digest(source_result_path),
              "source_scene_sha256": digest(source / "scene.mvs"),
              "source_mask_report_sha256": digest(mask_report_path),
              "cached_base_dmaps": len(base_hashes), "cached_base_dmap_sha256": base_hashes,
              "excluded_partial_geometric_dmaps": partial_geo,
              "cache_strategy": "copy complete masked full-resolution base DMAPs only; exclude partial geo; request both native geometric passes",
              "native_options": prior["native_options"], "binary_hashes": binary_hashes,
              "limits": {"max_output_bytes": 3 << 30, "timeout_minutes": args.timeout_minutes,
                         "max_child_rss_bytes": args.max_rss_gib * (1 << 30),
                         "min_free_bytes": RESERVE},
              "stages": [], "shipping_approved": False, "metric_scale_verified": False,
              "quality_accepted": False}
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    deadline = time.monotonic() + args.timeout_minutes * 60
    common = ["--max-threads", "2", "--working-folder", str(output)]
    def execute(name, command, validate):
        result = stage(output, name, [str(x) for x in command], deadline,
                       3 << 30, args.max_log_mib << 20, int(args.max_rss_gib * (1 << 30)))
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()
    try:
        dense = output / "dense"
        scene = output / "scene.mvs"
        execute("reimport", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                             "-o", scene, "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(scene)})
        execute("densify", [tool_path(binary_dir, "DensifyPointCloud"), "-i", scene,
                            "-o", output / "dense.mvs", "--resolution-level", "0",
                            "--max-resolution", "1600", "--geometric-iters", "2",
                            "--tower-mode", "4", "--mask-path", output / "masks",
                            "--ignore-mask-label", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("mesh", [tool_path(binary_dir, "ReconstructMesh"), "-i", scene,
                         "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        execute("refine", [tool_path(binary_dir, "RefineMesh"), "-i", scene,
                           "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                           "--resolution-level", "1", "--scales", "1", *common],
                lambda: {"faces": checked_ply(output / "refined.ply", "face")})
        execute("texture", [tool_path(binary_dir, "TextureMesh"), "-i", scene,
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
    try:
        validate_inputs(upstream, masks, mask_report)
        report["source_unchanged"] = (digest(source_result_path) == report["source_result_sha256"] and
                                      digest(source / "scene.mvs") == report["source_scene_sha256"] and
                                      digest(mask_report_path) == report["source_mask_report_sha256"] and
                                      all(digest(source / name) == expected for name, expected in base_hashes.items()))
    except Exception:
        report["source_unchanged"] = False
    if not report["source_unchanged"]:
        report.update(status="failed", failure="source camera/image/mask/depth evidence changed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=3)
    parser.add_argument("--max-rss-gib", type=float, default=10)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--timeout-minutes", type=float, default=10)
    args = parser.parse_args()
    result = resume(args)
    print(json.dumps({"status": result["status"], "result": str((args.output / "result.json").resolve()),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
