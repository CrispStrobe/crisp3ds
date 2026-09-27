"""Fixed-camera native OpenMVS mask ablation from a verified undistorted COLMAP run.

This is a composed-stage experiment: SfM and undistortion occurred in the
source run. It does not claim fresh photo-to-mesh timing or silhouette masks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

from scripts.classical_backend.dense_masks import model_hashes, native_mask_name
from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, obj_counts,
                                           stage, tool_path)


def validate_inputs(source: Path, masks: Path, mask_report: dict) -> dict:
    if (source / "result.json").is_symlink():
        raise ValueError("source report must not be a symlink")
    source_report = json.loads((source / "result.json").read_text())
    stages = {item["name"]: item for item in source_report.get("stages", [])}
    if (source_report.get("schema") != "classical_backend_v1" or
            any(stages.get(name, {}).get("status") != "complete" for name in ("reuse_sfm", "undistort", "import"))):
        raise ValueError("source must have completed imported SfM, undistortion and OpenMVS import")
    if mask_report.get("schema") != "classical_dense_masks_v1" or mask_report.get("status") != "complete":
        raise ValueError("mask report is not complete")
    if (Path(mask_report["source_model"]).resolve() != source / "sparse" / "0" or
            Path(mask_report["undistorted_model"]).resolve() != source / "dense" / "sparse" or
            mask_report["source_model_sha256"] != model_hashes(source / "sparse" / "0") or
            mask_report["undistorted_model_sha256"] != model_hashes(source / "dense" / "sparse") or
            mask_report.get("max_pose_matrix_difference", 1) > 1e-8 or
            mask_report.get("ignore_mask_label") != 0 or
            mask_report.get("native_filename_basis") != "image stem"):
        raise ValueError("mask report does not bind unchanged source/undistorted cameras")
    names = {item["name"] for item in source_report["inputs"]}
    if names != {item["name"] for item in mask_report["images"]}:
        raise ValueError("mask report does not cover exactly the selected photos")
    for item in mask_report["images"]:
        name = item["name"]
        image = source / "dense" / "images" / name
        if item.get("native_mask_name") != native_mask_name(name):
            raise ValueError(f"native OpenMVS mask filename differs from stem convention: {name}")
        mask = masks / item["native_mask_name"]
        if image.is_symlink() or mask.is_symlink() or not image.is_file() or not mask.is_file():
            raise ValueError(f"missing or symlink dense image/mask: {name}")
        if digest(image) != item["undistorted_image_sha256"] or digest(mask) != item["mask_sha256"]:
            raise ValueError(f"dense image or native mask differs from report: {name}")
    return source_report


def require_real_tree(root: Path, label: str) -> None:
    """Reject links and special files before copying an input tree.

    copytree's default follows symlinks, which would otherwise let a report
    describe a bounded directory while copying arbitrary outside data.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a real directory")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"{label} contains a symlink or special file: {path}")


def run(args: argparse.Namespace) -> dict:
    if (args.source.is_symlink() or args.masks.is_symlink() or args.output.is_symlink()
            or args.output.parent.is_symlink()
            or args.output.exists()):
        raise ValueError("source/masks must be real directories and output fresh")
    source = args.source.resolve()
    masks = args.masks.resolve()
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    if (not source.is_dir() or source.is_symlink() or not masks.is_dir() or masks.is_symlink()
            or output.exists()):
        raise ValueError("real source/mask directories and a fresh output are required")
    if (args.max_threads < 1 or args.max_threads > 2 or args.resolution_level not in (0, 1, 2) or
            args.geometric_iters < 0 or args.geometric_iters > 2 or not 640 <= args.max_resolution <= 4096 or
            not 0 <= args.tower_mode <= 4 or not 0 < args.max_log_mib <= 32 or
            not math.isfinite(args.max_gib) or not 0 < args.max_gib <= 2 or
            not math.isfinite(args.timeout_minutes) or not 0 < args.timeout_minutes <= 30 or
            not math.isfinite(args.max_rss_gib) or not 0 < args.max_rss_gib <= 12):
        raise ValueError("invalid bounded native settings")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be an existing real directory")
    require_real_tree(source / "dense", "source dense tree")
    require_real_tree(masks, "native mask tree")
    mask_report_path = masks / "report.json"
    if mask_report_path.is_symlink():
        raise ValueError("mask report must not be a symlink")
    mask_report = json.loads(mask_report_path.read_text())
    source_report = validate_inputs(source, masks, mask_report)
    binary_hashes = {name: digest(tool_path(binary_dir, name)) for name in TOOLS}
    if binary_hashes != source_report["software"]["openmvs_binaries"]:
        raise ValueError("OpenMVS binary bytes differ from source run")
    copy_bytes = folder_bytes(source / "dense") + folder_bytes(masks)
    if (copy_bytes > args.max_gib * (1 << 30) or
            shutil.disk_usage(output.parent).free < RESERVE + copy_bytes + (256 << 20)):
        raise ValueError("copy exceeds output cap or 10 GiB free-space reserve")
    output.mkdir(parents=True)
    shutil.copytree(source / "dense", output / "dense", symlinks=False)
    shutil.copytree(masks, output / "masks", symlinks=False)
    if digest(output / "masks" / "report.json") != digest(mask_report_path):
        raise ValueError("copied mask report differs")
    for item in mask_report["images"]:
        name = item["name"]
        if (digest(output / "dense" / "images" / name) != item["undistorted_image_sha256"] or
                digest(output / "masks" / item["native_mask_name"]) != item["mask_sha256"]):
            raise ValueError(f"copied dense image or mask differs: {name}")
    report = {"schema": "classical_masked_dense_v1", "status": "running",
              "runner_sha256": digest(Path(__file__)), "source_run": str(source),
              "source_result_sha256": digest(source / "result.json"),
              "mask_report_sha256": digest(mask_report_path),
              "mask_semantics": mask_report["semantics"],
              "mask_count": len(mask_report["images"]),
              "binary_hashes": binary_hashes,
              "native_options": {"resolution_level": args.resolution_level,
                                 "max_resolution": args.max_resolution,
                                 "geometric_iters": args.geometric_iters,
                                 "tower_mode": args.tower_mode,
                                 "ignore_mask_label": 0,
                                 "refine_resolution_level": 1, "refine_scales": 1},
              "limits": {"max_output_bytes": args.max_gib * (1 << 30),
                         "max_child_rss_bytes": args.max_rss_gib * (1 << 30),
                         "timeout_minutes": args.timeout_minutes, "min_free_bytes": RESERVE},
              "stages": [], "shipping_approved": False, "metric_scale_verified": False,
              "quality_accepted": False}
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
        scene = output / "scene.mvs"
        execute("import", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                           "-o", scene, "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(scene)})
        execute("densify", [tool_path(binary_dir, "DensifyPointCloud"), "-i", scene,
                            "-o", output / "dense.mvs", "--resolution-level", args.resolution_level,
                            "--max-resolution", args.max_resolution,
                            "--geometric-iters", args.geometric_iters,
                            "--tower-mode", args.tower_mode,
                            "--mask-path", output / "masks", "--ignore-mask-label", "0", *common],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        geometry_scene = output / "dense.mvs"
        execute("mesh", [tool_path(binary_dir, "ReconstructMesh"), "-i", geometry_scene,
                         "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        execute("refine", [tool_path(binary_dir, "RefineMesh"), "-i", geometry_scene,
                           "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                           "--resolution-level", "1", "--scales", "1", *common],
                lambda: {"faces": checked_ply(output / "refined.ply", "face")})
        execute("texture", [tool_path(binary_dir, "TextureMesh"), "-i", geometry_scene,
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
        validate_inputs(source, masks, mask_report)
        report["sources_unchanged"] = (digest(source / "result.json") == report["source_result_sha256"] and
                                       digest(mask_report_path) == report["mask_report_sha256"] and
                                       all(digest(tool_path(binary_dir, name)) == value
                                           for name, value in binary_hashes.items()))
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"]:
        report.update(status="failed", failure="source cameras/images/masks/binaries changed during run")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    parser.add_argument("--resolution-level", type=int, default=2)
    parser.add_argument("--max-resolution", type=int, default=1600)
    parser.add_argument("--geometric-iters", type=int, default=2)
    parser.add_argument("--tower-mode", type=int, default=4)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=2)
    parser.add_argument("--max-rss-gib", type=float, default=10)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--timeout-minutes", type=float, default=15)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str((args.output / "result.json").resolve()),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
