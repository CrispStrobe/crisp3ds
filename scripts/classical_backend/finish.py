"""Continue from a verified native dense cloud after a validator-only failure.

This is a composed-stage recovery, never a fresh image-to-mesh timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.classical_backend.run import (BIN, RESERVE, StageError, TOOLS, checked_ply,
                                           digest, folder_bytes, nonempty_bytes, obj_counts, stage, tool_path)


def verify_filter_files(filtering: dict, manifest: dict) -> None:
    model = Path(filtering["model_dir"])
    for name, expected in filtering["model_files_sha256"].items():
        if digest(model / name) != expected:
            raise ValueError(f"filtered-cloud model file changed: {name}")
    for image in manifest["images"]:
        name = image["name"]
        if digest(Path(image["path"])) != filtering["image_sha256"][name]:
            raise ValueError(f"filtered-cloud photo changed: {name}")
        if digest(Path(image["pose_support_mask"])) != filtering["mask_sha256"][name]:
            raise ValueError(f"filtered-cloud mask changed: {name}")


def finish(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    if not source.is_dir() or source.is_symlink() or output.exists():
        raise ValueError("source must be a real prior run and output must be fresh")
    source_result = source / "result.json"
    prior = json.loads(source_result.read_text())
    if (prior.get("schema") != "classical_backend_continuation_v1" or
            prior.get("status") != "failed" or not prior.get("stages") or
            prior["stages"][-1]["name"] != "densify" or
            prior["stages"][-1]["exit_code"] != 0):
        raise ValueError("source must be a native-success, validator-failed dense continuation")
    dense_cloud = source / "dense.ply"
    dense_points = checked_ply(dense_cloud, "vertex")
    if dense_points < 100:
        raise ValueError("dense source cloud has fewer than 100 points")
    chosen_cloud = dense_cloud
    filter_provenance = None
    if args.filtered_cloud:
        if not args.filter_report:
            raise ValueError("--filter-report required with --filtered-cloud")
        chosen_cloud = args.filtered_cloud.resolve()
        filter_path = args.filter_report.resolve()
        filtering = json.loads(filter_path.read_text())
        producer = json.loads((source / "source-result.json").read_text())
        producer_source = producer["sfm_source"]
        manifest_path = Path(producer_source["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        if (Path(filtering.get("source_cloud", "")).resolve() != dense_cloud or
                filtering.get("source_cloud_sha256") != digest(dense_cloud) or
                Path(filtering.get("filtered_cloud", "")).resolve() != chosen_cloud or
                filtering.get("filtered_cloud_sha256") != digest(chosen_cloud) or
                filtering.get("model_files_sha256") != producer_source["files_sha256"] or
                Path(filtering.get("model_dir", "")).resolve() != Path(producer_source["path"]) or
                filtering.get("manifest_sha256") != producer_source["manifest_sha256"] or
                digest(manifest_path) != producer_source["manifest_sha256"] or
                filtering.get("image_sha256") != {item["name"]: item["sha256"] for item in manifest["images"]} or
                filtering.get("mask_sha256") != {item["name"]: item["mask_sha256"] for item in manifest["images"]} or
                filtering.get("threshold_min_support") != 48 or
                filtering.get("camera_count") != 60 or filtering.get("input_vertices") != dense_points):
            raise ValueError("filtered cloud provenance does not bind source photos, masks, cameras and dense cloud")
        verify_filter_files(filtering, manifest)
        chosen_points = checked_ply(chosen_cloud, "vertex")
        if chosen_points != filtering.get("retained_vertices") or chosen_points >= dense_points:
            raise ValueError("filtered cloud point count does not match filter report")
        filter_provenance = {"report_path": str(filter_path), "report_sha256": digest(filter_path),
                             "filtered_cloud_path": str(chosen_cloud),
                             "filtered_cloud_sha256": digest(chosen_cloud),
                             "retained_vertices": chosen_points,
                             "threshold_min_support": 48}
    binary_hashes = prior["binary_hashes"]
    for name in TOOLS:
        if digest(tool_path(binary_dir, name)) != binary_hashes[name]:
            raise ValueError(f"OpenMVS executable changed: {name}")
    dense_input = source / "dense"
    if not (dense_input / "images").is_dir() or not (dense_input / "sparse").is_dir():
        raise ValueError("source lacks undistorted COLMAP input")
    copied_bytes = folder_bytes(dense_input) + chosen_cloud.stat().st_size
    if copied_bytes > args.max_gib * (1 << 30) or shutil.disk_usage(source).free < RESERVE + copied_bytes + (256 << 20):
        raise ValueError("copy would exceed output cap or disk reserve")
    output.mkdir(parents=True)
    shutil.copytree(dense_input, output / "dense", symlinks=False)
    shutil.copyfile(chosen_cloud, output / "dense.ply")
    if digest(output / "dense.ply") != digest(chosen_cloud):
        raise ValueError("copied dense cloud differs from source")
    report = {"schema": "classical_backend_dense_finish_v1", "status": "running",
              "runner_sha256": digest(Path(__file__)),
              "source_run": str(source), "source_result_sha256": digest(source_result),
              "source_failure": prior.get("failure"), "source_lineage": prior.get("source_run"),
              "source_dense_ply_sha256": digest(dense_cloud), "dense_points": dense_points,
              "selected_cloud_sha256": digest(chosen_cloud), "selected_points": checked_ply(chosen_cloud, "vertex"),
              "filter_provenance": filter_provenance,
              "binary_hashes": binary_hashes,
              "handoff": "fresh COLMAP reimport; explicit dense.ply to ReconstructMesh, mesh.ply to RefineMesh, refined.ply to TextureMesh",
              "stages": [], "limits": {"max_output_bytes": args.max_gib * (1 << 30),
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
        scene = output / "scene.mvs"
        execute("reimport", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                             "-o", scene, "--image-folder", dense / "images", *common],
                lambda: {"bytes": nonempty_bytes(scene)})
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
    report["source_result_unchanged"] = digest(source_result) == report["source_result_sha256"]
    if filter_provenance:
        try:
            verify_filter_files(filtering, manifest)
            report["filter_sources_unchanged"] = (digest(filter_path) == filter_provenance["report_sha256"]
                                                   and digest(chosen_cloud) == filter_provenance["filtered_cloud_sha256"]
                                                   and digest(dense_cloud) == report["source_dense_ply_sha256"])
        except Exception:
            report["filter_sources_unchanged"] = False
        if not report["filter_sources_unchanged"]:
            report.update(status="failed", failure="filtered-cloud source changed during finish")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary-dir", default=BIN, type=Path)
    parser.add_argument("--filtered-cloud", type=Path)
    parser.add_argument("--filter-report", type=Path)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=2.0)
    parser.add_argument("--max-rss-gib", type=float, default=10.0)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--timeout-minutes", type=float, default=15.0)
    args = parser.parse_args()
    result = finish(args)
    print(json.dumps({"status": result["status"], "result": str(args.output.resolve()),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
