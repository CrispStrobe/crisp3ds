"""Bounded mesh-only continuation of a calibrated dense cloud with audited DMAPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.classical_backend import calibrated_control
from scripts.classical_backend.calibrated_control import (OUTPUT_SCHEMA,
    depth_reprojection_diagnostic, validate_dmap_cameras)
from scripts.classical_backend.dense_masks import model_hashes, native_mask_name
from scripts.classical_backend.run import (BIN, RESERVE, StageError, checked_ply,
    digest, folder_bytes, obj_counts, stage, tool_path)


MAX_OUTPUT = 200 << 20
TIMEOUT_SECONDS = 300
MAX_RSS = 10 << 30
MAX_LOG = 32 << 20
NATIVE_TOOLS = ("ReconstructMesh", "RefineMesh", "TextureMesh")


def finish(source_arg: Path, output_arg: Path, binary_arg: Path = BIN) -> dict:
    if any(path.is_symlink() for path in (source_arg, output_arg, binary_arg)) or output_arg.exists():
        raise ValueError("source/binaries must be real paths and output fresh")
    source, output, binary_dir = source_arg.resolve(), output_arg.resolve(), binary_arg.resolve()
    if not source.is_dir() or not output.parent.is_dir():
        raise ValueError("source and output parent must exist")
    prior_path = source / "result.json"
    prior = json.loads(prior_path.read_text())
    stages = {item["name"]: item for item in prior.get("stages", [])}
    if (prior.get("schema") != OUTPUT_SCHEMA or prior.get("status") != "failed" or
            prior.get("registered_images") != 60 or prior.get("sources_unchanged") is not True or
            any(stages.get(name, {}).get("status") != "complete" for name in
                ("copy_inputs", "undistort", "warp_masks", "native_depth_preflight", "import", "densify")) or
            stages.get("dmap_camera_check", {}).get("status") != "failed" or
            any(name in stages for name in ("mesh", "refine", "texture"))):
        raise ValueError("source is not a calibrated completed dense cloud stopped at DMAP audit")
    files = {name: source / name for name in ("result.json", "dense.mvs", "dense.ply")}
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError("required dense continuation file missing or linked")
    dense_folder = source / "dense"
    if (dense_folder.is_symlink() or not dense_folder.is_dir() or
            any(path.is_symlink() for path in dense_folder.rglob("*"))):
        raise ValueError("dense image/model tree must not contain links")
    dense_points = checked_ply(files["dense.ply"], "vertex")
    if dense_points != stages["densify"].get("artifact", {}).get("points"):
        raise ValueError("native dense PLY differs from sealed densify count")
    if model_hashes(dense_folder / "sparse") != stages["undistort"]["artifact"]["model_sha256"]:
        raise ValueError("undistorted camera model differs from completed stage")
    mask_report_path = source / "masks" / "report.json"
    if mask_report_path.is_symlink() or not mask_report_path.is_file():
        raise ValueError("sealed warped-mask report missing or linked")
    mask_rows = json.loads(mask_report_path.read_text())["images"]
    photo_hashes = {row["name"]: row["undistorted_image_sha256"] for row in mask_rows}
    mask_hashes = {row["name"]: row["mask_sha256"] for row in mask_rows}
    if (len(photo_hashes) != 60 or
            any(digest(dense_folder / "images" / name) != value for name, value in photo_hashes.items()) or
            any(digest(source / "masks" / native_mask_name(name)) != value
                for name, value in mask_hashes.items())):
        raise ValueError("undistorted photos/masks differ from sealed mask report")
    camera_check = validate_dmap_cameras(source, dense_folder / "sparse")
    if camera_check["checked_views"] != 60 or camera_check["positive_depth_outside_mask"] != 0:
        raise ValueError("complete masked native camera audit required")
    camera_check["depth_reprojection"] = depth_reprojection_diagnostic(source)
    hashes = {name: digest(path) for name, path in files.items()}
    hashes["masks/report.json"] = digest(mask_report_path)
    dmap_hashes = {row["name"]: row["dmap_sha256"] for row in camera_check["images"]}
    dmap_files = sorted(source.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    if len(dmap_files) != 60 or any(digest(path) != row["dmap_sha256"]
                                    for path, row in zip(dmap_files, camera_check["images"])):
        raise ValueError("audited native DMAP set changed")
    dmap_file_hashes = {path.name: digest(path) for path in dmap_files}
    binaries = {name: digest(tool_path(binary_dir, name)) for name in NATIVE_TOOLS}
    if any(binaries[name] != prior["software"]["openmvs_binaries"][name] for name in NATIVE_TOOLS):
        raise ValueError("native binary differs from source run")
    copy_bytes = folder_bytes(dense_folder) + files["dense.mvs"].stat().st_size + files["dense.ply"].stat().st_size
    if copy_bytes + (32 << 20) >= MAX_OUTPUT:
        raise ValueError("dense inputs leave insufficient continuation output allowance")
    if shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT:
        raise ValueError("10 GiB disk floor plus 200 MiB continuation allowance unavailable")
    output.mkdir()
    shutil.copytree(dense_folder, output / "dense", symlinks=False)
    for name in ("dense.mvs", "dense.ply"):
        shutil.copyfile(files[name], output / name)
    if (any(digest(output / name) != hashes[name] for name in ("dense.mvs", "dense.ply")) or
            model_hashes(output / "dense" / "sparse") != stages["undistort"]["artifact"]["model_sha256"] or
            any(digest(output / "dense" / "images" / name) != value for name, value in photo_hashes.items())):
        raise ValueError("copied dense scene, point cloud, camera model or photos differ")
    (output / "dmap-camera-check.json").write_text(json.dumps(camera_check, indent=2) + "\n")
    report = {"schema": "classical_calibrated_finish_dense_v1", "status": "running",
              "provenance_class": prior["provenance_class"], "lane": prior["lane"],
              "composition": "hash-bound dense cloud from failed source; only mesh/refine/texture timed here",
              "source": str(source), "source_hashes": hashes,
              "source_dense_image_sha256": photo_hashes, "source_native_mask_sha256": mask_hashes,
              "source_dmap_sha256": dmap_hashes, "source_dmap_file_sha256": dmap_file_hashes,
              "source_runner_sha256": prior["software"]["runner_sha256"],
              "corrected_audit_runner_sha256": digest(Path(calibrated_control.__file__)),
              "binary_hashes": binaries, "dense_points": dense_points,
              "dmap_camera_check": {key: camera_check[key] for key in
                                    ("checked_views", "max_abs_difference", "positive_depth_outside_mask")},
              "depth_reprojection": camera_check["depth_reprojection"],
              "settings": {"refine_resolution_level": 1, "refine_scales": 1,
                           "max_threads": 2, "max_output_bytes": MAX_OUTPUT,
                           "timeout_seconds": TIMEOUT_SECONDS, "min_free_bytes": RESERVE},
              "stages": [], "shipping_approved": False, "metric_scale_verified": False,
              "quality_accepted": False}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(name, command, validate) -> None:
        result = stage(output, name, [str(arg) for arg in command], deadline,
                       MAX_OUTPUT, MAX_LOG, MAX_RSS)
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    save()
    deadline = time.monotonic() + TIMEOUT_SECONDS
    common = ["--max-threads", "2", "--working-folder", str(output)]
    try:
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
    report["source_unchanged"] = (
        all(digest(source / name) == value for name, value in hashes.items()) and
        all(digest(dense_folder / "images" / name) == value for name, value in photo_hashes.items()) and
        all(digest(source / "masks" / native_mask_name(name)) == value
            for name, value in mask_hashes.items()) and
        all(digest(source / name) == value for name, value in dmap_file_hashes.items()) and
        all(digest(tool_path(binary_dir, name)) == value for name, value in binaries.items()))
    if not report["source_unchanged"]:
        report.update(status="failed", failure="source scene/photos/binaries changed during continuation")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    result = finish(args.source, args.output, args.binary_dir)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
