"""Bounded refine/texture continuation of a completed OpenMVS rough mesh."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

from scripts.classical_backend.run import (BIN, RESERVE, StageError, checked_ply,
                                           digest, folder_bytes, obj_counts, stage, tool_path)


def finish(args: argparse.Namespace) -> dict:
    if args.source.is_symlink() or args.output.is_symlink() or args.output.exists():
        raise ValueError("source must not be a symlink and output must be fresh")
    source, output, binary_dir = args.source.resolve(), args.output.resolve(), args.binary_dir.resolve()
    if not source.is_dir() or not output.parent.is_dir():
        raise ValueError("source and output parent must be existing directories")
    if (not math.isfinite(args.max_gib) or not 0 < args.max_gib <= 1 or
            not math.isfinite(args.timeout_minutes) or not 0 < args.timeout_minutes <= 10 or
            not math.isfinite(args.max_rss_gib) or not 0 < args.max_rss_gib <= 10 or
            not 0 < args.max_log_mib <= 32 or args.max_threads != 2):
        raise ValueError("invalid bounded continuation settings")
    prior = json.loads((source / "result.json").read_text())
    stages = {item["name"]: item for item in prior.get("stages", [])}
    if (prior.get("schema") != "classical_backend_v1" or prior.get("status") != "failed" or
            any(stages.get(name, {}).get("status") != "complete" for name in
                ("reuse_sfm", "undistort", "import", "densify", "mesh")) or
            stages.get("refine", {}).get("status") != "failed"):
        raise ValueError("source is not a complete rough mesh with failed refinement")
    mesh = source / "mesh.ply"
    dense_scene = source / "dense.mvs"
    faces = checked_ply(mesh, "face")
    if not dense_scene.is_file() or dense_scene.stat().st_size == 0:
        raise ValueError("missing dense scene")
    photo_hashes = {item["name"]: digest(source / "dense" / "images" / item["name"])
                    for item in prior["inputs"]}
    hashes = {"result.json": digest(source / "result.json"), "mesh.ply": digest(mesh),
              "dense.mvs": digest(dense_scene)}
    binaries = {name: digest(tool_path(binary_dir, name)) for name in ("RefineMesh", "TextureMesh")}
    if any(binaries[name] != prior["software"]["openmvs_binaries"][name] for name in binaries):
        raise ValueError("native binary differs from baseline")
    copy_bytes = folder_bytes(source / "dense") + mesh.stat().st_size + dense_scene.stat().st_size
    if copy_bytes > args.max_gib * (1 << 30):
        raise ValueError("scene and images exceed continuation output cap")
    if shutil.disk_usage(output.parent).free < RESERVE + (1 << 30):
        raise ValueError("10 GiB disk reserve plus output allowance unavailable")
    output.mkdir()
    shutil.copytree(source / "dense", output / "dense", symlinks=False)
    shutil.copyfile(mesh, output / "mesh.ply")
    shutil.copyfile(dense_scene, output / "dense.mvs")
    if (digest(output / "mesh.ply") != hashes["mesh.ply"] or
            digest(output / "dense.mvs") != hashes["dense.mvs"] or
            any(digest(output / "dense" / "images" / name) != value for name, value in photo_hashes.items())):
        raise ValueError("copied scene, mesh or images differ")
    report = {"schema": "classical_finish_rough_v1", "status": "running",
              "source": str(source), "source_hashes": hashes, "source_dense_image_hashes": photo_hashes,
              "binary_hashes": binaries, "source_rough_faces": faces,
              "composition": "cached complete baseline rough mesh; only refine and texture stages timed here",
              "settings": {"refine_resolution_level": 1, "refine_scales": 1,
                           "max_threads": 2, "max_output_gib": args.max_gib,
                           "timeout_minutes": args.timeout_minutes},
              "stages": [], "shipping_approved": False, "metric_scale_verified": False,
              "quality_accepted": False}

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

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

    save()
    deadline = time.monotonic() + 60 * args.timeout_minutes
    common = ["--max-threads", "2", "--working-folder", str(output)]
    try:
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
        all(digest(source / "dense" / "images" / name) == value for name, value in photo_hashes.items()) and
        all(digest(tool_path(binary_dir, name)) == value for name, value in binaries.items()))
    if not report["source_unchanged"]:
        report.update(status="failed", failure="source or binary changed during continuation")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=1)
    parser.add_argument("--max-rss-gib", type=float, default=10)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--timeout-minutes", type=float, default=10)
    args = parser.parse_args()
    result = finish(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
