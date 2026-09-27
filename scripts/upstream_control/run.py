#!/usr/bin/env python3
"""Bounded OpenMVS 2.4 CPU replay from the pinned upstream Sceaux scene.

This is a stage-oracle lane: upstream ``scene.mvs`` supplies SfM cameras and
sparse points. It does not test image-only pose recovery or ground truth shape.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.classical_backend import geometry
from scripts.classical_backend.run import obj_counts
from scripts.upstream_control.fetch import COMMIT, OUTPUT as SAMPLE, SELECTED, git_blob, sha256

BIN = ROOT / ".local-tools/classical-backend/bin"
OUTPUT = ROOT / "build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001"
MAX_TOTAL_BYTES = 1536 * 1024 ** 2
RESERVE = 10 * 1024 ** 3
MAX_SECONDS = 900
LOG_CAP = 4 * 1024 ** 2


def verify_sample(sample):
    sample = Path(sample)
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "upstream_openmvs_sample_v23_v1" or manifest.get("commit") != COMMIT:
        raise ValueError("unexpected upstream sample manifest")
    rows = {r["path"]: r for r in manifest["files"]}
    if set(rows) != set(SELECTED):
        raise ValueError("upstream sample file set differs")
    for name, record in rows.items():
        path = sample / name
        if path.is_symlink() or path.stat().st_size != record["bytes"] or (
                sha256(path) != record["sha256"] or git_blob(path) != record["git_blob_sha1"]):
            raise ValueError(f"changed upstream sample file: {name}")
    return manifest, sha256(manifest_path)


def folder_bytes(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:  # Native OpenMVS atomically replaces some outputs.
                pass
    return total


def drain(pipe, target, state):
    saved = 0
    with target.open("xb") as log:
        for chunk in iter(lambda: pipe.read(1 << 16), b""):
            n = min(len(chunk), max(0, LOG_CAP - saved))
            if n:
                log.write(chunk[:n])
                saved += n
            if n < len(chunk):
                state["truncated"] = True
        if state["truncated"]:
            log.write(b"\n[bounded log truncated; child output drained]\n")


def run_stage(name, args, output, sample_bytes, deadline):
    start = time.monotonic()
    state = {"truncated": False}
    env = os.environ.copy()
    env["TMPDIR"] = str(ROOT / ".local-tools/tmp")
    process = subprocess.Popen([str(x) for x in args], cwd=output,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    reader = threading.Thread(target=drain, args=(process.stdout, output / f"{name}.log", state), daemon=True)
    reader.start()
    status = None
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                status = "timeout"
                process.kill()
                break
            if shutil.disk_usage(output).free < RESERVE:
                status = "disk_floor"
                process.kill()
                break
            if sample_bytes + folder_bytes(output) > MAX_TOTAL_BYTES:
                status = "output_cap"
                process.kill()
                break
            time.sleep(2)
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=10)
        process.stdout.close()
    if status is None:
        status = "completed" if code == 0 else "failed"
    return {"name": name, "status": status, "returncode": code,
            "elapsed_seconds": time.monotonic() - start, "log_truncated": state["truncated"],
            "command": [str(x) for x in args], "output_bytes": folder_bytes(output)}


def replay(sample, output):
    sample, output = Path(sample), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest, manifest_hash = verify_sample(sample)
    sample_bytes = manifest["total_bytes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if sample_bytes >= MAX_TOTAL_BYTES or shutil.disk_usage(output.parent).free < RESERVE + MAX_TOTAL_BYTES:
        raise RuntimeError("insufficient space for bounded upstream control")
    for name in ("DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"):
        if not (BIN / name).is_file():
            raise ValueError(f"missing pinned OpenMVS CLI {name}")
    output.mkdir(parents=True)
    (ROOT / ".local-tools/tmp").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    for name in ["scene.mvs"] + [f"images/{i:05}.jpg" for i in range(11)]:
        source, dest = sample / name, output / name
        shutil.copyfile(source, dest)
        if sha256(dest) != sha256(source):
            raise ValueError(f"staged photo or scene differs: {name}")
    specs = [
        ("densify", "DensifyPointCloud", ["-i", output / "scene.mvs", "-o", output / "scene_dense.mvs",
            "--max-resolution", "640", "--min-resolution", "640", "--resolution-level", "2"],
         output / "scene_dense.ply", "vertex"),
        ("mesh", "ReconstructMesh", ["-i", output / "scene_dense.mvs", "-p", output / "scene_dense.ply",
            "-o", output / "scene_dense_mesh.mvs"], output / "scene_dense_mesh.ply", "face"),
        ("refine", "RefineMesh", ["-i", output / "scene_dense.mvs", "-m", output / "scene_dense_mesh.ply",
            "-o", output / "scene_dense_mesh_refine.mvs", "--resolution-level", "2", "--scales", "1"],
         output / "scene_dense_mesh_refine.ply", "face"),
        ("texture", "TextureMesh", ["-i", output / "scene_dense.mvs", "-m", output / "scene_dense_mesh_refine.ply",
            "-o", output / "scene_dense_mesh_refine_texture.mvs", "--resolution-level", "2",
            "--export-type", "obj"], output / "scene_dense_mesh_refine_texture.obj", "obj"),
    ]
    report = {"schema": "upstream_sceaux_v23_stage_oracle_v24_cpu_v1",
              "lane": "stage-oracle: supplied upstream image-derived scene.mvs cameras/sparse points",
              "same_input_image_only_sfm": False, "reference_geometry_is_ground_truth": False,
              "sample_manifest_sha256": manifest_hash, "sample_commit": COMMIT,
              "sample_total_bytes": sample_bytes,
              "binary_sha256": {name: sha256(BIN / name) for name in (
                  "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")},
              "caps": {"max_download_plus_output_bytes": MAX_TOTAL_BYTES,
                       "reserve_bytes": RESERVE, "deadline_seconds": MAX_SECONDS,
                       "threads": 2, "depth_max_min_resolution": 640},
              "stages": [], "status": "running"}
    report_path = output / "report.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    deadline = time.monotonic() + MAX_SECONDS
    common = ["--working-folder", output, "--max-threads", "2"]
    try:
        for stage_name, binary, args, artifact, kind in specs:
            result = run_stage(stage_name, [BIN / binary, *args, *common], output, sample_bytes, deadline)
            report["stages"].append(result)
            save()
            if result["status"] != "completed":
                raise RuntimeError(f"{stage_name} {result['status']}")
            if kind == "obj":
                vertices, faces = obj_counts(artifact)
                counts = {"vertices": vertices, "faces": faces}
            else:
                counts = geometry.inspect(artifact)
            if counts["vertices" if kind == "vertex" else "faces"] < 1:
                raise ValueError(f"empty {stage_name} geometry")
            result["artifact"] = {"path": str(artifact), "sha256": sha256(artifact), **counts}
            save()
        reference = {}
        for name in ("scene_dense.ply", "scene_dense_mesh.ply"):
            reference[name] = {"sha256": sha256(sample / name), **geometry.inspect(sample / name)}
        report["upstream_regression_oracles"] = reference
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        raise
    finally:
        report["output_bytes"] = folder_bytes(output)
        save()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = replay(args.sample, args.output)
    print(json.dumps({"status": report["status"], "stages": [s["name"] for s in report["stages"]]}))


if __name__ == "__main__":
    main()
