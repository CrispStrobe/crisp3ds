"""Gated, bounded OpenMVG HIGH bunny to OpenMVS continuation.

The sparse producer and this continuation are separate runs. A separate camera
review must approve the exact sparse model before this module creates output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts.classical_backend.run import checked_ply, folder_bytes, obj_counts


SCHEMA = "openmvg_bunny_high_openmvs_v1"
SOURCE_SCHEMA = "openmvg_bunny_high_photo_control_v1"
GATE_SCHEMA = "bunny_independent_camera_review_v1"
RESERVE = 11 << 30
CAP = 1 << 30
RSS_CAP_KIB = 4 << 20
LOG_CAP = 16 << 20
DEADLINE_SECONDS = 20 * 60
EXPECTED_NAMES = {f"frame_{i:04}.png" for i in range(73)}
TOOLS = ("openMVG_main_openMVG2openMVS", "DensifyPointCloud",
         "ReconstructMesh", "RefineMesh", "TextureMesh")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _real_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked file: {path}")
    return path.resolve()


def validate(args: argparse.Namespace) -> dict:
    source = _real_file(args.source_receipt)
    gate_path = _real_file(args.camera_gate)
    if sha256(gate_path) != args.camera_gate_sha256:
        raise ValueError("camera review receipt SHA-256 mismatch")
    producer = json.loads(source.read_text())
    gate = json.loads(gate_path.read_text())
    model = _real_file(source.parent / "sparse" / "sfm_data.bin")
    model_hash = sha256(model)
    producer_hash = sha256(source)
    if (producer.get("schema") != SOURCE_SCHEMA or
            producer.get("status") != "completed_pending_geometry_review"):
        raise ValueError("HIGH image-only sparse producer has not completed")
    staged = producer.get("staged_photo_sha256", {})
    originals = producer.get("photos", {}).get("photo_sha256", {})
    if (not isinstance(staged, dict) or set(staged) != EXPECTED_NAMES or
            staged != originals):
        raise ValueError("producer does not bind all 73 processed source photos")
    source_images = Path(producer.get("source_images", ""))
    if (source_images.resolve() != Path(args.expected_images).resolve() or
            source_images.is_symlink() or not source_images.is_dir()):
        raise ValueError("producer image source differs from expected processed photo series")
    stages = {row.get("name"): row for row in producer.get("stages", []) if isinstance(row, dict)}
    if (producer.get("output_inventory", {}).get("sparse/sfm_data.bin", {}).get("sha256") != model_hash or
            stages.get("sfm", {}).get("status") != "completed" or
            stages.get("sfm", {}).get("artifact_sha256") != model_hash):
        raise ValueError("producer receipt does not seal the sparse model")
    prepare = _real_file(source_images / "prepare-manifest.json")
    if sha256(prepare) != producer.get("photos", {}).get("prepare_manifest_sha256"):
        raise ValueError("source ordering manifest differs from producer receipt")
    rows = json.loads(prepare.read_text()).get("images", [])
    mapping = {row.get("output"): row.get("source") for row in rows}
    if (len(rows) != 73 or set(mapping) != EXPECTED_NAMES or
            mapping != producer.get("photos", {}).get("source_frame_by_photo") or
            set(mapping.values()) != {f"bunny_{i}_rgb.png" for i in range(73)}):
        raise ValueError("numeric bunny source-frame mapping is not sealed")
    if (gate.get("schema") != GATE_SCHEMA or
            gate.get("decision") != "approved_for_openmvs" or
            gate.get("camera_geometry_reviewed") is not True or
            gate.get("object_region_consistent") is not True or
            not isinstance(gate.get("reviewer"), str) or not gate["reviewer"].strip() or
            gate.get("source_receipt_sha256") != producer_hash or
            gate.get("sparse_model_sha256") != model_hash or
            gate.get("prepare_manifest_sha256") != sha256(prepare)):
        raise ValueError("independent camera review does not approve this exact sparse run")
    for name, expected in staged.items():
        photo = _real_file(source_images / name)
        copied = _real_file(source.parent / "images" / name)
        if sha256(photo) != expected or sha256(copied) != expected:
            raise ValueError(f"processed source photo changed: {name}")
    binaries = {name: _real_file(Path(args.openmvg_binary if name == TOOLS[0]
                                  else args.openmvs_binary_dir / name)) for name in TOOLS}
    if not all(os.access(path, os.X_OK) for path in binaries.values()):
        raise ValueError("converter or OpenMVS tool is not executable")
    output = args.output
    if output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a fresh path under a real existing directory")
    # The converter writes images and OpenMVS writes depth maps beside its scene.
    # Keep 11 GiB free on both the workspace and output volumes throughout.
    for volume in {output.parent.resolve(), source.parent.resolve(), Path(__file__).resolve().parents[2]}:
        if shutil.disk_usage(volume).free < RESERVE + CAP:
            raise ValueError(f"11 GiB reserve plus 1 GiB headroom unavailable: {volume}")
    return {"source_receipt": str(source), "source_receipt_sha256": producer_hash,
            "sparse_model": str(model), "sparse_model_sha256": model_hash,
            "camera_gate": str(gate_path), "camera_gate_sha256": sha256(gate_path),
            "source_images": str(source_images.resolve()),
            "staged_images": str((source.parent / "images").resolve()),
            "prepare_manifest_sha256": sha256(prepare),
            "source_frame_by_photo": mapping,
            "source_photo_sha256": staged,
            "binary_sha256": {name: sha256(path) for name, path in binaries.items()},
            "binaries": {name: str(path) for name, path in binaries.items()}}


def commands(output: Path, bound: dict) -> list[tuple[str, list[str], str]]:
    b = bound["binaries"]
    common = ["--max-threads", "2", "--working-folder", str(output)]
    return [
        ("convert", [b[TOOLS[0]], "-i", bound["sparse_model"], "-o", str(output / "scene.mvs"),
                     "-d", str(output / "images")], "scene.mvs"),
        ("densify", [b["DensifyPointCloud"], "-i", str(output / "scene.mvs"),
                     "-o", str(output / "dense.mvs"), "--resolution-level", "3",
                     "--min-resolution", "640", "--max-resolution", "1280",
                     "--geometric-iters", "2", *common], "dense.ply"),
        ("mesh", [b["ReconstructMesh"], "-i", str(output / "dense.mvs"),
                  "-p", str(output / "dense.ply"), "-o", str(output / "mesh.mvs"), *common], "mesh.ply"),
        ("refine", [b["RefineMesh"], "-i", str(output / "dense.mvs"),
                    "-m", str(output / "mesh.ply"), "-o", str(output / "refined.mvs"),
                    "--resolution-level", "2", "--scales", "1", *common], "refined.ply"),
        ("texture", [b["TextureMesh"], "-i", str(output / "dense.mvs"),
                     "-m", str(output / "refined.ply"), "-o", str(output / "textured.mvs"),
                     "--export-type", "obj", *common], "textured.obj"),
    ]


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _stage(output: Path, name: str, command: list[str], deadline: float,
           reserve_paths: tuple[Path, ...]) -> dict:
    log_path = output / f"{name}.log"
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TMPDIR=str(output / "tmp"))
    started = time.monotonic()
    reason = None
    peak_kib = 0
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, cwd=output, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    reason = "global deadline reached"
                elif folder_bytes(output) > CAP:
                    reason = "1 GiB output cap reached"
                elif log_path.stat().st_size > LOG_CAP:
                    reason = "16 MiB log cap reached"
                elif any(shutil.disk_usage(p).free < RESERVE for p in reserve_paths):
                    reason = "11 GiB disk reserve reached"
                else:
                    sample = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)],
                                            capture_output=True, text=True, check=False)
                    if sample.stdout.strip().isdigit():
                        peak_kib = max(peak_kib, int(sample.stdout.strip()))
                    if peak_kib > RSS_CAP_KIB:
                        reason = "4 GiB child RSS cap reached"
                if reason:
                    break
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                _kill_group(proc)
    code = proc.wait()
    if code != 0 and reason is None:
        reason = f"exit status {code}"
    if folder_bytes(output) > CAP or any(shutil.disk_usage(p).free < RESERVE for p in reserve_paths):
        reason = reason or "post-stage resource cap reached"
    return {"name": name, "status": "complete" if reason is None else "failed",
            "failure": reason, "returncode": code, "command": command,
            "elapsed_seconds": time.monotonic() - started, "peak_child_rss_kib": peak_kib,
            "log": str(log_path)}


def run(args: argparse.Namespace) -> dict:
    bound = validate(args)  # No output exists before independent review passes.
    output = args.output.resolve()
    output.mkdir()
    (output / "tmp").mkdir()
    reserve_paths = tuple({output, Path(__file__).resolve().parents[2]})
    report = {"schema": SCHEMA, "status": "running", "composition":
              "OpenMVG HIGH image-only sparse producer plus separately gated OpenMVS continuation",
              "source": bound, "limits": {"output_bytes": CAP, "reserve_bytes": RESERVE,
              "threads": 2, "deadline_seconds": DEADLINE_SECONDS}, "stages": [],
              "quality_accepted": False, "metric_scale_verified": False, "shipping_approved": False}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    deadline = time.monotonic() + DEADLINE_SECONDS
    save()
    try:
        for name, command, artifact_name in commands(output, bound):
            if any(sha256(Path(bound["binaries"][tool])) != digest for tool, digest in bound["binary_sha256"].items()):
                raise ValueError("tool binary changed before stage")
            result = _stage(output, name, command, deadline, reserve_paths)
            report["stages"].append(result)
            save()
            if result["status"] != "complete":
                raise RuntimeError(f"{name}: {result['failure']}")
            artifact = output / artifact_name
            if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError(f"missing native artifact: {artifact_name}")
            if name == "convert":
                image_dir = output / "images"
                if (image_dir.is_symlink() or not image_dir.is_dir() or
                        {path.name for path in image_dir.iterdir()} != EXPECTED_NAMES or
                        any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0
                            for path in image_dir.iterdir())):
                    raise ValueError("OpenMVG converter did not export all 73 posed images")
                result["exported_images"] = 73
            elif name == "densify":
                result["points"] = checked_ply(artifact, "vertex")
            elif name in ("mesh", "refine"):
                result["faces"] = checked_ply(artifact, "face")
            elif name == "texture":
                result["vertices"], result["faces"] = obj_counts(artifact)
            result["artifact_sha256"] = sha256(artifact)
            save()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = {str(p): shutil.disk_usage(p).free for p in reserve_paths}
    try:
        report["source_unchanged"] = (sha256(Path(bound["source_receipt"])) == bound["source_receipt_sha256"] and
            sha256(Path(bound["sparse_model"])) == bound["sparse_model_sha256"] and
            sha256(Path(bound["camera_gate"])) == bound["camera_gate_sha256"] and
            sha256(Path(bound["source_images"]) / "prepare-manifest.json") == bound["prepare_manifest_sha256"] and
            all(sha256(Path(bound["source_images"]) / name) == value and
                sha256(Path(bound["staged_images"]) / name) == value
                for name, value in bound["source_photo_sha256"].items()) and
            all(sha256(Path(bound["binaries"][name])) == value for name, value in bound["binary_sha256"].items()))
    except Exception as error:
        report["source_unchanged"] = False
        report["source_postcheck_error"] = str(error)
    if (not report["source_unchanged"] or report["output_bytes"] > CAP or
            any(free < RESERVE for free in report["free_bytes_after"].values())):
        report.update(status="failed", failure="source or resource postcheck failed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--camera-gate", required=True, type=Path)
    parser.add_argument("--camera-gate-sha256", required=True)
    parser.add_argument("--expected-images", required=True, type=Path)
    parser.add_argument("--openmvg-binary", required=True, type=Path)
    parser.add_argument("--openmvs-binary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({"status": "abstained", "reason": str(error)}))
        return 2
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
