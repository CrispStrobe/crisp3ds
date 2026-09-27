#!/usr/bin/env python3
"""Fresh, bounded replay of the proven image-only turntable sparse profile.

This intentionally invokes the existing PyCOLMAP foreground worker unchanged.
It does not use Berkeley poses, calibration, depth, or reference geometry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from scripts.object_motion import run


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = Path("/Volumes/backups/code/crisp3ds-data")
OUTPUT_CAP = 512 * 1024**2
FREE_FLOOR = 10 * 1024**3
LOG_CAP = 4 * 1024**2
RSS_CAP = 4 * 1024**3
POLL_SECONDS = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_seal(manifest_path):
    """Revalidate every original photo and mask, then bind the input file bytes."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text())
    images = run.validate(manifest)
    return {"manifest": digest(manifest_path),
            "source_manifest": manifest["source_manifest_sha256"],
            "photos": {r["name"]: r["sha256"] for r in images},
            "masks": {r["name"]: r["mask_sha256"] for r in images}}


def is_external_filesystem(root):
    return os.stat(root).st_dev != os.stat(ROOT).st_dev


def validate_output(output, external_root):
    """Require a new directory on the selected external filesystem."""
    if Path(external_root).is_symlink():
        raise ValueError("external root must not be a symlink")
    root = Path(external_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("external root must be a real directory")
    if not is_external_filesystem(root):
        raise ValueError("external root must be on a different filesystem")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or not parent.is_relative_to(root):
        raise ValueError("output parent must be inside external root")
    if shutil.disk_usage(parent).free < FREE_FLOOR + OUTPUT_CAP:
        raise RuntimeError("external output lacks 10 GiB reserve plus 512 MiB budget")
    if shutil.disk_usage(ROOT).free < FREE_FLOOR:
        raise RuntimeError("less than 10 GiB free on internal workspace")
    return parent / output.name


def output_bytes(path):
    size = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                size += p.stat().st_size
        except FileNotFoundError:  # native worker atomically replaces temporary files
            continue
    return size


def verify_model_hashes(summary, destination):
    expected = summary.get("model_files_sha256") or {}
    model_dir = Path(summary.get("model_dir", ""))
    names = {"cameras.bin", "images.bin", "points3D.bin"}
    return (set(expected) == names and model_dir.is_dir() and
            model_dir.resolve().is_relative_to(destination.resolve()) and
            all((model_dir / name).is_file() and not (model_dir / name).is_symlink() and
                digest(model_dir / name) == expected[name] for name in names))


def _drain(pipe, log_path, state):
    used = 0
    with log_path.open("wb") as log:
        for chunk in iter(lambda: pipe.read(65536), b""):
            room = max(0, LOG_CAP - used)
            if room:
                part = chunk[:room]
                log.write(part)
                used += len(part)
            if len(chunk) > room:
                state["log_truncated"] = True
        if state["log_truncated"]:
            log.write(b"\n[log capped at 4 MiB; child output fully drained]\n")


def supervise(command, output, timeout_seconds):
    """Bound the fresh worker by wall time, output bytes, and remaining space."""
    output.mkdir()
    temp = output / "tmp"
    temp.mkdir()
    env = os.environ.copy()
    env.update(TMPDIR=str(temp), TMP=str(temp), OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
    state = {"log_truncated": False}
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=output, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    drain = threading.Thread(target=_drain, args=(process.stdout, output / "worker.log", state), daemon=True)
    drain.start()
    reason = None
    peak_rss = 0
    try:
        while process.poll() is None:
            if time.monotonic() - start >= timeout_seconds:
                reason = "timeout"
            elif output_bytes(output) > OUTPUT_CAP:
                reason = "output_cap"
            elif shutil.disk_usage(output).free < FREE_FLOOR:
                reason = "external_disk_floor"
            elif shutil.disk_usage(ROOT).free < FREE_FLOOR:
                reason = "internal_disk_floor"
            if process.poll() is None:
                sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                        capture_output=True, text=True, check=False)
                if sample.returncode == 0 and sample.stdout.strip().isdigit():
                    peak_rss = max(peak_rss, int(sample.stdout.strip()) * 1024)
                    if peak_rss > RSS_CAP:
                        reason = "rss_cap"
            if reason:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(POLL_SECONDS)
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    finally:
        drain.join(timeout=10)
        process.stdout.close()
    if reason is None:
        if output_bytes(output) > OUTPUT_CAP:
            reason = "output_cap"
        elif shutil.disk_usage(output).free < FREE_FLOOR:
            reason = "external_disk_floor"
        elif shutil.disk_usage(ROOT).free < FREE_FLOOR:
            reason = "internal_disk_floor"
    return {"status": reason or ("completed" if code == 0 else "worker_failed"),
            "returncode": code, "wall_seconds": time.monotonic() - start,
            "timeout_seconds": timeout_seconds, "output_bytes": output_bytes(output),
            "peak_sampled_child_rss_bytes": peak_rss, **state}


def run_profile(manifest_path, output, external_root, timeout_seconds):
    if not 1 <= timeout_seconds <= 3600:
        raise ValueError("timeout must be 1..3600 seconds")
    before = input_seal(manifest_path)
    manifest_path = Path(manifest_path).resolve(strict=True)
    destination = validate_output(output, external_root)
    command = [str(run.VENV_PYTHON), str(ROOT / "scripts/object_motion/run.py"),
               "--worker", "--arm", "foreground", "--manifest", str(manifest_path),
               "--output", str(destination)]
    profile = {"schema": "turntable_sparse_profile_v1", "profile": "prior_foreground_60_jpeg_replay",
               "input_seal": before, "runner_sha256": digest(__file__),
               "worker_sha256": digest(ROOT / "scripts/object_motion/run.py"),
               "diagnose_sha256": digest(ROOT / "scripts/object_motion/diagnose.py"),
               "command": command, "output_cap_bytes": OUTPUT_CAP, "free_floor_bytes": FREE_FLOOR,
               "child_rss_cap_bytes": RSS_CAP,
               "timeout_seconds": timeout_seconds,
               "implemented": ["coarse photo-derived feature masks", "single shared SIMPLE_RADIAL camera",
                               "ordered sequential matching with overlap 8", "one incremental model"],
               "unimplemented_upstream_guidance": ["Meshroom FeatureMatching Minimal 2D Motion 2px",
                                                    "dense-stage masks"],
               "qa_scope": "image-only sparse diagnostics; no independent pose truth, scale, or mesh acceptance",
               "independent_camera_gate_passed": None}
    status = supervise(command, destination, timeout_seconds)
    after = input_seal(manifest_path)
    status["inputs_unchanged"] = before == after
    if not status["inputs_unchanged"]:
        status["status"] = "input_changed"
    summary = destination / "summary.json"
    if status["status"] == "completed":
        if not summary.is_file():
            status["status"] = "missing_summary"
        else:
            result = json.loads(summary.read_text())
            status["registered"] = result.get("registered", 0)
            status["points3D"] = result.get("points3D", 0)
            status["sparse_coverage_passed"] = result.get("registered") == len(before["photos"])
            status["summary_sha256"] = digest(summary)
            status["model_files_sha256"] = result.get("model_files_sha256")
            status["model_hashes_verified"] = verify_model_hashes(result, destination)
            if not status["sparse_coverage_passed"] or not status["model_hashes_verified"]:
                status["status"] = "sparse_qa_failed"
    profile["status"] = status
    (destination / "turntable-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=run.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--check-only", action="store_true", help="validate inputs and destination without writing")
    args = parser.parse_args()
    if args.check_only:
        seal = input_seal(args.manifest)
        output = validate_output(args.output, args.external_root)
        print(json.dumps({"ready": True, "output": str(output), "images": len(seal["photos"]),
                          "input_manifest_sha256": seal["manifest"]}))
        return
    report = run_profile(args.manifest, args.output, args.external_root, args.timeout_seconds)
    print(json.dumps(report["status"]))
    if report["status"]["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
