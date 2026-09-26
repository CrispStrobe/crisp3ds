#!/usr/bin/env python3
"""Guard and run one configured CPU quality command on the Linux VPS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import time
import zipfile

STORAGE = Path("/mnt/storage")
FAST = Path("/mnt/volume1")
MAX_SCRATCH = 8 * 1024**3
MIN_FREE = 2 * 1024**3
MAX_INPUT = 250 * 1024**2
MAX_LOG = 64 * 1024**2


def inside(path: Path, base: Path) -> bool:
    return path.resolve().is_relative_to(base.resolve())


def has_linked_parent(path: Path, stop: Path) -> bool:
    current = path
    while current != stop and current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def tree_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        if any((Path(root) / name).is_symlink() for name in dirs + files):
            raise ValueError("symlink appeared in run output")
        total += sum((Path(root) / name).stat().st_size for name in files)
    return total


def unpack_checked(archive: Path, work: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        names = [entry.filename for entry in entries]
        if not names or names[0] != "remote-quality-manifest.json" or len(names) > 257 or len(set(names)) != len(names):
            raise ValueError("invalid package layout")
        if entries[0].file_size > 1024 * 1024:
            raise ValueError("manifest too large")
        manifest = json.loads(z.read(entries[0]))
        if manifest.get("schema") != 1 or set(names[1:]) != set(manifest.get("files", {})):
            raise ValueError("package manifest mismatch")
        total = 0
        for entry in entries[1:]:
            name = entry.filename
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name or entry.is_dir() or \
                    ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("unsafe package path")
            declared = manifest["files"][name]
            if entry.file_size != declared["bytes"] or entry.file_size > MAX_INPUT:
                raise ValueError("package member size mismatch")
            total += entry.file_size
            if total > MAX_INPUT:
                raise ValueError("package exceeds input cap")
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied = 0
            with z.open(entry) as src, target.open("xb") as dst:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    copied += len(block)
                    if copied > entry.file_size:
                        raise ValueError("package member expanded past declared size")
                    digest.update(block)
                    dst.write(block)
            if copied != entry.file_size or digest.hexdigest() != declared["sha256"]:
                raise ValueError("package hash mismatch")


def preflight(config: dict, *, system: str | None = None) -> dict:
    system = platform.system() if system is None else system
    if system != "Linux":
        raise ValueError("CPU runner is Linux VPS only")
    for path in (STORAGE, FAST):
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"required preexisting mount absent: {path}")
    source = Path(config["input_archive"])
    work = Path(config["work_dir"])
    output = Path(config["output_dir"])
    if not source.is_file() or source.is_symlink() or not inside(source, STORAGE):
        raise ValueError("input archive must be a regular file under /mnt/storage")
    if not inside(work, FAST) or not inside(output, STORAGE):
        raise ValueError("work_dir must be under /mnt/volume1 and output_dir under /mnt/storage")
    if work.exists() or output.exists() or work.is_symlink() or output.is_symlink() or \
            has_linked_parent(work, FAST) or has_linked_parent(output, STORAGE):
        raise ValueError("work and output directories must be fresh")
    cmd = config["command"]
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        raise ValueError("command must be a nonempty JSON string array")
    if not Path(cmd[0]).is_absolute() or not Path(cmd[0]).is_file():
        raise ValueError("command executable must be an existing absolute file")
    threads = config.get("threads", 2)
    timeout = config.get("timeout_seconds", 1800)
    scratch = config.get("scratch_cap_bytes", MAX_SCRATCH)
    if not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("threads must be 1..16")
    if not isinstance(timeout, int) or not 1 <= timeout <= 7200:
        raise ValueError("timeout_seconds must be 1..7200")
    if not isinstance(scratch, int) or not 1 <= scratch <= MAX_SCRATCH:
        raise ValueError("scratch_cap_bytes must be 1..8 GiB")
    result_file = config.get("result_file")
    if not isinstance(result_file, str) or not result_file or Path(result_file).is_absolute() or \
            ".." in Path(result_file).parts or Path(result_file).as_posix() != result_file:
        raise ValueError("result_file must be an output-relative JSON path")
    result_fields = config.get("result_fields")
    if not isinstance(result_fields, dict) or not result_fields or any(
            not isinstance(key, str) or not key or kind not in {"string", "number", "object", "array", "boolean"}
            for key, kind in result_fields.items()):
        raise ValueError("result_fields must declare required top-level fields and types")
    if source.stat().st_size > MAX_INPUT:
        raise ValueError("input archive exceeds 250 MiB")
    if shutil.disk_usage(FAST).free < scratch + MIN_FREE:
        raise ValueError("fast volume lacks scratch cap plus 2 GiB reserve")
    if shutil.disk_usage(STORAGE).free < MIN_FREE:
        raise ValueError("storage volume lacks 2 GiB reserve")
    return {"status": "ready", "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "threads": threads, "timeout_seconds": timeout, "scratch_cap_bytes": scratch}


def _run(config: dict, ready: dict) -> dict:
    work = Path(config["work_dir"])
    output = Path(config["output_dir"])
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    archive = Path(config["input_archive"])
    unpack_checked(archive, work)
    env = os.environ.copy()
    env.update({key: str(ready["threads"]) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env["TMPDIR"] = str(work)
    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (ready["scratch_cap_bytes"], ready["scratch_cap_bytes"]))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()
    started = time.monotonic()
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(config["command"], cwd=work, env=env, stdout=stdout, stderr=stderr,
                                preexec_fn=limits)
        status = "execution_complete"
        try:
            while proc.poll() is None:
                if time.monotonic() - started > ready["timeout_seconds"]:
                    status = "timeout"
                elif tree_bytes(work) + tree_bytes(output) > ready["scratch_cap_bytes"]:
                    status = "scratch_cap_exceeded"
                elif stdout_path.stat().st_size > MAX_LOG or stderr_path.stat().st_size > MAX_LOG:
                    status = "log_cap_exceeded"
                elif shutil.disk_usage(FAST).free < MIN_FREE or shutil.disk_usage(STORAGE).free < MIN_FREE:
                    status = "free_space_reserve_breached"
                if status != "execution_complete":
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(1)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        code = proc.wait()
    if status == "execution_complete" and code != 0:
        status = "failed"
    if status == "execution_complete":
        result_path = output / config["result_file"]
        if not result_path.is_file() or has_linked_parent(result_path, output) or \
                not inside(result_path, output) or result_path.stat().st_size > 1024 * 1024:
            status = "missing_or_large_result"
        else:
            try:
                metrics = json.loads(result_path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                types = {"string": str, "number": (int, float), "object": dict,
                         "array": list, "boolean": bool}
                if not isinstance(metrics, dict) or any(
                        key not in metrics or not isinstance(metrics[key], types[kind]) or
                        (kind == "number" and (isinstance(metrics[key], bool) or not math.isfinite(metrics[key])))
                        for key, kind in config["result_fields"].items()):
                    status = "invalid_result"
                else:
                    status = "execution_succeeded_with_artifact"
            except (OSError, ValueError):
                status = "invalid_result"
    result = {**ready, "status": status, "quality_acceptance": "not_evaluated", "returncode": code,
              "wall_seconds": round(time.monotonic() - started, 3),
              "finished_utc": datetime.now(timezone.utc).isoformat(), "command": config["command"]}
    (output / "status.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def run(config: dict, ready: dict) -> dict:
    try:
        return _run(config, ready)
    except Exception as exc:
        output = Path(config["output_dir"])
        if output.is_dir() and not output.is_symlink():
            report = {**ready, "status": "failed", "quality_acceptance": "not_evaluated",
                      "error_type": type(exc).__name__}
            (output / "status.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true", help="default performs preflight only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    ready = preflight(config)
    result = run(config, ready) if args.execute else ready
    print(json.dumps(result, sort_keys=True))
    if args.execute and result["status"] != "execution_succeeded_with_artifact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
