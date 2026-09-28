"""Private Kaggle AliceVision loader smoke. No photographs, upload, or mesh work."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import selectors
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request


URL = "https://github.com/alicevision/AliceVision/releases/download/v3.3.0/AliceVision-3.3.0-Linux.tar.gz"
SHA256 = "f43f498312859af627f2f7f65a6d33c2a3411b37989b8b680c04c8c690dcb640"
ARCHIVE_BYTES = 1_505_191_867
MAX_EXPANDED = 8 << 30
MIN_FREE_AFTER = 4 << 30
MAX_SECONDS = 15 * 60
MAX_DOWNLOAD_SECONDS = 8 * 60
MAX_EXTRACT_SECONDS = 5 * 60
OUTPUT = Path("/kaggle/working/alicevision-binary-smoke.json")
# /tmp is ephemeral and is not published as a Kaggle kernel output.
SCRATCH = Path("/tmp/alicevision-binary-smoke")


def short_command(argv: list[str], timeout: int = 20, env: dict[str, str] | None = None,
                  required_tokens: tuple[str, ...] = ()) -> dict:
    start = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        assert process.stdout is not None
        output = bytearray()
        deadline = start + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"timed_out": True, "seconds": round(time.monotonic() - start, 3)}
                for key, _ in selector.select(remaining):
                    block = os.read(key.fd, min(4096, 65_537 - len(output)))
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        output.extend(block)
                        if len(output) > 65_536:
                            return {"output_capped": True, "seconds": round(time.monotonic() - start, 3)}
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        return {"returncode": process.returncode,
                "seconds": round(time.monotonic() - start, 3),
                "missing_libraries": b"not found" in output,
                "required_tokens_present": {token: token.encode() in output
                                            for token in required_tokens},
                "output_tail": output[-4096:].decode("utf-8", "replace")}
    except subprocess.TimeoutExpired:
        return {"timed_out": True, "seconds": round(time.monotonic() - start, 3)}
    except OSError as exc:
        return {"unavailable": True, "error": f"{type(exc).__name__}: {exc}"[:256],
                "seconds": round(time.monotonic() - start, 3)}
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()


def download(path: Path, deadline: float) -> dict:
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(URL, timeout=30) as source, path.open("xb") as target:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("archive download exceeded deadline")
            block = source.read(1 << 20)
            if not block:
                break
            total += len(block)
            if total > ARCHIVE_BYTES:
                raise ValueError("archive exceeds pinned release size")
            digest.update(block)
            target.write(block)
    if total != ARCHIVE_BYTES or digest.hexdigest() != SHA256:
        raise ValueError("official release asset size or SHA-256 differs")
    return {"bytes": total, "sha256": digest.hexdigest()}


def inspect_tar(path: Path) -> dict:
    files = 0
    expanded = 0
    with tarfile.open(path, "r:gz") as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if (name.is_absolute() or ".." in name.parts or
                    not (item.isfile() or item.isdir() or item.issym() or item.islnk())):
                raise ValueError("archive has unsafe member")
            if item.issym() or item.islnk():
                target = PurePosixPath(item.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError("archive has unsafe link")
            files += 1
            expanded += item.size if item.isfile() else 0
            if files > 100_000 or expanded > MAX_EXPANDED:
                raise ValueError("archive exceeds member or expanded-byte cap")
    return {"members": files, "expanded_regular_bytes": expanded}


def bundled_runtime_paths(extracted: Path) -> tuple[Path, Path]:
    lib_dirs = {path.parent for path in extracted.rglob("libaliceVision_cmdline.so.3")
                if path.is_file()}
    if len(lib_dirs) != 1:
        raise RuntimeError("bundled AliceVision library directory missing or ambiguous")
    lib_dir = lib_dirs.pop()
    install_root = lib_dir.parent
    if not (install_root / "share" / "aliceVision" / "config.ocio").is_file():
        raise RuntimeError("bundled AliceVision OCIO configuration missing")
    return lib_dir, install_root


def main() -> None:
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("overall deadline exceeded")))
    signal.alarm(MAX_SECONDS)
    report: dict = {"schema": "alicevision_linux_binary_smoke_v1",
                    "status": "running", "release_url": URL,
                    "expected_release_sha256": SHA256,
                    "input_photographs": 0, "no_uploaded_photographs": True}
    try:
        if OUTPUT.exists() or SCRATCH.exists():
            raise FileExistsError("fresh scratch and output required")
        if not SCRATCH.parent.is_dir() or not OUTPUT.parent.is_dir():
            raise FileNotFoundError("Kaggle scratch/output directories missing")
        free = shutil.disk_usage(SCRATCH.parent).free
        report["starting_free_bytes"] = free
        if free < ARCHIVE_BYTES + MAX_EXPANDED + MIN_FREE_AFTER:
            raise RuntimeError("insufficient disk for bounded download, extraction and floor")
        SCRATCH.mkdir()
        archive = SCRATCH / "alicevision.tar.gz"
        report["archive"] = download(archive, start + MAX_DOWNLOAD_SECONDS)
        report["tar_inventory"] = inspect_tar(archive)
        if shutil.disk_usage(SCRATCH).free < report["tar_inventory"]["expanded_regular_bytes"] + MIN_FREE_AFTER:
            raise RuntimeError("not enough free space to extract while retaining 4 GiB")
        extracted = SCRATCH / "release"
        extracted.mkdir()
        command = ["tar", "-xzf", str(archive), "-C", str(extracted)]
        stage = short_command(command, MAX_EXTRACT_SECONDS)
        report["extract"] = {key: value for key, value in stage.items() if key != "output_tail"}
        if stage.get("returncode") != 0:
            raise RuntimeError("pinned archive extraction failed or timed out")
        names = ("aliceVision_cameraInit", "aliceVision_depthMapEstimation")
        located = {name: list(extracted.rglob(name)) for name in names}
        if any(len(paths) != 1 or not paths[0].is_file() for paths in located.values()):
            raise RuntimeError("expected CLI inventory missing or ambiguous")
        lib_dir, install_root = bundled_runtime_paths(extracted)
        report["bundled_lib_dir"] = str(lib_dir.relative_to(extracted))
        report["bundled_install_root"] = str(install_root.relative_to(extracted))
        cli_env = os.environ.copy()
        cli_env["LD_LIBRARY_PATH"] = str(lib_dir) + (
            ":" + cli_env["LD_LIBRARY_PATH"] if cli_env.get("LD_LIBRARY_PATH") else "")
        cli_env["ALICEVISION_ROOT"] = str(install_root)
        report["cli"] = {}
        for name, paths in located.items():
            executable = paths[0]
            report["cli"][name] = {
                "relative_path": str(executable.relative_to(extracted)),
                "ldd": short_command(["ldd", str(executable)], 20, env=cli_env),
                "help": short_command([str(executable), "--help"], 20, env=cli_env),
            }
        report["gpu"] = short_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv,noheader"], 10)
        report["status"] = "loader_smoke_complete"
        report["runtime_compatible"] = all(
            item["help"].get("returncode") == 0
            and item["ldd"].get("returncode") == 0
            and not item["ldd"].get("missing_libraries", True)
            for item in report["cli"].values())
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"[:512]
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    report["ending_free_bytes"] = shutil.disk_usage(OUTPUT.parent).free
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if len(payload.encode()) > 64 << 10:
        raise RuntimeError("smoke report exceeds 64 KiB")
    with OUTPUT.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"status": report["status"], "runtime_compatible": report.get("runtime_compatible"),
                      "receipt": str(OUTPUT)}))


if __name__ == "__main__":
    main()
