#!/usr/bin/env python3
"""Create a bounded, explicit-file transport archive for remote quality runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 250 * 1024 * 1024
RESERVE_BYTES = 10 * 1024**3
MAX_FILES = 256
SECRET_NAME = re.compile(r"(?:^|[._/-])(?:\.env|kaggle|credentials?|secrets?|tokens?|id_rsa|\.ssh)(?:$|[._/-])", re.I)
SECRET_VALUE = re.compile(rb"KGAT_[A-Za-z0-9]+|hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}")
BLOCKED_PARTS = {"build", "build-opencv", "node_modules", ".local-tools", ".git", "__pycache__", "datasets"}


def validate_file(root: Path, raw: str) -> tuple[Path, str]:
    p = PurePosixPath(raw)
    if p.is_absolute() or not raw or raw != p.as_posix() or "\\" in raw or any(x in ("", ".", "..") for x in p.parts):
        raise ValueError(f"unsafe relative path: {raw!r}")
    if any(part in BLOCKED_PARTS for part in p.parts) or SECRET_NAME.search(raw):
        raise ValueError(f"blocked path: {raw!r}")
    source = root / raw
    if any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts) + 1)) or not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing, linked, or outside source: {raw!r}")
    return source, raw


def make_package(root: Path, selected: list[str], output: Path, cap: int = MAX_BYTES) -> dict:
    if not selected or len(selected) > MAX_FILES or len(set(selected)) != len(selected):
        raise ValueError("file list must be nonempty, unique, and at most 256 files")
    if "remote-quality-manifest.json" in selected:
        raise ValueError("reserved manifest name cannot be selected")
    if cap <= 0 or cap > MAX_BYTES:
        raise ValueError("cap must be within 1..250 MiB")
    entries = []
    total = 0
    for raw in sorted(selected):
        source, name = validate_file(root, raw)
        size = source.stat().st_size
        total += size
        if total > cap:
            raise ValueError("selected files exceed byte cap")
        data = source.read_bytes()
        if SECRET_VALUE.search(data):
            raise ValueError(f"possible credential in {name}")
        entries.append((name, data, hashlib.sha256(data).hexdigest()))
    if output.exists() or output.is_symlink():
        raise ValueError("output must be fresh")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive output must be outside the repository")
    if not output.parent.is_dir() or any(parent.is_symlink() for parent in (output.parent, *output.parent.parents)):
        raise ValueError("output parent must preexist and not be a symlink")
    if shutil.disk_usage(output.parent).free < RESERVE_BYTES + cap:
        raise ValueError("less than 10 GiB free after worst-case archive")
    manifest = {"schema": 1, "files": {name: {"bytes": len(data), "sha256": digest} for name, data, digest in entries}}
    try:
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("remote-quality-manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            for name, data, _ in entries:
                archive.writestr(name, data)
        if output.stat().st_size > cap:
            raise ValueError("compressed archive exceeds byte cap")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {"status": "ready", "archive": str(output), "bytes": output.stat().st_size, "files": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="JSON list of repository-relative file names")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        parser.error("selection must be a JSON string array")
    print(json.dumps(make_package(ROOT, selected, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
