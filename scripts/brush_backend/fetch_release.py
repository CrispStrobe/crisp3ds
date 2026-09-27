"""Opt-in, bounded fetch of the audited Brush v0.3.0 Apple Silicon archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

from .preflight import MIN_FREE, RELEASE_COMMIT, RELEASE_SHA256, RELEASE_URL, sha256

MAX_ARCHIVE = 50 * 1024**2
MAX_UNPACK = 150 * 1024**2
MAX_TOTAL = 220 * 1024**2


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if len(members) > 10_000:
        raise ValueError("too many archive entries")
    total = 0
    for member in members:
        parts = PurePosixPath(member.name)
        if parts.is_absolute() or ".." in parts.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe archive entry: {member.name}")
        total += member.size
    if total > MAX_UNPACK:
        raise ValueError("unpacked archive exceeds 150 MiB")
    return members


def fetch(target: Path) -> dict:
    if target.exists() or target.is_symlink() or target.parent.is_symlink():
        raise ValueError("release target must be a fresh non-symlink path")
    if not target.parent.is_dir():
        raise ValueError("release parent must already exist")
    if shutil.disk_usage(target.parent).free < MIN_FREE + MAX_TOTAL:
        raise ValueError("less than 10 GiB reserve plus 220 MiB release allowance")
    target.mkdir()
    archive_path = target / "brush-app-aarch64-apple-darwin.tar.xz"
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urllib.request.urlopen(RELEASE_URL, timeout=30) as response, archive_path.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_ARCHIVE:
                    raise ValueError("archive exceeds 50 MiB")
                if shutil.disk_usage(target).free < MIN_FREE + MAX_UNPACK:
                    raise ValueError("disk floor before unpack would be exceeded")
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != RELEASE_SHA256:
            raise ValueError("GitHub release digest mismatch")
        with tarfile.open(archive_path, "r:xz") as archive:
            members = safe_members(archive)
            for member in members:
                dest = target / member.name
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, dest.open("xb") as output:
                    if source is None:
                        raise ValueError("unreadable archive member")
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if shutil.disk_usage(target).free < MIN_FREE:
                    raise ValueError("10 GiB disk floor reached")
                if member.mode & 0o111:
                    dest.chmod(0o755)
        files = [p for p in target.rglob("*") if p.is_file() and p != archive_path]
        unpacked = sum(p.stat().st_size for p in files)
        if unpacked + downloaded > MAX_TOTAL:
            raise ValueError("total release footprint exceeds 220 MiB")
        return {
            "schema": "brush_release_fetch_v1",
            "url": RELEASE_URL,
            "release_commit": RELEASE_COMMIT,
            "archive_sha256": sha256(archive_path),
            "archive_bytes": downloaded,
            "unpacked_bytes": unpacked,
            "files": {str(p.relative_to(target)): sha256(p) for p in files},
            "free_bytes_after": shutil.disk_usage(target).free,
            "license_review": "pending; not shipping approved",
        }
    except Exception:
        # Preserve the failed artifact for forensics; never silently replace it.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.target), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
