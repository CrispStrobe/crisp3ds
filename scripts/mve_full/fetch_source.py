"""Fetch the exact MVE source archive for an isolated CI build."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request

from scripts.mve_full.verify_source import ARCHIVE_SHA256, REVISION


URL = f"https://github.com/simonfuhrmann/mve/archive/{REVISION}.tar.gz"
MAX_BYTES = 20 << 20
MAX_EXTRACTED = 100 << 20
RESERVE = 10 << 30


def no_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlink path is not allowed: {component}")


def fetch(destination: Path) -> Path:
    destination = destination.absolute()
    no_symlink_components(destination)
    if not destination.parent.is_dir():
        raise ValueError("destination parent does not exist")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size > MAX_BYTES:
            raise ValueError("existing MVE archive is not a bounded regular file")
        if hashlib.sha256(destination.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("existing MVE archive has wrong SHA-256")
        return destination
    if shutil.disk_usage(destination.parent).free < RESERVE + MAX_BYTES + MAX_EXTRACTED:
        raise ValueError("insufficient free space for source fetch/extraction plus 10 GiB reserve")
    part = destination.with_name(destination.name + ".partial")
    if part.exists() or part.is_symlink():
        raise ValueError("partial MVE archive already exists")
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(URL, timeout=30) as response, part.open("xb") as output:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("source archive exceeds 20 MiB cap")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("downloaded MVE archive SHA-256 mismatch")
        part.replace(destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return destination


def extract(archive: Path, root: Path) -> Path:
    archive = archive.absolute()
    root = root.absolute()
    no_symlink_components(archive)
    no_symlink_components(root)
    if not archive.is_file() or archive.stat().st_size > MAX_BYTES:
        raise ValueError("MVE archive is not a bounded regular file")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("MVE archive checksum mismatch before extraction")
    source = root / f"mve-{REVISION}"
    if source.exists() or source.is_symlink():
        raise ValueError("source extraction target must be fresh")
    if not root.is_dir():
        raise ValueError("extraction root must exist")
    if shutil.disk_usage(root).free < RESERVE + MAX_EXTRACTED:
        raise ValueError("insufficient extraction allowance plus 10 GiB reserve")
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        if sum(member.size for member in members) > MAX_EXTRACTED:
            raise ValueError("uncompressed source archive exceeds 100 MiB cap")
        prefix = f"mve-{REVISION}/"
        for member in members:
            relative = member.name
            if relative != prefix[:-1] and not relative.startswith(prefix):
                raise ValueError("unexpected source archive root")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("unexpected link or special archive member")
            if ".." in Path(relative).parts:
                raise ValueError("archive path traversal")
        package.extractall(path=root, filter="data")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--extract-root", type=Path)
    args = parser.parse_args()
    archive = fetch(args.destination)
    print(extract(archive, args.extract_root) if args.extract_root else archive)


if __name__ == "__main__":
    main()
