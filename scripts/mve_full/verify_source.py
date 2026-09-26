"""Verify every selected MVE source input against the pinned archive.

The three intentional SIFT-only edits have fixed resulting hashes. This also
rejects unexpected .cc files that wildcard Make/CMake source lists might add.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile


REVISION = "bf2279f161ba962072ecac85224c15e82bc5f52e"
ARCHIVE_SHA256 = "0e33d40b150313617d28ab4297459939e9ad59c47f21566ed649daf3b34c47f6"
PATCHED = {
    "apps/sfmrecon/sfmrecon.cc": "d6498b777ce19df27bbf7adcd1493ee78e06ba554eb7380722fa921b8f43b0bc",
    "libs/sfm/Makefile": "8d30f254c761a2068b57a67f343bfc1b04211c569d0f25c7bb745a8f1e916e2d",
    "libs/sfm/feature_set.cc": "f53f994cc7c6631c718bc77c0ab40de65315a0941978bceb4baa083d2adcf51d",
}
LIBRARIES = {"math", "util", "mve", "sfm", "dmrecon", "fssr"}
APPS = {"makescene", "sfmrecon", "dmrecon", "scene2pset", "fssrecon", "meshclean"}
EXTENSIONS = {".cc", ".h", ".hpp", ".inl", ".inc", ".c"}


def selected(relative: str) -> bool:
    parts = Path(relative).parts
    if relative in {"LICENSE.txt", "Makefile.inc"}:
        return True
    return (len(parts) >= 3 and
            ((parts[0] == "libs" and parts[1] in LIBRARIES) or
             (parts[0] == "apps" and parts[1] in APPS)) and
            (Path(relative).suffix in EXTENSIONS or parts[-1] == "Makefile"))


def verify(archive: Path, source: Path) -> int:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("pinned MVE archive SHA-256 mismatch")
    expected = {}
    with tarfile.open(archive, "r:gz") as package:
        prefix = f"mve-{REVISION}/"
        for member in package:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if not selected(relative):
                continue
            expected[relative] = PATCHED.get(relative) or hashlib.sha256(package.extractfile(member).read()).hexdigest()
    if len(expected) < 100 or not PATCHED.keys() <= expected.keys():
        raise ValueError("selected source inventory unexpectedly small")
    for relative, checksum in expected.items():
        target = source / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"missing or linked selected source: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"selected source differs from pinned archive/patch: {relative}")
    for family, names in (("libs", LIBRARIES), ("apps", APPS)):
        for name in names:
            for path in (source / family / name).glob("*.cc"):
                relative = path.relative_to(source).as_posix()
                if relative not in expected:
                    raise ValueError(f"unexpected compilable source: {relative}")
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    print(f"verified {verify(args.archive, args.source)} selected source files")


if __name__ == "__main__":
    main()
