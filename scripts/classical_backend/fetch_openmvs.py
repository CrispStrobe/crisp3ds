"""Fetch the exact upstream OpenMVS arm64 research binaries into .local-tools."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".local-tools/classical-backend"
TMP = ROOT / ".local-tools/tmp"
URL = "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_macOS_arm64.zip"
SIZE = 68_347_008
SHA256 = "3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b"
SELECTED = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")
RESERVE = 10 << 30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    archive = DEST / "OpenMVS_macOS_arm64-v2.4.0.zip"
    if archive.is_symlink():
        raise ValueError("OpenMVS archive path is a symlink")
    if archive.exists():
        if archive.is_symlink() or archive.stat().st_size != SIZE or sha256(archive) != SHA256:
            raise ValueError("existing OpenMVS archive differs from pin")
    else:
        if shutil.disk_usage(DEST).free < RESERVE + SIZE + 200_000_000:
            raise OSError("OpenMVS download/extraction would violate disk reserve")
        temporary = TMP / "OpenMVS_macOS_arm64-v2.4.0.part"
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with urllib.request.urlopen(URL, timeout=30) as response, temporary.open("xb") as out:
                count = 0
                while chunk := response.read(1 << 20):
                    count += len(chunk)
                    if count > SIZE or shutil.disk_usage(DEST).free < RESERVE + 200_000_000:
                        raise ValueError("download exceeded expected size or disk reserve")
                    out.write(chunk)
            if count != SIZE or sha256(temporary) != SHA256:
                raise ValueError("OpenMVS archive size or SHA-256 mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    binary_dir = DEST / "bin"
    if binary_dir.is_symlink():
        raise ValueError("OpenMVS binary directory is a symlink")
    binary_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in SELECTED:
            info = source.getinfo(name)
            if info.is_dir() or info.file_size > 32_000_000:
                raise ValueError(f"unexpected archive member: {name}")
            target = binary_dir / name
            if target.is_symlink():
                raise ValueError(f"OpenMVS executable path is a symlink: {target}")
            if target.exists():
                if target.is_symlink() or target.stat().st_size != info.file_size:
                    raise ValueError(f"existing executable differs: {target}")
                with source.open(info) as input_stream:
                    expected = hashlib.sha256()
                    for chunk in iter(lambda: input_stream.read(1 << 20), b""):
                        expected.update(chunk)
                if sha256(target) != expected.hexdigest():
                    raise ValueError(f"existing executable hash differs: {target}")
                continue
            if shutil.disk_usage(DEST).free < RESERVE + info.file_size:
                raise OSError("extraction would violate disk reserve")
            with source.open(info) as input_stream, target.open("xb") as output:
                shutil.copyfileobj(input_stream, output, length=1 << 20)
            target.chmod(0o755)
    print(f"Verified OpenMVS v2.4.0 archive and {len(SELECTED)} CLI executables in {binary_dir}")


if __name__ == "__main__":
    fetch()
