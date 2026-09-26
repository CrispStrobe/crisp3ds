"""Fetch and prepare one bounded, real-photo YCB object with a separate-scanner reference.

Only Berkeley NP3 JPEGs are reconstruction inputs. Google geometry is reference-only.
The exact upstream archives below were inspected on 2026-09-27; a changed archive
must be reviewed and this script updated rather than silently accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / ".local-tools/test-data/ycb-cracker-box"
TMP = REPO / ".local-tools/tmp"
MIN_FREE = 10 * 1024**3
MAX_TRANSFER = 750_000_000
MAX_EXTRACT = 200_000_000
MAX_SECONDS = 900
OBJECT = "003_cracker_box"
SOURCES = {
    "berkeley_rgbd": {
        "filename": "003_cracker_box_berkeley_rgbd.tgz",
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/003_cracker_box/003_cracker_box_berkeley_rgbd.tgz",
        "bytes": 663_689_364,
        "etag": '"0d41a398679d242541fd18d348633974-80"',
        "sha256": "15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5",
    },
    "google_64k": {
        "filename": "003_cracker_box_google_64k.tgz",
        "url": "https://ycb-benchmarks.s3.amazonaws.com/data/google/003_cracker_box_google_64k.tgz",
        "bytes": 16_809_373,
        "etag": '"dcd346be307288c58ac2ed474f0e0cd4-3"',
        "sha256": "a4d7173ef4a43c51af5df166deed85ca2b7014626ec5f6f2895561cdbf7c2cd5",
    },
}
ANGLES = tuple(range(0, 360, 6))
PHOTO_MEMBERS = {f"{OBJECT}/NP3_{angle}.jpg": angle for angle in ANGLES}
SCAN_MEMBER = f"{OBJECT}/google_64k/nontextured.ply"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(root: Path, info: dict, deadline: float) -> Path:
    path = root / info["filename"]
    if path.is_symlink():
        raise ValueError(f"archive symlink refused: {path}")
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError(f"existing archive is not the pinned source: {path}")
        return path
    if shutil.disk_usage(root).free < MIN_FREE + info["bytes"] + MAX_EXTRACT:
        raise OSError("fetch would violate 10 GiB free-space reserve")
    request = Request(info["url"], method="HEAD")
    with urlopen(request, timeout=30) as response:
        if (response.status != 200 or int(response.headers.get("Content-Length", "0")) != info["bytes"]
                or response.headers.get("ETag") != info["etag"]):
            raise ValueError("upstream archive size or ETag changed")
    TMP.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix="ycb-", suffix=".part", dir=TMP)
    os.close(descriptor)
    temporary = Path(temp_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with urlopen(info["url"], timeout=30) as response, temporary.open("wb") as output:
            if response.status != 200 or int(response.headers.get("Content-Length", "0")) != info["bytes"]:
                raise ValueError("download response size changed")
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("YCB acquisition deadline exceeded")
                chunk = response.read(min(1024 * 1024, info["bytes"] - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > info["bytes"] or shutil.disk_usage(root).free < MIN_FREE + MAX_EXTRACT:
                    raise ValueError("download exceeded bound or free-space reserve")
                output.write(chunk)
                digest.update(chunk)
        if count != info["bytes"] or digest.hexdigest() != info["sha256"]:
            raise ValueError("archive size or SHA-256 mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def check_member(member: tarfile.TarInfo, seen: set[str]) -> None:
    name = PurePosixPath(member.name)
    if (member.name in seen or name.is_absolute() or any(p in ("", ".", "..") for p in name.parts)
            or not member.isfile() or member.size < 0 or member.size > 32 * 1024**2):
        raise ValueError(f"unsafe, duplicate, or oversized tar member: {member.name}")
    seen.add(member.name)


def extract_selected(archive: Path, selected: dict[str, Path], destination: Path,
                     max_members: int, remaining: int, deadline: float | None = None) -> list[dict]:
    seen: set[str] = set()
    records = []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("YCB extraction deadline exceeded")
            if len(seen) >= max_members:
                raise ValueError("tar member-count limit exceeded")
            check_member(member, seen)
            if member.name not in selected:
                continue
            if member.size > remaining or shutil.disk_usage(destination).free < MIN_FREE + member.size:
                raise ValueError("selected extraction exceeds bound or free-space reserve")
            relative = selected[member.name]
            target = destination / relative
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied = 0
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("missing regular member payload")
            with source, target.open("xb") as output:
                while copied < member.size:
                    if deadline is not None and time.monotonic() > deadline:
                        raise TimeoutError("YCB extraction deadline exceeded")
                    chunk = source.read(min(1024 * 1024, member.size - copied))
                    if not chunk:
                        raise ValueError("truncated tar member")
                    output.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
            remaining -= copied
            records.append({"source_member": member.name, "path": relative.as_posix(),
                            "bytes": copied, "sha256": digest.hexdigest()})
    if seen.intersection(selected) != set(selected):
        raise ValueError(f"missing selected members: {sorted(set(selected) - seen)}")
    return records


def convert_ascii_scan(source: Path, target: Path, deadline: float | None = None) -> dict:
    """Preserve vertex positions as binary float64 and triangular face indices exactly."""
    with source.open("rb") as input_file:
        header = []
        for _ in range(32):
            line = input_file.readline(256)
            if not line:
                raise ValueError("truncated scan header")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        else:
            raise ValueError("scan header too long")
        if header[:2] != ["ply", "format ascii 1.0"] or header[-1] != "end_header":
            raise ValueError("expected original ASCII PLY")
        vertex_line = next((x for x in header if x.startswith("element vertex ")), None)
        face_line = next((x for x in header if x.startswith("element face ")), None)
        if not vertex_line or not face_line:
            raise ValueError("scan lacks vertex/face counts")
        nv, nf = int(vertex_line.split()[-1]), int(face_line.split()[-1])
        if not 0 < nv <= 100_000 or not 0 < nf <= 200_000:
            raise ValueError("scan geometry exceeds conversion limits")
        vertex_section = header[header.index(vertex_line) + 1:header.index(face_line)]
        if vertex_section[:3] != ["property float x", "property float y", "property float z"] or any(
                not x.startswith("property float ") for x in vertex_section):
            raise ValueError("unexpected scan vertex properties")
        if header[header.index(face_line) + 1:-1] != ["property list uchar int vertex_indices"]:
            raise ValueError("unexpected scan face representation")
        with target.open("xb") as output:
            output.write(("ply\nformat binary_little_endian 1.0\n"
                          f"element vertex {nv}\nproperty double x\nproperty double y\nproperty double z\n"
                          f"element face {nf}\nproperty list uchar int vertex_indices\n"
                          "end_header\n").encode("ascii"))
            for _ in range(nv):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("YCB scan conversion deadline exceeded")
                fields = input_file.readline(1024).split()
                if len(fields) != len(vertex_section):
                    raise ValueError("malformed scan vertex")
                xyz = tuple(float(x) for x in fields[:3])
                if not all(math.isfinite(x) for x in xyz):
                    raise ValueError("nonfinite scan vertex")
                output.write(struct.pack("<ddd", *xyz))
            for _ in range(nf):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("YCB scan conversion deadline exceeded")
                fields = input_file.readline(256).split()
                if len(fields) != 4 or fields[0] != b"3":
                    raise ValueError("scan contains malformed or nontriangular face")
                indices = tuple(int(x) for x in fields[1:])
                if min(indices) < 0 or max(indices) >= nv:
                    raise ValueError("scan face index outside vertex range")
                output.write(struct.pack("<Biii", 3, *indices))
            if input_file.read().strip():
                raise ValueError("unexpected trailing scan data")
    return {"path": target.name, "bytes": target.stat().st_size,
            "sha256": sha256_file(target), "vertices": nv, "faces": nf,
            "conversion": "ASCII PLY decimal XYZ to IEEE-754 float64 binary XYZ; triangle indices unchanged; no rescale or cleanup"}


def run(root: Path = ROOT, fetch: bool = False) -> dict:
    started = time.monotonic()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("symlink destination path refused")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "photos").exists() or (root / "reference").exists() or (root / "manifest.json").exists():
        raise FileExistsError("prepared destination must be fresh")
    if sum(x["bytes"] for x in SOURCES.values()) > MAX_TRANSFER:
        raise ValueError("compressed transfer budget exceeded")
    if shutil.disk_usage(root).free < MIN_FREE + MAX_EXTRACT + (MAX_TRANSFER if fetch else 0):
        raise OSError("cannot preserve 10 GiB free-space reserve")
    paths = {}
    for name, info in SOURCES.items():
        if not fetch and not (root / info["filename"]).exists():
            raise FileNotFoundError(f"archive absent; rerun with --fetch: {info['filename']}")
        paths[name] = fetch_archive(root, info, started + MAX_SECONDS)
    TMP.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="ycb-prep-", dir=TMP))
    try:
        photos = {name: Path("photos") / f"NP3_{angle:03d}.jpg" for name, angle in PHOTO_MEMBERS.items()}
        records = extract_selected(paths["berkeley_rgbd"], photos, stage, 2500, MAX_EXTRACT,
                                   started + MAX_SECONDS)
        if len(records) != 60 or sum(x["bytes"] for x in records) > MAX_EXTRACT:
            raise ValueError("wrong photo count or extraction size")
        for record in records:
            image = stage / record["path"]
            with image.open("rb") as stream:
                if stream.read(3) != b"\xff\xd8\xff":
                    raise ValueError("selected member is not JPEG")
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise ValueError("truncated JPEG marker")
            record["camera"] = "NP3"
            record["turntable_angle_degrees"] = int(Path(record["path"]).stem.split("_")[1])
            record["role"] = "reconstruction-input-real-photograph"
        scan_record = extract_selected(paths["google_64k"],
                                       {SCAN_MEMBER: Path("reference/google_64k_original_ascii.ply")},
                                       stage, 32, MAX_EXTRACT - sum(x["bytes"] for x in records),
                                       started + MAX_SECONDS)[0]
        converted = convert_ascii_scan(stage / scan_record["path"],
                                       stage / "reference/google_64k_geometry_f64.ply",
                                       started + MAX_SECONDS)
        if sum(x["bytes"] for x in records) + scan_record["bytes"] + converted["bytes"] > MAX_EXTRACT:
            raise ValueError("prepared subset exceeds expanded limit")
        manifest = {"schema_version": 1, "dataset": "YCB Object and Model Set",
                    "object_id": OBJECT, "dataset_url": "https://ycb-benchmarks.s3.amazonaws.com/index.html",
                    "license": "CC BY 4.0", "license_scope": "YCB dataset data, per official dataset page",
                    "attribution": "Calli, Singh, Walsman, Srinivasa, Abbeel, and Dollar, YCB Object and Model Set",
                    "sources": SOURCES, "selection": {"camera": "Berkeley RGB-D NP3",
                    "angles_degrees": list(ANGLES), "views": 60, "source_modality": "real RGB JPEG",
                    "excluded_from_reconstruction": ["depth", "masks", "provided poses", "Google scanner reference"]},
                    "photos": sorted(records, key=lambda x: x["turntable_angle_degrees"]),
                    "reference_original": {**scan_record, "role": "separate-Google-scanner-reference-only"},
                    "reference_binary_geometry": {**converted, "path": "reference/" + converted["path"],
                    "role": "converted-reference-only"},
                    "evaluation_caveat": "Google scan is a separate-sensor shape oracle, not independently certified metrology ground truth; frames are not registered to scan by this subset; reference never enters reconstruction."}
        for folder in ("photos", "reference"):
            (stage / folder).replace(root / folder)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="fetch pinned official archives if absent")
    args = parser.parse_args()
    result = run(fetch=args.fetch)
    print(f"Prepared {len(result['photos'])} real photos and separate Google reference under {ROOT}")
