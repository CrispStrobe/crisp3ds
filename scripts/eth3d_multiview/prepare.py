#!/usr/bin/env python3
"""Verify, extract, and validate the pinned ETH3D pipes archives offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
import zlib

try:
    from . import fetch
except ImportError:
    import fetch


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / ".local-tools/test-data/eth3d-multiview-pipes-v2"
DEST = REPO / "build-opencv/pipes-prepare/run-001"
TMP = REPO / ".local-tools/tmp"
INVENTORY_SHA256 = "142c62c1c623ae581854f41b4f41d9ef6b85a0b13d08ff5b2029f6ddfdc500c6"
ARCHIVES = {
    "pipes_dslr_undistorted.7z": (145321540, "718981351c14e84759fcc73215e7251fce93d6e9ea1fe24f9e15f1028232c12c"),
    "pipes_dslr_scan_eval.7z": (53851871, "d900ecf9ab3bfda3dddbd78174f1ef90103aa8060d1f403253ad7c9e3dc54f4c"),
}
MAX_EXPANDED = 500 * 1024**2
MIN_FREE = 10 * 1024**3
TIME_LIMIT = 300
JPEG_PREFIX = "pipes/images/dslr_images_undistorted/"
CALIB_PREFIX = "pipes/dslr_calibration_undistorted/"
SCAN_PREFIX = "pipes/dslr_scan_eval/"


def _plain_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"directory missing or symlinked: {path}")


def _file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"file missing or symlinked: {path}")


def _safe_relative(name: str) -> PurePosixPath:
    fetch.safe_member(name)
    if ":" in name or name.endswith("/"):
        raise ValueError(f"unsafe member path: {name!r}")
    return PurePosixPath(name)


def _inventory_records(reviewed: dict) -> tuple[dict[str, int], set[str]]:
    records = reviewed.get("archives")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("reviewed inventory needs two archives")
    files: dict[str, int] = {}
    directories: set[str] = set()
    seen_archives = set()
    total = 0
    for record in records:
        name = record.get("filename")
        if name not in ARCHIVES or name in seen_archives:
            raise ValueError("unapproved or repeated archive")
        seen_archives.add(name)
        if record.get("bytes") != ARCHIVES[name][0] or record.get("sha256") != ARCHIVES[name][1]:
            raise ValueError(f"archive pin mismatch: {name}")
        listing = record.get("inventory", {})
        members = listing.get("members")
        if not isinstance(members, list) or listing.get("member_count") != len(members):
            raise ValueError("inventory member count mismatch")
        subtotal = 0
        local = set()
        for member in members:
            path = member.get("path")
            if not isinstance(path, str):
                raise ValueError("missing member path")
            _safe_relative(path)
            size = member.get("size_bytes")
            directory = member.get("directory")
            if path in local or type(size) is not int or size < 0 or type(directory) is not bool:
                raise ValueError(f"invalid or duplicate member: {path}")
            local.add(path)
            if directory:
                if size or path in files:
                    raise ValueError(f"invalid directory: {path}")
                directories.add(path)
            else:
                if path in files or path in directories:
                    raise ValueError(f"file collision: {path}")
                files[path] = size
                subtotal += size
        if listing.get("expanded_bytes") != subtotal:
            raise ValueError(f"expanded size mismatch: {name}")
        total += subtotal
    if total != reviewed.get("total_expanded_bytes") or total != 286754145 or total > MAX_EXPANDED:
        raise ValueError("expanded size outside approved bound")
    if len(files) != 19 or len(directories) != 5:  # pipes/ is intentionally shared.
        raise ValueError("unexpected file or directory count")
    for path in files:
        if any(str(parent) in files for parent in PurePosixPath(path).parents if str(parent) != "."):
            raise ValueError(f"file is parent of a member: {path}")
    for path in directories:
        if path in files or any(str(parent) in files for parent in PurePosixPath(path).parents if str(parent) != "."):
            raise ValueError(f"directory collides with file: {path}")
    jpegs = {p for p in files if p.startswith(JPEG_PREFIX) and p.endswith(".JPG")}
    expected_other = {CALIB_PREFIX + n for n in ("cameras.txt", "images.txt", "points3D.txt")}
    expected_other |= {SCAN_PREFIX + n for n in ("scan1.ply", "scan_alignment.mlp")}
    if len(jpegs) != 14 or set(files) != jpegs | expected_other:
        raise ValueError("unexpected approved file set")
    return files, directories


def preflight(source: Path = SOURCE, destination: Path = DEST, tmp: Path = TMP) -> tuple[dict, dict[str, int], set[str]]:
    _plain_directory(source)
    _plain_directory(tmp)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination must be fresh: {destination}")
    if any(p.is_symlink() for p in destination.parents):
        raise ValueError("destination has a symlinked parent")
    _file(source / "inventory-reviewed.json")
    if fetch.sha256_file(source / "inventory-reviewed.json") != INVENTORY_SHA256:
        raise ValueError("reviewed inventory SHA-256 changed")
    reviewed = json.loads((source / "inventory-reviewed.json").read_text())
    files, directories = _inventory_records(reviewed)
    seven_zip = shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("7z is required")
    deadline = time.monotonic() + 60
    for record in reviewed["archives"]:
        name = record["filename"]
        archive = source / name
        _file(archive)
        if archive.stat().st_size != ARCHIVES[name][0] or fetch.sha256_file(archive) != ARCHIVES[name][1]:
            raise ValueError(f"archive SHA-256 or size changed: {name}")
        if fetch.inventory(archive, deadline) != record["inventory"]:
            raise ValueError(f"raw 7z listing changed: {name}")
    existing_parent = next((p for p in destination.parents if p.exists()), None)
    if existing_parent is None or not existing_parent.is_dir():
        raise ValueError("destination has no usable parent")
    if shutil.disk_usage(existing_parent).free < MIN_FREE + MAX_EXPANDED:
        raise OSError("insufficient space for 10 GiB floor and expanded bound")
    return reviewed, files, directories


def _tree(destination: Path) -> tuple[dict[str, Path], set[str], int]:
    files, directories, total = {}, set(), 0
    for root, dirs, names in os.walk(destination, followlinks=False):
        parent = Path(root)
        for name in dirs:
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"extracted symlink: {path}")
            directories.add(path.relative_to(destination).as_posix())
        for name in names:
            path = parent / name
            _file(path)
            rel = path.relative_to(destination).as_posix()
            files[rel] = path
            total += path.stat().st_size
    return files, directories, total


def _extract_one(archive: Path, destination: Path, deadline: float, expected: dict[str, int], dirs: set[str], tmp: Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp)
    command = [shutil.which("7z"), "x", "-y", "-aos", f"-o{destination}", str(archive)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        while process.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("300-second extraction limit exceeded")
            if shutil.disk_usage(destination).free < MIN_FREE:
                raise OSError("free space fell below 10 GiB")
            actual_files, actual_dirs, total = _tree(destination)
            if total > MAX_EXPANDED or actual_dirs - dirs or set(actual_files) - set(expected):
                raise ValueError("extraction exceeded approved paths or expanded limit")
            if any(path.stat().st_size > expected[name] for name, path in actual_files.items()):
                raise ValueError("extracted file exceeded approved size")
            time.sleep(0.2)
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode:
            raise ValueError(f"7z extraction failed: {(stderr or stdout)[-500:].decode(errors='replace')}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise


def _crc32_file(path: Path) -> str:
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return f"{crc:08X}"


def verify_extracted(destination: Path, files: dict[str, int], directories: set[str], source: Path, reviewed: dict) -> dict[str, str]:
    actual_files, actual_dirs, total = _tree(destination)
    if set(actual_files) != set(files) or actual_dirs != directories or total != sum(files.values()):
        raise ValueError("extracted tree differs from inventory")
    crc_by_path = {}
    for record in reviewed["archives"]:
        archive = source / record["filename"]
        result = subprocess.run([shutil.which("7z"), "l", "-slt", str(archive)], capture_output=True, text=True, timeout=30, check=True)
        sections = result.stdout.split("----------", 1)[1].strip().split("\n\n")
        for section in sections:
            fields = dict(line.split(" = ", 1) for line in section.splitlines() if " = " in line)
            if fields.get("Path") in files:
                crc_by_path[fields["Path"]] = fields.get("CRC", "")
    digests = {}
    for name, path in actual_files.items():
        if path.stat().st_size != files[name] or _crc32_file(path) != crc_by_path.get(name):
            raise ValueError(f"extracted size or archive CRC mismatch: {name}")
        digests[name] = fetch.sha256_file(path)
    return digests


def verify_source_unchanged(source: Path, reviewed: dict) -> None:
    if fetch.sha256_file(source / "inventory-reviewed.json") != INVENTORY_SHA256:
        raise ValueError("source inventory changed during extraction")
    deadline = time.monotonic() + 60
    for record in reviewed["archives"]:
        archive = source / record["filename"]
        _file(archive)
        if archive.stat().st_size != record["bytes"] or fetch.sha256_file(archive) != record["sha256"]:
            raise ValueError(f"archive changed during extraction: {archive.name}")
        if fetch.inventory(archive, deadline) != record["inventory"]:
            raise ValueError(f"archive listing changed during extraction: {archive.name}")


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG: {path}")
        while True:
            marker = stream.read(1)
            if marker != b"\xff":
                raise ValueError(f"invalid JPEG marker: {path}")
            while marker == b"\xff":
                marker = stream.read(1)
            code = marker[0] if marker else 0
            if code in (0, 0xD9, 0xDA):
                raise ValueError(f"JPEG has no dimensions: {path}")
            if code in range(0xD0, 0xD8) or code == 0x01:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment")
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2 or length > path.stat().st_size:
                raise ValueError("invalid JPEG segment length")
            if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                frame = stream.read(5)
                if len(frame) != 5:
                    raise ValueError("truncated JPEG frame")
                height, width = struct.unpack(">HH", frame[1:])
                if width <= 0 or height <= 0:
                    raise ValueError("invalid JPEG dimensions")
                stream.seek(-2, os.SEEK_END)
                if stream.read(2) != b"\xff\xd9":
                    raise ValueError("JPEG missing EOI marker")
                return width, height
            stream.seek(length - 2, os.SEEK_CUR)


def validate_cameras(path: Path) -> dict[int, tuple[int, int, list[float]]]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8 or parts[1] != "PINHOLE":
            raise ValueError("unsupported camera model or fields")
        ident, width, height = int(parts[0]), int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        if ident in cameras or ident < 0 or width < 1 or height < 1 or not all(map(math.isfinite, params)) or min(params[:2]) <= 0:
            raise ValueError("invalid PINHOLE camera")
        cameras[ident] = (width, height, params)
    if not cameras:
        raise ValueError("no cameras")
    return cameras


def validate_images(path: Path, cameras: dict, jpeg_paths: dict[str, Path]) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    if len(lines) % 2:
        raise ValueError("COLMAP image records must have paired observation lines")
    images, seen_names, seen_ids = [], set(), set()
    for index in range(0, len(lines), 2):
        parts = lines[index].split()
        if len(parts) != 10:
            raise ValueError("invalid COLMAP image record")
        ident, camera_id = int(parts[0]), int(parts[8])
        q = [float(x) for x in parts[1:5]]
        t = [float(x) for x in parts[5:8]]
        name = parts[9]
        _safe_relative(name)
        matches = [p for p in jpeg_paths if p == name or p.endswith("/" + name)]
        if ident < 0 or ident in seen_ids or name in seen_names or camera_id not in cameras or len(matches) != 1:
            raise ValueError("invalid image id, camera, or contained path")
        if not all(map(math.isfinite, q + t)) or abs(sum(x*x for x in q) - 1) > 1e-3:
            raise ValueError("nonfinite or nonunit world-to-camera pose")
        if jpeg_dimensions(jpeg_paths[matches[0]]) != cameras[camera_id][:2]:
            raise ValueError("camera dimensions differ from JPEG")
        seen_ids.add(ident)
        seen_names.add(name)
        camera = cameras[camera_id]
        images.append({"id": ident, "name": PurePosixPath(name).name, "camera_id": camera_id,
                       "path": matches[0], "size_bytes": jpeg_paths[matches[0]].stat().st_size,
                       "camera": {"model": "PINHOLE", "width": camera[0], "height": camera[1], "params": camera[2]},
                       "qvec": q, "tvec": t, "pose_convention": "world_to_camera"})
    if len(images) != 14 or len(seen_names) != len(jpeg_paths):
        raise ValueError("expected 14 distinct camera-linked images")
    return sorted(images, key=lambda image: image["name"])


PLY_TYPES = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1, "short": 2, "ushort": 2,
             "int16": 2, "uint16": 2, "int": 4, "uint": 4, "int32": 4, "uint32": 4,
             "float": 4, "float32": 4, "double": 8, "float64": 8}


def validate_ply(path: Path) -> dict:
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline(4097)
            if not line or len(line) > 4096 or len(header) + len(line) > 65536:
                raise ValueError("invalid or oversized PLY header")
            header.extend(line)
    lines = header.decode("ascii").splitlines()
    if len(lines) < 4 or lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
        raise ValueError("expected binary little-endian PLY")
    elements = []
    for line in lines[2:-1]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        if parts[0] == "element" and len(parts) == 3:
            count = int(parts[2])
            if count < 0:
                raise ValueError("negative PLY element count")
            elements.append({"name": parts[1], "count": count, "stride": 0, "properties": []})
        elif parts[0] == "property" and len(parts) == 3 and elements and parts[1] in PLY_TYPES:
            elements[-1]["stride"] += PLY_TYPES[parts[1]]
            elements[-1]["properties"].append((parts[1], parts[2]))
        else:
            raise ValueError("unsupported PLY header schema")
    vertices = [e for e in elements if e["name"] == "vertex"]
    if len(vertices) != 1 or vertices[0]["count"] < 1 or not {"x", "y", "z"}.issubset({p[1] for p in vertices[0]["properties"]}):
        raise ValueError("PLY needs vertex positions and positive count")
    payload = path.stat().st_size - len(header)
    expected = sum(e["count"] * e["stride"] for e in elements)
    if payload != expected or payload < 1:
        raise ValueError("PLY payload does not match declared point count and stride")
    return {"format": "binary_little_endian", "vertex_count": vertices[0]["count"],
            "vertex_stride_bytes": vertices[0]["stride"], "header_bytes": len(header), "payload_bytes": payload,
            "elements": elements}


def validate_alignment(path: Path, scan_path: Path) -> dict:
    root = ET.parse(path).getroot()
    meshes = root.findall(".//MLMesh")
    if len(meshes) != 1:
        raise ValueError("alignment must declare exactly one scan")
    name = meshes[0].get("filename", "")
    _safe_relative(name)
    declared = (path.parent / name).resolve()
    if declared != scan_path.resolve() or not declared.is_relative_to(path.parent.resolve()):
        raise ValueError("alignment scan path escapes or differs from extracted PLY")
    matrix_node = meshes[0].find("MLMatrix44")
    if matrix_node is None or not matrix_node.text:
        raise ValueError("missing scan alignment matrix")
    matrix = [float(x) for x in matrix_node.text.split()]
    if len(matrix) != 16 or not all(map(math.isfinite, matrix)):
        raise ValueError("alignment requires 16 finite values")
    rows = [matrix[i:i+4] for i in range(0, 16, 4)]
    if any(abs(a-b) > 1e-4 for a, b in zip(rows[3], [0, 0, 0, 1])):
        raise ValueError("alignment homogeneous row is invalid")
    rotation = [row[:3] for row in rows[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-3:
                raise ValueError("alignment rotation is not orthonormal")
    a, b, c = rotation
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det - 1) > 1e-3:
        raise ValueError("alignment rotation must be proper")
    return {"scan_path": scan_path.name, "matrix_row_major": rows}


def validate_metadata(destination: Path, files: dict[str, int]) -> dict:
    cameras = validate_cameras(destination / (CALIB_PREFIX + "cameras.txt"))
    jpeg_paths = {name: destination / name for name in files if name.startswith(JPEG_PREFIX)}
    images = validate_images(destination / (CALIB_PREFIX + "images.txt"), cameras, jpeg_paths)
    scan = destination / (SCAN_PREFIX + "scan1.ply")
    return {"cameras": {str(k): {"width": v[0], "height": v[1], "params": v[2]} for k, v in cameras.items()},
            "images": images, "ply": validate_ply(scan),
            "alignment": validate_alignment(destination / (SCAN_PREFIX + "scan_alignment.mlp"), scan)}


def run(source: Path = SOURCE, destination: Path = DEST, tmp: Path = TMP) -> dict:
    reviewed, files, dirs = preflight(source, destination, tmp)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    deadline = time.monotonic() + TIME_LIMIT
    for record in reviewed["archives"]:
        _extract_one(source / record["filename"], destination, deadline, files, dirs, tmp)
    digests = verify_extracted(destination, files, dirs, source, reviewed)
    verify_source_unchanged(source, reviewed)
    metadata = validate_metadata(destination, files)
    for image in metadata["images"]:
        image["sha256"] = digests[image["path"]]
    report = {"status": "validated", "source_inventory_sha256": INVENTORY_SHA256,
              "expanded_bytes": sum(files.values()), "file_sha256": digests, **metadata}
    with (destination / "prepare-metadata.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="verify pins and listings without extracting")
    args = parser.parse_args()
    if args.preflight_only:
        reviewed, files, dirs = preflight()
        print(json.dumps({"status": "preflight passed", "files": len(files), "directories": len(dirs),
                          "expanded_bytes": reviewed["total_expanded_bytes"]}, indent=2))
    else:
        print(json.dumps(run(), indent=2))
