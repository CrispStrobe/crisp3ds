#!/usr/bin/env python3
"""Bounded HTTP-range extraction of real bunny photos and independent mesh."""
import binascii
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / ".local-tools/test-data/3dlf-scan-bunny"
TMP = REPO / ".local-tools/tmp"
SOURCE = "https://data.mendeley.com/public-files/datasets/ngvgpsvd8b/files/d3b7c0c5-d043-45f0-80db-d4d7ccd884c6/file_downloaded"
SOURCE_BYTES = 9_829_049_614
SOURCE_SHA256 = "95ce4c76397c47bd8091fb85a529e38896284d163de5536fd27bdc2495d02f87"
MAX_TOTAL = 2 * 1024**3
MIN_FREE = 10 * 1024**3
MAX_MEMBER = 32 * 1024**2
MAX_SECONDS = 900
PREFIX = "3DLF-Scan_dataset/"


def http_range(start, length):
    if start < 0 or length <= 0 or length > MAX_MEMBER or start + length > SOURCE_BYTES:
        raise ValueError("source range exceeds bounds")
    TMP.mkdir(parents=True, exist_ok=True)
    descriptor, header_name = tempfile.mkstemp(prefix="3dlf-range-", suffix=".headers", dir=TMP)
    header = Path(header_name)
    import os
    os.close(descriptor)
    try:
        cmd = ["curl", "-LfsS", "--max-time", "120", "--max-filesize", str(length + 8192),
               "-r", f"{start}-{start + length - 1}", "-D", str(header), SOURCE]
        result = subprocess.run(cmd, capture_output=True, timeout=130)
        if result.returncode:
            raise RuntimeError(result.stderr[-300:].decode(errors="replace"))
        expected = f"content-range: bytes {start}-{start + length - 1}/{SOURCE_BYTES}"
        if len(result.stdout) != length or expected not in header.read_text().lower():
            raise ValueError("server did not honor exact byte range")
        return result.stdout
    finally:
        header.unlink(missing_ok=True)


def index_zip():
    tail = http_range(SOURCE_BYTES - 65536, 65536)
    marker = tail.rfind(b"PK\x06\x07")
    if marker < 0:
        raise ValueError("ZIP64 locator missing")
    disk, record_offset, disks = struct.unpack_from("<IQI", tail, marker + 4)
    if disk or disks != 1:
        raise ValueError("multi-disk ZIP unsupported")
    record = http_range(record_offset, 56)
    if record[:4] != b"PK\x06\x06":
        raise ValueError("ZIP64 record missing")
    _, _, _, _, _, on_disk, count, size, offset = struct.unpack_from("<QHHIIQQQQ", record, 4)
    if on_disk != count or count > 30000 or size > 8 * 1024**2:
        raise ValueError("ZIP index exceeds limits")
    data = http_range(offset, size)
    entries = {}
    cursor = 0
    for _ in range(count):
        if data[cursor:cursor + 4] != b"PK\x01\x02":
            raise ValueError("malformed ZIP index")
        f = struct.unpack_from("<HHHHHHIIIHHHHHII", data, cursor + 4)
        flags, method, crc, compressed, expanded = f[2], f[3], f[6], f[7], f[8]
        name_len, extra_len, comment_len, local = f[9], f[10], f[11], f[15]
        name = data[cursor + 46:cursor + 46 + name_len].decode("utf8")
        extra = data[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
        cursor += 46 + name_len + extra_len + comment_len
        if name in entries or not name.startswith(PREFIX) or flags & 1 or method not in (0, 8):
            raise ValueError(f"unsafe or unsupported ZIP member: {name}")
        if 0xFFFFFFFF in (expanded, compressed, local):
            pos = 0
            while pos + 4 <= len(extra):
                tag, part_len = struct.unpack_from("<HH", extra, pos)
                part = extra[pos + 4:pos + 4 + part_len]
                pos += 4 + part_len
                if tag == 1:
                    read = 0
                    if expanded == 0xFFFFFFFF:
                        expanded = struct.unpack_from("<Q", part, read)[0]; read += 8
                    if compressed == 0xFFFFFFFF:
                        compressed = struct.unpack_from("<Q", part, read)[0]; read += 8
                    if local == 0xFFFFFFFF:
                        local = struct.unpack_from("<Q", part, read)[0]
                    break
            else:
                raise ValueError(f"ZIP64 member metadata missing: {name}")
        entries[name] = dict(local=local, compressed=compressed, expanded=expanded,
                             crc=crc, method=method, flags=flags)
    if cursor != len(data):
        raise ValueError("ZIP index length mismatch")
    return entries


def selection(entries):
    photos = sorted(n for n in entries if n.startswith(PREFIX + "pro/bunny/rgb/") and n.endswith(".png"))
    if not 50 <= len(photos) <= 100:
        raise ValueError(f"unexpected photo count: {len(photos)}")
    other = [PREFIX + "calib_pro/intrinsics/rgb_optic.json",
             PREFIX + "pro/bunny/pcd/poses_metric.json",
             PREFIX + "revopoint/bunny/fuse_mesh_rgb.ply"]
    names = other + photos
    if any(n not in entries or entries[n]["compressed"] > MAX_MEMBER or
           entries[n]["expanded"] > MAX_MEMBER for n in names):
        raise ValueError("required member absent or oversized")
    if (sum(entries[n]["compressed"] + 8192 for n in names) > MAX_TOTAL or
            sum(entries[n]["expanded"] for n in names) > MAX_TOTAL):
        raise ValueError("selected transfer exceeds 2 GiB")
    return names


def fetch_member(name, entry, root):
    relative = name.removeprefix(PREFIX)
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in ("", ".", "..") for p in parts.parts):
        raise ValueError("unsafe output path")
    target = root.joinpath(*parts.parts)
    if target.exists():
        raise FileExistsError(target)
    # Overfetch up to 4 KiB to cover the local header's variable extra field.
    packet = http_range(entry["local"], 30 + len(name.encode()) + 4096 + entry["compressed"])
    if packet[:4] != b"PK\x03\x04":
        raise ValueError(f"local ZIP header missing: {name}")
    _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<HHHHHIIIHH", packet, 4)
    if flags != entry["flags"] or method != entry["method"] or extra_len > 4096:
        raise ValueError(f"local ZIP metadata differs: {name}")
    if packet[30:30 + name_len].decode("utf8") != name:
        raise ValueError(f"local ZIP name differs: {name}")
    begin = 30 + name_len + extra_len
    compressed = packet[begin:begin + entry["compressed"]]
    if len(compressed) != entry["compressed"]:
        raise ValueError("short ZIP member")
    if method == 0:
        plain = compressed
    else:
        decoder = zlib.decompressobj(-15)
        plain = decoder.decompress(compressed, entry["expanded"] + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError(f"ZIP member exceeds declared size or has trailing data: {name}")
    if len(plain) != entry["expanded"] or binascii.crc32(plain) != entry["crc"]:
        raise ValueError(f"ZIP CRC or size mismatch: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(plain)
    return {"path": relative, "zip_member": name, "bytes": len(plain),
            "sha256": hashlib.sha256(plain).hexdigest(),
            "source_crc32": f"{entry['crc']:08x}", "source_compressed_bytes": entry["compressed"],
            "source_local_offset": entry["local"]}


def run(root=ROOT):
    if root.exists() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise FileExistsError(f"destination must be fresh and real: {root}")
    if shutil.disk_usage(root.parent).free < MIN_FREE + MAX_TOTAL:
        raise OSError("cannot preserve 10 GiB free space")
    entries = index_zip()
    names = selection(entries)
    root.mkdir(parents=True)
    records = []
    started = time.monotonic()
    written = 0
    for name in names:
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("overall fetch deadline exceeded")
        if written + entries[name]["expanded"] > MAX_TOTAL:
            raise ValueError("expanded output exceeds 2 GiB")
        if shutil.disk_usage(root).free < MIN_FREE + entries[name]["expanded"]:
            raise OSError("free space below reserve")
        records.append(fetch_member(name, entries[name], root))
        written += records[-1]["bytes"]
    manifest = {"schema_version": 1, "dataset": "3DLF-Scan bunny",
                "dataset_url": "https://data.mendeley.com/datasets/ngvgpsvd8b/1",
                "doi": "10.17632/ngvgpsvd8b.1", "license": "CC BY 4.0",
                "commercial_rights_status": "third-party Stanford model/physical-print derivative rights require review before shipping or commercial redistribution",
                "third_party_rights_source": "https://graphics.stanford.edu/data/3Dscanrep/",
                "source_zip_url": SOURCE, "source_zip_bytes": SOURCE_BYTES,
                "source_zip_sha256_published": SOURCE_SHA256,
                "source_zip_downloaded": False,
                "frame_source": "PhotonicSense apiCAM PRO real turntable photographs",
                "reference_source": "Revopoint Miraco independently acquired fused mesh",
                "reference_training_use": "never reconstruction input; may be used for explicitly labeled post hoc reference-fit diagnostic",
                "alignment": "PRO camera poses and Revopoint mesh have no supplied cross-sensor registration",
                "files": records}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    m = run()
    print(json.dumps({"root": str(ROOT), "frames": len(m["files"]) - 3,
                      "bytes": sum(f["bytes"] for f in m["files"])}, indent=2))
