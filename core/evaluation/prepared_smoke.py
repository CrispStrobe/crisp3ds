#!/usr/bin/env python3
"""Test-only prepared-input CLI contract with tiny generated stereo images."""

import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import zlib


def png(path: Path, width: int, height: int, pixels: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = b"".join(b"\0" + pixels[y * width:(y + 1) * width] for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def pfm(path: Path, width: int, height: int, disparity: float) -> None:
    with path.open("wb") as stream:
        stream.write(f"Pf\n{width} {height}\n-0.003922\n".encode())
        row = struct.pack(f"<{width}f", *([disparity] * width))
        for _ in range(height):
            stream.write(row)


def run(evaluator: Path, *args: object, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([str(evaluator), *map(str, args)], capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepared_smoke.py EVALUATOR TEMP_PARENT")
    evaluator, parent = map(Path, sys.argv[1:])
    parent.mkdir(parents=True, exist_ok=True)
    width, height, shift = 128, 64, 8
    rng = random.Random(2309)
    left_pixels = bytes(rng.randrange(256) for _ in range(width * height))
    right_pixels = bytes(left_pixels[y * width + min(x + shift, width - 1)]
                         for y in range(height) for x in range(width))
    with tempfile.TemporaryDirectory(prefix="prepared-smoke-", dir=parent) as name:
        folder = Path(name)
        left, right, truth, mask = (folder / item for item in ("left.png", "right.png", "truth.pfm", "zero.png"))
        png(left, width, height, left_pixels)
        png(right, width, height, right_pixels)
        png(mask, width, height, bytes(width * height))
        pfm(truth, width, height, float(shift))
        common = ("--left", left, "--right", right, "--gt", truth, "--ndisp", "32", "--downsample", "2")
        prepared = folder / "prepared"
        run(evaluator, "--prepare-only", "yes", *common, "--output-dir", prepared)
        manifest = json.loads((prepared / "inputs.json").read_text())
        assert manifest["ndisp"] == 16 and manifest["calibration"] is None
        assert manifest["source_calib"] is None and manifest["source_mask"] is None
        assert manifest["width"] == 64 and manifest["height"] == 32
        assert manifest["left"] == "left.pgm" and manifest["truth"] == "truth.pfm"
        original_hash = hash_file(prepared / "left.pgm")
        run(evaluator, "--prepare-only", "yes", *common, "--output-dir", prepared, success=False)
        assert hash_file(prepared / "left.pgm") == original_hash

        direct = folder / "direct"
        via_pgm = folder / "via-pgm"
        run(evaluator, *common, "--output-dir", direct)
        run(evaluator, "--left", prepared / "left.pgm", "--right", prepared / "right.pgm",
            "--gt", prepared / "truth.pfm", "--ndisp", "16", "--output-dir", via_pgm)
        assert hash_file(direct / "disparity.pfm") == hash_file(via_pgm / "disparity.pfm")
        direct_metrics = json.loads((direct / "metrics.json").read_text())
        via_metrics = json.loads((via_pgm / "metrics.json").read_text())
        for key in ("gt_valid", "matched", "coverage", "bad2_all_valid", "mae_matched_px"):
            assert direct_metrics[key] == via_metrics[key], key
        assert not (direct / "depth.pfm").exists() and direct_metrics["depth_formula"] is None

        calib = folder / "calib.txt"
        calib.write_text("cam0=[100 0 64; 0 100 32; 0 0 1]\n"
                         "baseline=10\ndoffs=1\nndisp=32\nwidth=128\nheight=64\n")
        masked = folder / "masked"
        run(evaluator, "--prepare-only", "yes", "--left", left, "--right", right,
            "--gt", truth, "--calib", calib, "--downsample", "2",
            "--mask", mask, "--output-dir", masked)
        masked_manifest = json.loads((masked / "inputs.json").read_text())
        assert masked_manifest["mask"] == "mask.pgm" and masked_manifest["source_mask"] == str(mask)
        assert masked_manifest["source_calib"] == str(calib)
        assert masked_manifest["calibration"] == {"fx": 50, "baseline": 10, "doffs": 0.5, "ndisp": 16}
        run(evaluator, "--prediction", via_pgm / "disparity.pfm", "--gt", masked / "truth.pfm",
            "--mask", masked / "mask.pgm", "--output-dir", folder / "empty-roi", success=False)

        broken = folder / "broken.pgm"
        broken.write_bytes(b"P5\n64 32\n255\n")
        run(evaluator, "--left", broken, "--right", prepared / "right.pgm",
            "--gt", prepared / "truth.pfm", "--ndisp", "16",
            "--output-dir", folder / "broken-result", success=False)
        print("prepared input, mask, pixel-only, parity, and rejection checks passed")


if __name__ == "__main__":
    main()
