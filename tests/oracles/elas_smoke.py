#!/usr/bin/env python3
"""Independent hand-shift check for the separately built libELAS oracle."""

import json
import hashlib
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile


def pgm(path: Path, width: int, height: int, pixels: bytes) -> None:
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels)


def pfm(path: Path, width: int, height: int, values: list[float]) -> None:
    with path.open("wb") as stream:
        stream.write(f"Pf\n{width} {height}\n-1.0\n".encode())
        for y in range(height - 1, -1, -1):
            stream.write(struct.pack(f"<{width}f", *values[y * width : (y + 1) * width]))


def read_pfm(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as stream:
        assert stream.readline() == b"Pf\n"
        width, height = map(int, stream.readline().split())
        assert stream.readline() == b"-1.0\n"
        values = struct.unpack(f"<{width * height}f", stream.read())
    rows = [values[y * width : (y + 1) * width] for y in range(height)]
    return width, height, [v for row in reversed(rows) for v in row]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: elas_smoke.py ELAS_ORACLE EVALUATOR TMP_PARENT")
    oracle, evaluator, temp_parent = map(Path, sys.argv[1:])
    rng = random.Random(307)
    width, height, shift = 192, 112, 8
    left = bytes(rng.randrange(256) for _ in range(width * height))
    right = bytes(left[y * width + min(x + shift, width - 1)] for y in range(height) for x in range(width))
    truth = [float(shift) if 24 <= x < width - 16 and 12 <= y < height - 12 else math.inf
             for y in range(height) for x in range(width)]
    with tempfile.TemporaryDirectory(prefix="elas-smoke-", dir=temp_parent) as name:
        directory = Path(name)
        pgm(directory / "left.pgm", width, height, left)
        pgm(directory / "right.pgm", width, height, right)
        pfm(directory / "truth.pfm", width, height, truth)
        subprocess.run([str(oracle), "--left", str(directory / "left.pgm"),
                        "--right", str(directory / "right.pgm"), "--ndisp", "32",
                        "--output", str(directory / "disparity.pfm")], check=True, timeout=60)
        before = hashlib.sha256((directory / "disparity.pfm").read_bytes()).hexdigest()
        overwrite = subprocess.run([str(oracle), "--left", str(directory / "left.pgm"),
                                    "--right", str(directory / "right.pgm"), "--ndisp", "32",
                                    "--output", str(directory / "disparity.pfm")],
                                   capture_output=True, text=True, timeout=60)
        assert overwrite.returncode != 0 and "exists" in overwrite.stderr
        assert hashlib.sha256((directory / "disparity.pfm").read_bytes()).hexdigest() == before
        link = directory / "linked-output.pfm"
        link.symlink_to(directory / "truth.pfm")
        protected = hashlib.sha256((directory / "truth.pfm").read_bytes()).hexdigest()
        linked = subprocess.run([str(oracle), "--left", str(directory / "left.pgm"),
                                 "--right", str(directory / "right.pgm"), "--ndisp", "32",
                                 "--output", str(link)], capture_output=True, text=True, timeout=60)
        assert linked.returncode != 0 and "exists" in linked.stderr
        assert hashlib.sha256((directory / "truth.pfm").read_bytes()).hexdigest() == protected
        output_width, output_height, predicted = read_pfm(directory / "disparity.pfm")
        assert (output_width, output_height) == (width, height)
        central = [predicted[y * width + x] for y in range(24, height - 24)
                   for x in range(40, width - 32) if math.isfinite(predicted[y * width + x])]
        assert len(central) > 3000, f"Only {len(central)} valid central pixels"
        central.sort()
        assert abs(central[len(central) // 2] - shift) <= 1, "Wrong disparity sign, scale, or shift"
        score_dir = directory / "score"
        subprocess.run([str(evaluator), "--prediction", str(directory / "disparity.pfm"),
                        "--gt", str(directory / "truth.pfm"), "--output-dir", str(score_dir)],
                       check=True, timeout=30)
        metrics = json.loads((score_dir / "metrics.json").read_text())
        assert metrics["coverage"] > 0.5, metrics
        assert metrics["bad2_all_valid"] < 0.35, metrics
        print(json.dumps({"central_median": central[len(central) // 2],
                          "coverage": metrics["coverage"],
                          "bad2_all_valid": metrics["bad2_all_valid"]}))


if __name__ == "__main__":
    main()
