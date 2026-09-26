"""Known-shift sanity check for the separate project-authored Census matcher."""

import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / ".local-tools/oracles/census/census"
SCRATCH = ROOT / ".local-tools/tmp"


def write_pgm(path: Path, rows: list[list[int]]) -> None:
    height, width = len(rows), len(rows[0])
    with path.open("wb") as output:
        output.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        output.write(bytes(pixel for row in rows for pixel in row))


def read_pfm(path: Path) -> list[list[float]]:
    with path.open("rb") as source:
        assert source.readline() == b"Pf\n"
        width, height = map(int, source.readline().split())
        assert source.readline() == b"-1.0\n"
        raw = source.read()
    assert len(raw) == width * height * 4
    pixels = struct.unpack(f"<{width * height}f", raw)
    return [list(pixels[y * width:(y + 1) * width]) for y in reversed(range(height))]


class CensusKnownShiftTest(unittest.TestCase):
    def test_shift_and_output_collision(self) -> None:
        self.assertTrue(ENGINE.is_file(), "Run python3 scripts/oracles/census/build.py first")
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.assertGreater(shutil.disk_usage(ROOT).free, 10 * 1024**3 + 64 * 1024**2)
        with tempfile.TemporaryDirectory(prefix="census-shift-", dir=SCRATCH) as temporary:
            root = Path(temporary)
            rng = random.Random(1469)
            left = [[rng.randrange(256) for _ in range(128)] for _ in range(48)]
            shift = 7
            right = [[left[y][min(127, x + shift)] for x in range(128)] for y in range(48)]
            write_pgm(root / "left.pgm", left)
            write_pgm(root / "right.pgm", right)
            output = root / "disparity.pfm"
            command = [str(ENGINE), "--left", str(root / "left.pgm"), "--right", str(root / "right.pgm"),
                       "--ndisp", "16", "--output", str(output)]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            disparity = read_pfm(output)
            tested = [disparity[y][x] for y in range(10, 38) for x in range(35, 112)]
            correct = sum(value == shift for value in tested)
            self.assertGreaterEqual(correct / len(tested), 0.95)
            self.assertTrue(math.isinf(disparity[0][0]))
            original = output.read_bytes()
            duplicate = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("already exists", duplicate.stderr)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
