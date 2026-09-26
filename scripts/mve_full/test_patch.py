"""Exercise the exact SIFT-only patch on a fresh pinned source extraction."""

import subprocess
import tempfile
from pathlib import Path
import unittest

from scripts.mve_full.fetch_source import extract
from scripts.mve_full.verify_source import verify


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / ".local-tools/mve-spike/mve-bf2279f.tar.gz"
PATCH = ROOT / "scripts/mve_full/sift_only.patch"
TEST_TEMP = ROOT / ".local-tools/tmp"


class SelectedPatchTests(unittest.TestCase):
    def test_clean_extraction_applies_and_matches_selected_hashes(self):
        if not ARCHIVE.is_file():
            self.skipTest("pinned source archive unavailable")
        self.assertNotIn(b"\r", PATCH.read_bytes(), "patch checkout must retain LF line endings")
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            source = extract(ARCHIVE, Path(directory))
            relative = source.relative_to(ROOT).as_posix()
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--unidiff-zero",
                            f"--directory={relative}", str(PATCH)], cwd=ROOT, check=True)
            self.assertEqual(verify(ARCHIVE, source), 215)


if __name__ == "__main__":
    unittest.main()
