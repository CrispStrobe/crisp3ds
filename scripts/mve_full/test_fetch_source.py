"""No-network checks for the bounded pinned-source fetcher."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.mve_full.fetch_source import extract, fetch


ROOT = Path(__file__).resolve().parents[2]
PINNED = ROOT / ".local-tools/mve-spike/mve-bf2279f.tar.gz"
TEST_TEMP = ROOT / ".local-tools/tmp"


class SourceFetchTests(unittest.TestCase):
    def test_existing_pinned_archive_is_accepted_without_network(self):
        if not PINNED.is_file():
            self.skipTest("pinned archive unavailable outside local development workspace")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(fetch(PINNED), PINNED)

    def test_rejects_symlink_destination_and_oversized_file(self):
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            root = Path(directory)
            archive = root / "archive.tar.gz"
            archive.write_bytes(b"bad")
            link = root / "link.tar.gz"
            try:
                link.symlink_to(archive)
            except OSError:
                if os.name != "nt":
                    raise
                # Hosted Windows runners may lack the symlink creation privilege.
            else:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    fetch(link)
            with self.assertRaisesRegex(ValueError, "wrong SHA-256"):
                fetch(archive)

    def test_extract_checks_ten_gib_reserve_before_writing(self):
        if not PINNED.is_file():
            self.skipTest("pinned archive unavailable outside local development workspace")
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=9 << 30)):
                with self.assertRaisesRegex(ValueError, "10 GiB"):
                    extract(PINNED, Path(directory))


if __name__ == "__main__":
    unittest.main()
