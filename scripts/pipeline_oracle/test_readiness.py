"""Focused tests for the read-only oracle preflight."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import readiness


SCRATCH = readiness.REPO / ".local-tools/tmp"


class ReadinessTests(unittest.TestCase):
    def test_mock_executable_and_missing_tool(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            executable = Path(directory) / "fake-colmap"
            executable.write_text("#!/bin/sh\necho 'COLMAP 3.99 test'\n")
            executable.chmod(0o755)
            found = readiness.probe("colmap", lambda _: str(executable))
            self.assertEqual(found["status"], "available")
            self.assertEqual(found["version"], "COLMAP 3.99 test")
            self.assertEqual(readiness.probe("missing", lambda _: None)["status"], "unavailable")

    def test_mock_executable_bad_version_and_timeout(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            executable = Path(directory) / "fake-openmvs"
            executable.write_text("#!/bin/sh\nexit 2\n")
            executable.chmod(0o755)
            self.assertEqual(readiness.probe("DensifyPointCloud", lambda _: str(executable))["status"],
                             "version_unverified")
            with mock.patch.object(readiness.subprocess, "run", side_effect=readiness.subprocess.TimeoutExpired("fake", 3)):
                self.assertEqual(readiness.probe("DensifyPointCloud", lambda _: str(executable))["probe_error"],
                                 "TimeoutExpired")

    def test_mock_executable_output_is_capped(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            executable = Path(directory) / "fake-noisy-tool"
            executable.write_text("#!/bin/sh\n/usr/bin/yes noisy-version\n")
            executable.chmod(0o755)
            result = readiness.probe("colmap", lambda _: str(executable))
            self.assertEqual(result["status"], "version_unverified")
            self.assertEqual(result["probe_error"], "output_limit")

    def test_dataset_integrity_and_budget(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "colmap").mkdir()
            contents = {
                "images/a.jpg": b"image",
                "colmap/cameras.txt": b"1 PINHOLE 8 6 4 4 4 3\n",
                "colmap/images.txt": b"1 1 0 0 0 0 0 0 1 a.jpg\n\n",
            }
            for name, data in contents.items():
                (root / name).write_bytes(data)
            def record(name):
                data = contents[name]
                return {"path": name, "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()}
            manifest = {"schema_version": 1, "source_files": [record("colmap/cameras.txt"),
                        record("colmap/images.txt")],
                        "selected_images": [{**record("images/a.jpg"), "id": 1, "name": "a.jpg",
                            "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0],
                            "camera_id": 1, "camera": {"model": "PINHOLE", "width": 8,
                            "height": 6, "params": [4.0, 4.0, 4.0, 3.0]}}],
                        "colmap": {"cameras": "colmap/cameras.txt", "images": "colmap/images.txt",
                                   "tracks_available": False}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(readiness.verify_dataset(root)["selected_image_count"], 1)
            with self.assertRaisesRegex(ValueError, "output budget"):
                readiness.build_report(root, 9)
            with self.assertRaisesRegex(ValueError, "output budget"):
                readiness.build_report(root, float("nan"))
            (root / "images/a.jpg").write_bytes(b"alter")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                readiness.verify_dataset(root)


if __name__ == "__main__":
    unittest.main()
