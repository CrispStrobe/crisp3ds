"""Guard against accepting empty artifacts or nonregistered views."""

import tempfile
from pathlib import Path
import struct
import sys
import time
import unittest

from scripts.mve_full.run import ply_counts, registered_views, selected_images, stage


class RunnerValidationTests(unittest.TestCase):
    @staticmethod
    def sample_ply(vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                   face=(0, 1, 2)) -> bytes:
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "element face 1\nproperty list uchar int vertex_indices\nend_header\n").encode()
        return header + b"".join(struct.pack("<fff", *vertex) for vertex in vertices) + b"\x03" + struct.pack("<iii", *face)

    def test_registered_focal_below_one_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            for index, focal in enumerate((0.8, 0.0, 1.2)):
                view = scene / "views" / f"view_{index:04d}.mve"
                view.mkdir(parents=True)
                (view / "meta.ini").write_text(f"[view]\nid = {index}\n[camera]\nfocal_length = {focal}\n")
            self.assertEqual(registered_views(scene), 2)

    def test_rejects_truncated_binary_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n" + b"\0" * 12)
            with self.assertRaisesRegex(ValueError, "truncated"):
                ply_counts(path)

    def test_validates_full_mesh_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            path.write_bytes(self.sample_ply())
            self.assertEqual(ply_counts(path), (3, 1))
            cases = [
                (self.sample_ply(vertices=((float("nan"), 0, 0), (1, 0, 0), (0, 1, 0))), "nonfinite"),
                (self.sample_ply(face=(0, 1, 3)), "indices"),
                (self.sample_ply(vertices=((0, 0, 0), (1, 0, 0), (2, 0, 0))), "degenerate"),
                (self.sample_ply() + b"extra", "trailing"),
            ]
            for payload, message in cases:
                with self.subTest(message=message):
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, message):
                        ply_counts(path)
            path.write_bytes(self.sample_ply(vertices=((0, 0, 0), (1, 0, 0), (2, 0, 0))))
            self.assertEqual(ply_counts(path, allow_degenerate=True), (3, 1))

    def test_image_list_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory)
            (images / "a.jpg").write_bytes(b"a")
            (images / "b.jpg").write_bytes(b"b")
            (images / "c.jpg").write_bytes(b"c")
            names = images / "names.txt"
            names.write_text("a.jpg\n../b.jpg\nc.jpg\n")
            with self.assertRaisesRegex(ValueError, "basenames"):
                selected_images(images, names, 3)

    def test_log_cap_kills_running_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            command = [sys.executable, "-c", "import sys,time; sys.stdout.write('x'*20000); sys.stdout.flush(); time.sleep(5)"]
            start = time.monotonic()
            result = stage(command, "oversized", run, 1 << 20, 4, start, 1024)
            self.assertEqual(result["status"], "log_cap")
            self.assertLess(time.monotonic() - start, 3)

    def test_nonzero_and_final_output_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            failure = stage([sys.executable, "-c", "raise SystemExit(7)"], "failure",
                            run, 1 << 20, 3, time.monotonic(), 1 << 20)
            self.assertEqual(failure["status"], "nonzero_exit")
            self.assertEqual(failure["returncode"], 7)
            command = [sys.executable, "-c", "open('large.bin','wb').write(b'x'*4096)"]
            result = stage(command, "size", run, 1024, 3, time.monotonic(), 1 << 20)
            self.assertEqual(result["status"], "output_cap")

    def test_total_timeout_kills_running_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            result = stage([sys.executable, "-c", "import time; time.sleep(5)"], "timeout",
                           Path(directory), 1 << 20, 0.1, start, 1 << 20)
            self.assertEqual(result["status"], "total_timeout")
            self.assertLess(time.monotonic() - start, 3)


if __name__ == "__main__":
    unittest.main()
