"""Offline safety tests for the private AliceVision Kaggle binary smoke."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.remote_quality.alicevision_binary_smoke import alicevision_binary_smoke as smoke


class ArchiveSafetyTests(unittest.TestCase):
    def _archive(self, members: list[tarfile.TarInfo]) -> Path:
        self.addCleanup(self._cleanup_temp)
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "example.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for member in members:
                archive.addfile(member, io.BytesIO(b"a" * member.size) if member.isfile() else None)
        return path

    def _cleanup_temp(self) -> None:
        self.temp.cleanup()

    def test_safe_regular_file(self) -> None:
        member = tarfile.TarInfo("release/bin/example")
        member.size = 3
        self.assertEqual(smoke.inspect_tar(self._archive([member])),
                         {"members": 1, "expanded_regular_bytes": 3})

    def test_rejects_parent_traversal(self) -> None:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        with self.assertRaisesRegex(ValueError, "unsafe member"):
            smoke.inspect_tar(self._archive([member]))

    def test_rejects_external_symlink(self) -> None:
        member = tarfile.TarInfo("release/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        with self.assertRaisesRegex(ValueError, "unsafe link"):
            smoke.inspect_tar(self._archive([member]))

    def test_rejects_expanded_cap(self) -> None:
        member = tarfile.TarInfo("release/large")
        member.size = 1
        original = smoke.MAX_EXPANDED
        try:
            smoke.MAX_EXPANDED = 0
            with self.assertRaisesRegex(ValueError, "expanded-byte cap"):
                smoke.inspect_tar(self._archive([member]))
        finally:
            smoke.MAX_EXPANDED = original

    def test_runtime_root_requires_bundled_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory)
            lib_dir = extracted / "aliceVision" / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libaliceVision_cmdline.so.3").touch()
            with self.assertRaisesRegex(RuntimeError, "OCIO configuration missing"):
                smoke.bundled_runtime_paths(extracted)
            config = extracted / "aliceVision" / "share" / "aliceVision" / "config.ocio"
            config.parent.mkdir(parents=True)
            config.touch()
            self.assertEqual(smoke.bundled_runtime_paths(extracted),
                             (lib_dir, extracted / "aliceVision"))


class CommandSafetyTests(unittest.TestCase):
    def test_missing_optional_gpu_utility(self) -> None:
        result = smoke.short_command(["/nonexistent/nvidia-smi"], 1)
        self.assertTrue(result["unavailable"])

    def test_output_is_capped(self) -> None:
        result = smoke.short_command(["/bin/sh", "-c", "yes x"], 2)
        self.assertTrue(result["output_capped"])

    def test_timeout_is_capped(self) -> None:
        result = smoke.short_command(["/bin/sh", "-c", "sleep 5"], 1)
        self.assertTrue(result["timed_out"])

    def test_loader_environment_is_scoped_to_command(self) -> None:
        result = smoke.short_command(["/usr/bin/env"], 1,
                                     env={"LD_LIBRARY_PATH": "/synthetic/bundled/lib"})
        self.assertEqual(result["returncode"], 0)
        self.assertIn("LD_LIBRARY_PATH=/synthetic/bundled/lib", result["output_tail"])


class MetadataSafetyTests(unittest.TestCase):
    def test_private_cpu_only_no_sources(self) -> None:
        metadata = json.loads((Path(smoke.__file__).with_name("kernel-metadata.json")).read_text())
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_internet"], "true")
        for key in ("competition_sources", "dataset_sources", "kernel_sources", "model_sources"):
            self.assertEqual(metadata[key], [])
        self.assertEqual(smoke.SCRATCH.parent, Path("/tmp"))
        self.assertEqual(smoke.OUTPUT.parent, Path("/kaggle/working"))


if __name__ == "__main__":
    unittest.main()
