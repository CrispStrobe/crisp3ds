"""Pure setup contracts; never transfer, install, or infer in CI."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.sam_m1_setup import prepare as setup


class SetupTests(unittest.TestCase):
    def test_exact_fixed_commands_are_binary_only_and_not_inference(self):
        ai = Path("/external/ai")
        source = Path("/external/source")
        data = Path("/external/data")
        stages = dict(setup.commands(ai, source, data))
        self.assertEqual(len(stages), 16)
        self.assertIn("--only-binary=:all:", stages["wheel-install"])
        self.assertIn("--no-deps", stages["wheel-install"])
        self.assertIn("--require-hashes", stages["wheel-install"])
        self.assertEqual(stages["wheel-install"][-1], str(setup.LOCK))
        self.assertEqual({name for name in stages if name.startswith("photo-")},
                         {"photo-NP3_000", "photo-NP3_066", "photo-NP3_108"})
        for name, command in stages.items():
            self.assertNotIn("cuda", " ".join(command).lower())
            self.assertNotIn("predict(", " ".join(command).lower())
            self.assertNotIn("build_sam2(", " ".join(command).lower())
            if name.startswith("vendor-"):
                self.assertTrue(command[-1].startswith(str(ai / "vendor-pure")))

    def test_preflight_rejects_existing_path_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            existing = base / "already"
            existing.mkdir()
            with patch.object(setup.sys, "platform", "darwin"), \
                 patch.object(setup.sys, "version_info", (3, 11, 12, "final", 0)), \
                 patch.object(setup.platform, "machine", return_value="arm64"), \
                 patch.object(setup.Path, "is_mount", return_value=True):
                with self.assertRaisesRegex(ValueError, "external mount|fresh"):
                    setup.preflight(existing, base / "source", base / "data", base, setup.ROOT)
            self.assertEqual(list(existing.iterdir()), [])

    def test_vendor_rejects_native_and_tampered_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in setup.VENDOR_DIRS:
                (root / name).mkdir()
            for folder_name, package_name, version in (
                ("iopath-0.1.10.dist-info", "iopath", "0.1.10"),
                ("antlr4_python3_runtime-4.9.3.dist-info", "antlr4-python3-runtime", "4.9.3"),
            ):
                (root / folder_name / "METADATA").write_text(f"Name: {package_name}\nVersion: {version}\n")
            (root / "iopath" / "module.py").write_text("x=1\n")
            with self.assertRaisesRegex(ValueError, "inventory"):
                setup.vendor_inventory(root)
            (root / "iopath" / "native.so").write_bytes(b"elf")
            with self.assertRaisesRegex(ValueError, "native"):
                setup.vendor_inventory(root)

    def test_vendor_inventory_uses_reviewed_path_component_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in setup.VENDOR_DIRS:
                (root / name).mkdir()
            for folder_name, package_name, version in (
                ("iopath-0.1.10.dist-info", "iopath", "0.1.10"),
                ("antlr4_python3_runtime-4.9.3.dist-info", "antlr4-python3-runtime", "4.9.3"),
            ):
                (root / folder_name / "METADATA").write_text(f"Name: {package_name}\nVersion: {version}\n")
            for index in range(76):
                package = "iopath" if index < 38 else "antlr4"
                (root / package / f"module_{index:02d}.py").write_text(f"value={index}\n")
            files = sorted((p for p in root.rglob("*") if p.is_file()))
            digest = hashlib.sha256()
            for path in files:
                digest.update(path.relative_to(root).as_posix().encode() + bytes([0]) +
                              bytes.fromhex(setup.sha(path)))
            with patch.object(setup, "VENDOR_SHA", digest.hexdigest()):
                self.assertEqual(setup.vendor_inventory(root)["inventory_sha256"], digest.hexdigest())

    def test_wheel_lock_is_18_hash_pinned_entries(self):
        lines = [line for line in setup.LOCK.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 18)
        self.assertEqual(len({line.split("==")[0].lower() for line in lines}), 18)
        for line in lines:
            name, hash_part = line.split(" --hash=sha256:")
            self.assertIn("==", name)
            self.assertEqual(len(hash_part), 64)
            self.assertEqual(set(hash_part) - set("0123456789abcdef"), set())
        self.assertEqual(hashlib.sha256(setup.LOCK.read_bytes()).hexdigest(),
                         "3c07dc415436bf22c09d597ca39aead5709f935e04729a9ad6aa25deed9b9a3d")


if __name__ == "__main__":
    unittest.main()
