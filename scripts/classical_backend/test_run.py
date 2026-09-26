import sys
import argparse
import json
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
import zipfile

from scripts.classical_backend import run
from scripts.classical_backend import fetch_openmvs


class SelectionAndFailureTests(unittest.TestCase):
    def test_selection_rejects_duplicate_or_escaping_names(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (root / name).write_bytes(b"photo")
            names = root / "list.txt"
            names.write_text("a.jpg\na.jpg\nb.jpg\n")
            with self.assertRaisesRegex(ValueError, "unique basenames"):
                run.photos(root, names, 3)
            names.write_text("a.jpg\n../escape.jpg\nc.jpg\n")
            with self.assertRaisesRegex(ValueError, "unique basenames"):
                run.photos(root, names, 3)

    def test_stage_timeout_is_recorded_and_process_stopped(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            with self.assertRaises(run.StageError) as raised:
                run.stage(root, "timeout", [sys.executable, "-c", "import time; time.sleep(30)"],
                          time.monotonic() + 0.1, 1 << 20, 1 << 20, 1 << 30)
            result = raised.exception.result
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"], "deadline exceeded")
            self.assertTrue((root / "timeout.log").is_file())

    def test_artifact_header_rejects_empty_geometry(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            path = Path(tmp) / "empty.ply"
            path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n"
                             b"element face 0\nend_header\n" + b" " * 40)
            with self.assertRaises(ValueError):
                run.checked_ply(path, "face")

    def test_pinned_fetch_rejects_same_size_binary_tamper(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            destination = root / "tools"
            destination.mkdir()
            archive = destination / "OpenMVS_macOS_arm64-v2.4.0.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("InterfaceCOLMAP", b"real binary")
            with (mock.patch.object(fetch_openmvs, "DEST", destination),
                  mock.patch.object(fetch_openmvs, "TMP", root),
                  mock.patch.object(fetch_openmvs, "SIZE", archive.stat().st_size),
                  mock.patch.object(fetch_openmvs, "SHA256", fetch_openmvs.sha256(archive)),
                  mock.patch.object(fetch_openmvs, "SELECTED", ("InterfaceCOLMAP",))):
                fetch_openmvs.fetch()
                target = destination / "bin" / "InterfaceCOLMAP"
                self.assertEqual(target.read_bytes(), b"real binary")
                target.write_bytes(b"fake binary")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    fetch_openmvs.fetch()

    def test_pinned_fetch_rejects_oversized_member(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            destination = root / "tools"
            destination.mkdir()
            archive = destination / "OpenMVS_macOS_arm64-v2.4.0.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("InterfaceCOLMAP", b"x" * 32_000_001)
            with (mock.patch.object(fetch_openmvs, "DEST", destination),
                  mock.patch.object(fetch_openmvs, "TMP", root),
                  mock.patch.object(fetch_openmvs, "SIZE", archive.stat().st_size),
                  mock.patch.object(fetch_openmvs, "SHA256", fetch_openmvs.sha256(archive)),
                  mock.patch.object(fetch_openmvs, "SELECTED", ("InterfaceCOLMAP",))):
                with self.assertRaisesRegex(ValueError, "unexpected archive member"):
                    fetch_openmvs.fetch()

    def test_shared_sfm_photo_hash_comparison_reaches_stage(self):
        """Regression: a photo Path must not shadow the producer provenance record."""
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            images = root / "photos"
            images.mkdir()
            photo_hashes = {}
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                path = images / name
                path.write_bytes(name.encode())
                photo_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = root / "models" / "0"
            model.mkdir(parents=True)
            model_hashes = {}
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                path = model / name
                path.write_bytes(name.encode())
                model_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            mask_hashes = {name: "mask-" + name for name in photo_hashes}
            manifest.write_text(json.dumps({"images": [{"name": name, "sha256": photo_hashes[name],
                                                         "mask_sha256": mask_hashes[name]}
                                                        for name in photo_hashes]}))
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({"schema": "object_motion_run_v1", "arm": "raw",
                                              "input_manifest_sha256": run.digest(manifest),
                                              "image_hashes": photo_hashes, "mask_hashes": mask_hashes}))
            summary = root / "summary.json"
            summary.write_text(json.dumps({"arm": "raw", "model_count": 1,
                                           "model_dir": str(model.resolve()),
                                           "model_files_sha256": model_hashes,
                                           "producer_provenance_sha256": run.digest(provenance)}))
            args = argparse.Namespace(images=images, output=root / "out", image_list=None,
                                      sparse_model=model, sparse_provenance=provenance,
                                      sparse_manifest=manifest, sparse_result=summary,
                                      binary_dir=root, python=Path(sys.executable), max_views=3,
                                      max_image_size=1200, max_threads=2, camera_model="SIMPLE_RADIAL",
                                      min_registered_fraction=0.7, max_gib=0.1,
                                      max_rss_gib=1, max_log_mib=1, timeout_minutes=1)
            failure = {"name": "reuse_sfm", "status": "failed", "exit_code": 99,
                       "failure": "intentional unit stop"}
            with (mock.patch.object(run, "check_toolchain", return_value={"openmvs_binaries": {}}),
                  mock.patch.object(run, "stage", side_effect=run.StageError(failure))):
                result = run.run(args)
            self.assertEqual(result["sfm_source"]["kind"], "external_raw_image_only_verified_model")
            self.assertEqual(len(result["inputs"]), 3)
            self.assertEqual(result["stages"][-1]["name"], "reuse_sfm")
            self.assertEqual(result["failure"], "reuse_sfm failed: intentional unit stop")


if __name__ == "__main__":
    unittest.main()
