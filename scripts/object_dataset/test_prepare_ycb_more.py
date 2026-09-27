"""Network-free acquisition preflight and transaction tests."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.object_dataset import prepare_ycb_more as more


def archive_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class Response:
    def __init__(self, payload, etag):
        self.payload = io.BytesIO(payload)
        self.status = 200
        self.headers = {"Content-Length": str(len(payload)), "ETag": etag}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size):
        return self.payload.read(size)


class MoreYCBTests(unittest.TestCase):
    def setUp(self):
        more.base.TMP.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=more.base.TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        name = "test_obj"
        photo_stream = io.BytesIO()
        Image.new("RGB", (20, 10), (10, 20, 30)).save(photo_stream, format="JPEG")
        photos = {f"{name}/NP3_{angle}.jpg": photo_stream.getvalue()
                  for angle in range(0, 360, 6)}
        self.photo_size_patch = patch.object(more, "PHOTO_SIZE", (20, 10))
        self.photo_size_patch.start()
        self.addCleanup(self.photo_size_patch.stop)
        scan = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
                b"property float y\nproperty float z\nelement face 1\n"
                b"property list uchar int vertex_indices\nend_header\n"
                b"0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
        self.payloads = {"berkeley_rgbd": archive_bytes(photos),
                         "google": archive_bytes({f"{name}/google_16k/nontextured.ply": scan})}
        self.config = {role: (f"test/{role}.tgz", len(payload), f'"{role}-etag"')
                       for role, payload in self.payloads.items()}
        self.config["scan_resolution"] = "16k"
        self.object_patch = patch.dict(more.OBJECTS, {name: self.config})
        self.object_patch.start()
        self.addCleanup(self.object_patch.stop)
        self.name = name

    def _urlopen(self, request, timeout):
        url = request.full_url if hasattr(request, "full_url") else request
        role = "berkeley_rgbd" if "berkeley_rgbd" in url else "google"
        return Response(self.payloads[role], self.config[role][2])

    def test_head_only_and_transactional_fetch_observed_hashes(self):
        with patch.object(more, "urlopen", side_effect=self._urlopen):
            head = more.preflight(self.name)
            self.assertTrue(head["head_only"])
            self.assertFalse(head["sources"]["google"]["etag_is_cryptographic_hash"])
            self.assertEqual(list(self.root.iterdir()), [])
            result = more.fetch(self.name, self.root)
        dest = self.root / self.name
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["source_authentication"],
                         "observed_sha256_only_unverified_upstream_identity")
        self.assertEqual(len(result["photos"]), 60)
        self.assertEqual([x["turntable_angle_degrees"] for x in result["photos"]],
                         list(range(0, 360, 6)))
        self.assertTrue((dest / "reference/google_geometry_f64.ply").is_file())
        self.assertEqual(result["archives"]["google"]["observed_sha256"],
                         hashlib.sha256(self.payloads["google"]).hexdigest())
        self.assertEqual(sorted(x.name for x in self.root.iterdir()), [self.name])

    def test_declared_metadata_drift_and_expected_hash_failure_leave_no_destination(self):
        bad_config = dict(self.config)
        bad_config["google"] = (self.config["google"][0],
                                self.config["google"][1] + 1, self.config["google"][2])
        with patch.dict(more.OBJECTS, {self.name: bad_config}), patch.object(
                more, "urlopen", side_effect=self._urlopen):
            with self.assertRaisesRegex(ValueError, "HEAD metadata drift"):
                more.preflight(self.name)
        with patch.object(more, "urlopen", side_effect=self._urlopen):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                more.fetch(self.name, self.root, expected_google_sha256="0" * 64)
        self.assertFalse((self.root / self.name).exists())
        reports = list(self.root.glob("*.failure.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())["status"], "failed_no_final_object")
        self.assertEqual([p for p in self.root.glob(".ycb-*") if p.is_dir()], [])

    def test_unsafe_member_fails_without_partial_final_directory(self):
        self.payloads["google"] = archive_bytes({"../escape": b"bad"})
        self.config["google"] = (self.config["google"][0], len(self.payloads["google"]),
                                 self.config["google"][2])
        with patch.object(more, "urlopen", side_effect=self._urlopen):
            with self.assertRaisesRegex(ValueError, "unsafe"):
                more.fetch(self.name, self.root)
        self.assertFalse((self.root / self.name).exists())
        self.assertEqual(len(list(self.root.glob("*.failure.json"))), 1)
        self.assertEqual([p for p in self.root.glob(".ycb-*") if p.is_dir()], [])

    def test_get_open_phase_is_distinct_from_storage_open(self):
        info = more.source_info(self.name, "google")
        with patch.object(more, "urlopen", side_effect=OSError(12, "Cannot allocate memory")):
            with self.assertRaises(more.AcquisitionError) as caught:
                more._download(self.root, info, None, float("inf"))
        self.assertEqual(caught.exception.phase, "open_https_get")
        self.assertEqual(caught.exception.__cause__.errno, 12)


if __name__ == "__main__":
    unittest.main()
