"""Frozen YCB split and immutable package checks; no network or large assets."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.object_dataset import evaluation_package as ep


def load_local_manifest(object_id):
    path, digest = ep.MANIFESTS[object_id]
    assert ep.sha256_file(path) == digest
    return json.loads(path.read_text()), digest


def synthetic_manifest(object_id):
    resolution = "16k" if object_id == "006_mustard_bottle" else "64k"
    vertices, faces = ((8194, 16384) if resolution == "16k" else (32770, 65536))
    photos = []
    for angle in ep.ANGLES:
        name = f"photos/NP3_{angle:03}.jpg"
        photos.append({"source_member": f"{object_id}/NP3_{angle}.jpg", "path": name,
                       "bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest(),
                       "role": "reconstruction-input-real-photograph",
                       "turntable_angle_degrees": angle})
    return {"schema": "ycb_more_prepared_v1", "object_id": object_id, "status": "complete",
            "source_authentication": "observed_sha256_only_unverified_upstream_identity",
            "selection": {"camera": "Berkeley NP3", "views": 60, "angles": list(ep.ANGLES),
                          "excluded_from_reconstruction": ["Google scanner reference"]},
            "archives": {name: {"observed_sha256": hashlib.sha256(name.encode()).hexdigest(),
                                "independently_pinned": False}
                         for name in ("berkeley_rgbd", "google")},
            "photos": photos,
            "reference_original": {"source_member": f"{object_id}/google_{resolution}/nontextured.ply",
                                   "path": "reference/google_original_ascii.ply", "bytes": 5,
                                   "sha256": hashlib.sha256(b"original").hexdigest(),
                                   "role": "Google scanner reference only"},
            "reference_binary_geometry": {"path": "reference/google_geometry_f64.ply", "bytes": 6,
                                          "sha256": hashlib.sha256(b"geometry").hexdigest(),
                                          "vertices": vertices, "faces": faces,
                                          "role": "converted Google scanner reference only"}}


class EvaluationPackageTests(unittest.TestCase):
    def setUp(self):
        self.object_id = "006_mustard_bottle"
        self.manifest = synthetic_manifest(self.object_id)
        self.manifest_digest = hashlib.sha256(b"synthetic-fixture").hexdigest()
        ep.ROOT.joinpath(".local-tools/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ep.ROOT / ".local-tools/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fixtures_freeze_disjoint_split_and_reference_role(self):
        for object_id in ep.MANIFESTS:
            manifest = synthetic_manifest(object_id)
            with self.subTest(object_id=object_id):
                plan = ep.build_protocol(manifest, object_id, self.manifest_digest)
                self.assertEqual(len(plan["training_inputs"]), 48)
                self.assertEqual(len(plan["heldout_photos"]), 12)
                self.assertEqual([r["angle_degrees"] for r in plan["heldout_photos"]],
                                 list(ep.HELDOUT))
                train = {r["path"] for r in plan["training_inputs"]}
                heldout = {r["path"] for r in plan["heldout_photos"]}
                references = {r["path"] for r in plan["evaluation_only"].values()}
                self.assertFalse(train & heldout)
                self.assertFalse((train | heldout) & references)
                self.assertTrue(all(path.startswith("photos/") for path in train | heldout))
                self.assertFalse(plan["runnable_image_only_training_package"])
                self.assertEqual(plan["mask_status"], "pending_photo_derived_review")

    def test_pinned_local_manifests_when_available(self):
        if not all(path.is_file() for path, _ in ep.MANIFESTS.values()):
            self.skipTest("ignored copied acquisition manifests not present in clean checkout")
        for object_id in ep.MANIFESTS:
            manifest, digest = load_local_manifest(object_id)
            with self.subTest(object_id=object_id):
                ep.build_protocol(manifest, object_id, digest)

    def test_duplicate_photo_content_is_rejected_even_across_splits(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["photos"][4]["sha256"] = duplicate["photos"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            ep.build_protocol(duplicate, self.object_id, self.manifest_digest)

    def test_missing_duplicate_and_leaked_inputs_are_rejected(self):
        mutations = (
            lambda m: m["photos"].pop(),
            lambda m: m["photos"][4].update(turntable_angle_degrees=0),
            lambda m: m["photos"][0].update(path="reference/google_original_ascii.ply"),
            lambda m: m["photos"][0].update(role="Google scanner reference only"),
            lambda m: m["reference_original"].update(role="reconstruction-input-real-photograph"),
        )
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                ep.build_protocol(manifest, self.object_id, self.manifest_digest)

    def _synthetic_files(self):
        manifest = copy.deepcopy(self.manifest)
        for record in manifest["photos"] + [manifest["reference_original"],
                                             manifest["reference_binary_geometry"]]:
            path = self.root / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = record["path"].encode("ascii")
            path.write_bytes(payload)
            record["bytes"] = len(payload)
            record["sha256"] = hashlib.sha256(payload).hexdigest()
        return manifest

    def test_file_rehash_missing_drift_and_parent_symlink(self):
        manifest = self._synthetic_files()
        plan = ep.build_protocol(manifest, self.object_id, self.manifest_digest)
        checked = ep.verify_files(self.root, plan)
        self.assertEqual(checked["verified_file_count"], 62)
        self.assertEqual(checked["status"], "photos_and_references_rehashed_masks_pending")
        self.assertFalse(checked["runnable_image_only_training_package"])
        target = self.root / manifest["photos"][0]["path"]
        target.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "differ"):
            ep.verify_files(self.root, ep.build_protocol(manifest, self.object_id, self.manifest_digest))
        target.unlink()
        with self.assertRaises(FileNotFoundError):
            ep.verify_files(self.root, ep.build_protocol(manifest, self.object_id, self.manifest_digest))
        target.write_bytes(b"photos/NP3_000.jpg")
        photo_dir = self.root / "photos"
        real_dir = self.root / "real_photos"
        photo_dir.rename(real_dir)
        try:
            photo_dir.symlink_to(real_dir, target_is_directory=True)
        except OSError:
            self.skipTest("host does not permit test symlink creation")
        with self.assertRaisesRegex(ValueError, "symlink"):
            ep.verify_files(self.root, ep.build_protocol(manifest, self.object_id, self.manifest_digest))

    def test_cli_refuses_modified_pinned_manifest_and_existing_output(self):
        altered = self.root / "altered.json"
        altered.write_text(json.dumps(self.manifest, separators=(",", ":")))
        output = self.root / "package.json"
        args = ["evaluation_package.py", "--object", self.object_id,
                "--manifest", str(altered), "--output", str(output)]
        with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "pinned"):
            ep.main()
        self.assertFalse(output.exists())
        output.write_text("already here")
        with patch("sys.argv", args), self.assertRaises(FileExistsError):
            ep.main()


if __name__ == "__main__":
    unittest.main()
