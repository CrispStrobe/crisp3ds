import copy
import json
from pathlib import Path
import unittest

from scripts.classical_backend import recovered_control as target


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "build-opencv/object-motion"
PRODUCER = PARENT / "initialization-recovery-005-delayed-refinement/ba_trial/report.json"
SOURCE = PARENT / "initialization-recovery-004-fixed-intrinsics/cached_seed_trial/report.json"
MODEL = PARENT / "initialization-recovery-005-delayed-refinement/ba_trial/refined_model"
MANIFEST = PARENT / "prepare-001/manifest.json"
MASKS = PARENT / "prepare-001/masks"
PHOTOS = ROOT / ".local-tools/test-data/ycb-cracker-box/photos"
CACHE_AUDIT = PARENT / "initialization-recovery-cache-audit-002/report.json"


class RecoveredControlTests(unittest.TestCase):
    def test_assisted_or_unsealed_producer_rejected_before_artifact_io(self):
        producer = {"schema": target.PRODUCER_SCHEMA, "status": "complete",
                    "lane": "exploratory_image_only_cached_sparse_refinement",
                    "supplied_intrinsics_or_poses_used": False,
                    "features_or_matches_recomputed": False,
                    "reference_mesh_used": False, "mapping_rerun": False,
                    "gates": {name: True for name in target.REQUIRED_GATES}}
        nonexistent = Path("/definitely-not-a-recovery-producer")
        for changed in ({"supplied_intrinsics_or_poses_used": True}, {"gates": {}},
                        {"schema": "other"}):
            suspect = {**producer, **changed}
            with self.assertRaisesRegex(ValueError, "image-only"):
                target.validate_chain(suspect, {}, nonexistent, nonexistent, nonexistent,
                                      nonexistent, nonexistent, nonexistent, nonexistent)

    def test_native_preflight_keeps_two_generation_hard_gate(self):
        sizes = {f"NP3_{i:03}.jpg": (1277, 1015) for i in range(0, 360, 6)}
        with self.assertRaisesRegex(ValueError, "peak exceeds"):
            target.audit.native_depth_preflight(sizes, 80 << 20,
                                                output_cap=700 << 20, minimum=600)
        result = target.preflight_existing(sizes, 80 << 20)
        self.assertEqual(len(result["images"]), 60)
        self.assertLess(result["predicted_two_generation_peak_bytes"], target.CAP)
        self.assertEqual({row["actual_level"] for row in result["images"].values()}, {1})

    def test_sealed_local_chain_and_tamper_rejection(self):
        if not all(path.exists() for path in (PRODUCER, SOURCE, MODEL, MANIFEST, MASKS, PHOTOS, CACHE_AUDIT)):
            self.skipTest("ignored YCB producer artifacts unavailable")
        producer, source = json.loads(PRODUCER.read_text()), json.loads(SOURCE.read_text())
        result = target.validate_chain(producer, source, PRODUCER, SOURCE, MODEL, PHOTOS,
                                       MASKS, MANIFEST, CACHE_AUDIT)
        self.assertEqual(len(result["registered_names"]), 60)
        self.assertEqual(result["points3D"], 3921)
        changed = copy.deepcopy(producer)
        changed["refined_model_files_sha256"]["images.bin"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hashes do not bind"):
            target.validate_chain(changed, source, PRODUCER, SOURCE, MODEL, PHOTOS,
                                  MASKS, MANIFEST, CACHE_AUDIT)
        changed = copy.deepcopy(producer)
        changed["supplied_intrinsics_or_poses_used"] = True
        with self.assertRaisesRegex(ValueError, "image-only"):
            target.validate_chain(changed, source, PRODUCER, SOURCE, MODEL, PHOTOS,
                                  MASKS, MANIFEST, CACHE_AUDIT)
        changed = copy.deepcopy(producer)
        changed["gates"] = {}
        with self.assertRaisesRegex(ValueError, "image-only"):
            target.validate_chain(changed, source, PRODUCER, SOURCE, MODEL, PHOTOS,
                                  MASKS, MANIFEST, CACHE_AUDIT)
        changed = copy.deepcopy(producer)
        changed["after_camera_params"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "invalid camera"):
            target.validate_chain(changed, source, PRODUCER, SOURCE, MODEL, PHOTOS,
                                  MASKS, MANIFEST, CACHE_AUDIT)


if __name__ == "__main__":
    unittest.main()
