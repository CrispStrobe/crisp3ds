"""Focused invariants for the evaluation-only six-pair matcher pilot."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.object_motion import mustard_lightglue_six_pair as pilot


class SixPairPilotTests(unittest.TestCase):
    def test_rootsift_is_finite_unit_float_and_does_not_mutate_input(self):
        source = np.arange(4 * 128, dtype=np.float32).reshape(4, 128)
        source[0] = 0
        before = source.copy()
        result = pilot.rootsift(source)
        self.assertEqual(result.shape, (4, 128))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.array_equal(source, before))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)

    def test_shared_artifact_hash_and_float_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            features = {"keypoints": np.array([[1, 2], [3, 4]], np.float32),
                        "scales": np.ones(2, np.float32),
                        "oris": np.zeros(2, np.float32),
                        "descriptors": (np.ones((2, 128), np.float32) / np.float32(np.sqrt(128))),
                        "image_size": np.array([1280, 1024], np.float32)}
            np.savez_compressed(path, **features)
            sealed = pilot.digest(path)
            arm_a = pilot.read_feature(path, sealed)
            arm_b = pilot.read_feature(path, sealed)
            for key in features:
                np.testing.assert_array_equal(arm_a[key], arm_b[key])
            features["descriptors"][0, 0] = 0
            np.savez_compressed(path, **features)
            with self.assertRaisesRegex(ValueError, "shared feature artifact changed"):
                pilot.read_feature(path, sealed)

    def test_frozen_panel_and_train_seal(self):
        path = pilot.STAGE / "train-names.txt"
        if not path.is_file():
            self.skipTest("sealed mustard stage unavailable")
        self.assertEqual(pilot.digest(path), pilot.NAMES_SHA)
        names = path.read_text().splitlines()
        pilot.validate_panel(names)
        with self.assertRaisesRegex(ValueError, "outside sealed"):
            pilot.validate_panel(names[:-1])
        pair_member = pilot.PAIRS[0][1]
        with self.assertRaisesRegex(ValueError, "outside sealed"):
            pilot.validate_panel(["heldout_or_unknown.jpg" if n == pair_member else n for n in names])

    def test_matcher_indices_and_duplicates(self):
        pilot.validate_matches(np.array([[0, 0], [1, 2]], np.uint32), 2, 3)
        for bad in (np.array([[2, 0]], np.uint32),
                    np.array([[0, 3]], np.uint32),
                    np.array([[0, 0], [0, 0]], np.uint32),
                    np.array([[0, 0]], np.int64)):
            with self.subTest(bad=bad.tolist()), self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                pilot.validate_matches(bad, 2, 3)
        descriptors = pilot.rootsift(np.eye(4, 128, dtype=np.float32))
        matches = pilot.classical({"descriptors": descriptors}, {"descriptors": descriptors})
        np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1], [2, 2], [3, 3]], np.uint32))

    def test_order_sensitivity_is_reported_without_reindexing_features(self):
        points = np.column_stack((np.arange(20), np.arange(20) + 1)).astype(np.float32)
        features = {"keypoints": points}
        matches = np.column_stack((np.arange(19, -1, -1), np.arange(19, -1, -1))).astype(np.uint32)
        observed = []

        def fake_fit(row):
            first = float(row["points_left"][0, 0])
            observed.append(first)
            return {"status": "sample", "first_x": first}

        with mock.patch.object(pilot, "fit_pair", side_effect=fake_fit):
            as_returned = pilot.fit("middle", "a", "b", 5, features, features, matches, "as_returned")
            canonical = pilot.fit("middle", "a", "b", 5, features, features, matches, "canonical")
            reverse = pilot.fit("middle", "a", "b", 5, features, features, matches, "reverse")
        self.assertEqual(as_returned["first_x"], reverse["first_x"] + 19 / 1536)
        self.assertEqual(canonical, reverse)
        self.assertEqual(len(observed), 3)

    @unittest.skipUnless(pilot.WEIGHT.is_file(), "external evaluation weight unavailable")
    def test_model_load_does_not_download(self):
        with mock.patch.object(torch.hub, "load_state_dict_from_url", side_effect=AssertionError("implicit download")) as fetch:
            model = pilot.load_model()
        fetch.assert_not_called()
        self.assertEqual(model.conf.input_dim, 128)
        self.assertTrue(model.conf.add_scale_ori)


if __name__ == "__main__":
    unittest.main()
