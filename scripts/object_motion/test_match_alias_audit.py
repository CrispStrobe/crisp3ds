"""Cyclic-index grouping and COLMAP pair-ID regression tests."""

from unittest import TestCase

import numpy as np

from scripts.object_motion import match_alias_audit as audit


class MatchAliasAuditTests(TestCase):
    def test_pair_id_roundtrip_and_rejection(self):
        pair = 3 * audit.MAX_IMAGE_ID + 6
        self.assertEqual(audit.pair_ids(pair), (3, 6))
        with self.assertRaises(ValueError):
            audit.pair_ids(6 * audit.MAX_IMAGE_ID + 3 * audit.MAX_IMAGE_ID)
        with self.assertRaises(ValueError):
            audit.pair_ids(-1)

    def test_cyclic_neighbor_middle_long_and_wrap(self):
        self.assertEqual(audit.cyclic_separation(0, 47, 48), 1)
        self.assertEqual(audit.cyclic_separation(2, 6, 48), 4)
        self.assertEqual(audit.cyclic_separation(0, 24, 48), 24)
        self.assertEqual(audit.bucket(4), "neighbor_1_to_4")
        self.assertEqual(audit.bucket(5), "middle_5_to_15")
        self.assertEqual(audit.bucket(15), "middle_5_to_15")
        self.assertEqual(audit.bucket(16), "long_16_to_24")
        with self.assertRaises(ValueError):
            audit.cyclic_separation(7, 7, 48)

    def test_recovered_full_rotation_comparison(self):
        identity = np.eye(3)
        quarter_turn = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(audit.rotation_degrees(identity, quarter_turn), 90)
        self.assertAlmostEqual(audit.rotation_degrees(quarter_turn, identity), 90)
