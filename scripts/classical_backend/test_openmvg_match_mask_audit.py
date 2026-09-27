"""Synthetic-only contracts for the read-only OpenMVG match/mask parser."""
from __future__ import annotations

import struct
import unittest

from scripts.classical_backend import openmvg_match_mask_audit as audit


def archive(*, endian: str = "<") -> bytes:
    marker = 1 if endian == "<" else 0
    return (bytes([marker]) + struct.pack(endian + "Q", 2) +
            struct.pack(endian + "IIQIIII", 0, 1, 2, 3, 4, 5, 6) +
            struct.pack(endian + "IIQII", 1, 3, 1, 7, 8))


class MatchMaskAuditTest(unittest.TestCase):
    def test_portable_binary_little_and_big_endian(self) -> None:
        expected = [(0, 1, [(3, 4), (5, 6)]), (1, 3, [(7, 8)])]
        self.assertEqual(audit.decode_matches(archive()), expected)
        self.assertEqual(audit.decode_matches(archive(endian=">")), expected)

    def test_truncation_trailing_bytes_and_bad_pairs_fail_closed(self) -> None:
        for blob in (archive()[:-1], archive() + b"x", bytes([2]) + archive()[1:],
                     archive().replace(struct.pack("<II", 1, 3), struct.pack("<II", 3, 1))):
            with self.subTest(blob=blob[-8:]), self.assertRaises(ValueError):
                audit.decode_matches(blob)

    def test_nearest_pixel_and_coarse_mask_classes(self) -> None:
        self.assertEqual(audit.nearest_pixel(0.49, 0.49, 4, 3), 0)
        self.assertEqual(audit.nearest_pixel(0.5, 1.5, 4, 3), 9)
        self.assertIsNone(audit.nearest_pixel(-0.51, 1.0, 4, 3))
        self.assertIsNone(audit.nearest_pixel(3.5, 1.0, 4, 3))
        self.assertEqual([audit.match_class(*pair) for pair in
                          ((True, True), (False, True), (False, False), (None, True))],
                         ["both_in", "one_in", "neither_in", "out_of_frame"])

    def test_cyclic_bins_have_distinct_opposite_slot(self) -> None:
        self.assertEqual([audit.distance_bin(x) for x in (1, 2, 4, 5, 9, 10, 19, 20, 29, 30)],
                         ["1", "2-4", "2-4", "5-9", "5-9", "10-19", "10-19",
                          "20-29", "20-29", "30"])


if __name__ == "__main__":
    unittest.main()
