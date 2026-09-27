from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend.validate_openmvs_fusion_trace import HEADER, validate


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "fusion-trace.tsv"

    def _write(self, pixels=5, event_count=5, reason="accepted", truncated=0):
        rows = ["\t".join(HEADER),
                f"0\t27\t1\t10\t11\t1.0\t2.0\t3.0\t{pixels}\t2\t{truncated}"]
        rows += [f"event\t0\t{1 if i % 2 == 0 else 2}\t10\t11\t3.0\t0.9\t3.0\t0.0\t1.0\t{reason}"
                 for i in range(event_count)]
        self.path.write_text("\n".join(rows) + "\n")

    def test_valid_point_counts_accepted_pixels_and_views(self):
        self._write()
        report = validate(self.path)
        self.assertEqual(report["point_count"], 1)
        self.assertTrue(report["complete_event_records"])

    def test_zero_based_view_and_single_pixel_are_valid(self):
        self.path.write_text("\t".join(HEADER) + "\n" +
                             "0\t0\t0\t0\t0\t1\t2\t3\t1\t1\t0\n" +
                             "event\t0\t0\t0\t0\t3\t0.9\t0\t0\t0\taccepted\n")
        self.assertTrue(validate(self.path)["complete_event_records"])

    def test_rejects_missing_accepted_pixel(self):
        self._write(event_count=4)
        with self.assertRaisesRegex(ValueError, "accounting"):
            validate(self.path)

    def test_rejects_unknown_reason(self):
        self._write(reason="surprise")
        with self.assertRaisesRegex(ValueError, "event"):
            validate(self.path)

    def test_truncation_is_reported(self):
        self._write(event_count=4, truncated=1)
        self.assertFalse(validate(self.path)["complete_event_records"])


if __name__ == "__main__":
    unittest.main()
