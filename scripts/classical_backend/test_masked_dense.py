"""Preflight-only safety contracts; no OpenMVS process is launched."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import masked_dense


class MaskedDenseSafetyTests(unittest.TestCase):
    def setUp(self):
        tmp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "source"
        (self.source / "dense").mkdir(parents=True)
        self.masks = self.base / "masks"
        self.masks.mkdir()

    def args(self, **changes):
        values = dict(source=self.source, masks=self.masks, output=self.base / "output",
                      binary_dir=self.base, resolution_level=2, max_resolution=2560,
                      geometric_iters=2, tower_mode=4, max_threads=2, max_gib=2,
                      max_rss_gib=10, max_log_mib=32, timeout_minutes=15)
        values.update(changes)
        return Namespace(**values)

    def test_raw_source_output_and_parent_links_rejected(self):
        alias = self.base / "alias"
        try:
            alias.symlink_to(self.source, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "real directories"):
            masked_dense.run(self.args(source=alias))
        output_link = self.base / "output-link"
        output_link.symlink_to(self.base / "not-yet-created", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "output fresh"):
            masked_dense.run(self.args(output=output_link))
        parent_link = self.base / "parent-link"
        parent_link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "output fresh"):
            masked_dense.run(self.args(output=parent_link / "fresh"))

    def test_internal_copy_link_and_missing_parent_rejected(self):
        alias = self.source / "dense" / "outside"
        try:
            alias.symlink_to(self.base, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "source dense tree contains"):
            masked_dense.run(self.args())
        with self.assertRaisesRegex(ValueError, "output parent"):
            masked_dense.run(self.args(output=self.base / "missing" / "fresh"))

    def test_finite_bounds_and_legitimate_2560_profile(self):
        for override in (dict(timeout_minutes=float("nan")), dict(timeout_minutes=31),
                         dict(max_rss_gib=float("inf")), dict(max_rss_gib=13),
                         dict(max_log_mib=33), dict(max_resolution=5000)):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "bounded"):
                masked_dense.run(self.args(**override))
        self.assertIsNone(masked_dense.require_real_tree(self.source / "dense", "dense"))
        self.assertIsNone(masked_dense.require_real_tree(self.masks, "masks"))
        self.assertEqual(self.args().max_resolution, 2560)


if __name__ == "__main__":
    unittest.main()
