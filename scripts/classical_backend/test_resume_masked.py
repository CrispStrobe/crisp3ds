"""Masked continuation rejects unsafe cache inputs before native work."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import resume_masked


class ResumeMaskedSafetyTests(unittest.TestCase):
    def setUp(self):
        tmp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "source"
        (self.source / "dense").mkdir(parents=True)
        (self.source / "masks").mkdir()
        (self.source / "masks" / "report.json").write_text("{}")
        (self.source / "scene.mvs").write_bytes(b"scene")
        self.upstream = self.base / "upstream"
        self.upstream.mkdir()
        prior = {"schema": "classical_masked_dense_v1", "status": "failed",
                 "source_run": str(self.upstream), "mask_report_sha256": "same",
                 "binary_hashes": {name: "same" for name in resume_masked.TOOLS},
                 "native_options": {"resolution_level": 0, "max_resolution": 1600,
                                    "geometric_iters": 2, "tower_mode": 4,
                                    "ignore_mask_label": 0,
                                    "refine_resolution_level": 1, "refine_scales": 1},
                 "stages": [{"name": "densify", "failure": "output byte limit reached"}]}
        (self.source / "result.json").write_text(json.dumps(prior))

    def args(self, **changes):
        values = dict(source=self.source, output=self.base / "continuation",
                      binary_dir=self.base, max_threads=2, max_gib=3,
                      max_rss_gib=10, max_log_mib=32, timeout_minutes=10)
        values.update(changes)
        return Namespace(**values)

    def test_raw_output_symlink_and_missing_parent_rejected(self):
        link = self.base / "link"
        try:
            link.symlink_to(self.base / "future", target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "fresh output"):
            resume_masked.resume(self.args(output=link))
        with self.assertRaisesRegex(ValueError, "output parent"):
            resume_masked.resume(self.args(output=self.base / "absent" / "fresh"))

    def test_symlink_inside_copy_tree_rejected(self):
        link = self.source / "dense" / "outside"
        try:
            link.symlink_to(self.base, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "cached dense tree contains"):
            resume_masked.resume(self.args())

    def test_bounds_rejected_before_depth_map_parse(self):
        with (patch.object(resume_masked, "validate_inputs", return_value={}),
              patch.object(resume_masked, "digest", return_value="same"),
              patch.object(resume_masked, "tool_path", return_value=self.source / "scene.mvs"),
              patch.object(resume_masked, "complete_base_maps", side_effect=AssertionError("parsed DMAP"))):
            for change in (dict(timeout_minutes=float("nan")), dict(timeout_minutes=11),
                           dict(max_gib=4), dict(max_rss_gib=float("inf")),
                           dict(max_log_mib=33), dict(max_threads=3)):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "restricted"):
                    resume_masked.resume(self.args(**change))

    def test_depth_map_symlink_rejected_before_parser(self):
        link = self.source / "depth0000.dmap"
        try:
            link.symlink_to(self.source / "scene.mvs")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with (patch.object(resume_masked, "validate_inputs", return_value={}),
              patch.object(resume_masked, "digest", return_value="same"),
              patch.object(resume_masked, "tool_path", return_value=self.source / "scene.mvs"),
              patch.object(resume_masked, "complete_base_maps", side_effect=AssertionError("parsed DMAP"))):
            with self.assertRaisesRegex(ValueError, "cached DMAP set contains"):
                resume_masked.resume(self.args())


if __name__ == "__main__":
    unittest.main()
