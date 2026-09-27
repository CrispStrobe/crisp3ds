"""Focused option and input checks for the two sealed mapping diagnostics."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest

from scripts.upstream_control import fixed_intrinsics


class Options:
    def __init__(self):
        self.num_threads = -1
        self.init_image_id1 = -1
        self.init_image_id2 = -1
        self.ba_refine_focal_length = True
        self.ba_refine_principal_point = False
        self.ba_refine_extra_params = True
        self.mapper = types.SimpleNamespace(num_threads=-1,
                                            abs_pose_refine_focal_length=True,
                                            abs_pose_refine_extra_params=True)


class DiagnosticTests(unittest.TestCase):
    def test_fixed_arm_changes_only_intrinsic_refinement(self):
        fake = types.SimpleNamespace(IncrementalPipelineOptions=Options)
        options = fixed_intrinsics.mapping_options(fake, "fixed-automatic")
        self.assertEqual((options.init_image_id1, options.init_image_id2), (-1, -1))
        self.assertEqual((options.num_threads, options.mapper.num_threads), (2, 2))
        self.assertFalse(options.ba_refine_focal_length)
        self.assertFalse(options.ba_refine_principal_point)
        self.assertFalse(options.ba_refine_extra_params)
        self.assertFalse(options.mapper.abs_pose_refine_focal_length)
        self.assertFalse(options.mapper.abs_pose_refine_extra_params)
        self.assertEqual(vars(options).keys(), vars(Options()).keys())

    def test_native_seed_arm_preserves_refinement(self):
        fake = types.SimpleNamespace(IncrementalPipelineOptions=Options)
        options = fixed_intrinsics.mapping_options(fake, "native-seed30", (33, 28))
        self.assertEqual((options.init_image_id1, options.init_image_id2), (33, 28))
        self.assertTrue(options.ba_refine_focal_length)
        self.assertTrue(options.ba_refine_extra_params)
        self.assertTrue(options.mapper.abs_pose_refine_focal_length)
        self.assertTrue(options.mapper.abs_pose_refine_extra_params)
        with self.assertRaisesRegex(ValueError, "requires verified"):
            fixed_intrinsics.mapping_options(fake, "native-seed30", (28, 33))
        with self.assertRaisesRegex(ValueError, "automatic"):
            fixed_intrinsics.mapping_options(fake, "fixed-automatic", (33, 28))

    def test_seed_ids_resolved_from_database_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.executemany("INSERT INTO images VALUES (?, ?)",
                                       [(28, "NP3_162.jpg"), (33, "NP3_192.jpg")])
                connection.commit()
            self.assertEqual(fixed_intrinsics.seed_ids(database), (33, 28))
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("UPDATE images SET image_id=34 WHERE image_id=33")
                connection.commit()
            with self.assertRaisesRegex(ValueError, "ID mapping changed"):
                fixed_intrinsics.seed_ids(database)

    def test_existing_output_rejected_before_reading_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                fixed_intrinsics.run(root / "missing", root, "fixed-automatic")


if __name__ == "__main__":
    unittest.main()
