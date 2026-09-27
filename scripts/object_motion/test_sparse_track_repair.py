"""Deterministic duplicate-view selection and pinned native mutation contracts."""

import importlib.util
from pathlib import Path
import tempfile
from unittest import TestCase, skipUnless

import numpy as np

from scripts.object_motion import sparse_track_repair as repair


class TrackRepairTests(TestCase):
    def test_minimum_finite_residual_then_index_and_view_count(self):
        records = [{"image_id": 2, "point2D_idx": 5, "baseline_residual": 1.0},
                   {"image_id": 1, "point2D_idx": 8, "baseline_residual": 0.5},
                   {"image_id": 1, "point2D_idx": 3, "baseline_residual": 0.5},
                   {"image_id": 2, "point2D_idx": 6, "baseline_residual": None}]
        selected, redundant, reason = repair.select_observations(records)
        self.assertIsNone(reason)
        self.assertEqual([(r["image_id"], r["point2D_idx"]) for r in selected], [(1, 3), (2, 5)])
        self.assertEqual(len(redundant), 2)
        self.assertEqual(repair.select_observations(records[:1])[2], "fewer_than_two_distinct_views")
        self.assertEqual(repair.select_observations([records[1], records[3]])[2],
                         "no_finite_reprojection_in_view")

    @skipUnless(importlib.util.find_spec("pycolmap"), "optional pinned PyCOLMAP fixture")
    def test_native_delete_backlink_autodelete_and_retriangulate_fixed_camera(self):
        import pycolmap

        def make_model():
            model = pycolmap.Reconstruction()
            model.add_camera(pycolmap.Camera.create(
                1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 100, 100))
            # World point (0,0,2); camera translations shift image X.
            for image_id, tx, points in ((1, 0.0, [[50.0, 50.0], [50.1, 50.0]]),
                                         (2, -1.0, [[0.0, 50.0]]),
                                         (3, -2.0, [[-50.0, 50.0]])):
                image = pycolmap.Image(name=f"image{image_id}", keypoints=np.asarray(points),
                                       cam_from_world=pycolmap.Rigid3d(
                                           pycolmap.Rotation3d(), np.asarray([tx, 0.0, 0.0])),
                                       camera_id=1, id=image_id)
                model.add_image(image)
                model.register_image(image_id)
            point_id = model.add_point3D(np.asarray([0.0, 0.0, 2.0]), pycolmap.Track([
                pycolmap.TrackElement(1, 0), pycolmap.TrackElement(1, 1),
                pycolmap.TrackElement(2, 0), pycolmap.TrackElement(3, 0)]))
            return model, point_id

        baseline, point_id = make_model()
        candidate, _ = make_model()
        options = pycolmap.EstimateTriangulationOptions()
        options.ransac.max_num_trials = 10000
        pycolmap.set_random_seed(repair.SEED)

        def fixed_estimator(xy, poses, cameras, _options):
            self.assertEqual(xy.shape, (3, 2))
            return {"xyz": np.asarray([0.0, 0.0, 2.0]), "inliers": np.ones(3, dtype=bool)}

        result = repair.repair_model(baseline, candidate, fixed_estimator, options)
        self.assertEqual(result["baseline"]["observations"], 4)
        self.assertEqual(result["output"]["observations"], 3)
        self.assertEqual(result["removed_observations"], 1)
        self.assertEqual(result["output"]["duplicate_same_image_tracks"], 0)
        self.assertFalse(candidate.images[1].points2D[1].has_point3D())
        self.assertEqual(len(candidate.points3D[point_id].track.elements), 3)
        self.assertTrue(repair.frozen_cameras_and_poses(baseline, candidate))
        self.assertEqual(result["residuals"]["repaired_all_original_common"]["denominator"], 4)
        self.assertEqual(result["residuals"]["repaired_selected"]["denominator"], 3)
        self.assertFalse(result["gates"]["retained_selected_observations_at_least_90pct"])
        self.assertFalse(result["gates"]["dense_eligible_predeclared"])

        actual_base, _ = make_model()
        actual_candidate, _ = make_model()
        pycolmap.set_random_seed(repair.SEED)
        native = repair.repair_model(actual_base, actual_candidate,
                                     pycolmap.estimate_triangulation, options)
        self.assertEqual(native["output"]["observations"], 3)
        self.assertEqual(native["output"]["duplicate_same_image_tracks"], 0)
        self.assertEqual(native["residuals"]["repaired_selected"]["invalid_count"], 0)
        self.assertTrue(repair.frozen_cameras_and_poses(actual_base, actual_candidate))

        two, two_id = make_model()
        two.delete_observation(3, 0)  # 4 -> 3
        two.delete_observation(1, 1)  # 3 -> 2
        two.delete_observation(1, 0)  # 2 -> point auto-deleted
        self.assertNotIn(two_id, two.points3D)
        self.assertFalse(two.images[2].points2D[0].has_point3D())

    @skipUnless(importlib.util.find_spec("pycolmap"), "optional pinned PyCOLMAP fixture")
    def test_clean_track_geometry_unchanged_and_estimator_not_called(self):
        import pycolmap

        def model():
            rec = pycolmap.Reconstruction()
            rec.add_camera(pycolmap.Camera.create(
                1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 100, 100))
            for image_id, tx, x in ((1, 0.0, 50.0), (2, -1.0, 0.0)):
                image = pycolmap.Image(name=f"image{image_id}", keypoints=np.asarray([[x, 50.0]]),
                                       cam_from_world=pycolmap.Rigid3d(
                                           pycolmap.Rotation3d(), np.asarray([tx, 0.0, 0.0])),
                                       camera_id=1, id=image_id)
                rec.add_image(image)
                rec.register_image(image_id)
            rec.add_point3D(np.asarray([0.0, 0.0, 2.0]), pycolmap.Track([
                pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0)]))
            return rec

        original, candidate = model(), model()
        old_xyz = {point_id: np.asarray(point.xyz).copy() for point_id, point in original.points3D.items()}

        def forbidden(*_args):
            self.fail("clean tracks must not be re-triangulated")

        report = repair.repair_model(original, candidate, forbidden,
                                     pycolmap.EstimateTriangulationOptions())
        self.assertEqual(report["unchanged_clean_points"], 1)
        self.assertEqual(report["removed_observations"], 0)
        for point_id, xyz in old_xyz.items():
            np.testing.assert_array_equal(candidate.points3D[point_id].xyz, xyz)
            self.assertGreaterEqual(candidate.points3D[point_id].error, 0)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            candidate.write_binary(directory)
            reopened = pycolmap.Reconstruction(directory)
            self.assertTrue(repair.frozen_cameras_and_poses(original, reopened))
            for point_id, xyz in old_xyz.items():
                np.testing.assert_array_equal(reopened.points3D[point_id].xyz, xyz)
