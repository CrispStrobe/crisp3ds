"""Synthetic-only board-plane, distortion and fixed-pose track guards."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from scripts.object_motion.board_frame_candidate_filter import (
    CalibratedFrame, filter_candidate_tracks, prepare_frames,
)
from scripts.object_motion.board_pose_sparse_contract import ObjectTrack, Observation


class BoardFrameCandidateFilterTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
        self.d = (-.08, .015, .001, -.001, .005)
        self.centers = (-1., 0., 1.)
        self.frames = [CalibratedFrame(f'NP3_{i*6:03}.jpg', 640, 480,
                                       (400., 400., 320., 240.), self.d,
                                       ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                                       (-center, 0., 10.))
                       for i, center in enumerate(self.centers)]
        self.unposed = 'NP3_018.jpg'
        self.sources = {frame.name for frame in self.frames} | {self.unposed}

    def track(self, track_id, xyz, *, with_unposed=False):
        obs = []
        for frame in self.frames:
            point = cv2.projectPoints(np.asarray(xyz, dtype=float).reshape(1, 3),
                                      np.zeros(3), np.asarray(frame.translation),
                                      self.k, np.asarray(self.d))[0].reshape(2)
            obs.append(Observation(frame.name, float(point[0]), float(point[1])))
        if with_unposed:
            obs.append(Observation(self.unposed, 100., 100.))
        return ObjectTrack(track_id, tuple(obs))

    def filter(self, tracks, frames=None):
        return filter_candidate_tracks(tracks, frames or self.frames, self.sources,
                                       expected_posed_count=3, expected_source_count=4)

    def test_distorted_triangulation_to_virtual_pinhole_observations(self):
        result = self.filter([self.track(7, (.3, -.2, -2.), with_unposed=True)])
        self.assertEqual(result.rejected_by_reason, {})
        self.assertEqual(result.dropped_unposed_observations, 1)
        self.assertEqual(result.camera_facing_sign, -1)
        accepted = result.accepted[0]
        self.assertEqual(accepted.track_id, 7)
        np.testing.assert_allclose(accepted.xyz_board_squares, (.3, -.2, -2.), atol=1e-4)
        self.assertAlmostEqual(accepted.signed_height_squares, 2., places=4)
        self.assertGreater(accepted.parallax_degrees, 1.)
        self.assertLess(accepted.max_distorted_reprojection_px, .01)
        self.assertLess(accepted.max_virtual_reprojection_px, .01)
        self.assertNotAlmostEqual(accepted.virtual_observations[0].x,
                                  self.track(7, (.3, -.2, -2.)).observations[0].x, places=5)

    def test_reject_board_footprint_near_plane_outside_and_behind(self):
        tracks = [self.track(1, (0., 0., -.2)),
                  self.track(2, (6., 0., -.2)),
                  self.track(3, (0., 0., .2)),
                  self.track(4, (0., 0., -2.))]
        result = self.filter(tracks)
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].track_id, 4)
        self.assertEqual(result.rejected_by_reason,
                         {'board_plane_footprint': 1,
                          'near_plane_outside_footprint': 1,
                          'on_or_behind_board': 1})

    def test_reprojection_and_parallax_fail_closed(self):
        good = self.track(1, (0., 0., -2.))
        wrong = replace(good.observations[0], x=good.observations[0].x+12)
        changed = replace(good, observations=(wrong, *good.observations[1:]))
        self.assertEqual(self.filter([changed]).rejected_by_reason, {'reprojection': 1})
        tiny = [replace(frame, translation=(-center, 0., 10.))
                for frame, center in zip(self.frames, (-.002, 0., .002))]
        small_baseline = []
        for frame in tiny:
            point = cv2.projectPoints(np.array([[0., 0., -2.]]), np.zeros(3),
                                      np.asarray(frame.translation), self.k,
                                      np.asarray(self.d))[0].reshape(2)
            small_baseline.append(Observation(frame.name, float(point[0]), float(point[1])))
        self.assertEqual(self.filter([ObjectTrack(2, tuple(small_baseline))], tiny).rejected_by_reason,
                         {'parallax': 1})

    def test_parallax_requires_one_baseline_not_every_pair(self):
        centers = (-1., 0., .001)
        frames = [replace(frame, translation=(-center, 0., 10.))
                  for frame, center in zip(self.frames, centers)]
        observations = []
        for frame in frames:
            point = cv2.projectPoints(np.array([[0., 0., -2.]]), np.zeros(3),
                                      np.asarray(frame.translation), self.k,
                                      np.asarray(self.d))[0].reshape(2)
            observations.append(Observation(frame.name, float(point[0]), float(point[1])))
        result = self.filter([ObjectTrack(3, tuple(observations))], frames)
        self.assertEqual(len(result.accepted), 1)
        self.assertGreater(result.accepted[0].parallax_degrees, 1.)
        # The last two views are almost coincident: this is not an all-pairs gate.
        near_pair_degrees = np.degrees(np.arctan(.001 / 8.))
        self.assertLess(near_pair_degrees, 1.)

    def test_exact_posed_subset_and_invalid_camera_guards(self):
        short = ObjectTrack(1, (self.track(1, (0., 0., -2.)).observations[0],
                                Observation(self.unposed, 100., 100.)))
        self.assertEqual(self.filter([short]).rejected_by_reason,
                         {'fewer_than_three_posed_views': 1})
        foreign = ObjectTrack(2, (Observation('foreign.jpg', 1., 1.),))
        with self.assertRaisesRegex(ValueError, 'foreign'):
            self.filter([foreign])
        bad = replace(self.frames[0], distortion=(float('nan'), *self.d[1:]))
        with self.assertRaisesRegex(ValueError, 'invalid calibrated camera'):
            prepare_frames([bad, *self.frames[1:]], expected_count=3)
        opposite_side = replace(self.frames[0], translation=(-self.centers[0], 0., -10.))
        with self.assertRaisesRegex(ValueError, 'side unavailable'):
            prepare_frames([opposite_side, *self.frames[1:]], expected_count=3)


if __name__ == '__main__':
    unittest.main()
