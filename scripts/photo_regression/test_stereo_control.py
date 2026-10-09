import unittest
import numpy as np
from stereo_control import intrinsic, pair_depth


class StereoControlTests(unittest.TestCase):
    def test_independent_photos_of_plane_recover_camera_z(self):
        rng = np.random.default_rng(236)
        texture = rng.integers(20, 230, (100, 160, 3), dtype=np.uint8)
        # fx=80, baseline=.1, plane Z=1: exact eight-pixel disparity.
        right = np.zeros_like(texture);right[:, :-8] = texture[:, 8:]
        a = {'rotation': np.eye(3).tolist(), 'translation': [0, 0, 0], 'k': [80, 80, 80.5, 50.5], 'width': 160, 'height': 100}
        b = dict(a, translation=[-.1, 0, 0])
        sparse = np.array([[-.3, -.2, 1], [.3, .2, 1]])
        mask = np.full((100, 160), 255, np.uint8)
        depth, _ = pair_depth(a, b, [texture, right], [mask, mask], sparse, 8)
        interior = depth[10:90, 40:145];valid = interior > 0
        self.assertGreater(float(valid.mean()), .9)
        self.assertLess(float(np.median(np.abs(interior[valid]-1))), .005)
        np.testing.assert_array_equal(intrinsic(a), [[80, 0, 80], [0, 80, 50], [0, 0, 1]])

    def test_zero_baseline_is_refused(self):
        a = {'rotation': np.eye(3).tolist(), 'translation': [0, 0, 0], 'k': [80, 80, 80.5, 50.5], 'width': 160, 'height': 100}
        image = np.zeros((100, 160, 3), np.uint8);mask = np.full((100, 160), 255, np.uint8)
        with self.assertRaisesRegex(ValueError, 'baseline'):
            pair_depth(a, a, [image, image], [mask, mask], np.array([[0, 0, 1], [0, 0, 2]]), 8)


if __name__ == '__main__':
    unittest.main()
