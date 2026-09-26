import unittest
from pathlib import Path

import numpy as np

from scripts.fixed_camera_verify.check import (
    dlt_point, parallax_degrees, project, ray_normal_residual, sha256,
    verify_counts, verify_hash_manifest, verify_lane,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {}
        for name, center in (("a", (-1.0, 0.0, 0.0)),
                             ("b", (1.0, 0.0, 0.0)),
                             ("c", (0.0, 0.8, 0.0))):
            self.cameras[name] = {"R": np.eye(3), "t": -np.array(center),
                                  "f": 700.0, "cx": 384.0, "cy": 256.0}

    def test_exact_dlt_and_ray_optimum(self):
        truth = np.array((0.2, -0.3, 7.0))
        observations = [(name, project(cam, truth)) for name, cam in self.cameras.items()]
        np.testing.assert_allclose(dlt_point(observations, self.cameras), truth, atol=1e-10)
        residual, _, singular = ray_normal_residual(observations, self.cameras, truth)
        self.assertLess(residual, 1e-10)
        self.assertGreater(singular[-1], 0)

    def test_bad_point_fails_normal_equation(self):
        truth = np.array((0.2, -0.3, 7.0))
        observations = [(name, project(cam, truth)) for name, cam in self.cameras.items()]
        residual, _, _ = ray_normal_residual(observations, self.cameras, truth+[0.2, 0, 0])
        self.assertGreater(residual, 0.01)

    def test_behind_camera_rejected(self):
        with self.assertRaisesRegex(ValueError, "front"):
            project(self.cameras["a"], (0, 0, -2))

    def test_antiparallel_rays_have_zero_line_parallax(self):
        opposite = dict(self.cameras["b"], R=np.diag((-1, 1, -1)))
        cams = {"a": self.cameras["a"], "b": opposite}
        self.assertAlmostEqual(parallax_degrees(
            [("a", (384, 256)), ("b", (384, 256))], cams), 0, places=6)

    def test_summary_tamper_rejected(self):
        component = frozenset((('a', 0), ('b', 0)))
        point = {"observations": [{"image": "a", "original_feature_id": 0},
                                   {"image": "b", "original_feature_id": 0}]}
        graph = {"verified_geometry_pairs": 1, "verified_geometry_rows": 1,
                 "roi_eligible_rows": 1, "roi_excluded_rows": 0,
                 "unregistered_endpoint_rows": 0, "connected_components": 1,
                 "same_image_conflict": 0}
        saved = {**graph, "rejected_by_reason": {}, "accepted_tracks": 1,
                 "accepted_tracks_three_or_more_views": 0, "accepted_observations": 2,
                 "registered_images_covered": 2,
                 "accepted_observations_by_image": {"a": 1, "b": 1}}
        verify_counts([point], saved, graph, {component})
        saved["accepted_tracks"] = 2
        with self.assertRaisesRegex(ValueError, "accounting"):
            verify_counts([point], saved, graph, {component})

    def test_camera_hash_tamper_rejected(self):
        source = str(Path(__file__).resolve())
        actual = sha256(Path(source))
        verify_hash_manifest({source: actual}, {source: actual}, {source})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_hash_manifest({source: "0"*64}, {source: "0"*64}, {source})

    def test_full_point_contract_and_mutations(self):
        truth = np.array((0.2, -0.3, 7.0))
        pixels = {name: project(cam, truth) for name, cam in self.cameras.items()}
        observations = [(name, xy) for name, xy in pixels.items()]
        _, _, singular = ray_normal_residual(observations, self.cameras, truth)
        point = {"id": 1, "xyz": truth.tolist(),
                 "observations": [{"image": name, "original_feature_id": 0,
                                   "reprojection_px": 0.0} for name in self.cameras],
                 "source_track": [[name, 0] for name in self.cameras],
                 "max_reprojection_px": 0.0,
                 "max_ray_parallax_deg": parallax_degrees(observations, self.cameras),
                 "normal_matrix_condition": float(singular[0]/singular[-1])}
        features = {name: [xy.tolist()] for name, xy in pixels.items()}
        polygons = {name: [[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]]
                    for name in pixels}
        heldout = {name: set() for name in pixels}
        result = verify_lane([point], self.cameras, features, polygons, heldout)
        self.assertEqual(result["points_checked"], 1)
        heldout["a"].add(0)
        with self.assertRaisesRegex(ValueError, "held-out"):
            verify_lane([point], self.cameras, features, polygons, heldout)
        heldout["a"].clear()
        point["xyz"] = (truth + [0.1, 0, 0]).tolist()
        with self.assertRaisesRegex(ValueError, "reprojection"):
            verify_lane([point], self.cameras, features, polygons, heldout)
        point["xyz"] = truth.tolist()
        point["observations"][0]["original_feature_id"] = 1
        with self.assertRaisesRegex(ValueError, "invalid feature reference"):
            verify_lane([point], self.cameras, features, polygons, heldout)


if __name__ == "__main__":
    unittest.main()
