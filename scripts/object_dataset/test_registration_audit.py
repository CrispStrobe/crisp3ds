import unittest

import numpy as np

from scripts.object_dataset import registration_audit as audit


class RegistrationAuditTests(unittest.TestCase):
    def test_known_sim3_is_proper_and_round_trips(self):
        truth = audit.known_sim3()
        self.assertAlmostEqual(np.linalg.det(truth[:3, :3]) ** (1 / 3), 1.7, places=12)
        self.assertGreater(np.linalg.det(truth[:3, :3]), 0)
        points = audit.synthetic_meshes()["asymmetric_full"][0][0]
        np.testing.assert_allclose(audit.apply(audit.apply(points, truth), np.linalg.inv(truth)),
                                   points, atol=1e-12)
        errors = audit.pose_errors(truth, truth, points, 10)
        self.assertLess(errors["rotation_error_degrees"], 1e-6)
        self.assertLess(errors["heldout_corresponding_rms"], 1e-12)
        independently_sampled_reference_frame = points[[1, 3, 0, 2]]
        output_native = audit.synthesize_output(independently_sampled_reference_frame, truth)
        np.testing.assert_allclose(audit.apply(output_native, truth),
                                   independently_sampled_reference_frame, atol=1e-12)

    def test_known_correspondence_and_mirror_not_proper_roundtrip(self):
        truth = audit.known_sim3()
        body = audit.synthetic_meshes()["asymmetric_full"][0]
        controls = audit.paired_controls(truth, body)
        self.assertLess(controls["exact_landmark_umeyama"]["heldout_corresponding_rms"], 1e-10)
        self.assertLess(controls["exact_landmark_umeyama"]["rotation_error_degrees"], 1e-5)
        self.assertGreater(controls["exact_landmark_source_singular_values"][-1], 1e-3)
        self.assertGreater(controls["mirrored_landmark_proper_fit_residual"], 0.01)
        self.assertAlmostEqual(controls["mirrored_landmark_fitted_rotation_determinant"], 1, places=12)
        self.assertEqual(controls["collinear_rank1_ambiguity"]["rank"], 1)
        self.assertLess(controls["collinear_rank1_ambiguity"]["point_fit_rms"], 1e-10)

    def test_identical_samples_unknown_correspondence(self):
        truth = audit.known_sim3()
        body = audit.synthetic_meshes()["asymmetric_full"][0]
        control = audit.identical_unknown_correspondence(truth, body)
        self.assertLess(control["errors"]["heldout_corresponding_rms"], 1e-8)
        self.assertLess(control["errors"]["scale_relative_error"], 1e-8)

    def test_fixture_faces_valid_and_deterministic(self):
        first = audit.synthetic_meshes()
        second = audit.synthetic_meshes()
        for name in first:
            for mesh, repeat in zip(first[name], second[name]):
                vertices, faces = mesh
                np.testing.assert_array_equal(vertices, repeat[0])
                np.testing.assert_array_equal(faces, repeat[1])
                self.assertGreater(len(faces), 0)
                self.assertGreaterEqual(faces.min(), 0)
                self.assertLess(faces.max(), len(vertices))


if __name__ == "__main__":
    unittest.main()
