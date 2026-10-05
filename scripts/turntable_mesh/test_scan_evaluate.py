import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .scan_evaluate import (
    closest_on_triangles,
    evaluate,
    remove_platform,
    rotation_angle,
    surface_distance,
    umeyama,
)


def lumpy_shape(rings=48, segments=96):
    """Closed star-shaped blob without any symmetry, as vertices and triangle indices."""
    theta = np.linspace(0, np.pi, rings + 1)[1:-1, None]
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)[None]
    radius = (1 + 0.35 * np.sin(theta) * np.cos(phi - 0.4) + 0.25 * np.cos(2 * theta + 0.7)
              + 0.2 * np.sin(theta) ** 2 * np.sin(3 * phi + theta)
              + 0.6 * np.exp(-((theta - 0.9) ** 2 + (phi - 1.0) ** 2) / 0.08)
              + 0.4 * np.exp(-((theta - 2.0) ** 2 + (phi - 4.0) ** 2) / 0.05))
    body = np.stack((1.0 * radius * np.sin(theta) * np.cos(phi),
                     0.6 * radius * np.sin(theta) * np.sin(phi),
                     0.8 * radius * np.cos(theta) * np.ones_like(phi)), -1).reshape(-1, 3)
    top = [0.0, 0.0, 0.8 * (1 + 0.25 * np.cos(0.7))]
    bottom = [0.0, 0.0, -0.8 * (1 + 0.25 * np.cos(2 * np.pi + 0.7))]
    vertices = np.vstack((body, top, bottom))
    faces = []
    for ring in range(rings - 2):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            faces += [(a, a + segments, b), (b, a + segments, b + segments)]
    last = (rings - 2) * segments
    for segment in range(segments):
        following = (segment + 1) % segments
        faces.append((len(body), segment, following))
        faces.append((len(body) + 1, last + following, last + segment))
    return vertices, np.array(faces)


def disc(radius, height, cells=60):
    """Flat circular platform at z = height as a triangulated grid."""
    axis = np.linspace(-radius, radius, cells + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel(), np.full(x.size, height)), -1)
    index = np.arange(x.size).reshape(x.shape)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    faces = np.concatenate((np.stack((a, b, c), -1).reshape(-1, 3),
                            np.stack((b, d, c), -1).reshape(-1, 3)))
    inside = (np.linalg.norm(vertices[faces][:, :, :2], axis=2) <= radius).all(1)
    return vertices, faces[inside]


def scene():
    """Reference (object standing on a platform) and the same object in another frame."""
    vertices, faces = lumpy_shape()
    platform_vertices, platform_faces = disc(3.5, vertices[:, 2].min())
    scale = 37.5
    rotation = Rotation.from_rotvec([0.9, -1.4, 2.1]).as_matrix()
    translation = np.array([12.0, -40.0, 7.0])
    to_reference = (scale, rotation, translation)
    reference_vertices = scale * np.vstack((vertices, platform_vertices)) @ rotation.T + translation
    reference_faces = np.vstack((faces, platform_faces + len(vertices)))
    # The "reconstruction" lives in its own frame and scale.
    mesh_rotation = Rotation.from_rotvec([-0.3, 2.0, 0.5]).as_matrix()
    mesh_scale, mesh_translation = 0.013, np.array([0.2, 0.1, -0.4])
    mesh_vertices = mesh_scale * vertices @ mesh_rotation.T + mesh_translation
    expected = (scale / mesh_scale, rotation @ mesh_rotation.T,
                translation - scale / mesh_scale * rotation @ mesh_rotation.T @ mesh_translation)
    return (mesh_vertices[faces], reference_vertices, reference_faces, len(faces), expected,
            to_reference)


class ScanEvaluateTests(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(50, 3))
        rotation = Rotation.from_rotvec([0.2, 1.1, -0.7]).as_matrix()
        scale, recovered, translation = umeyama(points, 2.5 * points @ rotation.T + [1, 2, 3])
        self.assertAlmostEqual(scale, 2.5, places=9)
        np.testing.assert_allclose(recovered, rotation, atol=1e-9)
        np.testing.assert_allclose(translation, [1, 2, 3], atol=1e-9)

    def test_triangle_distance_matches_dense_sampling(self):
        rng = np.random.default_rng(2)
        triangles = rng.normal(size=(40, 3, 3))
        points = rng.normal(size=(40, 3)) * 2
        u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
        inside = (u + v <= 1).ravel()
        u, v = u.ravel()[inside], v.ravel()[inside]
        dense = (triangles[:, None, 0] + u[None, :, None] * (triangles[:, None, 1] - triangles[:, None, 0])
                 + v[None, :, None] * (triangles[:, None, 2] - triangles[:, None, 0]))
        brute = np.linalg.norm(dense - points[:, None], axis=2).min(1)
        exact = np.linalg.norm(closest_on_triangles(points, triangles) - points, axis=1)
        self.assertTrue((exact <= brute + 1e-12).all())
        np.testing.assert_allclose(exact, brute, atol=0.02)
        # A point on a surface is at distance zero from it.
        on_surface = triangles.mean(1)
        np.testing.assert_allclose(surface_distance(on_surface, triangles), 0, atol=1e-12)

    def test_platform_is_removed_from_reference(self):
        _, vertices, faces, object_faces, _, (scale, rotation, _) = scene()
        keep, normal, _, report = remove_platform(vertices, faces, np.random.default_rng(3))
        self.assertFalse(keep[object_faces:].any())
        self.assertGreater(keep[:object_faces].mean(), 0.8)
        self.assertEqual(report["kept_triangles"], int(keep.sum()))
        self.assertEqual(report["removed_triangles_not_above_margin"]
                         + report["removed_triangles_in_small_components"]
                         + report["kept_triangles"], len(faces))
        # Normal is the platform's, pointing to the object's side.
        self.assertGreater(float(normal @ rotation[:, 2]), 0.9999)
        self.assertGreater(report["kept_height_range"][0], 0)

    def test_known_similarity_is_recovered_with_near_zero_distances(self):
        mesh, vertices, faces, _, (scale, rotation, translation), _ = scene()
        result = evaluate(mesh, vertices, faces, seed=7, samples=12000, random_starts=4,
                          handedness="proper")
        alignment = result["alignment"]
        self.assertEqual(alignment["handedness"]["scored"], "proper")
        self.assertFalse(result["warnings"])
        self.assertAlmostEqual(alignment["scale"] / scale, 1, delta=0.005)
        self.assertLess(rotation_angle(np.array(alignment["rotation"]), rotation), 0.5)
        diagonal = result["reference_object"]["bbox_diagonal"]
        # Compare where the object is, not at the far-away mesh-frame origin.
        corners = mesh.reshape(-1, 3)[::97]
        recovered = (alignment["scale"] * corners @ np.array(alignment["rotation"]).T
                     + alignment["translation"])
        error = np.linalg.norm(recovered - (scale * corners @ rotation.T + translation), axis=1)
        self.assertLess(error.max(), 0.003 * diagonal)
        self.assertEqual(alignment["starts"], 28)
        above = result["metrics"]["above_margin"]
        self.assertLess(above["accuracy_mesh_to_reference"]["fraction_of_diagonal"]["p95"], 0.002)
        self.assertLess(above["completeness_reference_to_mesh"]["fraction_of_diagonal"]["p95"], 0.002)
        self.assertGreater(above["thresholds"]["0.005"]["f1"], 0.99)
        everything = result["metrics"]["all"]
        self.assertLess(everything["accuracy_mesh_to_reference"]["fraction_of_diagonal"]["median"], 0.002)
        self.assertEqual(everything["accuracy_mesh_to_reference"]["samples"], 6000)

    def test_mirror_image_is_reported_not_hidden(self):
        mesh, vertices, faces, _, _, _ = scene()
        result = evaluate(mesh * [1.0, -1.0, 1.0], vertices, faces, seed=7, samples=6000,
                          random_starts=0)
        residual = result["alignment"]["handedness"]["best_trimmed_rms"]
        self.assertGreater(residual["proper"], 1.5 * residual["mirrored"])
        self.assertEqual(result["alignment"]["handedness"]["scored"], "mirrored")
        self.assertTrue(result["alignment"]["handedness"]["mesh_mirrored_before_transform"])
        self.assertTrue(any("CHIRALITY" in warning for warning in result["warnings"]))
        self.assertLess(np.linalg.det(np.array(result["alignment"]["linear_part_including_mirror"])), 0)
        above = result["metrics"]["above_margin"]
        self.assertLess(above["accuracy_mesh_to_reference"]["fraction_of_diagonal"]["p95"], 0.002)


if __name__ == "__main__":
    unittest.main()
