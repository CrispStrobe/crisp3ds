"""Tiny native regression for the zero-distortion sparse-to-dense handoff."""

import importlib.util
from pathlib import Path
import tempfile
from unittest import TestCase, skipUnless

import numpy as np

from scripts.classical_backend import sparse_masked_dense as handoff


@skipUnless(importlib.util.find_spec("pycolmap"), "optional PyCOLMAP fixture")
class ZeroDistortionHandoffTests(TestCase):
    def make_model(self, distortion: float):
        import pycolmap

        model = pycolmap.Reconstruction()
        camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1280, height=1024,
                                 params=[1536.0, 640.0, 512.0, distortion], camera_id=1)
        model.add_camera(camera)
        for image_id, tx, x in ((1, 0.0, 640.0), (2, -0.1, 563.2)):
            image = pycolmap.Image(
                name=f"NP3_{image_id:03d}.jpg", keypoints=np.asarray([[x, 512.0]]),
                cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(),
                                                np.asarray([tx, 0.0, 0.0])),
                camera_id=1, id=image_id)
            model.add_image(image)
            model.register_image(image_id)
        model.add_point3D(np.asarray([0.0, 0.0, 2.0]), pycolmap.Track([
            pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0)]))
        model.check()
        return model

    def test_native_zero_radial_preserves_geometry_and_identity_mask_warp(self):
        import pycolmap

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            sparse = output / "dense" / "sparse"
            sparse.mkdir(parents=True)
            original = self.make_model(0.0)
            original.write_binary(str(sparse))
            before_hashes = handoff.model_hashes(sparse)

            result = handoff.normalize_undistorted_camera(output)
            converted = pycolmap.Reconstruction(str(sparse))
            camera = next(iter(converted.cameras.values()))
            self.assertEqual(result["conversion"], "SIMPLE_RADIAL_zero_distortion_to_PINHOLE")
            self.assertEqual(camera.model.name, "PINHOLE")
            self.assertEqual((camera.width, camera.height), (1280, 1024))
            np.testing.assert_array_equal(camera.params, [1536.0, 1536.0, 640.0, 512.0])
            self.assertLessEqual(result["max_projection_difference_pixels"], 1e-9)
            self.assertTrue(result["poses_measurements_tracks_xyz_identical"])
            self.assertEqual(before_hashes["images.bin"], result["normalized_model_sha256"]["images.bin"])
            self.assertEqual(before_hashes["points3D.bin"], result["normalized_model_sha256"]["points3D.bin"])
            self.assertNotEqual(before_hashes["cameras.bin"], result["normalized_model_sha256"]["cameras.bin"])
            for image_id, old in original.images.items():
                new = converted.images[image_id]
                np.testing.assert_array_equal(old.cam_from_world.matrix(), new.cam_from_world.matrix())
                self.assertEqual(old.points2D[0].point3D_id, new.points2D[0].point3D_id)
                np.testing.assert_array_equal(old.points2D[0].xy, new.points2D[0].xy)
            for point_id, old in original.points3D.items():
                new = converted.points3D[point_id]
                np.testing.assert_array_equal(old.xyz, new.xyz)
                self.assertEqual([(e.image_id, e.point2D_idx) for e in old.track.elements],
                                 [(e.image_id, e.point2D_idx) for e in new.track.elements])

            if importlib.util.find_spec("cv2"):
                from scripts.classical_backend.dense_masks import remap_binary_mask
                mask = np.zeros((1024, 1280), dtype=np.uint8)
                mask[250:650, 480:760] = 255
                mask[300:450, 550:620] = 0
                source_camera = next(iter(original.cameras.values()))
                np.testing.assert_array_equal(remap_binary_mask(mask, source_camera, camera), mask)

    def test_nonzero_radial_rejected_without_model_replacement(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            sparse = output / "dense" / "sparse"
            sparse.mkdir(parents=True)
            self.make_model(0.01).write_binary(str(sparse))
            hashes = handoff.model_hashes(sparse)
            with self.assertRaisesRegex(ValueError, "zero-distortion"):
                handoff.normalize_undistorted_camera(output)
            self.assertEqual(hashes, handoff.model_hashes(sparse))
            self.assertFalse((output / "dense" / "pinhole-candidate").exists())
