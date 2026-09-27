"""Synthetic-only tests; never load checkerboard or dataset inputs."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion.board_pose_sparse_contract import (
    ContractError, Frame, ObjectTrack, Observation, build_sparse_model,
    write_sparse_model,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BoardPoseSparseContractTest(unittest.TestCase):
    def setUp(self):
        try:
            import pycolmap  # noqa: F401
        except ImportError:
            self.skipTest("PyCOLMAP unavailable")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.images, self.masks = base / "rgb", base / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        self.frames = []
        self.tracks = []
        xyzs = [np.array((0.04 * (i - 4), 0.035 * ((i * 3) % 7 - 3), 4.0 + i * 0.04))
                for i in range(8)]
        centers = (-0.35, 0.0, 0.35)
        for index, center in enumerate(centers):
            name = f"train_{index:03d}.jpg"
            source, mask = self.images / name, self.masks / (name + ".png")
            Image.new("RGB", (128, 128), (100, 120, 140)).save(source)
            Image.new("L", (128, 128), 255).save(mask)
            self.frames.append(Frame(name, digest(source), digest(mask), 128, 128,
                                     (400, 400, 64, 64),
                                     ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (-center, 0, 0)))
        for track_id, xyz in enumerate(xyzs):
            obs = []
            for frame, center in zip(self.frames, centers):
                obs.append(Observation(frame.name, 400 * (xyz[0] - center) / xyz[2] + 64,
                                       400 * xyz[1] / xyz[2] + 64))
            self.tracks.append(ObjectTrack(track_id, tuple(obs)))

    def build(self, frames=None, tracks=None):
        return build_sparse_model(self.images, self.masks, frames or self.frames, tracks or self.tracks)

    def test_builds_fixed_pose_colmap_model(self):
        model = self.build()
        self.assertEqual(len(model.reg_image_ids()), 3)
        self.assertEqual(len(model.points3D), 8)
        self.assertEqual(model.images[1].name, "train_000.jpg")
        self.assertEqual(list(model.cameras[1].params), [400, 400, 64.5, 64.5])
        self.assertAlmostEqual(model.images[1].points2D[0].xy[0], self.tracks[0].observations[0].x + 0.5)

    def test_rejects_reflection_and_bad_source_hash(self):
        bad = replace(self.frames[0], rotation=((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
        with self.assertRaisesRegex(ContractError, "right-handed"):
            self.build([bad, *self.frames[1:]])
        bad = replace(self.frames[0], image_sha256="0" * 64)
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            self.build([bad, *self.frames[1:]])

    def test_rejects_reprojection_and_reused_observation(self):
        first = self.tracks[0]
        perturbed = replace(first.observations[0], x=first.observations[0].x + 15)
        bad = replace(first, observations=(perturbed, *first.observations[1:]))
        with self.assertRaisesRegex(ContractError, "reprojection"):
            self.build(tracks=[bad, *self.tracks[1:]])
        duplicate = replace(self.tracks[1], observations=self.tracks[0].observations)
        with self.assertRaisesRegex(ContractError, "reused observation"):
            self.build(tracks=[self.tracks[0], duplicate, *self.tracks[2:]])

    def test_rejects_background_mask_and_inventory(self):
        mask = self.masks / (self.frames[0].name + ".png")
        Image.new("L", (128, 128), 0).save(mask)
        adjusted = replace(self.frames[0], mask_sha256=digest(mask))
        with self.assertRaisesRegex(ContractError, "non-object"):
            self.build([adjusted, *self.frames[1:]])
        (self.images / "extra.jpg").touch()
        with self.assertRaisesRegex(ContractError, "inventory"):
            self.build()

    def test_rejects_bad_intrinsics_and_tiny_baseline(self):
        bad = replace(self.frames[0], k=(-400, 400, 64, 64))
        with self.assertRaisesRegex(ContractError, "PINHOLE K"):
            self.build([bad, *self.frames[1:]])
        centers = (-0.001, 0.0, 0.001)
        tiny_poses = [replace(frame, translation=(-center, 0, 0))
                      for frame, center in zip(self.frames, centers)]
        tiny_tracks = []
        for track in self.tracks:
            xyz = np.array((0.04 * (track.track_id - 4),
                            0.035 * ((track.track_id * 3) % 7 - 3),
                            4.0 + track.track_id * 0.04))
            observations = tuple(Observation(frame.name,
                                             400 * (xyz[0] - center) / xyz[2] + 64,
                                             400 * xyz[1] / xyz[2] + 64)
                                 for frame, center in zip(self.frames, centers))
            tiny_tracks.append(replace(track, observations=observations))
        with self.assertRaisesRegex(ContractError, "parallax"):
            self.build(tiny_poses, tiny_tracks)

    def test_export_requires_fresh_path_and_disk_reserve(self):
        destination = Path(self.temp.name) / "new_sparse"
        class NoWrite:
            def write_binary(self, _):
                self.fail("must not write")
        with patch("scripts.object_motion.board_pose_sparse_contract.shutil.disk_usage") as usage:
            usage.return_value.free = 10 * 1024**3
            with self.assertRaisesRegex(ContractError, "insufficient disk"):
                write_sparse_model(NoWrite(), destination, self.images)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
