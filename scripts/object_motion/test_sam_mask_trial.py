import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import sam_mask_trial as sam
from scripts.object_motion import ycb_object_masks as common


class SAMMaskTrialTests(unittest.TestCase):
    def test_binary_full_frame_and_bounds(self):
        array = np.zeros((1024, 1280), dtype=bool)
        array[320:530, 520:650] = True
        mask, stats = sam.bounded_mask(array[None])
        self.assertEqual(mask.size, (1280, 1024))
        self.assertEqual(stats["support_pixels"], 27300)
        self.assertEqual(stats["bbox_xyxy_exclusive"], [520, 320, 650, 530])
        float_mask, _ = sam.bounded_mask(array.astype("float32")[None])
        self.assertEqual(float_mask.tobytes(), mask.tobytes())
        array = array.astype("uint8")
        array[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "boundary"):
            sam.bounded_mask(array)
        array[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "not binary"):
            sam.bounded_mask(array)
        array[0, 0] = 0
        floating = array.astype("float32")
        floating[500, 500] = float("nan")
        with self.assertRaisesRegex(ValueError, "not binary"):
            sam.bounded_mask(floating)

    def test_smoke_only_with_fake_predictor_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            package = base / "package.json"
            dataset = base / "dataset"
            photos = dataset / "photos"
            photos.mkdir(parents=True)
            rows = []
            for index, angle in enumerate([a for a in range(0, 360, 6) if (a // 6) % 5 != 4]):
                name = f"NP3_{angle:03}.jpg"
                # Training fixture uses unique bytes per view; held-out files are absent.
                path = photos / name
                if name in sam.SMOKE_NAMES:
                    Image.new("RGB", (1280, 1024), (index + 1, 100, 200)).save(path)
                else:
                    path.write_bytes(f"unused-train-photo-{name}".encode())
                rows.append({"angle_degrees": angle, "path": f"photos/{name}",
                             "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            package.write_text(json.dumps({"schema": "ycb_object_evaluation_package_v1",
                                           "object_id": sam.OBJECT_ID, "training_inputs": rows}))
            checkpoint = base / "model.pt"
            checkpoint.write_bytes(b"test")
            source = base / "sam2"
            source.mkdir()
            output = base / "result"
            seen = []

            def fake_predictor(image, box):
                seen.append(tuple(box))
                mask = np.zeros((1024, 1280), dtype=bool)
                mask[320:530, 520:650] = True
                return mask

            # Synthetic package/model-source fixtures still exercise exact hash binding.
            model_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            source_sha = "a" * 64
            with (patch.object(common, "load_package", return_value=json.loads(package.read_text())),
                  patch.object(sam, "checkpoint_metadata", return_value={"checkpoint_sha256": model_sha,
                                                                          "checkpoint_bytes": 156_000_000}),
                  patch.object(sam, "source_digest", return_value=source_sha),
                  patch.object(common, "MIN_FREE_BYTES", 1)):
                report = sam.run_candidate(package, dataset, output, checkpoint, source,
                                           model_sha, source_sha, fake_predictor)
            self.assertEqual(report["status"], "complete_unreviewed")
            self.assertEqual([x["name"] for x in report["images"]], list(sam.SMOKE_NAMES))
            self.assertEqual(seen, [sam.BOX] * 3)
            self.assertEqual(len(list((output / "masks").iterdir())), 3)
            self.assertEqual(common.digest(output / "smoke_overlay.jpg"), report["overlay_sha256"])
            with self.assertRaises(FileExistsError):
                sam.run_candidate(package, dataset, output, checkpoint, source,
                                  model_sha, source_sha, fake_predictor)

    def test_checkpoint_pin_and_supervisor_freshness(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "100-180 MB"):
                sam.checkpoint_metadata(path, "0" * 64)
            prior = Path(temp) / "prior"
            prior.mkdir()
            (prior / "supervisor.json").write_text("untouched")
            with self.assertRaises(FileExistsError):
                sam.supervise([], prior)
            self.assertEqual((prior / "supervisor.json").read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
