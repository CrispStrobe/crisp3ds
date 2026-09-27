import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import sam_m1_resume as m1
from scripts.object_motion import sam_point_resume as core
from scripts.object_motion import ycb_object_masks as common


def fixture():
    rows = [{"path": f"photos/NP3_{i * 6:03}.jpg", "sha256": f"{i:064x}"} for i in range(48)]
    prompts = [{"points_xy_label": [[600, 525, 1], [720, 620, 0], [520, 630, 0]]} for _ in rows]
    return rows, prompts


def valid_png():
    mask = np.zeros((1024, 1280), dtype=np.uint8)
    mask[470:580, 550:670] = 255
    data = io.BytesIO()
    Image.fromarray(mask, "L").save(data, "PNG")
    return mask, data.getvalue()


class SAMM1ResumeTests(unittest.TestCase):
    def test_resource_terminal_caps(self):
        allowed = (m1.BATCH_SECONDS, False, m1.mac.MAX_RSS_KIB, m1.mac.MIN_AVAILABLE_KIB,
                   m1.MAX_OUTPUT, m1.MAX_LOG, common.MIN_FREE_BYTES, 0)
        self.assertIsNone(m1.terminal_guard(*allowed))
        changed = [(0, m1.BATCH_SECONDS + 0.01), (1, True), (2, m1.mac.MAX_RSS_KIB + 1),
                   (3, m1.mac.MIN_AVAILABLE_KIB - 1), (4, m1.MAX_OUTPUT + 1),
                   (5, m1.MAX_LOG + 1), (6, common.MIN_FREE_BYTES - 1), (7, 1)]
        for index, value in changed:
            case = list(allowed)
            case[index] = value
            with self.subTest(index=index):
                self.assertIsNotNone(m1.terminal_guard(*case))

    def test_fake_predictor_publishes_mac_frame_and_preserves_failed_raw(self):
        rows, prompts = fixture()
        mask, _ = valid_png()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            (root / "contract.json").write_text("{}")
            with patch.object(m1.common, "verified_training_photo", return_value=None):
                record = m1.infer_and_publish(root, "001", rows[16], prompts[16],
                                              lambda *_: mask, root)
                self.assertEqual(record["producer"]["runner_sha256"], common.digest(m1.__file__))
                self.assertEqual(record["producer"]["host_backend"], "darwin_arm64_cpu")
                self.assertTrue((root / "receipts" / "NP3_096.jpg.json").is_file())
                with self.assertRaisesRegex(ValueError, "point labels"):
                    m1.infer_and_publish(root, "001", rows[17], prompts[17],
                                         lambda *_: np.zeros_like(mask), root)
            failed = root / "failures" / "NP3_102.jpg.json"
            self.assertEqual(json.loads(failed.read_text())["raw_mask_sha256"],
                             common.digest(root / "failures" / "NP3_102.jpg.raw.png"))

    def test_mac_full_inventory_after_partial_failure_and_resume(self):
        rows, prompts = fixture()
        _, data = valid_png()
        digest = hashlib.sha256(data).hexdigest()
        sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": digest,
                   "cleaned_mask_sha256": digest} for row in rows[:16]]
        qa = {"qa_sha256": "a" * 64}
        source = (rows, prompts, sealed, "b" * 64, "c" * 64, qa)
        calls = 0

        def prepare(root, expected):
            root = Path(root)
            if not root.exists():
                root.mkdir()
                for name in ("frames", "staging", "failures", "invocations", "receipts"):
                    (root / name).mkdir()
                core.exclusive_json(root / "contract.json", expected)
            elif json.loads((root / "contract.json").read_text()) != expected:
                raise ValueError("contract changed")

        def publish(root, invocation, row, prompt, prior=False, index=0):
            producer = ({"kind": "sealed_prior", "parent_sha256": core.PRIOR_SHA,
                         "batch_sha256": core.PRIOR_BATCH_SHA[index // 8]} if prior else
                        {"kind": "resume_inference", "invocation_id": invocation,
                         "runner_sha256": common.digest(m1.__file__), "host_backend": "darwin_arm64_cpu",
                         "resource_adapter_sha256": common.digest(m1.mac.__file__),
                         "contract_sha256": common.digest(Path(root) / "contract.json")})
            record = core.frame_record(Path(row["path"]).name, row["sha256"], prompt["points_xy_label"],
                                       digest, digest, producer)
            core.publish_frame(root, invocation, record, data, data)

        def import_prior(root, _prior, _rows, _prompts, _sealed, invocation, _dataset):
            for index in range(16):
                publish(root, "import-" + invocation, rows[index], prompts[index], True, index)

        def batch(args, root, _deadline):
            nonlocal calls
            names = args[args.index("--names") + 1].split(",")
            calls += 1
            selected = names[:2] if calls == 2 else names
            lookup = {Path(row["path"]).name: (row, prompt) for row, prompt in zip(rows, prompts)}
            for name in selected:
                publish(root, args[args.index("--invocation-id") + 1], *lookup[name])
            return {"status": "failed" if calls == 2 else "complete", "reason": "simulated interruption" if calls == 2 else None,
                    "returncode": 1 if calls == 2 else 0, "seconds": 0.01, "peak_worker_rss_kib": 1}

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory) / "run"
            common_args = (root, "001", "package", "dataset", "prompts", "checkpoint", "source",
                           "prior", "setup", "parity", "stage", "qa")
            with (patch.object(m1.mac, "host_preflight"), patch.object(m1, "inputs", return_value=source),
                  patch.object(m1.core, "prepare_root", side_effect=prepare),
                  patch.object(m1.core, "import_prior", side_effect=import_prior),
                  patch.object(m1.mac, "disk_free_both", return_value=common.MIN_FREE_BYTES),
                  patch.object(m1.mac, "available_kib", return_value=m1.mac.MIN_AVAILABLE_KIB),
                  patch.object(m1, "accepted_parity", return_value=qa),
                  patch.object(m1, "bounded_batch", side_effect=batch),
                  patch.object(m1.core, "review_sheets", return_value={})):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    m1.supervise(*common_args)
                self.assertEqual(len(list((root / "frames").iterdir())), 22)
                self.assertTrue((root / "invocations" / "001.json").is_file())
                resumed = list(common_args)
                resumed[1] = "002"
                result = m1.supervise(*resumed)
                self.assertEqual(result["status"], "complete_unreviewed")
                self.assertEqual(result["frames_after"], 48)
                inventory = json.loads((root / "complete_inventory.json").read_text())
                self.assertEqual(inventory["schema"], "sam21_mustard_point_mask_inventory_v1")
                self.assertEqual(len(inventory["images"]), 48)
                self.assertFalse(inventory["full_training_package_ready"])
                frame = root / "frames" / "NP3_096.jpg" / "frame.json"
                changed = json.loads(frame.read_text())
                changed["producer"]["host_backend"] = "linux"
                frame.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "append receipt"):
                    m1.verified_mac_frames(root, rows, prompts, sealed, m1.contract(rows, "b" * 64, "c" * 64, qa))


if __name__ == "__main__":
    unittest.main()
