#!/usr/bin/env python3
"""Synthetic geometry, split-leak, and malformed-model checks."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    from scripts.sparse_verify.verify import images, project, pair_sampson, sha256, three_view, verify
    from scripts.sparse_verify.model_only import audit as audit_model
except ModuleNotFoundError:
    from verify import images, project, pair_sampson, sha256, three_view, verify
    from model_only import audit as audit_model


CAM = dict(model="SIMPLE_RADIAL", width=100, height=100,
           params=[100.0, 50.5, 50.5, 0.1])
RA = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
A = dict(R=RA, t=[0, 0, 0])
B = dict(R=RA, t=[-1, 0, 0])


def uv(xyz, camera):
    return project(CAM, [v+t for v, t in zip(xyz, camera["t"])])


class SparseVerifyTest(unittest.TestCase):
    def test_distorted_epipolar_and_outlier(self):
        p = [0.5, 0.2, 4]
        good = pair_sampson(A, B, CAM, CAM, uv(p, A), uv(p, B))
        bad = pair_sampson(A, B, CAM, CAM, uv(p, A),
                           (uv(p, B)[0], uv(p, B)[1]+10))
        self.assertLess(good, 1e-9)
        self.assertGreater(bad, 4)
        self.assertAlmostEqual(uv([0, 0, 5], A)[0], 50.5)

    def test_closed_three_view_prediction(self):
        p = [0.5, 0.2, 4]
        poses = {"a": dict(A, center=[0, 0, 0], camera_id=1),
                 "b": dict(B, center=[1, 0, 0], camera_id=1),
                 "c": dict(R=RA, t=[0, -1, 0], center=[0, 1, 0], camera_id=1)}
        f = {name: [list(uv(p, pose))+[1.0]] for name, pose in poses.items()}
        result = three_view([(("a", 0), ("b", 0), ("c", 0))], poses, {1: CAM}, f)
        self.assertEqual(result["registered_cycles"], 1)
        self.assertEqual(result["third_view_reprojection"]["finite_count"], 1)
        self.assertLess(result["third_view_reprojection"]["median_px"], 1e-8)
        f["c"][0][1] += 10
        bad = three_view([(("a", 0), ("b", 0), ("c", 0))], poses, {1: CAM}, f)
        self.assertGreater(bad["third_view_reprojection"]["median_px"], 4)

    def fixture(self, root):
        model = root/"model"; model.mkdir()
        (model/"cameras.txt").write_text("1 SIMPLE_RADIAL 100 100 100 50.5 50.5 0.1\n")
        pts = [[0, 0, 5], [1, 0, 5]]
        held = [0.5, 0.2, 4]
        manifest_images = []
        for i in range(10):
            name = f"{i:02}.png"; path = root/name; path.write_bytes(bytes([i]))
            cam = A if i == 0 else B
            xy = [list(uv(p, cam))+[1.0] for p in pts+[held]]
            manifest_images.append(dict(name=name, path=str(path), sha256=sha256(path),
                                        width=100, height=100, features=xy,
                                        heldout_ids=[2], compact_to_original=[0, 1]))
        headers = []
        for i, cam in ((0, A), (1, B)):
            xy = manifest_images[i]["features"]
            header = f"{i+1} 1 0 0 0 {' '.join(map(str, cam['t']))} 1 {i:02}.png"
            obs = " ".join(f"{xy[j][0]} {xy[j][1]} {j+1}" for j in range(2))
            headers.extend([header, obs])
        (model/"images.txt").write_text("\n".join(headers)+"\n")
        (model/"points3D.txt").write_text(
            "1 0 0 5 255 255 255 0 1 0 2 0\n"
            "2 1 0 5 255 255 255 0 1 1 2 1\n")
        manifest = dict(images=manifest_images,
                        training_matches=[dict(image1="00.png", feature1=j,
                                               image2="01.png", feature2=j) for j in range(2)],
                        heldout_pairs=[dict(image1="00.png", feature1=2,
                                            image2="01.png", feature2=2)])
        path = root/"manifest.json"; path.write_text(json.dumps(manifest))
        return path, model, manifest

    def test_full_audit_and_split_tamper(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path, model, manifest = self.fixture(Path(temp))
            report = verify(path, model)
            self.assertEqual(report["status"], "pass", report["errors"])
            self.assertEqual(report["model"]["track_observations"], 4)
            self.assertLess(report["heldout"]["all"]["median_px"], 1e-9)
            manifest["training_matches"].append(dict(image1="00.png", feature1=2,
                                                      image2="01.png", feature2=2))
            path.write_text(json.dumps(manifest))
            self.assertIn("heldout feature leaked", " ".join(verify(path, model)["errors"]))

    def test_behind_camera_fails(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path, model, _ = self.fixture(Path(temp))
            pts = (model/"points3D.txt").read_text().replace("1 0 0 5", "1 0 0 -5", 1)
            (model/"points3D.txt").write_text(pts)
            report = verify(path, model)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["model"]["nonpositive_depth_observations"], 2)

    def test_empty_observation_line_is_valid(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path = Path(temp)/"images.txt"
            path.write_text("# comment\n1 1 0 0 0 0 0 0 1 empty.png\n\n")
            parsed = images(path)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[1]["xy"], [])

    def test_multiple_same_image_observations_are_reported(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path, model, _ = self.fixture(Path(temp))
            images_path = model/"images.txt"
            records = images_path.read_text().splitlines()
            first = records[1].split()
            records[1] += f" {first[0]} {first[1]} 1"
            second = records[3].split()
            second[2] = "-1"
            records[3] = " ".join(second)
            images_path.write_text("\n".join(records)+"\n")
            p = model/"points3D.txt"
            p.write_text(p.read_text().replace("1 0 2 0", "1 0 1 2", 1))
            model_report = audit_model(model)
            self.assertEqual(model_report["multi_observation_same_image_tracks"], [1])
            self.assertEqual(model_report["integrity_status"], "pass", model_report["errors"])

    def test_model_only_audit(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            _, model, _ = self.fixture(Path(temp))
            result = audit_model(model)
            self.assertEqual(result["integrity_status"], "pass", result["errors"])
            self.assertFalse(result["quality_accepted"])

    def test_inverse_distortion_failure_is_counted(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path, model, manifest = self.fixture(Path(temp))
            (model/"cameras.txt").write_text("1 SIMPLE_RADIAL 100 100 100 50.5 50.5 -1\n")
            manifest["images"][0]["features"][2][:2] = [200.5, 50.5]
            path.write_text(json.dumps(manifest))
            report = verify(path, model)
            self.assertEqual(report["heldout"]["all"]["invalid_count"], 1)

    def test_colocated_orientation_clone_invalidates_holdout(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path, model, manifest = self.fixture(Path(temp))
            train_xy = manifest["images"][0]["features"][0]
            manifest["images"][0]["features"][2][:2] = [train_xy[0]+0.02,
                                                         train_xy[1]-0.02]
            path.write_text(json.dumps(manifest))
            report = verify(path, model)
            self.assertFalse(report["heldout_valid"])
            self.assertEqual(report["split_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()
