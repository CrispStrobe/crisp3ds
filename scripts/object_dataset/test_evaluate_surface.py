import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from scripts.object_dataset import evaluate


def write_triangle(path, z=0.0):
    header = ("ply\nformat binary_little_endian 1.0\n"
              "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
              "element face 1\nproperty list uchar int vertex_indices\nend_header\n").encode()
    vertices = b"".join(struct.pack("<fff", x, y, z) for x, y in ((0, 0), (1, 0), (0, 1)))
    path.write_bytes(header + vertices + struct.pack("<Biii", 3, 0, 1, 2))
    return path


def transform_record(reference, output, matrix=None, basis="external-calibration"):
    return {"matrix": matrix or [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "object_id": "test:triangle", "source_frame": "test reconstruction",
            "target_frame": "test reference", "reference_units": "millimetres",
            "provenance": "analytic test construction", "scale_provenance": "unit geometry in analytic test",
            "registration_basis": basis,
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


class SurfaceComparisonTests(unittest.TestCase):
    def test_explicit_width_ply_types_preserve_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = write_triangle(root / "legacy.ply", z=2.0)
            aliases = root / "aliases.ply"
            aliases.write_bytes(legacy.read_bytes().replace(b"property float ", b"property float32 ")
                                .replace(b"list uchar int ", b"list uint8 int32 "))
            _, lv, lf = evaluate.inspect_ply(legacy, geometry=True)
            report, av, af = evaluate.inspect_ply(aliases, geometry=True)
            import numpy as np
            np.testing.assert_array_equal(av, [[0, 0, 2], [1, 0, 2], [0, 1, 2]])
            np.testing.assert_array_equal(af, [[0, 1, 2]])
            np.testing.assert_array_equal(av, lv)
            np.testing.assert_array_equal(af, lf)
            self.assertEqual(report["zero_area_triangle_faces"], 0)

    def test_explicit_width_face_indices_keep_bounds_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_triangle(Path(temporary) / "mesh.ply")
            original = path.read_bytes()
            for name, payload in [(b"int32", struct.pack("<Biii", 3, 0, 1, -1)),
                                  (b"uint32", struct.pack("<BIII", 3, 0, 1, 2**32-1))]:
                header_and_vertices = original[:-13].replace(b"list uchar int ", b"list uint8 " + name + b" ")
                path.write_bytes(header_and_vertices + payload)
                with self.assertRaisesRegex(ValueError, "face index exceeds"):
                    evaluate.inspect_ply(path, geometry=True)

    def test_identical_mesh_samples_are_deterministically_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = write_triangle(root / "mesh.ply")
            report, vertices, faces = evaluate.inspect_ply(mesh, geometry=True)
            self.assertEqual(report["zero_area_triangle_faces"], 0)
            import numpy as np
            matrix = np.eye(4)
            a = evaluate.compare_meshes((vertices, faces), (vertices, faces), matrix,
                                        threshold=0.001, count=256, seed=42)
            b = evaluate.compare_meshes((vertices, faces), (vertices, faces), matrix,
                                        threshold=0.001, count=256, seed=42)
            self.assertEqual(a, b)
            self.assertLess(a["output_to_reference"]["rms"], 1e-7)
            self.assertEqual(a["f_score"], 1.0)

    def test_offset_planes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = write_triangle(root / "ref.ply")
            out = write_triangle(root / "out.ply", z=2)
            _, rv, rf = evaluate.inspect_ply(ref, geometry=True)
            _, ov, of = evaluate.inspect_ply(out, geometry=True)
            import numpy as np
            comparison = evaluate.compare_meshes((rv, rf), (ov, of), np.eye(4),
                                                  threshold=1, count=256)
            self.assertGreaterEqual(comparison["output_to_reference"]["median"], 2)
            self.assertEqual(comparison["f_score"], 0)

    def test_distinct_seed_reference_control_is_a_nonzero_sampling_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            mesh = write_triangle(Path(temporary) / "ref.ply")
            _, vertices, faces = evaluate.inspect_ply(mesh, geometry=True)
            control = evaluate.reference_sampling_control((vertices, faces),
                                                           threshold=0.001, count=128, seed=2027)
            self.assertEqual((control["first_seed"], control["second_seed"]), (2027, 2028))
            self.assertGreater(control["first_to_second"]["median"], 0)
            self.assertLess(control["f_score"], 1)

    def test_reflection_nonuniform_scale_and_wrong_object_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = write_triangle(root / "ref.ply")
            out = write_triangle(root / "out.ply")
            path = root / "transform.json"
            for linear in ((-1, 1, 1), (1, 2, 1)):
                record = transform_record(ref, out)
                record["matrix"][0][0], record["matrix"][1][1], record["matrix"][2][2] = linear
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "reflection|uniform"):
                    evaluate.load_sim3(path, ref, out)
            record = transform_record(ref, out)
            record["reference_sha256"] = "0" * 64
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "exact meshes"):
                evaluate.load_sim3(path, ref, out)

    def test_valid_external_scale_and_reference_fit_label_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = write_triangle(root / "ref.ply")
            out = write_triangle(root / "out.ply")
            path = root / "transform.json"
            record = transform_record(ref, out, basis="reference-fit")
            record["matrix"] = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]
            path.write_text(json.dumps(record))
            loaded, _, scale = evaluate.load_sim3(path, ref, out)
            self.assertEqual(scale, 2)
            self.assertEqual(loaded["registration_basis"], "reference-fit")

    def test_cli_scores_only_with_bound_transform_and_saves_fresh_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = write_triangle(root / "ref.ply")
            out = write_triangle(root / "out.ply")
            transform = root / "sim3.json"
            transform.write_text(json.dumps(transform_record(ref, out)))
            report = root / "report.json"
            command = [sys.executable, str(Path(evaluate.__file__)),
                       "--reference", str(ref), "--output", str(out),
                       "--transform", str(transform), "--threshold", "0.001",
                       "--samples", "128", "--seed", "2027", "--reference-self-control",
                       "--save-report", str(report)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            saved = json.loads(report.read_text())
            self.assertEqual(saved, json.loads(result.stdout))
            self.assertEqual(saved["sampled_surface_comparison"]["f_score"], 1.0)
            self.assertFalse(saved["sampled_surface_comparison"]["metric_accuracy_claim_allowed"])
            self.assertLess(saved["reference_sampling_control"]["f_score"], 1)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(saved, json.loads(report.read_text()))


if __name__ == "__main__":
    unittest.main()
