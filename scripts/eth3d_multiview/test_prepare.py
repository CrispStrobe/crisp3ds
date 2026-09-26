import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

try:
    from . import prepare
except ImportError:
    import prepare


class PrepareValidationTest(unittest.TestCase):
    def test_inventory_rejects_unsafe_paths_types_and_size(self):
        def record(name, members):
            return {"filename": name, "bytes": prepare.ARCHIVES[name][0],
                    "sha256": prepare.ARCHIVES[name][1],
                    "inventory": {"member_count": len(members),
                                  "expanded_bytes": sum(m["size_bytes"] for m in members),
                                  "members": members}}
        names = list(prepare.ARCHIVES)
        normal = {"path": "pipes/images/a.JPG", "size_bytes": 2, "directory": False}
        archive = record(names[0], [normal])
        other = record(names[1], [{"path": "pipes/dslr_scan_eval/scan1.ply", "size_bytes": 3, "directory": False}])
        reviewed = {"archives": [archive, other], "total_expanded_bytes": 5}
        for bad in ("../escape", "/absolute", "pipes/../../escape", "x\\escape", "x//escape"):
            with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "unsafe"):
                archive["inventory"]["members"] = [{**normal, "path": bad}]
                prepare._inventory_records(reviewed)
        archive["inventory"]["members"] = [normal]
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            archive["inventory"]["members"] = [{**normal, "directory": "false"}]
            prepare._inventory_records(reviewed)
        archive["inventory"]["members"] = [normal]
        with self.assertRaisesRegex(ValueError, "expanded size"):
            archive["inventory"]["expanded_bytes"] = 4
            prepare._inventory_records(reviewed)

    def test_ply_header_stride_and_payload(self):
        header = (b"ply\nformat binary_little_endian 1.0\n"
                  b"element vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scan.ply"
            path.write_bytes(header + struct.pack("<6f", 1, 2, 3, 4, 5, 6))
            result = prepare.validate_ply(path)
            self.assertEqual(result["vertex_count"], 2)
            self.assertEqual(result["vertex_stride_bytes"], 12)
            path.write_bytes(header + struct.pack("<3f", 1, 2, 3))
            with self.assertRaisesRegex(ValueError, "payload"):
                prepare.validate_ply(path)
            path.write_bytes(header.replace(b"binary_little_endian", b"ascii") + b"0" * 24)
            with self.assertRaisesRegex(ValueError, "binary"):
                prepare.validate_ply(path)

    def test_alignment_rigid_finite_and_contained(self):
        def xml(matrix, filename="scan1.ply"):
            return (f'<MeshLabProject><MeshGroup><MLMesh filename="{filename}">'
                    f'<MLMatrix44>{matrix}</MLMatrix44></MLMesh></MeshGroup></MeshLabProject>')
        identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            scan = directory / "scan1.ply"
            scan.write_bytes(b"ply")
            alignment = directory / "scan_alignment.mlp"
            alignment.write_text(xml(identity))
            self.assertEqual(prepare.validate_alignment(alignment, scan)["scan_path"], "scan1.ply")
            for invalid in ("2 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                            "1 0 0 nan 0 1 0 0 0 0 1 0 0 0 0 1",
                            "-1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    alignment.write_text(xml(invalid))
                    prepare.validate_alignment(alignment, scan)
            alignment.write_text(xml(identity, "../scan1.ply"))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                prepare.validate_alignment(alignment, scan)

    def test_nonfinite_camera_and_image_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cameras = directory / "cameras.txt"
            cameras.write_text("0 PINHOLE 2 3 1 1 nan 1\n")
            with self.assertRaisesRegex(ValueError, "invalid PINHOLE"):
                prepare.validate_cameras(cameras)
            cameras.write_text("0 PINHOLE 2 3 1 1 1 1\n")
            parsed = prepare.validate_cameras(cameras)
            self.assertIn(0, parsed)
            images = directory / "images.txt"
            images.write_text("0 nan 0 0 0 0 0 0 0 DSC_0634.JPG\n\n")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                prepare.validate_images(images, parsed, {"pipes/images/DSC_0634.JPG": directory / "absent.JPG"})


if __name__ == "__main__":
    unittest.main()
