import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from .fetch_tree_subset import (REPO, EXCLUDED, MAX_PHOTO_BYTES, MIN_FREE_BYTES,
                                    fetch_one, parse_cameras, parse_images, safe_path,
                                    select_images, verify_digest)
except ImportError:
    from fetch_tree_subset import (REPO, EXCLUDED, MAX_PHOTO_BYTES, MIN_FREE_BYTES,
                                   fetch_one, parse_cameras, parse_images, safe_path,
                                   select_images, verify_digest)


class FetchTreeSubsetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_root = REPO / ".local-tools/tmp"
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    def test_path_containment_and_symlinks(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            self.assertEqual(safe_path(root, "images/photo.jpg"), root / "images/photo.jpg")
            for name in ("../other", "/absolute", "images/../other", "images\\other", "images//other"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(root, name)
            (root / "images").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, "images/photo.jpg")

    def test_git_blob_digest_detects_tampering(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            path = Path(temporary) / "cameras.txt"
            payload = b"some pinned source bytes\n"
            path.write_bytes(payload)
            oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            self.assertTrue(verify_digest(path, {"oid": oid}))
            path.write_bytes(b"different source bytes\n")
            self.assertFalse(verify_digest(path, {"oid": oid}))

    def test_fetch_enforces_budget_disk_and_hash(self):
        class Response(io.BytesIO):
            def __init__(self, data, declared):
                super().__init__(data)
                self.headers = {"Content-Length": str(declared)}

        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            payload = b"five"
            item = {"size": len(payload), "lfs": {"oid": hashlib.sha256(payload).hexdigest()}}
            with self.assertRaises(ValueError):
                fetch_one(root, "images/photo.jpg", item, remaining=len(payload)-1)
            with patch(f"{fetch_one.__module__}.shutil.disk_usage", return_value=SimpleNamespace(free=MIN_FREE_BYTES)):
                with self.assertRaises(ValueError):
                    fetch_one(root, "images/photo.jpg", item, remaining=100)
            with patch(f"{fetch_one.__module__}.urllib.request.urlopen", return_value=Response(b"wrong", 5)):
                with self.assertRaises(ValueError):
                    fetch_one(root, "images/photo.jpg", item, remaining=100)
            self.assertFalse((root / "images/photo.jpg").exists())
            with patch(f"{fetch_one.__module__}.urllib.request.urlopen", return_value=Response(payload, 4)):
                record = fetch_one(root, "images/photo.jpg", item, remaining=100)
            self.assertEqual(record["sha256"], item["lfs"]["oid"])

    def test_colmap_requires_valid_poses_and_empty_observation_lines(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            cameras = root / "cameras.txt"
            cameras.write_text("1 PINHOLE 5464 3640 4000 4000 2732 1820\n")
            self.assertEqual(parse_cameras(cameras)[1]["model"], "PINHOLE")
            cameras.write_text("1 PINHOLE 5464 3640 -1 4000 2732 1820\n")
            with self.assertRaises(ValueError):
                parse_cameras(cameras)
            images = root / "images.txt"
            images.write_text("1 1 0 0 0 0 0 0 1 The_Tree-1.jpg\n\n")
            self.assertEqual(parse_images(images)[0]["name"], "The_Tree-1.jpg")
            images.write_text("1 1 0 0 0 0 0 0 1 The_Tree-1.jpg\n10 20 3\n")
            with self.assertRaises(ValueError):
                parse_images(images)

    def test_selection_excludes_suspect_and_respects_photo_cap(self):
        images = [{"id": index, "name": f"The_Tree-{index}.jpg", "qvec": [1, 0, 0, 0],
                   "tvec": [index / 10, 0, 0], "camera_id": index} for index in range(1, 14)]
        images.append({"id": 88, "name": "The_Tree-88.jpg", "qvec": [1, 0, 0, 0],
                       "tvec": [0, 0, 0], "camera_id": 88})
        files = {f'images/{image["name"]}': {"size": 20_000_000} for image in images}
        chosen = select_images(images, files, count=10)
        self.assertEqual(len(chosen), 10)
        self.assertTrue(all(image["name"] not in EXCLUDED for image in chosen))
        self.assertLessEqual(sum(files[f'images/{image["name"]}']["size"] for image in chosen), MAX_PHOTO_BYTES)
        for item in files.values():
            item["size"] = 30_000_000
        with self.assertRaises(ValueError):
            select_images(images, files, count=10)


if __name__ == "__main__":
    unittest.main()
