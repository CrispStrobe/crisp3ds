import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import fetch
except ImportError:
    import fetch


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, url: str, length: int):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(length)}

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, data: bytes, length: int | None = None):
        self.data = data
        self.length = len(data) if length is None else length

    def open(self, request, timeout):
        return FakeResponse(self.data, request.full_url, self.length)


class FetchTest(unittest.TestCase):
    def test_rejects_unsafe_inventory(self):
        template = ("Path = archive.7z\nType = 7z\n\n----------\nPath = {name}\n"
                    "Size = 4\nAttributes = A_ -rwxr-----\nEncrypted = -\n")
        good = fetch.parse_7z_listing(template.format(name="pipes/images/a.jpg"))
        self.assertEqual(good["expanded_bytes"], 4)
        for name in ("../bad", "/absolute", "x/../bad", "x\\bad", "x//bad"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fetch.parse_7z_listing(template.format(name=name))
        with self.assertRaises(ValueError):
            fetch.parse_7z_listing(template.format(name="a") +
                                    "\nPath = a\nSize = 1\nAttributes = A_ -rwxr-----\nEncrypted = -\n")
        with self.assertRaises(ValueError):
            fetch.parse_7z_listing(template.format(name="a").replace(
                "Attributes = A_ -rwxr-----", "Attributes = A_ lrwxrwxrwx"))
        with self.assertRaises(ValueError):
            fetch.parse_7z_listing(template.format(name="a").replace("Encrypted = -", "Encrypted = +"))

    def test_actual_7z_directory_and_special_metadata(self):
        listing = ("Path = archive.7z\nType = 7z\n\n----------\n"
                   "Path = pipes/scan1.ply\nSize = 7\nAttributes = A_ -rwxr-----\nEncrypted = -\n\n"
                   "Path = pipes\nSize = 0\nAttributes = D_ drwxr-x---\nEncrypted = -\n")
        result = fetch.parse_7z_listing(listing)
        self.assertEqual([member["directory"] for member in result["members"]], [False, True])
        for type_char in "lbcp":
            with self.subTest(type_char=type_char), self.assertRaises(ValueError):
                fetch.parse_7z_listing(listing.replace("A_ -rwxr-----",
                                                       f"A_ {type_char}rwxr-----"))

    def test_bounded_fetch_rejects_short_and_oversized_stream(self):
        name = next(iter(fetch.FILES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fetch, "FILES", {name: 4}):
                for payload, length in ((b"abc", 4), (b"abcde", 4), (b"abcd", 5)):
                    with self.subTest(payload=payload, length=length), self.assertRaises(ValueError):
                        fetch.fetch_one(root, name, 4, FakeOpener(payload, length),
                                        fetch.time.monotonic() + 10)
                    self.assertFalse((root / name).exists())
                    self.assertTrue((root / (name + ".partial")).exists())
                    (root / (name + ".partial")).unlink()
                record = fetch.fetch_one(root, name, 4, FakeOpener(b"abcd"),
                                         fetch.time.monotonic() + 10)
                self.assertEqual(record["sha256_measured"], hashlib.sha256(b"abcd").hexdigest())
                with self.assertRaises(FileExistsError):
                    fetch.fetch_one(root, name, 4, FakeOpener(b"abcd"),
                                    fetch.time.monotonic() + 10)
                self.assertEqual((root / name).read_bytes(), b"abcd")

    def test_fresh_destination_required_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "existing"
            root.mkdir()
            with self.assertRaises(FileExistsError):
                fetch.run(root, opener=FakeOpener(b""))

    def test_offline_review_preserves_original_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            name = "sample.7z"
            payload = b"archive"
            (root / name).write_bytes(payload)
            original = {"archives": [{"filename": name, "bytes": len(payload),
                                      "sha256_measured": hashlib.sha256(payload).hexdigest()}]}
            manifest_bytes = json.dumps(original).encode()
            (root / "manifest.json").write_bytes(manifest_bytes)
            fake_listing = {"member_count": 1, "expanded_bytes": 7,
                            "members": [{"path": "pipes/scan1.ply", "size_bytes": 7,
                                         "directory": False}]}
            with patch.object(fetch, "FILES", {name: len(payload)}), patch.object(
                    fetch, "inventory", return_value=fake_listing):
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    fetch.review_inventory(root, "0" * 64)
                reviewed = fetch.review_inventory(root, hashlib.sha256(manifest_bytes).hexdigest())
                self.assertEqual(reviewed["original_manifest_sha256"],
                                 hashlib.sha256(manifest_bytes).hexdigest())
                self.assertEqual((root / name).read_bytes(), payload)
                self.assertEqual((root / "manifest.json").read_bytes(), manifest_bytes)
                with self.assertRaises(FileExistsError):
                    fetch.review_inventory(root, hashlib.sha256(manifest_bytes).hexdigest())
            (root / "inventory-reviewed.json").unlink()
            (root / name).write_bytes(b"tamper!")
            with patch.object(fetch, "FILES", {name: len(payload)}), patch.object(
                    fetch, "inventory", return_value=fake_listing):
                with self.assertRaises(ValueError):
                    fetch.review_inventory(root, hashlib.sha256(manifest_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
