"""Pinned sample fetch and bounded-stage contracts."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.upstream_control import continue_run, fetch, run


class UpstreamControlTests(unittest.TestCase):
    def setUp(self):
        (fetch.ROOT / ".local-tools/tmp").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=fetch.ROOT / ".local-tools/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_git_blob_hash_and_no_existing_destination(self):
        path = self.base / "blob"
        path.write_bytes(b"sample")
        self.assertEqual(fetch.git_blob(path), hashlib.sha1(b"blob 6\0sample").hexdigest())
        with self.assertRaises(FileExistsError):
            fetch.fetch(self.base)

    def test_selected_tree_rejects_missing_or_oversized_blob(self):
        tree_sha = "a" * 40
        entries = [{"path": path, "type": "blob", "size": 100, "sha": "b" * 40}
                   for path in fetch.SELECTED]
        commit = {"sha": fetch.COMMIT, "commit": {"tree": {"sha": tree_sha}}}
        with patch.object(fetch, "read_json", side_effect=[commit, {"sha": tree_sha, "tree": entries}]) as mocked:
            _, selected = fetch.selected_tree()
        self.assertEqual(set(selected), set(fetch.SELECTED))
        self.assertEqual(mocked.call_count, 2)
        entries[0] = dict(entries[0], size=fetch.FILE_CAP + 1)
        with patch.object(fetch, "read_json", side_effect=[commit, {"sha": tree_sha, "tree": entries}]):
            with self.assertRaisesRegex(ValueError, "metadata"):
                fetch.selected_tree()

    def test_stage_deadline_stops_child_and_saves_log(self):
        cmd = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]
        result = run.run_stage("timeout", cmd, self.base, 0, time.monotonic() + 0.1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue((self.base / "timeout.log").is_file())

    def test_real_pinned_sample_when_available(self):
        if not fetch.OUTPUT.exists():
            self.skipTest("pinned local upstream sample has not been fetched")
        manifest, digest = run.verify_sample(fetch.OUTPUT)
        self.assertEqual(manifest["commit"], fetch.COMMIT)
        self.assertEqual(len(manifest["files"]), len(fetch.SELECTED))
        self.assertEqual(len(digest), 64)

    def test_continuation_rejects_nonfailed_source_before_copy(self):
        source = self.base / "source"
        source.mkdir()
        (source / "report.json").write_text(json.dumps({"status": "complete", "stages": []}))
        output = self.base / "continuation"
        with patch.object(continue_run, "verify_sample", return_value=({}, "a" * 64)):
            with self.assertRaisesRegex(ValueError, "unexpected original failed run"):
                continue_run.continue_run(source=source, output=output, sample=self.base / "sample")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
