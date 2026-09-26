"""Failure-mode checks for the bounded live dense runner."""

from array import array
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.tree_dense.run_dense import run


def write_depth(path, value):
    path.write_bytes(b"\x89MVE_IMAGE\n" + struct.pack("=4i",2,2,1,9) +
                     array("f",[value]*4).tobytes())


class RunDenseFailureTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path.cwd()/"build-opencv")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.scene=self.root/"scene"
        self.scene.mkdir()
        self.output=self.root/"run"
        source=self.root/"source.jpg"
        source.write_bytes(b"source")
        views=[]
        for index in range(5):
            folder=self.scene/"views"/f"view_{index:04d}.mve"
            folder.mkdir(parents=True)
            (folder/"meta.ini").write_text("[view]\n")
            (folder/"undistorted.png").write_bytes(b"image")
            views.append({"source":str(source)})
        (self.scene/"conversion.json").write_text(json.dumps({"views":views}))
        (self.scene/"synth_0.out").write_text("bundle")
        (self.scene/"measured-seeds.txt").write_text("seed")
        self.binary=Path("/usr/bin/true")

    def fake_runner(self, value, mutate=False):
        def callback(command, **_):
            index=int(next(part for part in command if part.startswith("--master-view=")).split("=")[1])
            write_depth(self.scene/"views"/f"view_{index:04d}.mve"/"depth-L0.mvei",value)
            if mutate and index==4:
                (self.scene/"synth_0.out").write_text("changed bundle")
            return SimpleNamespace(returncode=0)
        return callback

    def test_successful_exit_with_zero_depth_fails_after_both_maps(self):
        with patch("scripts.tree_dense.run_dense.subprocess.run",side_effect=self.fake_runner(0)):
            with self.assertRaisesRegex(RuntimeError,"dense_failed_empty_reference"):
                run(self.binary,self.scene,self.output)
        status=json.loads((self.output/"dense-status.json").read_text())
        self.assertEqual([x["valid_depth_pixels"] for x in status["results"]],[0,0])
        self.assertEqual(status["status"],"dense_failed_empty_reference")

    def test_refuses_existing_depth_before_running(self):
        write_depth(self.scene/"views"/"view_0001.mve"/"depth-L0.mvei",1)
        with patch("scripts.tree_dense.run_dense.subprocess.run") as called:
            with self.assertRaises(FileExistsError):
                run(self.binary,self.scene,self.output)
            called.assert_not_called()

    def test_mutated_input_fails_even_with_nonempty_depths(self):
        with patch("scripts.tree_dense.run_dense.subprocess.run",side_effect=self.fake_runner(1,True)):
            with self.assertRaisesRegex(RuntimeError,"dense_failed_input_changed"):
                run(self.binary,self.scene,self.output)
        status=json.loads((self.output/"dense-status.json").read_text())
        self.assertEqual([x["valid_depth_pixels"] for x in status["results"]],[4,4])
        self.assertFalse(status["inputs_unchanged"])
        self.assertEqual(status["status"],"dense_failed_input_changed")


if __name__=="__main__":
    unittest.main()
