"""Malformed fixture-input checks for the isolated optimizer executable."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/".local-tools/bundle-quality/quality-build-mpl/bundle_quality_optimize"
INPUT=ROOT/"build-opencv/bundle-quality-mpl/optimization-input.txt"


@unittest.skipUnless(BINARY.exists() and INPUT.exists(),"requires built guarded optimizer and fixture input")
class InputTests(unittest.TestCase):
    def run_bad(self,content,expected):
        with tempfile.TemporaryDirectory(dir=ROOT/".local-tools/tmp") as directory:
            source=Path(directory)/"input.txt"
            output=Path(directory)/"output.json"
            source.write_text(content)
            result=subprocess.run([str(BINARY),str(source),str(output)],capture_output=True,timeout=15)
            self.assertEqual(result.returncode,expected,result.stderr.decode())
            self.assertFalse(output.exists())

    def test_truncated_rotation(self):
        self.run_bad("C 3\n1 0\n",9)

    def test_bad_observation_index(self):
        lines=INPUT.read_text().splitlines()
        index=lines.index("O 2843")+1
        parts=lines[index].split()
        parts[0]="9"
        lines[index]=" ".join(parts)
        self.run_bad("\n".join(lines)+"\n",9)

    def test_nonpositive_initial_depth(self):
        lines=INPUT.read_text().splitlines()
        first_camera=lines[1].split()
        first_camera[-1]="0"
        lines[1]=" ".join(first_camera)
        self.run_bad("\n".join(lines)+"\n",13)


if __name__=="__main__":unittest.main()
