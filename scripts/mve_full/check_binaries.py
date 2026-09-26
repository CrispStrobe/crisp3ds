"""Bounded selected-CLI smoke check for native matrix builds."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


NAMES = ("makescene", "sfmrecon", "dmrecon", "scene2pset", "fssrecon", "meshclean")
TITLES = {"makescene": "MVE Makescene", "sfmrecon": "MVE SfM Reconstruction",
          "dmrecon": "MVE Depth Map Reconstruction", "scene2pset": "MVE Scene to Pointset",
          "fssrecon": "Floating Scale Surface Reconstruction", "meshclean": "MVE FSSR Mesh Cleaning"}


def check(paths: list[Path]) -> None:
    if len(paths) != len(NAMES):
        raise ValueError("expected six selected MVE executables")
    for name, path in zip(NAMES, paths):
        if name not in path.stem:
            raise ValueError(f"binary name mismatch: {path}")
        result = subprocess.run([str(path), "--help"], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=5, check=False)
        output = result.stdout[:65536].decode("utf-8", errors="replace")
        # Upstream help prints usage but exits 1 for unrecognized --help.
        if result.returncode not in (0, 1) or "Usage:" not in output or not output.startswith(TITLES[name]):
            raise ValueError(f"selected CLI did not report bounded usage: {name}, exit {result.returncode}")
        print(f"{name}: usage probe passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binaries", nargs=6, type=Path)
    args = parser.parse_args()
    check(args.binaries)


if __name__ == "__main__":
    main()
