#!/usr/bin/env python3
"""Record source, adapter, binary, and fixed libELAS settings for local results."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


source_dir, binary, destination = map(Path, sys.argv[1:])
repo_dir = Path(__file__).resolve().parents[3]
adapter = repo_dir / "scripts/oracles/elas/oracle.cpp"
settings = {
    "preset": "MIDDLEBURY",
    "disp_min": 0,
    "disp_max": "ndisp-1 inclusive per invocation",
    "support_threshold": 0.95,
    "support_texture": 10,
    "candidate_stepsize": 5,
    "incon_window_size": 5,
    "incon_threshold": 5,
    "incon_min_support": 5,
    "add_corners": True,
    "grid_size": 20,
    "beta": 0.02,
    "gamma": 5,
    "sigma": 1,
    "sradius": 3,
    "match_texture": 0,
    "lr_threshold": 2,
    "speckle_sim_threshold": 1,
    "speckle_size": 200,
    "ipol_gap_width": 5000,
    "filter_median": True,
    "filter_adaptive_mean": False,
    "postprocess_only_left": False,
    "subsampling": False,
}
record = {
    "oracle": "libELAS",
    "source_url": "https://github.com/kou1okada/libelas",
    "source_commit": "862ee4ef30a070753e9b92965855562a4482da05",
    "git_archive_sha256": "ae32ab63888994910e6813b65242a8f05ffc86e448048d6fe67dcd1eb4830a8d",
    "compiler": subprocess.check_output(["/usr/bin/clang++", "--version"], text=True).splitlines()[0],
    "build_flags": ["-arch", "x86_64", "-msse3", "-std=c++17", "-O2", "-DNDEBUG", "-Wno-c++11-narrowing"],
    "runtime": "x86_64 via macOS Rosetta on Apple Silicon",
    "adapter_sha256": digest(adapter),
    "binary_sha256": digest(binary),
    "settings": settings,
    "license": "libELAS GPL-3.0-or-later; bundled Triangle 1.6 separate noncommercial redistribution terms; local research evaluation only",
}
destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
