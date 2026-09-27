#!/usr/bin/env python3
"""Build/run the bounded standalone Metal compute oracle; no AliceVision sources."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(workspace):
    workspace = Path(workspace).resolve()
    if shutil.disk_usage(workspace).free < 10 * 1024**3 + 100 * 1024**2:
        raise RuntimeError("Insufficient free space: 10 GiB floor plus 100 MiB scratch allowance required")
    scratch = workspace / ".local-tools" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix(".mm")
    with tempfile.TemporaryDirectory(prefix="metal-smoke-", dir=scratch) as temporary:
        binary = Path(temporary) / "smoke"
        environment = os.environ.copy()
        environment["TMPDIR"] = temporary
        compile_command = ["xcrun", "clang++", "-x", "objective-c++", "-std=c++17",
                           str(source), "-framework", "Metal", "-framework", "Foundation",
                           "-o", str(binary)]
        compile_result = subprocess.run(compile_command, text=True, capture_output=True, timeout=45,
                                        check=False, env=environment)
        if compile_result.returncode:
            raise RuntimeError(f"Metal host compilation failed: {compile_result.stderr[-2000:]}")
        execute_result = subprocess.run([str(binary)], text=True, capture_output=True,
                                        timeout=15, check=False, env=environment)
        if execute_result.returncode:
            raise RuntimeError(f"Metal compute failed ({execute_result.returncode}): {execute_result.stderr[-2000:]}")
        return execute_result.stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    print(run(args.workspace))
