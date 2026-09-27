#!/usr/bin/env python3
"""Read-only host/toolchain inventory for the pinned Metal AliceVision audit.

This does not configure, compile, download, install, or claim that dependencies exist.
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path


def command(*argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def version_at_least(value, minimum):
    match = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", value or "")
    if not match:
        return False
    found = tuple(int(match.group(i) or 0) for i in (1, 2, 3))
    return found >= minimum


def inspect(workspace, runner=command, disk=shutil.disk_usage, system=platform.system, machine=platform.machine):
    workspace = Path(workspace).resolve()
    free_bytes = disk(workspace).free
    cmake = runner("cmake", "--version")
    ninja = runner("ninja", "--version")
    tools = {name: runner("xcrun", "--find", name) for name in ("metal", "metallib", "clang++")}
    sdk = runner("xcrun", "--sdk", "macosx", "--show-sdk-path")
    developer = runner("xcode-select", "-p")
    host = {"system": system(), "machine": machine(), "free_bytes": free_bytes}
    toolchain_present = (host["system"] == "Darwin" and host["machine"] == "arm64"
                         and bool(sdk and developer) and all(tools.values())
                         and version_at_least(cmake, (3, 28, 0)) and bool(ninja))
    return {
        "schema": "metal_alicevision_host_preflight_v1",
        "revision": "505197b6fc13e016718c12bc2731d3862593f611",
        "host": host,
        "tools": {"cmake": cmake, "ninja": ninja, "xcode_developer": developer,
                  "macos_sdk": sdk, **tools},
        "toolchain_present": toolchain_present,
        "disk_floor_10gib_met": free_bytes >= 10 * 1024**3,
        "build_ready": None,
        "build_ready_reason": "Unverified CMake dependencies, Metal shader compilation, and runtime; no build attempted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".", help="Existing path whose volume is checked")
    args = parser.parse_args()
    print(json.dumps(inspect(args.workspace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
