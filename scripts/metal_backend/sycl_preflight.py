#!/usr/bin/env python3
"""Read-only, pinned AliceVision/AdaptiveCpp SYCL CPU build-contract audit.

This never configures, downloads, installs, compiles, or runs AliceVision.
Passing checks are necessary, not sufficient, for a working backend.
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path


ALICEVISION_REVISION = "8e4f0be76b250f0dd04b11d2a9a457d59d71b42c"
ADAPTIVECPP_REVISION = "493644ef7024a6dc64ae103b890b80feb014c8c0"
MIN_FREE_BYTES = 10 * 1024 ** 3
TARGETS = ("aliceVision_depthMapEstimation", "aliceVision_depthMapFiltering",
           "aliceVision_meshing")
REQUIRED_CACHE = {
    "ALICEVISION_BUILD_DEPENDENCIES": "OFF",
    "ALICEVISION_BUILD_MVS": "ON",
    "ALICEVISION_BUILD_SFM": "ON",
    "ALICEVISION_BUILD_SOFTWARE": "ON",
    "ALICEVISION_BUILD_TESTS": "OFF",
    "ALICEVISION_USE_CUDA": "OFF",
    "ALICEVISION_USE_SYCL": "ON",
    "ALICEVISION_DEPTHMAP_BACKEND": "SYCL",
    "ALICEVISION_BUILD_HDR": "OFF",
    "ALICEVISION_BUILD_SEGMENTATION": "OFF",
    "ALICEVISION_BUILD_PHOTOMETRICSTEREO": "OFF",
    "ALICEVISION_BUILD_PANORAMA": "OFF",
    "ALICEVISION_BUILD_LIDAR": "OFF",
    "ALICEVISION_USE_ONNX": "OFF",
    "ALICEVISION_USE_CCTAG": "OFF",
    "ALICEVISION_USE_APRILTAG": "OFF",
    "ALICEVISION_USE_POPSIFT": "OFF",
    "ALICEVISION_USE_ALEMBIC": "OFF",
    "ALICEVISION_USE_USD": "OFF",
    "ALICEVISION_USE_MESHSDFILTER": "OFF",
    "ALICEVISION_USE_OPENCV": "OFF",
}
SOURCE_MARKERS = {
    "CMakeLists.txt": ("ALICEVISION_BUILD_DEPENDENCIES", "add_subdirectory(src)"),
    "src/CMakeLists.txt": ("set(ACPP_TARGETS generic)", "find_package(AdaptiveCpp)",
                           "ALICEVISION_DEPTHMAP_BACKEND_SYCL"),
    "src/aliceVision/depthMap_sycl/CMakeLists.txt": ("aliceVision_depthMap_sycl", "USE_SYCL"),
    "src/software/pipeline/CMakeLists.txt": TARGETS,
}


def run_command(argv, *, cwd=None):
    try:
        process = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                 timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def version_tuple(output):
    match = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", output or "")
    return tuple(int(part or 0) for part in match.groups()) if match else None


def source_contract(path, revision, markers, runner=run_command):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return ["source is missing, not a directory, or is a symlink"]
    if runner(["git", "rev-parse", "HEAD"], cwd=path) != revision:
        return ["source git HEAD does not match pinned revision"]
    top_level = runner(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if top_level is None or Path(top_level).resolve() != path.resolve():
        return ["source path is not the git checkout root"]
    cleanliness = runner(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path)
    if cleanliness is None or cleanliness:
        return ["source tracked files are dirty or cleanliness could not be checked"]
    issues = []
    for relative, required in markers.items():
        file = path / relative
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 2 * 1024 ** 2:
            issues.append(f"missing, linked, or oversized source contract: {relative}")
            continue
        content = file.read_text(encoding="utf-8")
        for marker in required:
            if marker not in content:
                issues.append(f"source contract changed: {relative}: {marker}")
    return issues


def parse_cache(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 ** 2:
        raise ValueError("CMakeCache.txt missing, linked, or oversized")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in values:
            raise ValueError(f"duplicate CMake cache key: {key}")
        values[key] = value
    return values


def cache_contract(path, source):
    try:
        values = parse_cache(path)
    except ValueError as error:
        return [str(error)]
    issues = [f"{key} must equal {expected} (found {values.get(key)!r})"
              for key, expected in REQUIRED_CACHE.items() if values.get(key) != expected]
    if values.get("CMAKE_HOME_DIRECTORY") != str(Path(source).resolve()):
        issues.append("CMake cache points at a different AliceVision source")
    if not values.get("AdaptiveCpp_DIR") or not Path(values["AdaptiveCpp_DIR"]).is_dir():
        issues.append("AdaptiveCpp_DIR does not point to an installed CMake package")
    return issues


def adaptivecpp_cache_contract(path, source):
    try:
        values = parse_cache(path)
    except ValueError as error:
        return [str(error)]
    issues = []
    if values.get("CMAKE_HOME_DIRECTORY") != str(Path(source).resolve()):
        issues.append("AdaptiveCpp cache points at a different source")
    if values.get("ACPP_COMPILER_FEATURE_PROFILE") != "full":
        issues.append("AdaptiveCpp must use the full compiler feature profile")
    return issues


def native_executable_file(size, mode, platform):
    """Windows uses .exe naming; POSIX additionally requires owner execute."""
    return size > 0 and (platform == "nt" or bool(mode & stat.S_IXUSR))


def artifact_contract(path, *, platform=os.name):
    """Check native executable naming/permissions without invoking artifacts."""
    if platform not in ("nt", "posix"):
        raise ValueError("artifact platform must be nt or posix")
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return ["artifact directory missing, linked, or invalid"]
    issues = []
    expected_names = {name: name + ".exe" if platform == "nt" else name for name in TARGETS}
    found = {name: [] for name in expected_names.values()}
    queue = [(path, 0)]
    entries = 0
    while queue:
        directory, depth = queue.pop()
        for entry in os.scandir(directory):
            entries += 1
            if entries > 10000:
                return ["artifact tree exceeds 10000 entries"]
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False) and depth < 5:
                queue.append((Path(entry.path), depth + 1))
            elif entry.name in found and entry.is_file(follow_symlinks=False):
                found[entry.name].append(Path(entry.path))
    for name in TARGETS:
        matches = found[expected_names[name]]
        if len(matches) != 1 or not native_executable_file(
                matches[0].stat().st_size, matches[0].stat().st_mode, platform):
            issues.append(f"expected one nonempty, regular executable artifact: {name}")
    return issues


def inspect(workspace, *, alicevision=None, adaptivecpp=None, cache=None, adaptivecpp_cache=None,
            min_free_gib=10,
            artifacts=None, runner=run_command, disk=shutil.disk_usage):
    workspace = Path(workspace)
    if not workspace.is_dir():
        raise ValueError("workspace must be an existing directory")
    if not isinstance(min_free_gib, int) or not 0 <= min_free_gib <= 1024:
        raise ValueError("min_free_gib must be an integer from 0 to 1024")
    free = disk(workspace).free
    checks = {}
    checks["disk"] = [] if free >= min_free_gib * 1024 ** 3 else [f"less than {min_free_gib} GiB free"]
    cmake = version_tuple(runner(["cmake", "--version"]))
    llvm = version_tuple(runner(["llvm-config", "--version"]))
    checks["tools"] = []
    if cmake is None or cmake < (3, 30, 0):
        checks["tools"].append("CMake >= 3.30 is required")
    if llvm is None or not 15 <= llvm[0] <= 21:
        checks["tools"].append("official LLVM 15-21 must be independently verified")
    if runner(["ninja", "--version"]) is None:
        checks["tools"].append("Ninja is missing")
    if runner(["acpp", "--version"]) is None:
        checks["tools"].append("installed AdaptiveCpp acpp driver is missing")
    checks["alicevision_source"] = (source_contract(alicevision, ALICEVISION_REVISION, SOURCE_MARKERS, runner)
                                    if alicevision else ["AliceVision source not supplied"])
    checks["adaptivecpp_source"] = (source_contract(adaptivecpp, ADAPTIVECPP_REVISION,
                                                    {"doc/installing.md": ("ACPP_COMPILER_FEATURE_PROFILE", "generic")}, runner)
                                     if adaptivecpp else ["AdaptiveCpp source not supplied"])
    checks["cache"] = cache_contract(cache, alicevision) if cache and alicevision else ["CMake cache not supplied"]
    checks["adaptivecpp_cache"] = (adaptivecpp_cache_contract(adaptivecpp_cache, adaptivecpp)
                                   if adaptivecpp_cache and adaptivecpp else ["AdaptiveCpp cache not supplied"])
    checks["artifacts"] = artifact_contract(artifacts) if artifacts else ["build artifacts not supplied"]
    return {"schema": "alicevision_sycl_cpu_preflight_v1",
            "revisions": {"AliceVision": ALICEVISION_REVISION, "AdaptiveCpp": ADAPTIVECPP_REVISION},
            "free_bytes": free, "min_free_gib": min_free_gib, "checks": checks,
            "contract_met": all(not failures for failures in checks.values()),
            "build_ready": None,
            "limitation": "Static and local tool checks only; dependency closure, no-download configure, compilation and runtime are untested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--alicevision")
    parser.add_argument("--adaptivecpp")
    parser.add_argument("--cache")
    parser.add_argument("--adaptivecpp-cache")
    parser.add_argument("--artifacts")
    parser.add_argument("--min-free-gib", type=int, default=10)
    args = parser.parse_args()
    report = inspect(args.workspace, alicevision=args.alicevision, adaptivecpp=args.adaptivecpp,
                     cache=args.cache, adaptivecpp_cache=args.adaptivecpp_cache,
                     artifacts=args.artifacts, min_free_gib=args.min_free_gib)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["contract_met"] else 2)


if __name__ == "__main__":
    main()
