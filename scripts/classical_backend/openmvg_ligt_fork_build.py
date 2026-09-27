"""Resume-only FOUR-target OpenMVG LiGT-off oracle build; default is read-only.

--build requires separate approval. It never stages photos or runs SfM and does
not make a shipping-license determination.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from scripts.classical_backend import openmvg_ligt_fork_supervisor as fork


ROOT = fork.ROOT
CONFIGURED_MANIFEST_SHA = "5dc2a067b492216166338d8035d6db0dd12149672965a09d7c89635f1878d00a"
CACHE_SHA = "e40359026ce9a4cff60e8e5dfa6781b042ff62d79ac6867948ad8ef0f9814769"
LOG_SHA = {
    "01-patch-check.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "02-patch.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "03-configure.log": "0493235a98712ac837a1690678c7dfe780ae5fc66125db0f3b72e81a850cf18c",
    "04-target-commands.log": "847b30da11a793671d480b22901ffc871b793728d178e07a231295ca9df710a2",
}
GRAPH_SHA = {
    "compile_commands_sha256": "1b02a4bbba2963559f18c530aa8654944850ace70b7b828ce45f8d79616579e8",
    "ninja_sha256": "4dc33b80221bdf9f9933d0fba33e207b12bfaa0009d0e7560d70263a4a8d808b",
    "target_commands_sha256": LOG_SHA["04-target-commands.log"],
}
PATCHED_CMAKE_SHA = "da54e79ecfee54b1eccb429095de5d7a4172a125b4bf67c8a2709a6eb846a1ae"
CMDLINE_SHA = "aa3f87d1c33822745eff5d1c993905b3791308e9823f34f044796459e9cab5ba"
COIN_PARSER_SHA = "c00f7cc23e4e9697c5eae094029c50c21ce8b6e0f7eb532dc7a92e0f10aa66c0"
COIN_LICENSE_SHA = "0076749b626931ea5aaee25ddd5019fbfd96da78243cc0d5bd24fe246500981f"
CERES_CMAKE_SHA = "38f4c59a4e6e51507775fe18871fc32c879a1fc31ee339e8b648819202643555"
BUILD_TIMEOUT = 5400
BUILD_LOG = ROOT / "logs/05-four-target-build.log"
BUILD_MANIFEST = ROOT / "build-attempt-manifest.json"
RECEIPT = ROOT / "license-closure-receipt.json"
FROZEN_NINJA = ROOT / "build/frozen-build.ninja"
FORBIDDEN_FETCH = re.compile(r"(?:\bcurl\b|\bwget\b|\bbrew\b|\bgit\s+clone\b|\bpip\s+install\b)", re.I)
EXPECTED_DRY_RUN_STEPS = 386


def build_command() -> list[str]:
    return ["ninja", "-C", str(ROOT / "build"), "-f", "frozen-build.ninja",
            "-j", "2", *fork.base.TARGETS]


def verify_output_layout(ninja: str, source: Path) -> None:
    for target in fork.base.TARGETS:
        expected = f"build Darwin-arm64-Release/{target}:"
        if expected not in ninja:
            raise RuntimeError(f"four-target executable path changed: {target}")
        main = source / "src/software/SfM" / f"main_{target.removeprefix('openMVG_main_')}.cpp"
        if not main.is_file():
            raise RuntimeError(f"four-target main source missing: {main}")
    for archive in ("libopenMVG_multiview.a", "liblib_CoinUtils.a"):
        if f"build Darwin-arm64-Release/{archive}:" not in ninja:
            raise RuntimeError(f"audited archive path changed: {archive}")
    if "CoinModelUseful2.cpp.o" not in ninja:
        raise RuntimeError("CoinUtils parser missing from generated archive graph")


def dry_run_graph(manifest: str = "build.ninja") -> dict:
    command = ["ninja", "-C", str(ROOT / "build"), "-f", manifest, "-n",
               *fork.base.TARGETS]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    output = result.stdout + result.stderr
    if len(output.encode()) > fork.LOG_CAP:
        raise RuntimeError("Ninja dry-run exceeds 16 MiB")
    if ("Re-running CMake" in output or "VerifyGlobs" in output or
            fork.FORBIDDEN_GRAPH.search(output)):
        raise RuntimeError("dry-run would regenerate CMake or compile LiGT/patented input")
    if (f"[1/{EXPECTED_DRY_RUN_STEPS}]" not in output or
            f"[{EXPECTED_DRY_RUN_STEPS}/{EXPECTED_DRY_RUN_STEPS}]" not in output or
            any(f"{target}" not in output for target in fork.base.TARGETS)):
        raise RuntimeError("four-target dry-run closure differs from frozen 386 steps")
    return {"steps": EXPECTED_DRY_RUN_STEPS, "cmake_regeneration_pending": False,
            "forbidden_inputs": 0}


def preflight() -> dict:
    fork.reject_git_context()
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise RuntimeError("configured fixed fork root is missing or symlinked")
    if BUILD_LOG.exists() or BUILD_MANIFEST.exists() or RECEIPT.exists() or FROZEN_NINJA.exists():
        raise RuntimeError("build attempt already exists; no retry or cleanup")
    if any((ROOT / name).exists() for name in ("images48", "matches", "sparse")):
        raise RuntimeError("unexpected photo/SfM output in fork root")
    expected_top = {"source", "build", "logs", "tmp", "fork-manifest.json"}
    if {item.name for item in ROOT.iterdir()} != expected_top:
        raise RuntimeError("configured fork root has unexpected top-level inputs")
    seals = {ROOT / "fork-manifest.json": CONFIGURED_MANIFEST_SHA,
             ROOT / "build/CMakeCache.txt": CACHE_SHA}
    seals.update({ROOT / "logs" / name: digest for name, digest in LOG_SHA.items()})
    for path, expected in seals.items():
        if fork.base.sha256(path) != expected:
            raise RuntimeError(f"configured manifest/cache/log seal mismatch: {path}")
    manifest = json.loads((ROOT / "fork-manifest.json").read_text())
    if (manifest.get("status") != "configured_graph_clean_no_compile" or
            [stage.get("status") for stage in manifest.get("stages", [])] != ["completed"] * 5 or
            manifest.get("input", {}).get("root") != str(ROOT) or
            manifest.get("input", {}).get("seals", {}).get("patch_sha256") != fork.PATCH_SHA):
        raise RuntimeError("configured manifest stage/source contract mismatch")
    live_seals = fork.seal_patch_and_source()
    if live_seals != manifest["input"]["seals"]:
        raise RuntimeError("original source/patch/Eigen seals changed")
    patched = fork.verify_patched_fork(ROOT / "source", live_seals["source"])
    if patched != manifest["patched_fork"] or patched["patched_cmake_sha256"] != PATCHED_CMAKE_SHA:
        raise RuntimeError("fork source differs from configured one-patch state")
    commands = (ROOT / "logs/04-target-commands.log").read_text(errors="replace")
    graph = fork.graph_gate(ROOT / "build", ROOT / "logs/03-configure.log", commands)
    if graph != manifest["graph_gate"] or any(graph[key] != value for key, value in GRAPH_SHA.items()):
        raise RuntimeError("four-target generated graph changed")
    if FORBIDDEN_FETCH.search(commands):
        raise RuntimeError("generated build commands contain fetch/install tool")
    verify_output_layout((ROOT / "build/build.ninja").read_text(errors="replace"), ROOT / "source")
    dry_run = dry_run_graph()
    # Ninja dry-run must not regenerate; verify the configured graph remained sealed.
    if fork.base.sha256(ROOT / "build/build.ninja") != GRAPH_SHA["ninja_sha256"]:
        raise RuntimeError("generated graph changed during dry-run")
    for target in fork.base.TARGETS:
        if (ROOT / "build/Darwin-arm64-Release" / target).exists():
            raise RuntimeError(f"target binary already exists: {target}")
    space = fork.capacity(ROOT, prospective=True)
    return {"root": str(ROOT), "input_sha256": {str(path): digest for path, digest in seals.items()},
            "patched_cmake_sha256": PATCHED_CMAKE_SHA, "graph": graph,
            "capacity": space, "command": build_command(), "timeout_seconds": BUILD_TIMEOUT,
            "dry_run": dry_run,
            "status": "read_only_preflight; evaluation_oracle_only"}


def save_json(path: Path, data: dict) -> None:
    pending = ROOT / "tmp" / (path.name + ".pending")
    pending.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    pending.replace(path)


def run_build_log(manifest: dict) -> None:
    command = build_command()
    record = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
              "log": str(BUILD_LOG), "timeout_seconds": BUILD_TIMEOUT, "status": "running"}
    manifest["stages"].append(record)
    save_json(BUILD_MANIFEST, manifest)
    started = time.monotonic()
    proc = None
    try:
        with BUILD_LOG.open("xb") as log:
            env = os.environ.copy()
            for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
                env[key] = str(ROOT / "tmp")
            for key in fork.GIT_CONTEXT_KEYS:
                env.pop(key, None)
            env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_CONFIG_GLOBAL"] = "/dev/null"
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - started > BUILD_TIMEOUT:
                    raise RuntimeError("four-target build timeout")
                if BUILD_LOG.stat().st_size > fork.LOG_CAP:
                    raise RuntimeError("16 MiB build log cap exceeded")
                if (fork.base.sha256(FROZEN_NINJA) != GRAPH_SHA["ninja_sha256"] or
                        fork.base.sha256(ROOT / "build/build.ninja") != GRAPH_SHA["ninja_sha256"]):
                    raise RuntimeError("generated/frozen graph changed during build")
                fork.capacity(ROOT)
                table = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
                if fork.base.process_rss_kib(proc.pid, table) > fork.RSS_CAP_KIB:
                    raise RuntimeError("4 GiB process-tree RSS cap exceeded")
                time.sleep(0.5)
            if proc.returncode or BUILD_LOG.stat().st_size > fork.LOG_CAP:
                raise RuntimeError(f"build exited {proc.returncode} or exceeded log cap")
        fork.capacity(ROOT)
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "stopped"
        record["reason"] = str(exc)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record["returncode"] = proc.returncode if proc else None
        if BUILD_LOG.exists() and BUILD_LOG.stat().st_size > fork.LOG_CAP:
            with BUILD_LOG.open("r+b") as stream:
                stream.truncate(fork.LOG_CAP)
            record["log_truncated_at_cap"] = True
        record["log_sha256"] = fork.base.sha256(BUILD_LOG) if BUILD_LOG.exists() else None
        save_json(BUILD_MANIFEST, manifest)


def checked_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    if len(result.stdout.encode()) > fork.LOG_CAP:
        raise RuntimeError("dependency inspection output exceeds 16 MiB")
    return result.stdout


def postbuild_seals() -> None:
    if fork.base.sha256(ROOT / "fork-manifest.json") != CONFIGURED_MANIFEST_SHA:
        raise RuntimeError("configured manifest changed during build")
    if fork.base.sha256(ROOT / "build/CMakeCache.txt") != CACHE_SHA:
        raise RuntimeError("CMake cache changed during build (possible reconfigure)")
    if fork.base.sha256(FROZEN_NINJA) != GRAPH_SHA["ninja_sha256"]:
        raise RuntimeError("frozen Ninja graph changed during build")
    for name, digest in LOG_SHA.items():
        if fork.base.sha256(ROOT / "logs" / name) != digest:
            raise RuntimeError(f"configured log changed during build: {name}")
    graph = fork.graph_gate(ROOT / "build", ROOT / "logs/03-configure.log",
                            (ROOT / "logs/04-target-commands.log").read_text(errors="replace"))
    if any(graph[key] != value for key, value in GRAPH_SHA.items()):
        raise RuntimeError("generated graph changed during build")
    live_seals = fork.seal_patch_and_source()
    configured = json.loads((ROOT / "fork-manifest.json").read_text())
    if live_seals != configured["input"]["seals"]:
        raise RuntimeError("original source/patch/Eigen seals changed during build")
    patched = fork.verify_patched_fork(ROOT / "source", live_seals["source"])
    if patched != configured["patched_fork"] or patched["patched_cmake_sha256"] != PATCHED_CMAKE_SHA:
        raise RuntimeError("patched fork source changed during build")


def closure_receipt() -> dict:
    source = ROOT / "source/src"
    build = ROOT / "build"
    binaries = {}
    for target in fork.base.TARGETS:
        path = build / "Darwin-arm64-Release" / target
        if not path.is_file() or path.is_symlink() or not os.access(path, os.X_OK):
            raise RuntimeError(f"missing exact four-target executable: {path}")
        binaries[target] = {"path": str(path), "sha256": fork.base.sha256(path),
                            "otool_L": checked_output(["/usr/bin/otool", "-L", str(path)])}
    notices = {
        "LGPLv3+ cmdLine header": (source / "third_party/cmdLine/cmdLine.h", CMDLINE_SHA),
        "Bison exception CoinUtils parser": (source / "dependencies/osi_clp/CoinUtils/src/CoinModelUseful2.cpp", COIN_PARSER_SHA),
        "EPL CoinUtils license": (source / "dependencies/osi_clp/CoinUtils/LICENSE", COIN_LICENSE_SHA),
        "vendored Ceres CMake": (source / "third_party/ceres-solver/CMakeLists.txt", CERES_CMAKE_SHA),
    }
    for label, (path, digest) in notices.items():
        if fork.base.sha256(path) != digest:
            raise RuntimeError(f"license evidence changed: {label}")
    mains = {}
    for target in fork.base.TARGETS:
        name = target.removeprefix("openMVG_main_")
        path = source / f"software/SfM/main_{name}.cpp"
        if '#include "third_party/cmdLine/cmdLine.h"' not in path.read_text():
            raise RuntimeError(f"cmdLine inclusion changed: {path}")
        mains[target] = fork.base.sha256(path)
    multiview = build / "Darwin-arm64-Release/libopenMVG_multiview.a"
    coin_archive = build / "Darwin-arm64-Release/liblib_CoinUtils.a"
    mv_members = checked_output(["/usr/bin/ar", "-t", str(multiview)]).splitlines()
    coin_members = checked_output(["/usr/bin/ar", "-t", str(coin_archive)]).splitlines()
    if any("LiGT" in item for item in mv_members):
        raise RuntimeError("LiGT member present in multiview archive")
    if not any("CoinModelUseful2" in item for item in coin_members):
        raise RuntimeError("audited CoinUtils parser missing from archive")
    return {"schema": "openmvg_four_target_license_receipt_v1", "binaries": binaries,
            "archives": {str(multiview): {"sha256": fork.base.sha256(multiview),
                                         "members": mv_members},
                         str(coin_archive): {"sha256": fork.base.sha256(coin_archive),
                                             "members": coin_members}},
            "notices_sha256": {label: digest for label, (_, digest) in notices.items()},
            "cli_mains_sha256": mains,
            "unresolved_for_shipping": ["LGPLv3+ cmdLine.h versus upstream MPL summary",
                                        "EPL Clp/Osi/CoinUtils and Bison exception",
                                        "Ceres/Eigen sparse-license and dynamic Homebrew deps",
                                        "VLFeat nonFree provenance/patent review"],
            "decision": "evaluation oracle only; no commercial/App Store clearance"}


def build_once() -> dict:
    checked = preflight()
    manifest = {"schema": "openmvg_ligt_four_target_build_v1", "input": checked,
                "tree_cap_bytes": fork.TREE_CAP, "floor_bytes": fork.FLOOR,
                "rss_cap_kib": fork.RSS_CAP_KIB, "log_cap_bytes": fork.LOG_CAP,
                "stages": [], "status": "running"}
    save_json(BUILD_MANIFEST, manifest)
    try:
        shutil.copyfile(ROOT / "build/build.ninja", FROZEN_NINJA)
        if fork.base.sha256(FROZEN_NINJA) != GRAPH_SHA["ninja_sha256"]:
            raise RuntimeError("frozen Ninja manifest differs from sealed graph")
        manifest["frozen_dry_run"] = dry_run_graph("frozen-build.ninja")
        save_json(BUILD_MANIFEST, manifest)
        run_build_log(manifest)
        postbuild_seals()
        receipt = closure_receipt()
        save_json(RECEIPT, receipt)
        fork.capacity(ROOT)
        manifest["receipt_sha256"] = fork.base.sha256(RECEIPT)
        manifest["status"] = "built_oracle_only"
        save_json(BUILD_MANIFEST, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "stopped"
        manifest["stop_reason"] = str(exc)
        save_json(BUILD_MANIFEST, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="approved one-shot four-target build")
    args = parser.parse_args()
    print(json.dumps(build_once() if args.build else preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
