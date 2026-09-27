"""One-shot, evaluation-only OpenMVG GeometricFilter extension; default read-only.

The already built four-target fork is immutable input. --build is separately
approved and never stages photos or runs SfM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from scripts.classical_backend import openmvg_ligt_fork_build as four


ROOT = four.ROOT
TARGET = "openMVG_main_GeometricFilter"
MAIN_REL = Path("src/software/SfM/main_GeometricFilter.cpp")
MAIN_SHA = "087875a3122f64c328c365e7e73db752830ddb709ed7ecbda1a075c241ff40fa"
FOUR_MANIFEST_SHA = "b9dc7aeb98403acde7377e2b635312b138604cee0d3d74def652803399ed83c6"
FOUR_RECEIPT_SHA = "9aec9a7ac01c76f683416cd8eb47c57bbcbfa006a642348568a04298570078c1"
FOUR_LOG_SHA = "fe58812988f1ac98b9df1359f37d69b792e310eac6a00d8ec78be576a527466d"
FIFTH_CLOSURE_SHA = "a680fe2c358c97c6b6a4e56592034c9d6a587871bb05a26b06b909d284928e6d"
FIFTH_DELTA_SHA = (
    "08be7d878fe108784cc87f07fff21b56eb203ab8e0090295380e0a5dbfa87d19",
    "43cea7df03550a35ad5803edc531a8026cb7304d8213f1aee6ca707061d82d1f",
)
TIMEOUT = 900
LOG = ROOT / "logs/06-geometric-filter-build.log"
MANIFEST = ROOT / "geometric-filter-build-manifest.json"
RECEIPT = ROOT / "geometric-filter-license-receipt.json"
BINARY = ROOT / "build/Darwin-arm64-Release/openMVG_main_GeometricFilter"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command() -> list[str]:
    return ["ninja", "-C", str(ROOT / "build"), "-f", "frozen-build.ninja",
            "-j", "2", TARGET]


def target_closure() -> tuple[str, list[str]]:
    result = subprocess.run(["ninja", "-C", str(ROOT / "build"), "-t", "commands", TARGET],
                            capture_output=True, timeout=60, check=True)
    if len(result.stdout) > four.fork.LOG_CAP:
        raise RuntimeError("fifth-target closure exceeds 16 MiB")
    if sha_bytes(result.stdout) != FIFTH_CLOSURE_SHA:
        raise RuntimeError("fifth-target command closure changed")
    text = result.stdout.decode(errors="replace")
    if four.fork.FORBIDDEN_GRAPH.search(text) or four.FORBIDDEN_FETCH.search(text):
        raise RuntimeError("LiGT/patented or fetch/install input in fifth-target closure")
    old = set((ROOT / "logs/04-target-commands.log").read_text().splitlines())
    lines = text.splitlines()
    extra = [line for line in lines if line not in old]
    if (len(lines) != 367 or len(extra) != 2 or
            tuple(sha_bytes(line.encode()) for line in extra) != FIFTH_DELTA_SHA or
            str(ROOT / "source" / MAIN_REL) not in extra[0] or
            "-o Darwin-arm64-Release/openMVG_main_GeometricFilter" not in extra[1]):
        raise RuntimeError("fifth-target delta is not exactly one main compile plus one link")
    old_link = "\n".join(line for line in old if " -o Darwin-arm64-Release/openMVG_main_" in line)
    libraries = set(re.findall(r"\S+\.(?:a|dylib)\b", extra[1]))
    novel = {item for item in libraries if item not in old_link}
    if novel or "-framework Accelerate" not in old_link:
        raise RuntimeError(f"new library/framework in fifth-target link: {sorted(novel)}")
    return text, extra


def dry_run() -> dict:
    result = subprocess.run(["ninja", "-C", str(ROOT / "build"), "-f",
                             "frozen-build.ninja", "-n", TARGET], capture_output=True,
                            text=True, timeout=60, check=True)
    text = result.stdout + result.stderr
    if (len(text.encode()) > four.fork.LOG_CAP or "[1/2]" not in text or
            "[2/2]" not in text or "Re-running CMake" in text or "VerifyGlobs" in text or
            four.fork.FORBIDDEN_GRAPH.search(text)):
        raise RuntimeError("fifth-target dry-run not exactly safe two-step closure")
    return {"steps": 2, "cmake_regeneration_pending": False, "forbidden_inputs": 0}


def sealed_four_state() -> dict:
    required = {ROOT / "build-attempt-manifest.json": FOUR_MANIFEST_SHA,
                ROOT / "license-closure-receipt.json": FOUR_RECEIPT_SHA,
                ROOT / "logs/05-four-target-build.log": FOUR_LOG_SHA}
    for path, digest in required.items():
        if four.fork.base.sha256(path) != digest:
            raise RuntimeError(f"successful four-target receipt/log changed: {path}")
    four.postbuild_seals()
    receipt = json.loads((ROOT / "license-closure-receipt.json").read_text())
    for target in four.fork.base.TARGETS:
        binary = ROOT / "build/Darwin-arm64-Release" / target
        if four.fork.base.sha256(binary) != receipt["binaries"][target]["sha256"]:
            raise RuntimeError(f"four-target executable changed: {target}")
    for path, info in receipt["archives"].items():
        if four.fork.base.sha256(Path(path)) != info["sha256"]:
            raise RuntimeError(f"four-target archive changed: {path}")
    return {str(path): digest for path, digest in required.items()}


def preflight() -> dict:
    four.fork.reject_git_context()
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise RuntimeError("fixed configured fork root missing or symlinked")
    if any(path.exists() for path in (LOG, MANIFEST, RECEIPT, BINARY)):
        raise RuntimeError("fifth-target attempt already exists; no retry or cleanup")
    if any((ROOT / part).exists() for part in ("images48", "matches", "sparse")):
        raise RuntimeError("photo/SfM output unexpectedly present")
    expected_top = {"source", "build", "logs", "tmp", "fork-manifest.json",
                    "build-attempt-manifest.json", "license-closure-receipt.json"}
    if {item.name for item in ROOT.iterdir()} != expected_top:
        raise RuntimeError("fork root contains unexpected top-level state")
    inputs = sealed_four_state()
    main = ROOT / "source" / MAIN_REL
    if (four.fork.base.sha256(main) != MAIN_SHA or
            '#include "third_party/cmdLine/cmdLine.h"' not in main.read_text() or
            "Mozilla Public" not in main.read_text()):
        raise RuntimeError("fifth main source/license/header seal changed")
    ninja = (ROOT / "build/build.ninja").read_text(errors="replace")
    if f"build Darwin-arm64-Release/{TARGET}:" not in ninja:
        raise RuntimeError("fifth executable output rule missing")
    closure, extra = target_closure()
    planned = dry_run()
    if (four.fork.base.sha256(ROOT / "build/build.ninja") != four.GRAPH_SHA["ninja_sha256"] or
            four.fork.base.sha256(four.FROZEN_NINJA) != four.GRAPH_SHA["ninja_sha256"]):
        raise RuntimeError("generated/frozen Ninja graph changed during preflight")
    space = four.fork.capacity(ROOT, prospective=True)
    return {"root": str(ROOT), "input_sha256": inputs,
            "main_sha256": MAIN_SHA, "closure_sha256": sha_bytes(closure.encode()),
            "delta_sha256": [sha_bytes(line.encode()) for line in extra],
            "new_dependency_sources_or_libraries": 0, "dry_run": planned,
            "capacity": space, "command": command(), "timeout_seconds": TIMEOUT,
            "status": "read_only_preflight; evaluation_oracle_only"}


def run_logged(manifest: dict) -> None:
    record = {"command": command(), "log": str(LOG), "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": TIMEOUT}
    manifest["stages"].append(record)
    four.save_json(MANIFEST, manifest)
    started = time.monotonic()
    proc = None
    try:
        with LOG.open("xb") as stream:
            env = os.environ.copy()
            for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
                env[key] = str(ROOT / "tmp")
            for key in four.fork.GIT_CONTEXT_KEYS:
                env.pop(key, None)
            env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_CONFIG_GLOBAL"] = "/dev/null"
            proc = subprocess.Popen(command(), env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - started > TIMEOUT:
                    raise RuntimeError("fifth-target build timeout")
                if LOG.stat().st_size > four.fork.LOG_CAP:
                    raise RuntimeError("16 MiB fifth-target log cap exceeded")
                if (four.fork.base.sha256(ROOT / "build/build.ninja") != four.GRAPH_SHA["ninja_sha256"] or
                        four.fork.base.sha256(four.FROZEN_NINJA) != four.GRAPH_SHA["ninja_sha256"]):
                    raise RuntimeError("generated/frozen graph changed during fifth build")
                four.fork.capacity(ROOT)
                table = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
                if four.fork.base.process_rss_kib(proc.pid, table) > four.fork.RSS_CAP_KIB:
                    raise RuntimeError("4 GiB process-tree RSS cap exceeded")
                time.sleep(0.5)
            if proc.returncode or LOG.stat().st_size > four.fork.LOG_CAP:
                raise RuntimeError(f"fifth-target build exited {proc.returncode} or exceeded log cap")
        four.fork.capacity(ROOT)
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
        if LOG.exists() and LOG.stat().st_size > four.fork.LOG_CAP:
            with LOG.open("r+b") as stream:
                stream.truncate(four.fork.LOG_CAP)
            record["log_truncated_at_cap"] = True
        record["log_sha256"] = four.fork.base.sha256(LOG) if LOG.exists() else None
        four.save_json(MANIFEST, manifest)


def closure_receipt() -> dict:
    sealed_four_state()
    if not BINARY.is_file() or BINARY.is_symlink() or not os.access(BINARY, os.X_OK):
        raise RuntimeError("fifth executable missing")
    dylibs = four.checked_output(["/usr/bin/otool", "-L", str(BINARY)])
    prior = json.loads((ROOT / "license-closure-receipt.json").read_text())
    def deps(output: str) -> set[str]:
        return {line.strip().split(" ")[0] for line in output.splitlines()[1:] if line.strip()}

    old_deps = set().union(*(deps(value["otool_L"]) for value in prior["binaries"].values()))
    novel = sorted(deps(dylibs) - old_deps)
    if novel:
        raise RuntimeError(f"new fifth-target dynamic dependency: {novel}")
    if four.fork.base.sha256(ROOT / "source" / MAIN_REL) != MAIN_SHA:
        raise RuntimeError("fifth main changed after build")
    return {"schema": "openmvg_geometric_filter_license_receipt_v1",
            "target": TARGET, "binary": {"path": str(BINARY),
                                         "sha256": four.fork.base.sha256(BINARY),
                                         "otool_L": dylibs},
            "new_source_sha256": {str(ROOT / "source" / MAIN_REL): MAIN_SHA},
            "direct_cmdline_header_sha256": four.CMDLINE_SHA,
            "new_static_or_dynamic_libraries": [],
            "four_target_receipt_sha256": FOUR_RECEIPT_SHA,
            "decision": "evaluation oracle only; no commercial/App Store clearance",
            "unresolved_for_shipping": prior["unresolved_for_shipping"]}


def build_once() -> dict:
    checked = preflight()
    manifest = {"schema": "openmvg_geometric_filter_build_v1", "input": checked,
                "tree_cap_bytes": four.fork.TREE_CAP, "floor_bytes": four.fork.FLOOR,
                "rss_cap_kib": four.fork.RSS_CAP_KIB, "log_cap_bytes": four.fork.LOG_CAP,
                "stages": [], "status": "running"}
    four.save_json(MANIFEST, manifest)
    try:
        run_logged(manifest)
        four.postbuild_seals()
        receipt = closure_receipt()
        four.save_json(RECEIPT, receipt)
        four.fork.capacity(ROOT)
        manifest["receipt_sha256"] = four.fork.base.sha256(RECEIPT)
        manifest["status"] = "built_fifth_oracle_only"
        four.save_json(MANIFEST, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "stopped"
        manifest["stop_reason"] = str(exc)
        four.save_json(MANIFEST, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="approved one-shot fifth target only")
    args = parser.parse_args()
    print(json.dumps(build_once() if args.build else preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
