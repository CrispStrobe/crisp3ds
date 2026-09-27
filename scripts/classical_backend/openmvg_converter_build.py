"""One-shot sixth OpenMVG target build for diagnostic SfM_Data conversion.

Default invocation is read-only. --build compiles only the sealed converter
target; it never reads mustard photos or runs a model conversion.
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

from scripts.classical_backend import openmvg_geometric_filter_build as fifth

ROOT = fifth.ROOT
TARGET = "openMVG_main_ConvertSfM_DataFormat"
MAIN_REL = Path("src/software/SfM/main_ConvertSfM_DataFormat.cpp")
MAIN_SHA = "f58a28392fe160059625c20727396e58f7b26111be6227f064e082fb1f3d7622"
FIFTH_MANIFEST_SHA = "4b3dc5ac03ea2232067bebabc0b4d41dc5bfb91076bf0e31211162f32842774b"
FIFTH_RECEIPT_SHA = "49db5dca401c827cac53a5b1337256effd198ed36de23de283b2b32548e53dc2"
FIFTH_LOG_SHA = "b1ca30c60c465046f9a31dda6f8f96b4df32c93a915beb29252c70547ab0d299"
FIFTH_BINARY_SHA = "49a5ca029356f3f8bb3eb68eaae58d98f77204b2b4cdf85f52aa0448f0ab1b77"
CLOSURE_SHA = "6dc06bbba6d3a285ce0dc3dfe0909da34408cdd3d2e3cc8cbd30baa7753121d0"
SORTED_DELTA_SHA = "5ed2caa1a42eb70991d008381855f4055a6cc057eae95f887e5a7a3a93a2afd9"
TIMEOUT = 900
LOG = ROOT / "logs/07-converter-build.log"
MANIFEST = ROOT / "converter-build-manifest.json"
RECEIPT = ROOT / "converter-license-receipt.json"
BINARY = ROOT / "build/Darwin-arm64-Release" / TARGET


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command() -> list[str]:
    return ["ninja", "-C", str(ROOT / "build"), "-f", "frozen-build.ninja",
            "-j", "2", TARGET]


def target_closure() -> dict:
    result = subprocess.run(["ninja", "-C", str(ROOT / "build"), "-t", "commands", TARGET],
                            capture_output=True, timeout=60, check=True)
    if len(result.stdout) > fifth.four.fork.LOG_CAP or sha_bytes(result.stdout) != CLOSURE_SHA:
        raise RuntimeError("converter command closure changed or exceeded 16 MiB")
    closure = result.stdout.decode(errors="replace")
    if fifth.four.fork.FORBIDDEN_GRAPH.search(closure) or fifth.four.FORBIDDEN_FETCH.search(closure):
        raise RuntimeError("LiGT/patented or fetch/install command in converter closure")
    prior_four = (ROOT / "logs/04-target-commands.log").read_text().splitlines()
    prior_fifth, _ = fifth.target_closure()
    prior = set(prior_four + prior_fifth.splitlines())
    lines = closure.splitlines()
    extra = sorted(line for line in lines if line not in prior)
    delta_bytes = ("\n".join(extra) + "\n").encode()
    if (len(lines) != 363 or len(extra) != 2 or sha_bytes(delta_bytes) != SORTED_DELTA_SHA or
            not any(str(ROOT / "source" / MAIN_REL) in line and " -c " in line for line in extra) or
            not any("-o Darwin-arm64-Release/" + TARGET in line for line in extra)):
        raise RuntimeError("converter delta is not one main compile plus one link")
    link = next(line for line in extra if "-o Darwin-arm64-Release/" + TARGET in line)
    old_links = "\n".join(line for line in prior if " -o Darwin-arm64-Release/openMVG_main_" in line)
    libraries = set(re.findall(r"\S+\.(?:a|dylib|tbd)\b", link))
    novel = sorted(libraries - set(re.findall(r"\S+\.(?:a|dylib|tbd)\b", old_links)))
    frameworks = set(re.findall(r"-framework\s+(\S+)", link))
    novel_frameworks = sorted(frameworks - set(re.findall(r"-framework\s+(\S+)", old_links)))
    link_flags = set(re.findall(r"(?<!\S)-l\S+", link))
    novel_flags = sorted(link_flags - set(re.findall(r"(?<!\S)-l\S+", old_links)))
    if novel or novel_frameworks or novel_flags or "Accelerate" not in frameworks:
        raise RuntimeError(f"converter link introduces library/framework: {novel + novel_frameworks + novel_flags}")
    return {"commands": len(lines), "closure_sha256": CLOSURE_SHA,
            "sorted_delta_sha256": SORTED_DELTA_SHA, "new_command_count": 2,
            "new_static_or_dynamic_libraries": []}


def dry_run() -> dict:
    result = subprocess.run(["ninja", "-C", str(ROOT / "build"), "-f",
                             "frozen-build.ninja", "-n", TARGET], capture_output=True,
                            text=True, timeout=60, check=True)
    output = result.stdout + result.stderr
    if (len(output.encode()) > fifth.four.fork.LOG_CAP or "[1/2]" not in output or
            "[2/2]" not in output or "Re-running CMake" in output or
            "VerifyGlobs" in output or fifth.four.fork.FORBIDDEN_GRAPH.search(output)):
        raise RuntimeError("converter dry-run not exactly two safe steps")
    return {"steps": 2, "cmake_regeneration_pending": False,
            "forbidden_inputs": 0}


def sealed_five_state() -> dict:
    four = fifth.sealed_four_state()
    fifth_files = {ROOT / "geometric-filter-build-manifest.json": FIFTH_MANIFEST_SHA,
                   ROOT / "geometric-filter-license-receipt.json": FIFTH_RECEIPT_SHA,
                   ROOT / "logs/06-geometric-filter-build.log": FIFTH_LOG_SHA}
    for item, digest in fifth_files.items():
        if item.is_symlink() or fifth.four.fork.base.sha256(item) != digest:
            raise RuntimeError(f"sealed fifth-target manifest/receipt/log changed: {item}")
    recorded = json.loads((ROOT / "geometric-filter-build-manifest.json").read_text())
    license_receipt = json.loads((ROOT / "geometric-filter-license-receipt.json").read_text())
    if (recorded.get("status") != "built_fifth_oracle_only" or
            recorded.get("receipt_sha256") != FIFTH_RECEIPT_SHA or
            license_receipt.get("binary", {}).get("sha256") != FIFTH_BINARY_SHA or
            fifth.four.fork.base.sha256(fifth.BINARY) != FIFTH_BINARY_SHA or
            fifth.closure_receipt() != license_receipt):
        raise RuntimeError("fifth-target binary/license closure changed")
    return {**four, **{str(path): digest for path, digest in fifth_files.items()},
            str(fifth.BINARY): FIFTH_BINARY_SHA}


def preflight() -> dict:
    fifth.four.fork.reject_git_context()
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise RuntimeError("sealed OpenMVG fork missing or symlinked")
    if any(item.exists() or item.is_symlink() for item in (LOG, MANIFEST, RECEIPT, BINARY)):
        raise RuntimeError("converter attempt already exists; no retry or cleanup")
    expected_top = {"source", "build", "logs", "tmp", "fork-manifest.json",
                    "build-attempt-manifest.json", "license-closure-receipt.json",
                    "geometric-filter-build-manifest.json", "geometric-filter-license-receipt.json"}
    if {item.name for item in ROOT.iterdir()} != expected_top:
        raise RuntimeError("fork tree has unexpected top-level state")
    inputs = sealed_five_state()
    main = ROOT / "source" / MAIN_REL
    if (main.is_symlink() or fifth.four.fork.base.sha256(main) != MAIN_SHA or
            '#include "third_party/cmdLine/cmdLine.h"' not in main.read_text() or
            "Mozilla Public" not in main.read_text()):
        raise RuntimeError("converter main source/license/header seal differs")
    for graph in (ROOT / "build/build.ninja", fifth.four.FROZEN_NINJA):
        if fifth.four.fork.base.sha256(graph) != fifth.four.GRAPH_SHA["ninja_sha256"]:
            raise RuntimeError(f"generated/frozen graph changed: {graph}")
    closure = target_closure()
    pending = dry_run()
    space = fifth.four.fork.capacity(ROOT, prospective=True)
    return {"root": str(ROOT), "target": TARGET, "input_sha256": inputs,
            "main_sha256": MAIN_SHA, "closure": closure, "dry_run": pending,
            "capacity": space, "command": command(), "timeout_seconds": TIMEOUT,
            "status": "read_only_preflight; evaluation_oracle_only"}


def run_logged(manifest: dict) -> None:
    record = {"command": command(), "log": str(LOG), "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": TIMEOUT}
    manifest["stages"].append(record)
    fifth.four.save_json(MANIFEST, manifest)
    started = time.monotonic()
    proc = None
    try:
        with LOG.open("xb") as stream:
            env = os.environ.copy()
            for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
                env[key] = str(ROOT / "tmp")
            for key in fifth.four.fork.GIT_CONTEXT_KEYS:
                env.pop(key, None)
            env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_CONFIG_GLOBAL"] = "/dev/null"
            proc = subprocess.Popen(command(), env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - started > TIMEOUT:
                    raise RuntimeError("converter build timeout")
                if LOG.stat().st_size > fifth.four.fork.LOG_CAP:
                    raise RuntimeError("16 MiB converter build log cap exceeded")
                for graph in (ROOT / "build/build.ninja", fifth.four.FROZEN_NINJA):
                    if fifth.four.fork.base.sha256(graph) != fifth.four.GRAPH_SHA["ninja_sha256"]:
                        raise RuntimeError("generated/frozen graph changed during build")
                fifth.four.fork.capacity(ROOT)
                table = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
                if fifth.four.fork.base.process_rss_kib(proc.pid, table) > fifth.four.fork.RSS_CAP_KIB:
                    raise RuntimeError("4 GiB process-tree RSS cap exceeded")
                time.sleep(0.5)
            if proc.returncode or LOG.stat().st_size > fifth.four.fork.LOG_CAP:
                raise RuntimeError(f"converter build exited {proc.returncode} or exceeded log cap")
        fifth.four.fork.capacity(ROOT)
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
        if LOG.exists() and LOG.stat().st_size > fifth.four.fork.LOG_CAP:
            with LOG.open("r+b") as stream:
                stream.truncate(fifth.four.fork.LOG_CAP)
            record["log_truncated_at_cap"] = True
        record["log_sha256"] = fifth.four.fork.base.sha256(LOG) if LOG.exists() else None
        fifth.four.save_json(MANIFEST, manifest)


def closure_receipt() -> dict:
    sealed_five_state()
    if not BINARY.is_file() or BINARY.is_symlink() or not os.access(BINARY, os.X_OK):
        raise RuntimeError("sixth executable missing")
    dylibs = fifth.four.checked_output(["/usr/bin/otool", "-L", str(BINARY)])
    four_receipt = json.loads((ROOT / "license-closure-receipt.json").read_text())
    fifth_receipt = json.loads((ROOT / "geometric-filter-license-receipt.json").read_text())
    def deps(output: str) -> set[str]:
        return {line.strip().split(" ")[0] for line in output.splitlines()[1:] if line.strip()}
    old = set().union(*(deps(value["otool_L"]) for value in four_receipt["binaries"].values()))
    old |= deps(fifth_receipt["binary"]["otool_L"])
    novel = sorted(deps(dylibs) - old)
    if novel:
        raise RuntimeError(f"new converter dynamic dependency: {novel}")
    return {"schema": "openmvg_converter_license_receipt_v1", "target": TARGET,
            "binary": {"path": str(BINARY), "sha256": fifth.four.fork.base.sha256(BINARY),
                       "otool_L": dylibs},
            "main_sha256": MAIN_SHA,
            "direct_cmdline_header_sha256": fifth.four.CMDLINE_SHA,
            "five_target_receipt_sha256": FIFTH_RECEIPT_SHA,
            "new_static_or_dynamic_libraries": [],
            "decision": "evaluation oracle only; no commercial/App Store clearance",
            "unresolved_for_shipping": four_receipt["unresolved_for_shipping"]}


def build_once() -> dict:
    checked = preflight()
    manifest = {"schema": "openmvg_converter_build_v1", "input": checked,
                "tree_cap_bytes": fifth.four.fork.TREE_CAP,
                "floor_bytes": fifth.four.fork.FLOOR,
                "rss_cap_kib": fifth.four.fork.RSS_CAP_KIB,
                "log_cap_bytes": fifth.four.fork.LOG_CAP,
                "stages": [], "status": "running"}
    fifth.four.save_json(MANIFEST, manifest)
    try:
        run_logged(manifest)
        fifth.four.postbuild_seals()
        receipt = closure_receipt()
        fifth.four.save_json(RECEIPT, receipt)
        fifth.four.fork.capacity(ROOT)
        manifest["receipt_sha256"] = fifth.four.fork.base.sha256(RECEIPT)
        manifest["status"] = "built_sixth_oracle_only"
        fifth.four.save_json(MANIFEST, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "stopped"
        manifest["stop_reason"] = str(exc)
        fifth.four.save_json(MANIFEST, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="one-shot sixth target only; requires review")
    args = parser.parse_args()
    print(json.dumps(build_once() if args.build else preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
