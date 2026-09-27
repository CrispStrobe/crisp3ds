"""Guarded OpenMVG v2.1 evaluation build and sealed 48-photo staging.

Default invocation is read-only. --build and --stage are deliberately separate
opt-in operations; neither runs the photo SfM pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


SOURCE_REV = "01193a245ee3c36458e650b1cf4402caad8983ef"
SUBMODULES = {
    "src/dependencies/cereal": ("ac168fe78ac499be0a745bf4a3253a9660572f8d", "https://github.com/openMVG-thirdparty/cereal.git"),
    "src/dependencies/glfw": ("eab31f228fd1872753c55fb438e3d88e51c0e9b2", "https://github.com/elmindreda/glfw.git"),
    "src/dependencies/osi_clp": ("a25a980c1af50cbd8962fe9d21035afa03653270", "https://github.com/openMVG-thirdparty/osi_clp.git"),
}
TARGETS = (
    "openMVG_main_SfMInit_ImageListing",
    "openMVG_main_ComputeFeatures",
    "openMVG_main_ComputeMatches",
    "openMVG_main_SfM",
)
GIB = 1024 ** 3
TREE_CAP = 2 * GIB
BUILD_CAP = 1536 * 1024 ** 2
STAGE_CAP = 512 * 1024 ** 2
FLOOR_WITH_MARGIN = 11 * GIB
RSS_CAP_KIB = 4 * 1024 ** 2
LOG_CAP = 16 * 1024 ** 2
TRAIN_LIST_SHA = "a81d1647109db27991c39c5fe1a9ae308c04081f8ada49d8b0f5ab1c15245544"
STAGE_REPORT_SHA = "bbcba624f6a51f82e0ebdebf6e7e4cda10804bc8df34cfd0d5b3242983a225d0"
FIRST_MANIFEST_SHA = "e328f483306fe0979291a8c74e0c3016a547283d37b725fe63e20958a97327e8"
FIRST_CONFIG_LOG_SHA = "c12becb2b57a76590f239730bc5960b19befb862cb310df65f50dc242d6696f6"
DEFAULT_ROOT = Path("/Volumes/backups/code/crisp3ds-data/openmvg-mustard-v21-001")
DEFAULT_INPUT = Path("/Volumes/backups/code/crisp3ds-data/mustard-sfm-train-001")
DEFAULT_INTERNAL = Path("/Users/christianstrobele/code/crisp3ds")
LICENSE_FILES = (
    "LICENSE", "COPYRIGHT.md", ".gitmodules", "src/CMakeLists.txt",
    "src/software/SfM/CMakeLists.txt", "src/nonFree/sift/SIFT_describer.hpp",
    "src/nonFree/sift/vl/sift.c",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            if item.is_symlink():
                raise RuntimeError(f"symlink in managed tree: {item}")
            if item.is_file():
                total += item.stat().st_size
    return total


def free_bytes(path: Path) -> int:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise RuntimeError(f"no existing parent for {path}")
        existing = existing.parent
    return shutil.disk_usage(existing).free


def capacity(root: Path, internal: Path, prospective: bool = False) -> dict:
    used = tree_bytes(root)
    external_free = free_bytes(root)
    internal_free = free_bytes(internal)
    # Reserve the remaining full cap before starting, then enforce the guard
    # floors continuously as the managed tree grows.
    remaining = max(0, TREE_CAP - used)
    required_external = FLOOR_WITH_MARGIN + (remaining if prospective else 0)
    if used > TREE_CAP:
        raise RuntimeError(f"managed tree cap exceeded: {used} > {TREE_CAP}")
    if external_free < required_external or internal_free < FLOOR_WITH_MARGIN:
        raise RuntimeError(
            f"disk guard: external {external_free}, need {required_external}; "
            f"internal {internal_free}, need {FLOOR_WITH_MARGIN}"
        )
    return {"managed_bytes": used, "external_free": external_free,
            "internal_free": internal_free, "external_required": required_external}


def paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in
            ("source", "build", "images48", "matches", "sparse", "logs", "tmp")}


def approved_root(root: Path) -> None:
    approved_parent = DEFAULT_ROOT.parent.resolve()
    if root.is_symlink() or root.resolve().parent != approved_parent:
        raise RuntimeError("managed root must be a direct child of the approved external data path")


def sealed_inputs(stage: Path) -> tuple[list[str], dict]:
    names_file, report_file = stage / "train-names.txt", stage / "stage-report.json"
    if sha256(names_file) != TRAIN_LIST_SHA or sha256(report_file) != STAGE_REPORT_SHA:
        raise RuntimeError("sealed TRAIN manifest/report hash mismatch")
    names = names_file.read_text().splitlines()
    report = json.loads(report_file.read_text())
    if (len(names) != 48 or len(set(names)) != 48 or
            any(not re.fullmatch(r"NP3_\d{3}\.jpg", n) for n in names) or
            names != report["train_names"] or
            set(names) != set(report["train_photo_sha256"]) or
            set(names) != set(report["cleaned_masks"])):
        raise RuntimeError("TRAIN names are not exactly the sealed 48")
    if set(stage.joinpath("images").iterdir()) != {stage / "images" / n for n in names}:
        raise RuntimeError("unexpected image in sealed input directory")
    if set(stage.joinpath("masks").iterdir()) != {stage / "masks" / f"{n}.png" for n in names}:
        raise RuntimeError("unexpected mask in sealed input directory")
    for name in names:
        image, mask = stage / "images" / name, stage / "masks" / f"{name}.png"
        if image.is_symlink() or mask.is_symlink():
            raise RuntimeError("symlink in sealed input")
        if sha256(image) != report["train_photo_sha256"][name]:
            raise RuntimeError(f"photo hash mismatch: {name}")
        if sha256(mask) != report["cleaned_masks"][name]["sha256"]:
            raise RuntimeError(f"mask hash mismatch: {name}")
    return names, report


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(["/usr/bin/git", "-C", str(source), *args], text=True).strip()


def source_inventory(source: Path) -> dict:
    if git_output(source, "rev-parse", "HEAD") != SOURCE_REV:
        raise RuntimeError("OpenMVG HEAD differs from sealed v2.1 revision")
    if git_output(source, "status", "--porcelain"):
        raise RuntimeError("OpenMVG source checkout is dirty")
    gitlinks = {}
    for rel, (revision, url) in SUBMODULES.items():
        listing = git_output(source, "ls-tree", "HEAD", rel).split()
        if len(listing) < 3 or listing[0] != "160000" or listing[2] != revision:
            raise RuntimeError(f"gitlink mismatch: {rel}")
        if git_output(source / rel, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"submodule checkout mismatch: {rel}")
        configured = git_output(source, "config", "--file", ".gitmodules", "--get", f"submodule.{rel}.url")
        if configured != url:
            raise RuntimeError(f"submodule URL mismatch: {rel}: {configured}")
        gitlinks[rel] = {"revision": revision, "url": url}
    files = {}
    for rel in LICENSE_FILES:
        item = source / rel
        if not item.is_file() or item.is_symlink():
            raise RuntimeError(f"source/license inventory file missing: {rel}")
        files[rel] = sha256(item)
    for rel in SUBMODULES:
        for item in sorted((source / rel).glob("*LICEN*")) + sorted((source / rel).glob("*COPYING*")):
            if item.is_file() and not item.is_symlink():
                files[str(item.relative_to(source))] = sha256(item)
    return {"source_revision": SOURCE_REV, "gitlinks": gitlinks, "files_sha256": files}


def reject_implicit_fetches(source: Path) -> None:
    forbidden = re.compile(r"ExternalProject_Add\s*\(|FetchContent_(?:Declare|MakeAvailable)\s*\(|file\s*\(\s*DOWNLOAD", re.I)
    for item in (source / "src").rglob("*"):
        if item.suffix.lower() not in (".cmake", ".txt") or not item.is_file():
            continue
        if item.name != "CMakeLists.txt" and item.suffix.lower() != ".cmake":
            continue
        if forbidden.search(item.read_text(errors="replace")):
            raise RuntimeError(f"CMake file can fetch during configure; review before build: {item}")


def configure_command(source: Path, build: Path) -> list[str]:
    return ["cmake", "-S", str(source / "src"), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DOpenMVG_BUILD_TESTS=OFF", "-DOpenMVG_BUILD_DOC=OFF",
            "-DOpenMVG_BUILD_EXAMPLES=OFF", "-DOpenMVG_BUILD_GUI_SOFTWARES=OFF",
            "-DOpenMVG_BUILD_OPENGL_EXAMPLES=OFF", "-DOpenMVG_USE_OPENCV=OFF",
            "-DOpenMVG_USE_LIGT=OFF", "-DOpenMVG_USE_OPENMP=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]


def build_command(build: Path) -> list[str]:
    return ["cmake", "--build", str(build), "--target", *TARGETS, "--parallel", "2"]


def resume_configure_command(source: Path, build: Path) -> list[str]:
    return configure_command(source, build) + ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]


def verify_build_outputs(build: Path) -> dict[str, str]:
    cache_file = build / "CMakeCache.txt"
    if not cache_file.is_file():
        raise RuntimeError("missing CMake cache; build is incomplete")
    cache = cache_file.read_text()
    if not re.search(r"^OpenMVG_USE_LIGT:BOOL=OFF$", cache, re.M):
        raise RuntimeError("LiGT CMake option was not disabled")
    binaries = {}
    for target in TARGETS:
        found = [p for p in build.rglob(target) if p.is_file() and os.access(p, os.X_OK)]
        if len(found) != 1:
            raise RuntimeError(f"expected exactly one built executable for {target}, got {len(found)}")
        binaries[target] = str(found[0])
    return binaries


def process_rss_kib(root_pid: int, process_table: str) -> int:
    rows = {}
    for line in process_table.splitlines():
        fields = line.split()
        if len(fields) >= 3 and all(part.isdigit() for part in fields[:3]):
            pid, ppid, rss = map(int, fields[:3])
            rows[pid] = (ppid, rss)
    active = {root_pid}
    while True:
        expanded = active | {pid for pid, (ppid, _) in rows.items() if ppid in active}
        if expanded == active:
            break
        active = expanded
    return sum(rows[pid][1] for pid in active if pid in rows)


def guarded_run(command: list[str], root: Path, internal: Path, timeout: int,
                env: dict, log_path: Path, manifest: dict,
                tree_limit: int = TREE_CAP) -> None:
    capacity(root, internal)
    if tree_bytes(root) > tree_limit:
        raise RuntimeError(f"build allocation exceeded: {tree_limit} bytes")
    record = {"command": command, "log": str(log_path), "timeout_seconds": timeout,
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    manifest["stages"].append(record)
    save_manifest(root, manifest)
    started = time.monotonic()
    proc = None
    try:
        with log_path.open("xb") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"timeout: {command[0]}")
                if log_path.stat().st_size > LOG_CAP:
                    raise RuntimeError(f"log cap exceeded: {log_path}")
                capacity(root, internal)
                if tree_bytes(root) > tree_limit:
                    raise RuntimeError(f"build allocation exceeded: {tree_limit} bytes")
                table = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
                rss = process_rss_kib(proc.pid, table)
                if rss > RSS_CAP_KIB:
                    raise RuntimeError(f"RSS cap exceeded: {rss} KiB")
                time.sleep(0.5)
            if log_path.stat().st_size > LOG_CAP:
                raise RuntimeError(f"log cap exceeded: {log_path}")
            if proc.returncode:
                raise RuntimeError(f"command failed ({proc.returncode}): {command}; see {log_path}")
        capacity(root, internal)
        if tree_bytes(root) > tree_limit:
            raise RuntimeError(f"build allocation exceeded: {tree_limit} bytes")
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
        record["returncode"] = proc.returncode if proc is not None else None
        if log_path.exists() and log_path.stat().st_size > LOG_CAP:
            with log_path.open("r+b") as log:
                log.truncate(LOG_CAP)
            record["log_truncated_at_cap"] = True
        record["log_bytes"] = log_path.stat().st_size if log_path.exists() else 0
        save_manifest(root, manifest)


def save_manifest(root: Path, manifest: dict) -> None:
    pending = root / "tmp" / "build-manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pending.replace(root / "build-manifest.json")


def build(root: Path, internal: Path) -> dict:
    approved_root(root)
    if root.exists():
        raise RuntimeError(f"refusing nonfresh build root: {root}")
    capacity(root, internal, prospective=True)
    if not shutil.which("cmake") or not shutil.which("ninja"):
        raise RuntimeError("cmake and ninja must already be installed; no package installation")
    p = paths(root)
    root.mkdir(parents=False)
    for key in ("logs", "tmp"):
        p[key].mkdir()
    manifest = {"schema": "openmvg_m1_build_v1", "source_revision": SOURCE_REV,
                "submodules": SUBMODULES, "tree_cap_bytes": TREE_CAP,
                "log_cap_bytes_each": LOG_CAP, "rss_cap_kib": RSS_CAP_KIB,
                "targets": TARGETS, "stages": []}
    save_manifest(root, manifest)
    env = os.environ.copy()
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        env[key] = str(p["tmp"])
    env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    guarded_run(["/usr/bin/git", "clone", "--no-checkout", "--filter=blob:none",
                 "https://github.com/openMVG/openMVG.git", str(p["source"])], root, internal, 1200, env,
                p["logs"] / "01-clone.log", manifest)
    guarded_run(["/usr/bin/git", "-C", str(p["source"]), "checkout", "--detach", SOURCE_REV], root, internal,
                300, env, p["logs"] / "02-checkout.log", manifest)
    guarded_run(["/usr/bin/git", "-C", str(p["source"]), "submodule", "update", "--init", "--depth", "1",
                 *SUBMODULES], root, internal, 1200, env, p["logs"] / "03-submodules.log", manifest)
    inventory = source_inventory(p["source"])
    manifest["source_inventory"] = inventory
    save_manifest(root, manifest)
    reject_implicit_fetches(p["source"])
    if tree_bytes(root) > BUILD_CAP:
        raise RuntimeError("source checkout exceeds 1.5 GiB build allocation")
    guarded_run(configure_command(p["source"], p["build"]), root, internal, 600, env,
                p["logs"] / "04-configure.log", manifest)
    cache = (p["build"] / "CMakeCache.txt").read_text()
    if not re.search(r"^OpenMVG_USE_LIGT:BOOL=OFF$", cache, re.M):
        raise RuntimeError("LiGT CMake option was not disabled")
    guarded_run(build_command(p["build"]), root, internal, 5400, env,
                p["logs"] / "05-build.log", manifest, BUILD_CAP)
    inventory["executables"] = verify_build_outputs(p["build"])
    manifest["executables"] = inventory["executables"]
    save_manifest(root, manifest)
    if tree_bytes(root) > BUILD_CAP:
        raise RuntimeError("build exceeds 1.5 GiB allocation")
    return inventory


def resume_preflight(root: Path, internal: Path) -> dict:
    approved_root(root)
    p = paths(root)
    if not root.is_dir() or not p["build"].is_dir() or p["images48"].exists():
        raise RuntimeError("resume requires the first incomplete build and no staged photos")
    if sha256(root / "build-manifest.json") != FIRST_MANIFEST_SHA:
        raise RuntimeError("first build manifest changed")
    if sha256(p["logs"] / "04-configure.log") != FIRST_CONFIG_LOG_SHA:
        raise RuntimeError("first configure log changed")
    if (p["logs"] / "05-build.log").exists() or (p["logs"] / "06-resume-configure.log").exists():
        raise RuntimeError("resume is not fresh or compile already attempted")
    manifest = json.loads((root / "build-manifest.json").read_text())
    if ([item["status"] for item in manifest["stages"]] !=
            ["completed", "completed", "completed", "stopped"] or
            len(manifest["stages"]) != 4):
        raise RuntimeError("unexpected first-attempt stage history")
    inventory = source_inventory(p["source"])
    if inventory != manifest["source_inventory"]:
        raise RuntimeError("source/license inventory changed since first attempt")
    reject_implicit_fetches(p["source"])
    space = capacity(root, internal, prospective=True)
    if tree_bytes(root) > BUILD_CAP:
        raise RuntimeError("incomplete build already exceeds 1.5 GiB allocation")
    return {"capacity": space, "inventory": inventory, "first_manifest_sha256": FIRST_MANIFEST_SHA,
            "first_config_log_sha256": FIRST_CONFIG_LOG_SHA,
            "resume_configure": resume_configure_command(p["source"], p["build"]),
            "resume_build": build_command(p["build"])}


def resume_configure_build(root: Path, internal: Path) -> dict:
    checked = resume_preflight(root, internal)
    p = paths(root)
    snapshot = root / "build-manifest-attempt1.json"
    if snapshot.exists():
        raise RuntimeError("first-manifest snapshot already exists")
    shutil.copyfile(root / "build-manifest.json", snapshot)
    if sha256(snapshot) != FIRST_MANIFEST_SHA:
        raise RuntimeError("first-manifest snapshot hash mismatch")
    manifest = json.loads(snapshot.read_text())
    manifest["resume_policy_override"] = "CMAKE_POLICY_VERSION_MINIMUM=3.5"
    manifest["first_attempt_manifest_sha256"] = FIRST_MANIFEST_SHA
    manifest["first_attempt_config_log_sha256"] = FIRST_CONFIG_LOG_SHA
    save_manifest(root, manifest)
    env = os.environ.copy()
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        env[key] = str(p["tmp"])
    env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    guarded_run(checked["resume_configure"], root, internal, 600, env,
                p["logs"] / "06-resume-configure.log", manifest, BUILD_CAP)
    cache = (p["build"] / "CMakeCache.txt").read_text()
    if (not re.search(r"^OpenMVG_USE_LIGT:BOOL=OFF$", cache, re.M) or
            not re.search(r"^CMAKE_POLICY_VERSION_MINIMUM:[^=]+=3\.5$", cache, re.M)):
        raise RuntimeError("LiGT/policy CMake cache seal failed")
    guarded_run(checked["resume_build"], root, internal, 5400, env,
                p["logs"] / "07-resume-build.log", manifest, BUILD_CAP)
    executables = verify_build_outputs(p["build"])
    manifest["executables"] = executables
    save_manifest(root, manifest)
    return {"inventory": checked["inventory"], "executables": executables}


def stage_photos(root: Path, internal: Path, stage: Path) -> dict:
    approved_root(root)
    p = paths(root)
    if not root.is_dir() or not p["source"].is_dir() or p["images48"].exists():
        raise RuntimeError("requires existing sealed build and fresh image destination")
    source_inventory(p["source"])
    verify_build_outputs(p["build"])
    names, report = sealed_inputs(stage)
    capacity(root, internal, prospective=True)
    p["images48"].mkdir()
    for name in names:
        for src, dst, expected in (
            (stage / "images" / name, p["images48"] / name, report["train_photo_sha256"][name]),
            (stage / "masks" / f"{name}.png", p["images48"] / f"{name[:-4]}_mask.png",
             report["cleaned_masks"][name]["sha256"]),
        ):
            shutil.copyfile(src, dst)
            if sha256(dst) != expected:
                raise RuntimeError(f"staged hash mismatch: {dst}")
            capacity(root, internal)
    if tree_bytes(p["images48"]) > STAGE_CAP:
        raise RuntimeError("staged images exceed 0.5 GiB allocation")
    return {"train_count": len(names), "staged_bytes": tree_bytes(p["images48"]),
            "train_names_sha256": TRAIN_LIST_SHA, "stage_report_sha256": STAGE_REPORT_SHA}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--internal", type=Path, default=DEFAULT_INTERNAL)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true", help="explicitly clone/configure/build; never implicit")
    action.add_argument("--resume-configure-build", action="store_true", help="explicitly retry only the sealed first configure with CMake 3.5 policy floor")
    action.add_argument("--stage", action="store_true", help="explicitly copy sealed 48 TRAIN inputs; no SfM")
    action.add_argument("--resume-preflight", action="store_true", help="read-only check of the sealed failed build")
    args = parser.parse_args()
    try:
        if args.build:
            result = {"action": "build", "inventory": build(args.root, args.internal)}
        elif args.resume_configure_build:
            result = {"action": "resume_configure_build", **resume_configure_build(args.root, args.internal)}
        elif args.stage:
            result = {"action": "stage", **stage_photos(args.root, args.internal, args.input)}
        elif args.resume_preflight:
            result = {"action": "read_only_resume_preflight", **resume_preflight(args.root, args.internal)}
        else:
            approved_root(args.root)
            p = paths(args.root)
            result = {"action": "read_only_preflight", "source_exists": p["source"].exists(),
                      "sealed_inputs": len(sealed_inputs(args.input)[0]),
                      "capacity": capacity(args.root, args.internal, prospective=True),
                      "source_revision": SOURCE_REV, "gitlinks": SUBMODULES,
                      "build_targets": TARGETS, "build_root": str(args.root),
                      "source_inventory": source_inventory(p["source"]) if p["source"].exists() else "pending clone"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"OpenMVG supervisor stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
