#!/usr/bin/env python3
"""Read-only, bounded preflight for a separate COLMAP/OpenMVS research oracle."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / ".local-tools/test-data/tree-subset"
GIB = 1024 ** 3
MIN_FREE = 10 * GIB
MAX_MANIFEST = 2 * 1024 * 1024
MAX_INPUT_BYTES = GIB
MAX_FILES = 256
MAX_SINGLE_BYTES = 60 * 1000 * 1000
MAX_PROBE_BYTES = 64 * 1024
TOOLS = ("colmap", "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
         "RefineMesh", "TextureMesh")
STAGES = (
    {"name": "colmap_sparse", "requires": ["colmap"], "settings": [
        "feature_extractor: --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 2000",
        "exhaustive_matcher: --FeatureMatching.use_gpu 0",
        "mapper: default settings; record actual installed version/help before execution"]},
    {"name": "colmap_undistort", "requires": ["colmap"], "settings": [
        "image_undistorter: --output_type COLMAP --max_image_size 2000"]},
    {"name": "openmvs_import", "requires": ["InterfaceCOLMAP"], "settings": [
        "InterfaceCOLMAP: import COLMAP undistorted sparse model and images"]},
    {"name": "openmvs_dense", "requires": ["DensifyPointCloud"], "settings": [
        "DensifyPointCloud: CPU build, default settings; verify version-specific options"]},
    {"name": "openmvs_mesh", "requires": ["ReconstructMesh"], "settings": [
        "ReconstructMesh: default settings; no geometry quality claim"]},
    {"name": "openmvs_refine", "requires": ["RefineMesh"], "settings": [
        "RefineMesh: default settings; optional if budget permits"]},
    {"name": "openmvs_texture", "requires": ["TextureMesh"], "settings": [
        "TextureMesh: default settings; optional if budget permits"]},
)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or str(relative) != name or
            "\\" in name or any(part in (".", "..") for part in relative.parts)):
        raise ValueError(f"unsafe manifest path: {name!r}")
    target = root.joinpath(*relative.parts)
    if root.is_symlink() or any(path.is_symlink() for path in (target, *target.parents) if path == root or root in path.parents):
        raise ValueError(f"symlink in manifest path: {name!r}")
    if not target.is_file():
        raise ValueError(f"missing manifest file: {name}")
    return target


def verify_dataset(root):
    manifest_path = checked_file(root, "manifest.json")
    if manifest_path.stat().st_size > MAX_MANIFEST:
        raise ValueError("manifest exceeds 2 MiB")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    selected = manifest.get("selected_images")
    source = manifest.get("source_files")
    if not isinstance(selected, list) or not selected or not isinstance(source, list):
        raise ValueError("manifest lacks selected images or source files")
    records = source + selected
    if len(records) > MAX_FILES:
        raise ValueError("too many manifest files")
    seen = set()
    total = 0
    verified = []
    for item in records:
        name = item.get("path")
        size = item.get("size_bytes")
        expected = item.get("sha256")
        if not isinstance(name, str) or name in seen:
            raise ValueError("duplicate or invalid manifest path")
        seen.add(name)
        if not isinstance(size, int) or size < 0 or size > MAX_SINGLE_BYTES or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid size/hash: {name}")
        total += size
        if total > MAX_INPUT_BYTES:
            raise ValueError("manifest inputs exceed 1 GiB")
        path = checked_file(root, name)
        if path.stat().st_size != size or digest_file(path) != expected:
            raise ValueError(f"input hash/size mismatch: {name}")
        verified.append({"path": name, "size_bytes": size, "sha256": expected})
    camera_path = manifest.get("colmap", {}).get("cameras")
    image_path = manifest.get("colmap", {}).get("images")
    if camera_path not in seen or image_path not in seen:
        raise ValueError("COLMAP camera/image records are not pinned")
    if checked_file(root, camera_path).stat().st_size > MAX_MANIFEST or checked_file(root, image_path).stat().st_size > MAX_MANIFEST:
        raise ValueError("COLMAP text record exceeds 2 MiB")
    cameras = {}
    for line in checked_file(root, camera_path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[1] == "PINHOLE":
            cameras[int(parts[0])] = (int(parts[2]), int(parts[3]), [float(v) for v in parts[4:8]])
    image_rows = {}
    for line in checked_file(root, image_path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0].isdigit():
            image_rows[int(parts[0])] = (int(parts[8]), parts[9], [float(v) for v in parts[1:8]])
    for item in selected:
        camera = item.get("camera", {})
        expected_camera = (camera.get("width"), camera.get("height"), camera.get("params"))
        if camera.get("model") != "PINHOLE" or cameras.get(item.get("camera_id")) != expected_camera:
            raise ValueError(f"camera record mismatch: {item['path']}")
        if image_rows.get(item.get("id")) != (item.get("camera_id"), item.get("name"),
                                                 item.get("qvec", []) + item.get("tvec", [])):
            raise ValueError(f"image pose record mismatch: {item['path']}")
        if item["path"] != "images/" + item["name"]:
            raise ValueError(f"image path/name mismatch: {item['path']}")
    return {"manifest_path": str(manifest_path), "manifest_sha256": digest_file(manifest_path),
            "source_revision": manifest.get("source", {}).get("revision"),
            "selected_image_count": len(selected), "input_bytes": total,
            "files": verified, "supplied_poses": True,
            "tracks_available": manifest.get("colmap", {}).get("tracks_available"),
            "physical_scale_verified": False}


def probe(name, search=None):
    executable = (search or shutil.which)(name)
    if not executable:
        return {"status": "unavailable", "path": None, "version": None}
    def cap_output():
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_PROBE_BYTES, MAX_PROBE_BYTES))

    try:
        with tempfile.TemporaryFile(dir=REPO / ".local-tools/tmp") as stream:
            result = subprocess.run([executable, "--version"], stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3, check=False,
                                    cwd=REPO / ".local-tools/tmp", preexec_fn=cap_output)
            stream.seek(0)
            output = stream.read(MAX_PROBE_BYTES).decode("utf-8", errors="replace").strip()
        if len(output.encode("utf-8")) >= MAX_PROBE_BYTES:
            return {"status": "version_unverified", "path": executable,
                    "version": None, "probe_error": "output_limit"}
        if result.returncode != 0 or not output:
            return {"status": "version_unverified", "path": executable,
                    "version": None, "probe_exit_code": result.returncode}
        return {"status": "available", "path": executable,
                "version": output.splitlines()[0][:200]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "version_unverified", "path": executable,
                "version": None, "probe_error": type(error).__name__}


def build_report(dataset, output_budget_gib, search=None):
    if not math.isfinite(output_budget_gib) or output_budget_gib <= 0 or output_budget_gib > 8:
        raise ValueError("output budget must be >0 and <=8 GiB")
    root = Path(dataset).absolute()
    data = verify_dataset(root)
    available = shutil.disk_usage(root).free
    budget = int(output_budget_gib * GIB)
    tools = {name: probe(name, search) for name in TOOLS}
    stages = [{**stage, "status": "pending_capability_verification" if all(
        tools[name]["path"] is not None for name in stage["requires"]) else "unavailable"}
              for stage in STAGES]
    return {"schema_version": 1, "purpose": "research_oracle_preflight_only",
            "comparison_executed": False,
            "host": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "cpu_count": os.cpu_count(),
                     "memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                     "build_tools": {name: shutil.which(name) for name in
                                     ("cmake", "ninja", "c++", "pkg-config", "brew")}},
            "dataset": data,
            "storage": {"free_bytes": available, "reserve_bytes": MIN_FREE,
                        "output_budget_bytes": budget,
                        "status": "sufficient" if available - budget >= MIN_FREE else "insufficient"},
            "tools": tools, "stages": stages,
            "readiness": "pending_installed_capability_verification" if available - budget >= MIN_FREE and all(
                tools[name]["path"] is not None for name in TOOLS) else "blocked",
            "notes": ["Output budget is an admission check, not an enforced runtime quota.",
                      "All stage settings require installed-version help validation before execution.",
                      "No masks or independent physical ground truth are pinned for this tree subset."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-budget-gib", type=float, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.dataset, args.output_budget_gib)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
