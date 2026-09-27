"""Read-only, hash-bound plan for a paired OpenMVS cached-depth fusion test.

This module deliberately does not copy files or launch native software. Both
planned arms consume the same completed depth maps; only --fusion-filter differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.classical_backend.run import BIN, RESERVE, ROOT, digest, tool_path

SOURCE = ROOT / "build-opencv/classical-ycb-native-masked-008"
SOURCE_RESULT_SHA256 = "8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9"
ARM_CAP = 700 << 20
BUFFER = 256 << 20
EXPECTED_OPTIONS = {
    "resolution_level": 2, "max_resolution": 1600, "geometric_iters": 2,
    "tower_mode": 4, "ignore_mask_label": 0,
}


def _real_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty or linked source file: {path}")


def _real_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"missing or linked source directory: {path}")


def _command(binary: Path, output: Path, fusion_filter: int) -> list[str]:
    if fusion_filter not in (1, 2):
        raise ValueError("paired test permits only fusion filters 1 and 2")
    return [str(binary), "-i", str(output / "scene.mvs"), "-o", str(output / "dense.mvs"),
            "--resolution-level", "2", "--max-resolution", "1600",
            "--min-resolution", "640", "--geometric-iters", "0",
            "--postprocess-dmaps", "0", "--fusion-mode", "0",
            "--tower-mode", "4", "--mask-path", str(output / "masks"),
            "--ignore-mask-label", "0", "--number-views-fuse", "2",
            "--fusion-filter", str(fusion_filter), "--max-threads", "2",
            "--working-folder", str(output)]


def build_plan(source: Path, control: Path, ablation: Path, binary_dir: Path,
               *, expected_result_sha256: str = SOURCE_RESULT_SHA256,
               free_bytes: int | None = None) -> dict:
    """Validate a sealed source and describe, but never execute, the two arms."""
    for path in (source, control, ablation, binary_dir):
        if path.is_symlink():
            raise ValueError(f"top-level symlink rejected: {path}")
    source, control, ablation, binary_dir = (p.resolve() for p in
                                              (source, control, ablation, binary_dir))
    _real_dir(source)
    _real_dir(binary_dir)
    if control == ablation or source in (control, ablation):
        raise ValueError("source and planned arms must be distinct")
    for output in (control, ablation):
        if output.exists() or not output.parent.is_dir() or output.parent.is_symlink():
            raise ValueError(f"planned output must be fresh under a real parent: {output}")
    result_path = source / "result.json"
    _real_file(result_path)
    if digest(result_path) != expected_result_sha256:
        raise ValueError("source result.json differs from reviewed 008 evidence")
    report = json.loads(result_path.read_text())
    if (report.get("schema") != "classical_masked_dense_v1" or
            report.get("status") != "complete" or report.get("mask_count") != 60):
        raise ValueError("source is not the complete 60-view masked control")
    options = report.get("native_options", {})
    if any(options.get(key) != value for key, value in EXPECTED_OPTIONS.items()):
        raise ValueError("source native depth profile differs from reviewed control")
    densify = [s for s in report.get("stages", []) if s.get("name") == "densify"]
    if len(densify) != 1 or densify[0].get("status") != "complete":
        raise ValueError("source densification was not complete")
    binary = tool_path(binary_dir, "DensifyPointCloud")
    _real_file(binary)
    if digest(binary) != report["binary_hashes"]["DensifyPointCloud"]:
        raise ValueError("native binary differs from reviewed source")

    scene = source / "scene.mvs"
    images_dir = source / "dense/images"
    masks_dir = source / "masks"
    _real_file(scene)
    _real_dir(source / "dense")
    _real_dir(images_dir)
    _real_dir(masks_dir)
    images = sorted(images_dir.glob("*.jpg"))
    masks = sorted(masks_dir.glob("*.mask.png"))
    maps = sorted(source.glob("depth[0-9][0-9][0-9][0-9].dmap"))
    if (len(images), len(masks), len(maps)) != (60, 60, 60):
        raise ValueError("expected exactly 60 images, masks, and final depth maps")
    if [p.name for p in maps] != [f"depth{i:04d}.dmap" for i in range(1, 61)]:
        raise ValueError("depth-map ID sequence is incomplete")
    if {p.name for p in masks} != {p.stem + ".mask.png" for p in images}:
        raise ValueError("mask/image names differ")
    files = [scene, *images, *masks, *maps]
    for path in files:
        _real_file(path)
    # Stream all consumed bytes, not just metadata. The manifest can be checked
    # again immediately before/after any later approved native execution.
    hashes = {str(p.relative_to(source)): digest(p) for p in files}
    copy_bytes = sum(p.stat().st_size for p in files)
    if copy_bytes >= ARM_CAP:
        raise ValueError("input copy alone exceeds per-arm output cap")
    if free_bytes is None:
        free_bytes = shutil.disk_usage(control.parent).free
    if free_bytes < RESERVE + 2 * ARM_CAP + BUFFER:
        raise ValueError("insufficient free space for both bounded arms and reserve")
    commands = {"control_dense_fuse": _command(binary, control, 2),
                "ablation_simple_fuse": _command(binary, ablation, 1)}
    # Normalize only the arm-local paths to prove the CLI has exactly one
    # experimental parameter difference.
    normalized = [[argument.replace(str(path), "<ARM>") for argument in command]
                  for path, command in ((control, commands["control_dense_fuse"]),
                                        (ablation, commands["ablation_simple_fuse"]))]
    differences = [(a, b) for a, b in zip(*normalized) if a != b]
    if differences != [("2", "1")]:
        raise AssertionError(f"unexpected native-option differences: {differences}")
    return {"schema": "openmvs_fusion_plan_v1", "execution": "not_authorized_or_run",
            "source": str(source), "source_result_sha256": expected_result_sha256,
            "binary_sha256": digest(binary), "input_sha256": hashes,
            "input_copy_bytes_per_arm": copy_bytes,
            "limits": {"max_bytes_per_arm": ARM_CAP, "min_free_bytes": RESERVE,
                       "preflight_buffer_bytes": BUFFER, "max_threads": 2},
            "free_bytes_at_preflight": free_bytes,
            "control": str(control), "ablation": str(ablation),
            "commands": commands, "sole_option_difference": {"--fusion-filter": [2, 1]},
            "cache_policy": "copy final 008 geometric DMAPs; geo-iters 0 in both arms"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source, args.control, args.ablation, args.binary_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
