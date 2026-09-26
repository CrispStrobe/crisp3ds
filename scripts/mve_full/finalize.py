"""Finalize the frozen bunny-002 raw mesh after its pre-clean validation failure.

This is a scoped continuation of one completed SfM/MVS/FSSR run. It never
reprocesses photographs or edits the failed run and cannot be called a fresh
end-to-end replay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from scripts.mve_full.fetch_source import no_symlink_components
from scripts.mve_full.run import ROOT, RESERVE, SOURCE, ply_counts, sha256, stage, tree_bytes
from scripts.mve_full.sanitize_mesh import remove_zero_area_faces
from scripts.mve_full.verify_source import verify as verify_selected_source


EXPECTED_RAW_SHA256 = "3bdb717736efd698d778abb371d4681f60581d95833cafd09b968717f72c988b"
EXPECTED_MESHCLEAN_SHA256 = "3334f2da0db4666ef57114bf2de3aa6e516a60534c81df650532c20afdf2454e"
EXPECTED_RESULT_SHA256 = "e61ae7ff607719c9c7a63e745b8caf57a430f3f4147ea328920a5eb50c4a3d2f"


def finalize(failed_run: Path, output: Path, max_gib: float = 1.0,
             timeout_minutes: float = 10.0) -> dict:
    failed_run = failed_run.absolute()
    output = output.absolute()
    no_symlink_components(failed_run)
    no_symlink_components(output)
    if not failed_run.is_dir() or not output.parent.is_dir() or output.exists():
        raise ValueError("failed run must exist; output must be a fresh directory")
    result_path = failed_run / "result.json"
    raw_mesh = failed_run / "mesh-raw.ply"
    image_dir = failed_run / "input"
    if (not result_path.is_file() or result_path.is_symlink() or
            not raw_mesh.is_file() or raw_mesh.is_symlink() or
            not image_dir.is_dir() or image_dir.is_symlink()):
        raise ValueError("failed run has missing or linked provenance files")
    if sha256(result_path) != EXPECTED_RESULT_SHA256:
        raise ValueError("failed run result differs from frozen bunny-002 report")
    old_result = json.loads(result_path.read_text())
    if (old_result.get("schema") != "mve_full_unknown_pose_v2" or
            old_result.get("status") != "failed" or
            old_result.get("error") != "ValueError: degenerate PLY triangle" or
            [entry["stage"] for entry in old_result["stages"]] !=
            ["makescene", "sfmrecon", "dmrecon", "scene2pset", "fssrecon"] or
            any(entry["status"] != "succeeded" for entry in old_result["stages"]) or
            old_result.get("registered_views") != 73 or
            old_result.get("source_commit") != "bf2279f161ba962072ecac85224c15e82bc5f52e"):
        raise ValueError("input is not the frozen bunny-002 pre-clean failure")
    if sha256(raw_mesh) != EXPECTED_RAW_SHA256:
        raise ValueError("raw mesh differs from frozen bunny-002 artifact")
    binary = SOURCE / "apps/meshclean/meshclean"
    if (sha256(binary) != EXPECTED_MESHCLEAN_SHA256 or
            old_result["binaries"]["meshclean"]["sha256"] != EXPECTED_MESHCLEAN_SHA256):
        raise ValueError("selected meshclean binary differs from frozen run")
    verified_sources = verify_selected_source(ROOT / ".local-tools/mve-spike/mve-bf2279f.tar.gz", SOURCE)
    if old_result.get("selected_source_file_count") not in (None, verified_sources):
        raise ValueError("selected source inventory differs from frozen run")
    for item in old_result["input"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("input basename escaped frozen run")
        copied = image_dir / item["name"]
        if copied.is_symlink() or not copied.is_file() or sha256(copied) != item["sha256"]:
            raise ValueError(f"copied input photo changed: {item['name']}")
    raw_vertices, raw_faces = ply_counts(raw_mesh, allow_degenerate=True)
    if raw_vertices < 3 or raw_faces < 1:
        raise ValueError("raw surface is empty")
    if (not math.isfinite(max_gib) or not math.isfinite(timeout_minutes) or
            max_gib <= 0 or timeout_minutes <= 0):
        raise ValueError("positive resource caps required")
    cap = int(max_gib * (1 << 30))
    if shutil.disk_usage(output.parent).free < RESERVE + cap:
        raise ValueError("output cap plus 10 GiB reserve unavailable")
    old_hash = EXPECTED_RESULT_SHA256
    report = {"schema": "mve_full_finalize_v1", "status": "started",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "continuation_of": str(failed_run), "failed_run_result_sha256": old_hash,
              "raw_mesh_sha256": EXPECTED_RAW_SHA256,
              "meshclean_binary_sha256": EXPECTED_MESHCLEAN_SHA256,
              "selected_source_file_count": verified_sources,
              "raw_mesh_vertices": raw_vertices, "raw_mesh_faces": raw_faces,
              "reused_stages": [entry["stage"] for entry in old_result["stages"]],
              "fresh_end_to_end_run": False, "scale_verified": False,
              "surface_quality_accepted": False, "shipping_approved": False,
              "resource_limits": {"max_gib": max_gib, "timeout_minutes": timeout_minutes,
                                  "reserve_gib": 10}, "stages": []}
    output.mkdir()
    started = time.monotonic()
    native_mesh = output / "mesh-native.ply"
    mesh = output / "mesh.ply"
    try:
        command = [str(binary), "--threshold=0", "--component-size=0", str(raw_mesh), str(native_mesh)]
        result = stage(command, "meshclean", output, cap, timeout_minutes * 60,
                       started, 64 << 20)
        report["stages"].append(result)
        if result["status"] != "succeeded":
            raise RuntimeError(f"meshclean: {result['status']}")
        report.update(remove_zero_area_faces(native_mesh, mesh))
        if report["mesh_faces"] < 1:
            raise ValueError("cleaned mesh has zero faces")
        if sha256(raw_mesh) != EXPECTED_RAW_SHA256 or sha256(result_path) != old_hash:
            raise ValueError("failed run changed during finalization")
        report["output_bytes"] = tree_bytes(output)
        if report["output_bytes"] > cap or shutil.disk_usage(output).free < RESERVE:
            raise ValueError("final output cap or disk reserve breached")
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-gib", type=float, default=1.0)
    parser.add_argument("--timeout-minutes", type=float, default=10.0)
    args = parser.parse_args()
    result = finalize(args.failed_run, args.output, args.max_gib, args.timeout_minutes)
    print(json.dumps({"status": result["status"], "mesh_faces": result["mesh_faces"],
                      "output": str(args.output.absolute())}, sort_keys=True))


if __name__ == "__main__":
    main()
