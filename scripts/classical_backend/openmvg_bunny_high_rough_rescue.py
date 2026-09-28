"""Post-hoc read-only validation of -003 native rough outputs.

The original failed receipt and native files are never modified. Default is
read-only; --write-sidecar creates a separately labeled exact-hash audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.classical_backend.geometry import digest, inspect
from scripts.classical_backend import openmvg_bunny_high_cache_fusion as fusion

SOURCE = fusion.OUTPUT
SIDE_CAR = Path(__file__).resolve().parents[2] / ".local-tools/openmvg-bunny-high-fusion-rough-rescue-003.json"
SCHEMA = "openmvg_bunny_high_rough_rescue_v1"
CORE = ("dense.mvs", "dense.ply", "mesh.ply")


def validate(source: Path = SOURCE) -> dict:
    if source.is_symlink() or not source.is_dir() or source.resolve() != SOURCE.resolve():
        raise ValueError("unexpected rough source")
    receipt_path = fusion.real_file(source / "result.json")
    receipt_sha = digest(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "openmvg_bunny_high_cached73_fusion_rough_v1" or
            receipt.get("status") != "failed" or
            receipt.get("failure") != f"[Errno 2] No such file or directory: '{source / 'mesh.mvs'}'" or
            receipt.get("source_unchanged") is not True or
            receipt.get("capacity_after_error") or
            receipt.get("real_wall_seconds_total", 99999) > fusion.TOTAL_SECONDS or
            receipt.get("output_bytes", 9999999999) > fusion.LOGICAL_CAP):
        raise ValueError("failed receipt is not the exact post-native mesh.mvs assumption failure")
    if (source / "mesh.mvs").exists() or (source / "mesh.mvs").is_symlink():
        raise ValueError("unexpected mesh.mvs appeared")
    source_seal = fusion.verify_source()
    if receipt.get("sealed_source") != source_seal:
        raise ValueError("failed receipt/source lineage differs")
    stages = receipt.get("stages", [])
    if ([stage.get("name") for stage in stages] != ["densify", "mesh"] or
            any(stage.get("status") != "complete" or stage.get("returncode") != 0
                for stage in stages) or
            stages[0].get("cache_fusion_verified") is not True):
        raise ValueError("native stages not both complete with cache fusion")
    inventory = receipt.get("output_inventory", {})
    artifacts = {}
    for name in CORE:
        path = fusion.real_file(source / name)
        size, sha = path.stat().st_size, digest(path)
        if (inventory.get(name, {}).get("bytes") != size or
                inventory.get(name, {}).get("sha256") != sha):
            raise ValueError(f"core artifact receipt/hash differs: {name}")
        artifacts[name] = {"path": str(path), "bytes": size, "sha256": sha}
    stage_artifacts = ((stages[0], "dense.ply"), (stages[1], "mesh.ply"))
    for stage, name in stage_artifacts:
        if (stage.get("artifact", {}).get("path") != artifacts[name]["path"] or
                stage["artifact"].get("bytes") != artifacts[name]["bytes"] or
                stage["artifact"].get("sha256") != artifacts[name]["sha256"]):
            raise ValueError(f"native stage artifact differs: {name}")
        log = fusion.real_file(source / f"{stage['name']}.log")
        if (digest(log) != stage.get("log_sha256") or
                log.stat().st_size != stage.get("log_bytes")):
            raise ValueError(f"native stage log differs: {stage['name']}")
    dense_report = inspect(source / "dense.ply")
    mesh_report = inspect(source / "mesh.ply")
    if (dense_report["vertices"] != stages[0]["artifact"].get("points") or
            mesh_report["faces"] != stages[1]["artifact"].get("faces") or
            mesh_report["faces"] <= 0 or mesh_report["zero_area_faces"] != 0):
        raise ValueError("native geometry differs or has zero-area faces")
    for name, row in source_seal["images"].items():
        path = fusion.real_file(source / "images" / name)
        if (path.stat().st_size != row["bytes"] or digest(path) != row["sha256"] or
                inventory.get("images/" + name) != row):
            raise ValueError(f"staged image differs: {name}")
    for name, row in source_seal["dmaps"].items():
        path = fusion.real_file(source / name)
        if (path.stat().st_size != row["bytes"] or digest(path) != row["sha256"] or
                inventory.get(name) != row):
            raise ValueError(f"cached base map differs: {name}")
    if list(source.glob("depth*.geo.dmap")):
        raise ValueError("geometric-consistency maps unexpectedly present")
    if shutil.disk_usage(source).free < fusion.FLOOR:
        raise ValueError("external 11GiB floor breached")
    return {"schema": SCHEMA, "status": "native_rough_complete_pending_visual_review",
            "rough_source": str(source), "failed_receipt_path": str(receipt_path),
            "failed_receipt_sha256": receipt_sha,
            "failure_classification": "receipt finalization expected mesh.mvs; native ReconstructMesh returned 0 and saved mesh.ply",
            "source_unchanged": True, "cached_base_dmaps_verified": 73,
            "converted_images_verified": 73, "native_mesh_valid": True,
            "native_artifacts": artifacts,
            "stage_status": {stage["name"]: {"status": stage["status"],
                                             "returncode": stage["returncode"],
                                             "log_sha256": stage["log_sha256"],
                                             "artifact_sha256": stage["artifact"]["sha256"]}
                             for stage in stages},
            "dense_geometry": dense_report, "mesh_geometry": mesh_report,
            "real_wall_seconds_total": receipt["real_wall_seconds_total"],
            "logical_output_bytes": receipt["output_bytes"],
            "physical_delta_bytes": receipt["capacity_after"]["physical_delta_bytes"],
            "visual_object_shape_reviewed": False, "quality_accepted": False,
            "shipping_approved": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sidecar", action="store_true")
    args = parser.parse_args()
    report = validate()
    if args.write_sidecar:
        if SIDE_CAR.exists() or SIDE_CAR.is_symlink():
            raise FileExistsError(SIDE_CAR)
        with SIDE_CAR.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "failed_receipt_sha256": report["failed_receipt_sha256"],
                      "mesh_ply_sha256": report["native_artifacts"]["mesh.ply"]["sha256"],
                      "sidecar": str(SIDE_CAR) if args.write_sidecar else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
