#!/usr/bin/env python3
"""Seal a calibrated YCB camera-only Sim(3) against Berkeley rig metadata."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.object_motion import ycb_camera_reference as base
from scripts.upstream_control.camera_compare import fit_centers


ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "build-opencv/object-motion/calibration-ablation-002/calibrated_shifted/report.json"
ORACLE = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/report.json"
METADATA = ORACLE.parent / "metadata"
OUTPUT = ROOT / "build-opencv/object-motion/calibrated-camera-reference-001"


def verified_inputs(producer_path=PRODUCER, oracle_path=ORACLE, metadata=METADATA):
    producer_path, oracle_path, metadata = map(Path, (producer_path, oracle_path, metadata))
    producer = json.loads(producer_path.read_text())
    oracle = json.loads(oracle_path.read_text())
    if (producer.get("schema") != "ycb_calibrated_intrinsics_sfm_v1" or
            producer.get("lane") != "calibrated_intrinsics_only" or
            producer.get("status") != "complete" or producer.get("registered") != 60 or
            producer.get("calibration_h5_sha256") != base.sha256(metadata / "calibration.h5") or
            producer.get("camera_reference_report_sha256") != base.sha256(oracle_path) or
            oracle.get("source_archive_sha256") != base.ARCHIVE_SHA256):
        raise ValueError("calibrated model or Berkeley metadata provenance mismatch")
    model = Path(producer["model_dir"])
    expected = producer.get("model_files_sha256", {})
    if set(expected) != {"cameras.bin", "images.bin", "points3D.bin"}:
        raise ValueError("missing binary model hash")
    for filename, value in expected.items():
        if base.sha256(model / filename) != value:
            raise ValueError(f"calibrated binary model changed: {filename}")
    members = oracle.get("metadata", {}).get("members", {})
    if set(members) != base.MEMBERS:
        raise ValueError("Berkeley metadata member inventory changed")
    for member, details in members.items():
        local = metadata / member.removeprefix(base.PREFIX)
        if base.sha256(local) != details.get("sha256"):
            raise ValueError(f"Berkeley metadata member changed: {member}")
    selected = json.loads(base.PHOTO_MANIFEST.read_text())
    photos = {Path(item["path"]).name: item["sha256"] for item in selected["photos"]}
    if (selected["sources"]["berkeley_rgbd"]["sha256"] != base.ARCHIVE_SHA256 or
            photos != producer.get("source_image_hashes")):
        raise ValueError("calibrated producer photos do not match Berkeley archive selection")
    return producer, oracle, model, photos


def evaluate(output=OUTPUT, producer_path=PRODUCER, oracle_path=ORACLE, metadata=METADATA):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    producer, oracle, model, photos = verified_inputs(producer_path, oracle_path, metadata)
    source_centers, source_rotations, camera = base.estimated_cameras(model)
    if camera.model.name != "FULL_OPENCV" or not np.allclose(
            camera.params, producer["full_opencv_params"], rtol=0, atol=1e-10):
        raise ValueError("calibrated producer camera is not the sealed fixed FULL_OPENCV camera")
    target_centers, target_rotations, k, d = base.reference_cameras(metadata)
    matrix, diagnostics = fit_centers(source_centers, target_centers)
    errors = base.rotation_errors(source_rotations, target_rotations, matrix)
    report = {"schema": "ycb_calibrated_berkeley_camera_oracle_v1",
              "lane": "camera-only post hoc oracle diagnostic of calibrated-intrinsics SfM",
              "source_archive_sha256": base.ARCHIVE_SHA256,
              "calibration_h5_sha256": base.sha256(Path(metadata) / "calibration.h5"),
              "producer_report_sha256": base.sha256(producer_path),
              "oracle_metadata_report_sha256": base.sha256(oracle_path),
              "photo_manifest_sha256": base.sha256(base.PHOTO_MANIFEST),
              "source_image_count": len(photos), "producer_model_files_sha256": producer["model_files_sha256"],
              "metadata_members_sha256": {name: info["sha256"] for name, info in oracle["metadata"]["members"].items()},
              "transform_convention": "proper Sim(3) from SfM world to per-angle Berkeley table frame, fit only to named camera centers",
              "matrix_estimated_world_to_berkeley_table": matrix.tolist(),
              "camera_centers": diagnostics,
              "rotation_error_degrees_by_name": errors,
              "rotation_error_median_degrees": float(np.median(list(errors.values()))),
              "rotation_error_p95_degrees": float(np.quantile(list(errors.values()), .95)),
              "intrinsics": {"model": camera.model.name, "estimated_params": list(map(float, camera.params)),
                             "berkeley_np3_rgb_K": k.tolist(), "berkeley_np3_rgb_d": d.tolist()},
              "reference_mesh_used": False, "supplied_poses_used_for_mapping": False,
              "metadata_table_pose_refinement_state_known": False,
              "metric_google_mesh_accuracy_claim_allowed": False}
    output.mkdir(parents=True)
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    if target.stat().st_size > 20_000:
        raise ValueError("camera report exceeded 20 KB cap")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps({"fit_rms": result["camera_centers"]["fit_rms"],
                      "rotation_p95_degrees": result["rotation_error_p95_degrees"]}))


if __name__ == "__main__":
    main()
