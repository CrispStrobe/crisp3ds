#!/usr/bin/env python3
"""Bind recovered image-only 005 cameras to Berkeley poses, post hoc only."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.object_motion import ycb_camera_reference as base
from scripts.upstream_control.camera_compare import fit_centers


ROOT = Path(__file__).resolve().parents[2]
PRODUCER_005 = ROOT / "build-opencv/object-motion/initialization-recovery-005-delayed-refinement/ba_trial/report.json"
PRODUCER_004 = ROOT / "build-opencv/object-motion/initialization-recovery-004-fixed-intrinsics/cached_seed_trial/report.json"
ORACLE = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/report.json"
PREPARED = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
METADATA = ORACLE.parent / "metadata"
OUTPUT = ROOT / "build-opencv/object-motion/recovered-camera-reference-001"
PRODUCER_005_SHA256 = "80241c6c8d72097ce38a9a046aadfa495c1027bbfbd05a6a9480ad09b02c7e10"
PRODUCER_004_SHA256 = "9a7ddeac8b99ab3ec41a512aec0b852b555aa6176860d2047460e3f87f910b63"
MODEL_NAMES = ("cameras.bin", "images.bin", "points3D.bin")


def verified_inputs(producer_005=PRODUCER_005, producer_004=PRODUCER_004,
                    oracle_path=ORACLE, metadata=METADATA,
                    photo_manifest=base.PHOTO_MANIFEST, prepared=PREPARED):
    producer_005, producer_004, oracle_path, metadata, photo_manifest = map(
        Path, (producer_005, producer_004, oracle_path, metadata, photo_manifest))
    prepared = Path(prepared)
    if (base.sha256(producer_005) != PRODUCER_005_SHA256 or
            base.sha256(producer_004) != PRODUCER_004_SHA256):
        raise ValueError("recovery producer reports differ from frozen 005/004")
    five = json.loads(producer_005.read_text())
    four = json.loads(producer_004.read_text())
    oracle = json.loads(oracle_path.read_text())
    selected = json.loads(photo_manifest.read_text())
    preparation = json.loads(prepared.read_text())
    photos = {Path(item["path"]).name: item["sha256"] for item in selected["photos"]}
    prepared_photos = {item["name"]: item["sha256"] for item in preparation["images"]}
    if (five.get("schema") != "ycb_image_only_delayed_self_calibration_v1" or
            five.get("status") != "complete" or five.get("registered") != 60 or
            five.get("source_producer_report_sha256") != PRODUCER_004_SHA256 or
            five.get("source_model_files_sha256") != four.get("model_files_sha256") or
            five.get("input_model_copy_sha256") != four.get("model_files_sha256") or
            five.get("source_database_sha256") != four.get("source_database_sha256") or
            five.get("source_cache_mutation_audit_sha256") != four.get("cache_mutation_audit_sha256") or
            five.get("seed_pair") != four.get("seed_pair") or
            five.get("supplied_intrinsics_or_poses_used") is not False or
            five.get("reference_mesh_used") is not False or
            four.get("schema") != "ycb_image_only_fixed_initial_intrinsics_v1" or
            four.get("status") != "complete" or four.get("registered") != 60 or
            four.get("source_image_hashes") != photos or prepared_photos != photos or
            four.get("source_manifest_sha256") != base.sha256(prepared) or
            preparation.get("source_manifest_sha256") != base.sha256(photo_manifest) or
            oracle.get("schema") != "ycb_berkeley_camera_oracle_v1" or
            oracle.get("source_archive_sha256") != base.ARCHIVE_SHA256 or
            selected["sources"]["berkeley_rgbd"]["sha256"] != base.ARCHIVE_SHA256 or
            len(photos) != 60):
        raise ValueError("005→004→Berkeley photo provenance differs from frozen lineage")
    if (five.get("refined_model_dir") != str(PRODUCER_005.parent / "refined_model") or
            four.get("model_dir") != str(PRODUCER_004.parent / "models/0") or
            set(five.get("refined_model_files_sha256", {})) != set(MODEL_NAMES) or
            set(four.get("model_files_sha256", {})) != set(MODEL_NAMES)):
        raise ValueError("recovery model paths or binary inventory differ")
    for folder, expected in ((Path(four["model_dir"]), four["model_files_sha256"]),
                             (Path(five["input_model_copy"]), five["input_model_copy_sha256"]),
                             (Path(five["refined_model_dir"]), five["refined_model_files_sha256"])):
        for filename in MODEL_NAMES:
            if base.sha256(folder / filename) != expected[filename]:
                raise ValueError(f"recovery model changed: {filename}")
    members = oracle.get("metadata", {}).get("members", {})
    if set(members) != base.MEMBERS:
        raise ValueError("Berkeley metadata inventory differs")
    for member, details in members.items():
        if base.sha256(metadata / member.removeprefix(base.PREFIX)) != details.get("sha256"):
            raise ValueError(f"Berkeley metadata changed: {member}")
    return five, four, oracle, photos


def evaluate(output=OUTPUT, producer_005=PRODUCER_005, producer_004=PRODUCER_004,
             oracle_path=ORACLE, metadata=METADATA, photo_manifest=base.PHOTO_MANIFEST,
             prepared=PREPARED):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    five, four, oracle, photos = verified_inputs(
        producer_005, producer_004, oracle_path, metadata, photo_manifest, prepared)
    model = Path(five["refined_model_dir"])
    source_centers, source_rotations, camera = base.estimated_cameras(model)
    if (camera.model.name != "SIMPLE_RADIAL" or camera.width != 1280 or camera.height != 1024 or
            not np.allclose(camera.params, five["after_camera_params"], rtol=0, atol=1e-10)):
        raise ValueError("recovered camera differs from sealed 005 intrinsics")
    target_centers, target_rotations, k, d = base.reference_cameras(metadata)
    matrix, diagnostics = fit_centers(source_centers, target_centers)
    errors = base.rotation_errors(source_rotations, target_rotations, matrix)
    report = {"schema": "ycb_recovered_berkeley_camera_oracle_v1",
              "lane": "post hoc named-camera fit of recovered image-only 005; no mesh fit",
              "source_archive_sha256": base.ARCHIVE_SHA256,
              "runner_sha256": base.sha256(Path(__file__)),
              "calibration_h5_sha256": base.sha256(Path(metadata) / "calibration.h5"),
              "producer_005_report_sha256": base.sha256(producer_005),
              "producer_004_report_sha256": base.sha256(producer_004),
              "oracle_metadata_report_sha256": base.sha256(oracle_path),
              "photo_manifest_sha256": base.sha256(photo_manifest),
              "prepared_manifest_sha256": base.sha256(prepared),
              "source_image_hashes": photos,
              "producer_model_files_sha256": five["refined_model_files_sha256"],
              "metadata_members_sha256": {name: info["sha256"] for name, info in oracle["metadata"]["members"].items()},
              "transform_convention": "proper Sim(3) from recovered SfM world to per-angle Berkeley table frame, fitted only to named camera centers",
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
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 100_000:
        raise ValueError("camera report exceeded 100 KB cap")
    if (base.sha256(producer_005) != PRODUCER_005_SHA256 or
            base.sha256(producer_004) != PRODUCER_004_SHA256 or
            base.sha256(oracle_path) != report["oracle_metadata_report_sha256"] or
            base.sha256(photo_manifest) != report["photo_manifest_sha256"] or
            base.sha256(prepared) != report["prepared_manifest_sha256"]):
        raise ValueError("source report changed during camera evaluation")
    # Rehash the actual model and all calibration/pose members after fitting,
    # not merely the reports that named them before fitting.
    verified_inputs(producer_005, producer_004, oracle_path, metadata,
                    photo_manifest, prepared)
    output.mkdir(parents=True)
    with (output / "report.json").open("xb") as target:
        target.write(encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.output)
    print(json.dumps({"fit_rms": report["camera_centers"]["fit_rms"],
                      "rotation_p95_degrees": report["rotation_error_p95_degrees"]}))


if __name__ == "__main__":
    main()
