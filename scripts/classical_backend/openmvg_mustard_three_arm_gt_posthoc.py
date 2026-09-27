"""Read-only-first, posthoc 44-name Berkeley camera comparison of three SfM arms.

No GT H5 dataset is opened by default. --run requires separate review and
recomputes all three Sim3 fits on the common 44; prior 46-fit errors are not used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from scripts.classical_backend import openmvg_mustard_global_pose_export as global_export
from scripts.classical_backend import openmvg_mustard_gt_posthoc as two
from scripts.classical_backend import openmvg_mustard_json_pose_validator as poses

OUTPUT = two.BASE / "openmvg-mustard-gt-posthoc-three-arm-001"
GLOBAL_EXPORT = global_export.OUTPUT
GLOBAL_RECEIPT_SHA = "ac20487ca1932e7e42046c43b5fbae768383051f53a61729f4bb5d49dac7bebd"
GLOBAL_JSON_SHA = "6382b68e07815d730f5b428e23004bac99a210fced8b5c9a5706e8503a2da709"
PRIOR = two.OUTPUT
PRIOR_RECEIPT_SHA = "863472bc0af7e69d0b16602dd8ea05ed59ead1a106d076b1f8ad01f002a1643e"
PRIOR_REPORT_SHA = "e6a791a8e16deba1ac6856b8f4eb90df7ce6ec4e2a092a53eb4b955892e8d035"
EXCLUDED = ("NP3_042.jpg", "NP3_048.jpg", "NP3_252.jpg", "NP3_276.jpg")
COMMON = tuple(name for name in poses.TRAIN_NAMES if name not in EXCLUDED)
OUTPUT_CAP = 1 << 20
FLOOR = 11 << 30
WALL_SECONDS = 300


def named_global(data: dict) -> tuple[dict, dict]:
    checked = poses.parse_export(data, poses.TRAIN_NAMES)
    if checked["missing_pose_names"] != ["NP3_252.jpg", "NP3_276.jpg"]:
        raise ValueError("GLOBAL named pose inventory differs")
    by_id = {row["key"]: row["value"] for row in data["extrinsics"]}
    centers, rotations = {}, {}
    for row in data["views"]:
        view = row["value"]["ptr_wrapper"]["data"]
        pose = by_id.get(view["id_pose"])
        if pose is not None:
            centers[view["filename"]] = pose["center"]
            rotations[view["filename"]] = pose["rotation"]
    expected = set(poses.TRAIN_NAMES) - {"NP3_252.jpg", "NP3_276.jpg"}
    if set(centers) != expected or set(rotations) != expected:
        raise ValueError("GLOBAL cereal view/pose ID mapping differs")
    return centers, rotations


def audit_exports() -> tuple[dict, dict]:
    adjust, fixed, previous = two.audit_exports()
    if previous["common_names"] != list(two.EXPECTED_NAMES):
        raise ValueError("sealed ADJUST_ALL/NONE 46-name set differs")
    receipt_path = GLOBAL_EXPORT / "receipt.json"
    json_path = GLOBAL_EXPORT / "sfm_camera_poses.json"
    for path, digest in ((receipt_path, GLOBAL_RECEIPT_SHA), (json_path, GLOBAL_JSON_SHA)):
        if path.is_symlink() or not path.is_file() or two.sha(path) != digest:
            raise ValueError(f"sealed GLOBAL export differs: {path.name}")
    if json_path.stat().st_size > poses.MAX_JSON_BYTES:
        raise ValueError("GLOBAL export JSON exceeds bound")
    global_model = global_export.verify_global_model()
    receipt = poses.load_json(receipt_path)
    data = poses.load_json(json_path)
    centers, rotations = named_global(data)
    exported = receipt.get("exported", {})
    if (receipt.get("schema") != "openmvg_mustard_global_pose_export_v1" or
            receipt.get("status") != "exported_global_model_diagnostic_only" or
            receipt.get("output") != str(GLOBAL_EXPORT) or receipt.get("returncode") != 0 or
            receipt.get("command") != global_export.command(GLOBAL_EXPORT) or
            receipt.get("json_sha256") != GLOBAL_JSON_SHA or
            receipt.get("json_bytes") != json_path.stat().st_size or
            receipt.get("global_model") != global_model or
            receipt.get("converter") != global_export.prior.verify_converter() or
            exported.get("views") != 48 or exported.get("poses") != 46 or
            exported.get("intrinsics") != 1 or exported.get("structure") != 0 or
            exported.get("control_points") != 0 or
            exported.get("missing_names") != ["NP3_252.jpg", "NP3_276.jpg"] or
            exported.get("posed_names") != sorted(centers)):
        raise ValueError("sealed GLOBAL export receipt/model linkage differs")
    global_arm = {"receipt_sha256": GLOBAL_RECEIPT_SHA, "json_sha256": GLOBAL_JSON_SHA,
                  "model_sha256": global_export.MODEL_SHA,
                  "source_receipt_sha256": global_export.RECEIPT_SHA,
                  "centers": centers, "rotations": rotations}
    arms = {"adjust_all": adjust, "fixed_none": fixed, "global_adjust_all": global_arm}
    for arm in arms.values():
        if not set(COMMON) <= set(arm["centers"]) or set(arm["centers"]) != set(arm["rotations"]):
            raise ValueError("three arms do not share exact 44 named TRAIN poses")
        arm["centers"] = {name: arm["centers"][name] for name in COMMON}
        arm["rotations"] = {name: arm["rotations"][name] for name in COMMON}
    return arms, {"common_names": list(COMMON), "excluded_names": list(EXCLUDED),
                  "population_count": 44, "fit_policy": "refit each arm from its own 44 camera centers"}


def audit_reference() -> dict:
    receipt_path, report_path = PRIOR / "receipt.json", PRIOR / "report.json"
    if (PRIOR.is_symlink() or not PRIOR.is_dir() or
            any(path.is_symlink() or not path.is_file() for path in (receipt_path, report_path)) or
            two.sha(receipt_path) != PRIOR_RECEIPT_SHA or
            two.sha(report_path) != PRIOR_REPORT_SHA or
            two.METADATA_RECEIPT.is_symlink() or not two.METADATA_RECEIPT.is_file() or
            two.sha(two.METADATA_RECEIPT) != two.METADATA_RECEIPT_SHA):
        raise ValueError("prior two-arm result or Berkeley metadata receipt seal differs")
    receipt = poses.load_json(receipt_path)
    report = poses.load_json(report_path)
    if (receipt.get("schema") != "openmvg_mustard_two_arm_gt_posthoc_v1" or
            receipt.get("status") != "posthoc_46_view_diagnostic_only" or
            receipt.get("report_sha256") != PRIOR_REPORT_SHA or
            receipt.get("report_bytes") != report_path.stat().st_size or
            report.get("schema") != "openmvg_mustard_two_arm_gt_posthoc_result_v1" or
            report.get("status") != "posthoc_46_view_diagnostic_only" or
            receipt.get("names", {}).get("common_names") != list(two.EXPECTED_NAMES) or
            receipt.get("adjust_all", {}).get("json_sha256") != poses.APPROVED_JSON_SHA256 or
            receipt.get("fixed_none", {}).get("json_sha256") != two.FIXED_EXPORT_JSON_SHA or
            receipt.get("metadata", {}).get("receipt_sha256") != two.METADATA_RECEIPT_SHA or
            receipt.get("metadata", {}).get("archive_sha256") != two.ARCHIVE_SHA):
        raise ValueError("prior two-arm result provenance differs")
    metadata_receipt = poses.load_json(two.METADATA_RECEIPT)
    found = metadata_receipt.get("extraction", {}).get("members", {})
    if (metadata_receipt.get("schema") != "mustard_pose_metadata_acquisition_v1" or
            metadata_receipt.get("status") != "metadata_only_no_camera_evaluation" or
            len(found) != 40):
        raise ValueError("original Berkeley metadata receipt differs")
    extracted = receipt["metadata"].get("extracted_members", {})
    staged = receipt.get("staged_metadata", {})
    inventory = receipt.get("artifact_inventory", {})
    if (len(extracted) != 38 or len(staged) != 9 or
            extracted != {name: found[name] for name in extracted} or
            set(staged) != set(receipt["metadata"].get("archive_only_members", {})) or
            set(inventory) != {"report.json"} | {"metadata/" + Path(name).name for name in staged} or
            inventory.get("report.json") != {"bytes": report_path.stat().st_size,
                                             "sha256": PRIOR_REPORT_SHA}):
        raise ValueError("prior staged/extracted metadata manifest differs")
    needed = {two.PREFIX + "calibration.h5"} | {
        two.PREFIX + f"poses/NP5_{int(name[4:7])}_pose.h5" for name in COMMON}
    paths = {}
    for member in sorted(needed):
        if member in extracted:
            path, record = two.METADATA / Path(member).relative_to(two.PREFIX), extracted[member]
        elif member in staged:
            path, record = PRIOR / "metadata" / Path(member).name, staged[member]
        else:
            raise ValueError(f"common-44 Berkeley pose absent: {member}")
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or
                two.sha(path) != record["sha256"]):
            raise ValueError(f"common-44 Berkeley metadata seal differs: {member}")
        paths[member] = {"path": str(path), **record}
    for member, record in staged.items():
        path = PRIOR / "metadata" / Path(member).name
        if (path.is_symlink() or not path.is_file() or
                path.stat().st_size != record["bytes"] or two.sha(path) != record["sha256"] or
                inventory.get("metadata/" + path.name) != record):
            raise ValueError(f"prior staged NP5 seal differs: {member}")
    if len(paths) != 45:
        raise ValueError("common-44 calibration/pose metadata count differs")
    return {"prior_receipt_sha256": PRIOR_RECEIPT_SHA,
            "prior_report_sha256": PRIOR_REPORT_SHA,
            "metadata_receipt_sha256": two.METADATA_RECEIPT_SHA,
            "source_archive_sha256": two.ARCHIVE_SHA,
            "used_h5": paths, "used_h5_count": len(paths),
            "reuse_policy": "metadata only; never reuse prior 46-view fit errors"}


def capacity(output: Path, prospective: bool = False) -> dict:
    used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) if output.exists() else 0
    external = shutil.disk_usage(two.BASE).free
    internal = shutil.disk_usage(Path(__file__).resolve().parents[2]).free
    required = FLOOR + (OUTPUT_CAP - used if prospective else 0)
    if used > OUTPUT_CAP or external < required or internal < FLOOR:
        raise RuntimeError("three-arm posthoc disk/output cap breached")
    return {"output_bytes": used, "external_free": external,
            "external_required": required, "internal_free": internal}


def preflight(output: Path = OUTPUT) -> dict:
    if (output.exists() or output.is_symlink() or output.parent != two.BASE or
            output.name != OUTPUT.name or two.BASE.is_symlink() or not two.BASE.is_dir()):
        raise ValueError("three-arm output must be the fresh reviewed external root")
    arms, population = audit_exports()
    reference = audit_reference()
    for arm in arms.values():
        arm.pop("centers")
        arm.pop("rotations")
    return {"schema": "openmvg_mustard_three_arm_gt_posthoc_v1",
            "status": "read_only_preflight_no_gt_scoring", "output": str(output),
            "arms": arms, "population": population, "reference": reference,
            "algorithm_runtime": two.algorithm_runtime(),
            "capacity": capacity(output, prospective=True),
            "limits": {"output_bytes": OUTPUT_CAP, "disk_floor_bytes": FLOOR,
                       "wall_seconds": WALL_SECONDS},
            "separation": "posthoc only; no Berkeley reference input to any OpenMVG SfM arm",
            "license_scope": "evaluation only; no shipping or GPL/AGPL clearance"}


def reference_cameras(reference: dict, started: float, output: Path = OUTPUT) -> tuple[dict, dict, dict]:
    import numpy as np
    from scripts.object_motion import checkerboard_pose_reference_eval as shared
    from scripts.object_motion.ycb_camera_reference import dataset, rigid
    files = reference["used_h5"]
    calib = Path(files[two.PREFIX + "calibration.h5"]["path"])
    np3_from_np5 = rigid(dataset(calib, "/H_NP3_from_NP5", (4, 4)), "mustard NP3 calibration")
    k = dataset(calib, "/NP3_rgb_K", (3, 3))
    d = dataset(calib, "/NP3_rgb_d", (5,))
    if not np.allclose(k[2], [0, 0, 1], atol=1e-10) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("invalid supplied NP3 RGB calibration")
    centers, rotations = {}, {}
    for name in COMMON:
        if time.monotonic() - started > WALL_SECONDS:
            raise TimeoutError("three-arm metadata read wall-time cap exceeded")
        capacity(output)
        member = two.PREFIX + f"poses/NP5_{int(name[4:7])}_pose.h5"
        pose = rigid(dataset(Path(files[member]["path"]), "/H_table_from_reference_camera", (4, 4)), name)
        np3_from_table = rigid(np3_from_np5 @ np.linalg.inv(pose), name + " NP3-from-table")
        rotations[name] = np3_from_table[:3, :3]
        centers[name] = shared.camera_center(np3_from_table[:3, :3], np3_from_table[:3, 3])
    return centers, rotations, {"K": k.tolist(), "distortion": d.tolist()}


def compare_three(arms: dict, target_centers: dict, target_rotations: dict) -> dict:
    from scripts.object_motion import checkerboard_pose_reference_eval as shared
    compared = {}
    for label, arm in arms.items():
        result = shared.compare(arm["centers"], arm["rotations"], target_centers, target_rotations)
        if result["matched_names"] != sorted(COMMON) or result["matched_count"] != 44:
            raise ValueError("three-arm Sim3 population differs from common 44")
        fit = result["fit"]
        fit["scale_berkeley_pose_units_per_openmvg_unit"] = fit.pop("scale_supplied_units_per_square")
        fit["world_translation_berkeley_pose_units"] = fit.pop("world_translation_supplied_units")
        fit["berkeley_pose_median_center_radius"] = fit.pop("supplied_median_center_radius")
        result["center_residual_berkeley_pose_units"] = result.pop("center_residual_supplied_units")
        for row in result["per_frame"].values():
            row["center_residual_berkeley_pose_units"] = row.pop("center_residual_supplied_units")
        compared[label] = result
    fields = ("center_residual_over_radius", "leave_one_out_center_residual_over_radius",
              "orientation_residual_degrees")
    pairs = (("fixed_none", "adjust_all"), ("global_adjust_all", "adjust_all"),
             ("global_adjust_all", "fixed_none"))
    deltas = {high + "_minus_" + low: {
        name: {field: compared[high]["per_frame"][name][field] -
               compared[low]["per_frame"][name][field] for field in fields}
        for name in sorted(COMMON)} for high, low in pairs}
    return {"arms": compared, "paired_per_frame_deltas": deltas,
            "fit_policy": "three independent proper positive-scale Sim3 fits on the same 44 names"}


def run(output: Path = OUTPUT) -> dict:
    checked = preflight(output)
    output.mkdir()
    receipt = {**checked, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    started = time.monotonic()
    try:
        arms, population = audit_exports()
        target_centers, target_rotations, calibration = reference_cameras(checked["reference"], started, output)
        comparison = compare_three(arms, target_centers, target_rotations)
        post_arms, post_population = audit_exports()
        if (post_population != population or
                {label: {key: arm[key] for key in checked["arms"][label]}
                 for label, arm in post_arms.items()} != checked["arms"] or
                audit_reference() != checked["reference"] or
                two.algorithm_runtime() != checked["algorithm_runtime"]):
            raise ValueError("sealed exports/reference/algorithm changed during three-arm scoring")
        if time.monotonic() - started > WALL_SECONDS:
            raise TimeoutError("three-arm comparison wall-time cap exceeded")
        report = {"schema": "openmvg_mustard_three_arm_gt_posthoc_result_v1",
                  "status": "posthoc_common_44_diagnostic_only", "inputs": checked,
                  "reference_calibration_diagnostic": calibration,
                  "fit_units": "source centers: arbitrary OpenMVG units; target centers: Berkeley pose-translation units, not board squares or independently established meters",
                  "transform_convention": "H_NP3_from_table = H_NP3_from_NP5 @ inverse(H_table_from_reference_camera); world-to-camera OpenCV axes",
                  "caveats": ["All three SfM models remain 46/48 incomplete; scoring uses only their common 44.",
                              "Every arm is refit on 44; prior two-arm 46-view fit errors are not reused.",
                              "Separate Sim3 fits remove arbitrary world frame and scale but do not prove absolute frame or mesh accuracy.",
                              "Orientation depends on NP3/NP5 matrix-name/axis convention; board-frame offset and depth were not independently validated.",
                              "Near-planar orbit weakens third-axis alignment; inspect covariance singular values and leave-one-out errors.",
                              "Berkeley K/poses were opened only posthoc, never fed into SfM."],
                  "comparison": comparison}
        encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
        if len(encoded) > OUTPUT_CAP - (64 << 10):
            raise ValueError("three-arm report exceeds small output cap")
        with (output / "report.json").open("xb") as stream:
            stream.write(encoded)
        capacity(output)
        receipt["report_sha256"] = two.sha(output / "report.json")
        receipt["report_bytes"] = len(encoded)
        receipt["status"] = "posthoc_common_44_diagnostic_only"
    except BaseException as exc:
        receipt["status"] = "stopped"
        receipt["reason"] = str(exc)
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["wall_seconds"] = round(time.monotonic() - started, 3)
        report_path = output / "report.json"
        if report_path.is_file() and not report_path.is_symlink():
            receipt["artifact_inventory"] = {"report.json": {
                "bytes": report_path.stat().st_size, "sha256": two.sha(report_path)}}
        (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        capacity(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="posthoc GT scoring; requires separate review")
    args = parser.parse_args()
    print(json.dumps(run() if args.run else preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
