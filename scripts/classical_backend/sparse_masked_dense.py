"""Bounded verified sparse-model → masked OpenMVS rough-mesh continuation.

The producer, bounded supervisor and separate root repair-acceptance gates
must all pass before undistortion or native stages. No reference geometry or
held-out photograph is used to reconstruct the model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.classical_backend import calibrated_control as audit
from scripts.classical_backend.calibrated_control import native_depth_size
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes
from scripts.classical_backend.run import (BIN, RESERVE, StageError, checked_ply,
                                           digest, folder_bytes, nonempty_bytes,
                                           stage, tool_path)


NATIVE_LEVEL = 0
NATIVE_MIN_RESOLUTION = 600
NATIVE_MAX_RESOLUTION = 1280
OUTPUT_CAP = 3 << 30
RSS_CAP = 4 << 30
LOG_CAP = 32 << 20
TIMEOUT_SECONDS = 600
ROOT = Path(__file__).resolve().parents[2]
REPAIR_GATES = frozenset({
    "retained_points_at_least_90pct",
    "retained_selected_observations_at_least_90pct",
    "all_registered_poses_and_intrinsics_unchanged",
    "zero_repeated_image_tracks",
    "all_selected_reprojections_finite",
    "all_selected_depths_positive",
    "common_original_mean_within_110pct",
    "common_original_p95_within_110pct",
    "dense_eligible_predeclared",
})


def validate_repair_metrics(report: dict) -> dict:
    """Fail closed on the repair's predeclared, stage-independent quality gates."""
    if (report.get("schema") != "mustard_sparse_track_repair_v1" or
            report.get("status") != "candidate_unreviewed"):
        raise ValueError("not the candidate-only sparse-track repair report")
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != REPAIR_GATES or any(value is not True for value in gates.values()):
        raise ValueError("repair fails a predeclared dense-eligibility gate")
    baseline, output = report.get("baseline"), report.get("output")
    if not isinstance(baseline, dict) or not isinstance(output, dict):
        raise ValueError("repair lacks baseline/output accounting")
    for key in ("points", "observations"):
        before, after = baseline.get(key), output.get(key)
        if (not isinstance(before, int) or isinstance(before, bool) or before <= 0 or
                not isinstance(after, int) or isinstance(after, bool) or after < 0 or
                after < 0.9 * before):
            raise ValueError(f"repair retained fewer than 90% of {key}")
    if output.get("duplicate_same_image_tracks") != 0:
        raise ValueError("repair retains repeated-image tracks")
    residuals = report.get("residuals")
    if not isinstance(residuals, dict):
        raise ValueError("repair lacks common-observation residual comparison")
    for key in ("mean_pixels", "p95_pixels"):
        before = residuals.get("baseline_common_original", {}).get(key)
        after = residuals.get("repaired_all_original_common", {}).get(key)
        if (not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or
                not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0 or
                after > 1.10 * before):
            raise ValueError(f"repair worsens common-original {key} beyond 10%")
    repair = report.get("repair")
    common = repair.get("surviving_point_original_observations") if isinstance(repair, dict) else None
    selected = residuals.get("repaired_selected", {})
    if (not isinstance(common, int) or common < output["observations"] or
            repair.get("preserved_camera_and_pose") is not True or
            repair.get("backlink_valid") is not True or
            repair.get("duplicate_track_count_after") != 0 or
            any(residuals[part].get("denominator") != common or
                residuals[part].get("invalid_count") != 0
                for part in ("baseline_common_original", "repaired_all_original_common")) or
            selected.get("denominator") != output["observations"] or
            selected.get("invalid_count") != 0):
        raise ValueError("repair common/selected residual accounting is incomplete")
    return {"points": output["points"], "observations": output["observations"],
            "gates": sorted(gates)}


def mask_warp_manifest(stage: dict, expected_names: list[str]) -> dict:
    """Adapt a sealed TRAIN stage's mask map to dense_masks' exact-name contract."""
    names = stage.get("train_names")
    masks = stage.get("cleaned_masks")
    if (stage.get("schema") != "mustard_train_only_stage_v1" or
            stage.get("status") != "complete" or names != expected_names or
            not isinstance(masks, dict) or set(masks) != set(expected_names) or
            len(expected_names) != len(set(expected_names))):
        raise ValueError("stage does not contain exactly the expected TRAIN masks")
    rows = []
    for name in expected_names:
        mask_hash = masks[name].get("sha256") if isinstance(masks[name], dict) else None
        if (not isinstance(mask_hash, str) or len(mask_hash) != 64 or
                any(char not in "0123456789abcdef" for char in mask_hash)):
            raise ValueError(f"stage mask hash invalid: {name}")
        rows.append({"name": name, "mask_sha256": mask_hash})
    return {"schema": "classical_exact_source_masks_v1", "images": rows}


def validate_seals(report: dict, supervisor: dict, qa: dict, stage: dict, *,
                   report_sha: str, supervisor_sha: str,
                   baseline_result_sha: str, baseline_model_hashes: dict[str, str],
                   stage_report_sha: str, output_model_hashes: dict[str, str]) -> list[str]:
    """Bind accepted repair to frozen image-only SfM and exact TRAIN stage.

    File bytes and the binary reconstruction must also be inspected by the
    caller; self-reported producer fields are never sufficient by themselves.
    """
    validate_repair_metrics(report)
    if (supervisor.get("schema") != "mustard_sparse_track_repair_supervisor_v1" or
            supervisor.get("status") != "candidate_unreviewed" or
            supervisor.get("reason") is not None or
            not isinstance(supervisor.get("seconds"), (int, float)) or
            not 0 <= supervisor["seconds"] <= 600 or
            not isinstance(supervisor.get("peak_worker_rss_kib"), int) or
            not 0 <= supervisor["peak_worker_rss_kib"] <= 4 * 1024**2):
        raise ValueError("repair supervisor did not complete within approved bounds")
    if (qa.get("schema") != "mustard_sparse_track_repair_qa_v1" or
            qa.get("status") != "accepted" or
            qa.get("report_sha256") != report_sha or
            qa.get("supervisor_sha256") != supervisor_sha or
            qa.get("model_files_sha256") != output_model_hashes):
        raise ValueError("root repair QA does not accept these exact producer/model bytes")
    baseline, producer_stage, output = report["baseline"], report.get("stage"), report["output"]
    if (baseline.get("result_sha256") != baseline_result_sha or
            baseline.get("model_files_sha256") != baseline_model_hashes or
            not isinstance(producer_stage, dict) or
            producer_stage.get("report_sha256") != stage_report_sha or
            output.get("model_files_sha256") != output_model_hashes):
        raise ValueError("repair lineage differs from frozen SfM/stage/model bytes")
    names = stage.get("train_names")
    if not isinstance(names, list) or len(names) != 48 or len(set(names)) != 48:
        raise ValueError("repair does not cover exactly 48 TRAIN names")
    mask_manifest = mask_warp_manifest(stage, names)
    if (not isinstance(stage.get("train_photo_sha256"), dict) or
            set(stage["train_photo_sha256"]) != set(names) or
            baseline.get("registered_names") != names or output.get("registered_names") != names or
            producer_stage.get("train_photo_sha256") != stage.get("train_photo_sha256") or
            producer_stage.get("cleaned_mask_sha256") !=
            {row["name"]: row["mask_sha256"] for row in mask_manifest["images"]}):
        raise ValueError("repair and TRAIN stage do not bind identical 48 photos/masks")
    return names


def native_plan(output: Path, binary_dir: Path) -> list[tuple[str, list[str]]]:
    """Frozen, masked full-detail CPU import → depth → rough-mesh handoff."""
    output, binary_dir = output.resolve(), binary_dir.resolve()
    dense = output / "dense"
    common = ["--max-threads", "2", "--working-folder", str(output)]
    return [
        ("import", [str(tool_path(binary_dir, "InterfaceCOLMAP")), "-i", str(dense),
                    "-o", str(output / "scene.mvs"), "--image-folder", str(dense / "images"),
                    *common]),
        ("densify", [str(tool_path(binary_dir, "DensifyPointCloud")), "-i",
                     str(output / "scene.mvs"), "-o", str(output / "dense.mvs"),
                     "--resolution-level", str(NATIVE_LEVEL), "--min-resolution",
                     str(NATIVE_MIN_RESOLUTION), "--max-resolution",
                     str(NATIVE_MAX_RESOLUTION), "--geometric-iters", "2",
                     "--tower-mode", "4", "--mask-path", str(output / "masks"),
                     "--ignore-mask-label", "0", *common]),
        ("rough_mesh", [str(tool_path(binary_dir, "ReconstructMesh")), "-i",
                        str(output / "dense.mvs"), "-p", str(output / "dense.ply"),
                        "-o", str(output / "mesh.mvs"), *common]),
    ]


def depth_budget(image_sizes: dict[str, tuple[int, int]], existing_bytes: int,
                 cap_bytes: int = OUTPUT_CAP) -> dict:
    """Conservatively budget two OpenMVS geometric-depth generations.

    OpenMVS v2.4.0 stores depth, normal and confidence at 20 bytes/pixel.
    The estimate is a preflight rejection gate, not a prediction of peak RSS
    or a substitute for the runtime folder-size and disk-reserve guards.
    """
    if (not image_sizes or existing_bytes < 0 or cap_bytes <= 0 or
            len(image_sizes) != len(set(image_sizes))):
        raise ValueError("invalid native depth budget inputs")
    rows = {name: native_depth_size(*size, NATIVE_LEVEL, NATIVE_MIN_RESOLUTION,
                                    NATIVE_MAX_RESOLUTION)
            for name, size in image_sizes.items()}
    if any(row["actual_level"] != NATIVE_LEVEL for row in rows.values()):
        raise ValueError("native depth resolution differs from frozen level-0 profile")
    pixels = sum(math.prod(row["pixel_budget_size"]) for row in rows.values())
    predicted = existing_bytes + 2 * (20 * pixels + 4096 * len(rows)) + (96 << 20)
    if predicted > cap_bytes:
        raise ValueError("predicted two-generation native depth peak exceeds output cap")
    return {"source": "OpenMVS v2.4.0 TImage::computeMaxResolution",
            "requested_level": NATIVE_LEVEL,
            "min_resolution": NATIVE_MIN_RESOLUTION,
            "max_resolution": NATIVE_MAX_RESOLUTION,
            "predicted_two_generation_peak_bytes": predicted,
            "hard_output_cap_bytes": cap_bytes, "images": rows}


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite_positive_depth(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def worker_python() -> str:
    """Preserve the active venv executable even when it links to system Python."""
    return str(Path(sys.executable).absolute())


def _real(path: Path, *, directory: bool) -> Path:
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"missing, linked or wrong-kind input: {path}")
    return path.resolve()


def verify_binary_geometry(baseline_dir: Path, repaired_dir: Path,
                           names: list[str], expected_points: int,
                           expected_observations: int) -> dict:
    """Independently reject changed cameras, bad links and duplicate-view tracks."""
    import pycolmap
    original = pycolmap.Reconstruction(str(baseline_dir))
    repaired = pycolmap.Reconstruction(str(repaired_dir))
    source_images = {image.name: image for image in original.images.values() if image.has_pose}
    output_images = {image.name: image for image in repaired.images.values() if image.has_pose}
    if (set(source_images) != set(names) or set(output_images) != set(names) or
            len(original.cameras) != 1 or len(repaired.cameras) != 1 or
            repaired.num_points3D() != expected_points or
            set(original.images) != set(repaired.images)):
        raise ValueError("repaired binary changes registered images, cameras or point count")
    old_camera = next(iter(original.cameras.values()))
    new_camera = next(iter(repaired.cameras.values()))
    if (old_camera.model.name != "SIMPLE_RADIAL" or old_camera.width != 1280 or
            old_camera.height != 1024 or not np.array_equal(old_camera.params, new_camera.params) or
            old_camera.model.name != new_camera.model.name or
            (old_camera.width, old_camera.height) != (new_camera.width, new_camera.height) or
            not np.isfinite(new_camera.params).all() or not 640 <= new_camera.params[0] <= 3200 or
            abs(new_camera.params[3]) > 1.0):
        raise ValueError("repaired binary camera intrinsics changed or implausible")
    for name in names:
        old, new = source_images[name], output_images[name]
        if (old.image_id != new.image_id or old.camera_id != new.camera_id or
                not np.array_equal(old.cam_from_world.matrix(), new.cam_from_world.matrix()) or
                not np.isfinite(new.cam_from_world.matrix()).all() or
                len(old.points2D) != len(new.points2D) or
                not np.array_equal(np.asarray([item.xy for item in old.points2D]),
                                   np.asarray([item.xy for item in new.points2D]))):
            raise ValueError(f"repaired binary pose changed: {name}")
    clean_points = 0
    for point_id, old_point in original.points3D.items():
        old_links = [(int(item.image_id), int(item.point2D_idx))
                     for item in old_point.track.elements]
        if len({image_id for image_id, _ in old_links}) != len(old_links):
            continue
        if point_id not in repaired.points3D:
            raise ValueError("repair deleted an originally clean track")
        new_point = repaired.points3D[point_id]
        new_links = [(int(item.image_id), int(item.point2D_idx))
                     for item in new_point.track.elements]
        if (not np.array_equal(old_point.xyz, new_point.xyz) or
                sorted(old_links) != sorted(new_links)):
            raise ValueError("repair changed an originally clean point or track")
        clean_points += 1
    observed = 0
    for point_id, point in repaired.points3D.items():
        if not np.isfinite(point.xyz).all():
            raise ValueError("nonfinite repaired 3D point")
        seen = set()
        for item in point.track.elements:
            image_id, index = int(item.image_id), int(item.point2D_idx)
            if image_id in seen:
                raise ValueError("repaired 3D track repeats an image")
            seen.add(image_id)
            image = repaired.images[image_id] if image_id in repaired.images else None
            depth = (image.cam_from_world * point.xyz)[2] if image is not None else None
            if (image is None or not image.has_pose or not 0 <= index < len(image.points2D) or
                    int(image.points2D[index].point3D_id) != int(point_id) or
                    not np.isfinite(image.points2D[index].xy).all() or
                    not finite_positive_depth(depth)):
                raise ValueError("repaired track backlink, pixel or depth invalid")
            observed += 1
        if len(seen) < 2:
            raise ValueError("repaired 3D point lacks two distinct observations")
    if observed != expected_observations:
        raise ValueError("repaired binary observation count differs from producer")
    return {"registered": len(names), "points": expected_points, "observations": observed,
            "unchanged_clean_points": clean_points,
            "camera_model": new_camera.model.name,
            "camera_params": [float(value) for value in new_camera.params]}


def validate_inputs(args: argparse.Namespace) -> dict:
    paths = {name: _real(path, directory=directory) for name, path, directory in (
        ("repair_report", args.repair_report, False), ("repair_qa", args.repair_qa, False),
        ("baseline_run", args.baseline_run, True), ("stage_root", args.stage_root, True),
        ("model", args.model, True))}
    if any(not _sha(value) for value in (args.repair_report_sha256, args.supervisor_sha256,
                                          args.repair_qa_sha256,
                                          args.baseline_result_sha256, args.stage_report_sha256)):
        raise ValueError("all five source report SHA-256 seals are required")
    source_files = {"repair_report": paths["repair_report"], "repair_qa": paths["repair_qa"],
                    "supervisor": paths["repair_report"].parent / "supervisor.json",
                    "baseline_result": paths["baseline_run"] / "result.json",
                    "stage_report": paths["stage_root"] / "stage-report.json"}
    expected = {"repair_report": args.repair_report_sha256,
                "supervisor": args.supervisor_sha256, "repair_qa": args.repair_qa_sha256,
                "baseline_result": args.baseline_result_sha256,
                "stage_report": args.stage_report_sha256}
    if any(_real(path, directory=False) != path or digest(path) != expected[name]
           for name, path in source_files.items()):
        raise ValueError("sealed source report changed")
    report = json.loads(source_files["repair_report"].read_text())
    supervisor = json.loads(source_files["supervisor"].read_text())
    qa = json.loads(source_files["repair_qa"].read_text())
    baseline = json.loads(source_files["baseline_result"].read_text())
    train = json.loads(source_files["stage_report"].read_text())
    import pycolmap
    from scripts.object_motion import (mustard_sparse_support, sam_m1_parity,
                                       sparse_track_repair)
    producer_sources = {"runner_sha256": sparse_track_repair.__file__,
                        "helper_sha256": mustard_sparse_support.__file__,
                        "resource_helper_sha256": sam_m1_parity.__file__}
    if (any(report.get(key) != digest(Path(path)) for key, path in producer_sources.items()) or
            report.get("software") != {"pycolmap_version": pycolmap.__version__,
                                       "pycolmap_core_sha256": digest(Path(pycolmap._core.__file__))} or
            pycolmap.__version__ != "3.11.1"):
        raise ValueError("repair producer source or PyCOLMAP binary/version differs")
    if (baseline.get("schema") != "classical_backend_v1" or baseline.get("status") != "sparse_complete" or
            baseline.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap" or
            baseline["sfm_source"].get("intrinsics_policy") != "fixed-initial" or
            baseline["sfm_source"].get("matching") != "exhaustive"):
        raise ValueError("baseline must be sealed image-only fixed/exhaustive sparse SfM")
    baseline_model = paths["baseline_run"] / "sparse" / "0"
    _real(baseline_model, directory=True)
    if report.get("baseline", {}).get("sfm_sha256") != digest(paths["baseline_run"] / "sfm.json"):
        raise ValueError("repair does not bind the saved baseline SfM selection")
    base_hashes, output_hashes = model_hashes(baseline_model), model_hashes(paths["model"])
    names = validate_seals(report, supervisor, qa, train,
                           report_sha=expected["repair_report"], supervisor_sha=expected["supervisor"],
                           baseline_result_sha=expected["baseline_result"],
                           baseline_model_hashes=base_hashes,
                           stage_report_sha=expected["stage_report"],
                           output_model_hashes=output_hashes)
    staged_images = _real(paths["stage_root"] / "images", directory=True)
    staged_masks = _real(paths["stage_root"] / "masks", directory=True)
    if ({path.name for path in staged_images.iterdir()} != set(names) or
            {path.name for path in staged_masks.iterdir()} != {name + ".png" for name in names} or
            {path.name for path in paths["model"].iterdir()} != set(MODEL_FILES) or
            {path.name for path in baseline_model.iterdir()} != set(MODEL_FILES)):
        raise ValueError("source contains extra/missing photo, mask or model files")
    if Path(report["output"].get("model_dir", "")).resolve() != paths["model"]:
        raise ValueError("repair report model path differs")
    if ([row.get("name") for row in baseline.get("inputs", [])] != names or
            {row["name"]: row["sha256"] for row in baseline["inputs"]} != train["train_photo_sha256"] or
            {row["name"][:-4]: row["sha256"] for row in baseline.get("pose_masks", [])} !=
            {name: row["sha256"] for name, row in train["cleaned_masks"].items()}):
        raise ValueError("baseline photos/pose masks differ from accepted TRAIN stage")
    for name in names:
        photo = _real(paths["stage_root"] / "images" / name, directory=False)
        mask = _real(paths["stage_root"] / "masks" / (name + ".png"), directory=False)
        if (digest(photo) != train["train_photo_sha256"][name] or
                digest(mask) != train["cleaned_masks"][name]["sha256"]):
            raise ValueError(f"TRAIN photo or mask differs: {name}")
        with Image.open(photo) as image, Image.open(mask) as label:
            if image.size != (1280, 1024) or label.size != image.size or label.mode != "L":
                raise ValueError(f"TRAIN photo/mask dimensions or mask mode differ: {name}")
    geometry = verify_binary_geometry(baseline_model, paths["model"], names,
                                      report["output"]["points"], report["output"]["observations"])
    if geometry["unchanged_clean_points"] != report.get("repair", {}).get("unchanged_clean_points"):
        raise ValueError("clean-track preservation count differs from repaired binary")
    if any(digest(path) != expected[name] for name, path in source_files.items()):
        raise ValueError("sealed source report changed during validation")
    if model_hashes(baseline_model) != base_hashes or model_hashes(paths["model"]) != output_hashes:
        raise ValueError("source model changed during validation")
    return {"paths": paths, "report": report, "supervisor": supervisor, "qa": qa,
            "baseline": baseline, "stage": train,
            "source_files": source_files, "source_hashes": expected,
            "producer_source_files": producer_sources,
            "baseline_model_hashes": base_hashes, "output_model_hashes": output_hashes,
            "names": names, "geometry": geometry}


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    from scripts.object_motion import sam_m1_parity as mac
    mac.host_preflight(args.output, check_memory=False)
    if (args.output.exists() or args.output.is_symlink() or args.output.parent.is_symlink() or
            not args.output.parent.is_dir()):
        raise ValueError("native output must be fresh under an existing real parent")
    bound = validate_inputs(args)
    import pycolmap
    output = args.output.resolve()
    binary_dir = _real(args.binary_dir, directory=True)
    tools = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh")
    binary_hashes = {name: digest(_real(tool_path(binary_dir, name), directory=False)) for name in tools}
    input_bytes = sum((bound["paths"]["stage_root"] / "images" / name).stat().st_size +
                      (bound["paths"]["stage_root"] / "masks" / (name + ".png")).stat().st_size
                      for name in bound["names"])
    input_bytes += sum((bound["paths"]["model"] / name).stat().st_size for name in MODEL_FILES)
    if (input_bytes >= OUTPUT_CAP or
            shutil.disk_usage(output.parent).free < RESERVE + OUTPUT_CAP + (256 << 20) or
            shutil.disk_usage(ROOT).free < RESERVE + (256 << 20)):
        raise ValueError("3 GiB cap or dual 10 GiB disk floor plus margin unavailable")
    output.mkdir()
    report = {"schema": "classical_verified_sparse_masked_dense_v1", "status": "running",
              "provenance_class": "image-only masked fixed-initial exhaustive SfM; separately repaired and root-accepted",
              "repair_report_sha256": bound["source_hashes"]["repair_report"],
              "supervisor_sha256": bound["source_hashes"]["supervisor"],
              "repair_qa_sha256": bound["source_hashes"]["repair_qa"],
              "baseline_result_sha256": bound["source_hashes"]["baseline_result"],
              "stage_report_sha256": bound["source_hashes"]["stage_report"],
              "baseline_model_sha256": bound["baseline_model_hashes"],
              "source_model_sha256": bound["output_model_hashes"],
              "source_photo_sha256": bound["stage"]["train_photo_sha256"],
              "source_mask_sha256": {name: bound["stage"]["cleaned_masks"][name]["sha256"]
                                     for name in bound["names"]},
              "source_geometry": bound["geometry"],
              "software": {"runner_sha256": digest(Path(__file__)),
                           "python_path": worker_python(),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_core_sha256": digest(Path(pycolmap._core.__file__)),
                           "openmvs_binaries": binary_hashes},
              "native_options": {"resolution_level": NATIVE_LEVEL,
                                 "min_resolution": NATIVE_MIN_RESOLUTION,
                                 "max_resolution": NATIVE_MAX_RESOLUTION,
                                 "geometric_iters": 2, "tower_mode": 4,
                                 "ignore_mask_label": 0, "max_threads": 2},
              "limits": {"max_output_bytes": OUTPUT_CAP, "max_child_rss_bytes": RSS_CAP,
                         "max_log_bytes": LOG_CAP, "timeout_seconds": TIMEOUT_SECONDS,
                         "min_free_bytes_each_disk": RESERVE},
              "stages": [], "quality_accepted": False, "metric_scale_verified": False,
              "shipping_approved": False}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    deadline = started + TIMEOUT_SECONDS
    save()

    def execute(name: str, command: list[str], validate) -> None:
        result = stage(output, name, command, deadline, OUTPUT_CAP, LOG_CAP, RSS_CAP,
                       extra_reserve_paths=(ROOT,))
        try:
            result["artifact"] = validate()
        except Exception as error:
            result.update(status="failed", failure=str(error))
            raise StageError(result) from error
        report["stages"].append(result)
        save()

    try:
        copy = {"name": "copy_inputs", "status": "running"}
        report["stages"].append(copy)
        save()
        (output / "images").mkdir()
        (output / "source-masks").mkdir()
        sparse = output / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in bound["names"]:
            for source, target, checksum in (
                (bound["paths"]["stage_root"] / "images" / name, output / "images" / name,
                 bound["stage"]["train_photo_sha256"][name]),
                (bound["paths"]["stage_root"] / "masks" / (name + ".png"),
                 output / "source-masks" / (name + ".png"),
                 bound["stage"]["cleaned_masks"][name]["sha256"])):
                if (time.monotonic() > deadline or
                        shutil.disk_usage(output).free < RESERVE + source.stat().st_size or
                        shutil.disk_usage(ROOT).free < RESERVE or
                        folder_bytes(output) + source.stat().st_size > OUTPUT_CAP):
                    raise ValueError("copy deadline, disk floor or output cap reached")
                shutil.copyfile(source, target)
                if digest(target) != checksum:
                    raise ValueError(f"copied input hash differs: {name}")
        for name in MODEL_FILES:
            source, target = bound["paths"]["model"] / name, sparse / name
            if time.monotonic() > deadline or folder_bytes(output) + source.stat().st_size > OUTPUT_CAP:
                raise ValueError("model copy deadline or output cap reached")
            shutil.copyfile(source, target)
        if model_hashes(sparse) != bound["output_model_hashes"]:
            raise ValueError("copied repaired model differs")
        manifest = mask_warp_manifest(bound["stage"], bound["names"])
        (output / "source-mask-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        copy.update(status="complete", artifact={"photos": 48, "masks": 48, "model_files": 3})
        save()
        # The venv executable may be a symlink to the system interpreter.
        # Preserve its path so subprocesses inherit the same PyCOLMAP site-packages.
        python = worker_python()
        execute("undistort", [python, "-m", "scripts.classical_backend.calibrated_control",
                              "--worker", "undistort", "--output", str(output)],
                lambda: audit.validate_undistorted(output, bound["names"]))
        execute("warp_masks", [python, "-m", "scripts.classical_backend.dense_masks",
                               "--source-model", str(sparse),
                               "--undistorted-model", str(output / "dense" / "sparse"),
                               "--source-masks", str(output / "source-masks"),
                               "--undistorted-images", str(output / "dense" / "images"),
                               "--manifest", str(output / "source-mask-manifest.json"),
                               "--output", str(output / "masks")],
                lambda: {"views": audit.mask_count(output / "masks" / "report.json", 48)})
        sizes = {}
        for name in bound["names"]:
            with Image.open(output / "dense" / "images" / name) as image:
                sizes[name] = image.size
        budget = depth_budget(sizes, folder_bytes(output))
        if (shutil.disk_usage(output).free < RESERVE + OUTPUT_CAP - folder_bytes(output) or
                shutil.disk_usage(ROOT).free < RESERVE + (256 << 20)):
            raise ValueError("native preflight disk floor plus remaining cap unavailable")
        report["native_depth_preflight"] = budget
        save()
        plan = dict(native_plan(output, binary_dir))
        execute("import", plan["import"],
                lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
        execute("densify", plan["densify"],
                lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
        execute("dmap_camera_check", [python, "-m", "scripts.classical_backend.calibrated_control",
                                      "--worker", "dmap_check", "--output", str(output)],
                lambda: {"views": audit.checked_dmap_count(output / "dmap-camera-check.json", 48)})
        execute("rough_mesh", plan["rough_mesh"],
                lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        if report["stages"] and report["stages"][-1].get("status") == "running":
            report["stages"][-1].update(status="failed", failure=str(error))
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = {"output": shutil.disk_usage(output).free,
                                  "internal": shutil.disk_usage(ROOT).free}
    if (time.monotonic() > deadline or report["output_bytes"] > OUTPUT_CAP or
            min(report["free_bytes_after"].values()) < RESERVE):
        report.update(status="failed", failure="postflight time/output/disk cap exceeded")
    try:
        validate_inputs(args)
        report["sources_unchanged"] = (
            digest(Path(__file__)) == report["software"]["runner_sha256"] and
            digest(Path(pycolmap._core.__file__)) == report["software"]["pycolmap_core_sha256"] and
            all(digest(tool_path(binary_dir, name)) == value for name, value in binary_hashes.items()))
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"]:
        report.update(status="failed", failure="producer, photos, masks, model or native binary changed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repair-report", "repair-qa", "baseline-run", "stage-root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("repair-report-sha256", "supervisor-sha256", "repair-qa-sha256", "baseline-result-sha256",
                 "stage-report-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
