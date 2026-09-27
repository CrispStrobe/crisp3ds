#!/usr/bin/env python3
"""Candidate-only duplicate-view sparse track repair with frozen cameras."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from scripts.object_motion import sam_m1_parity as mac
from scripts.object_motion import mustard_sparse_support as support
from scripts.object_motion import ycb_object_masks as common


FILES = ("cameras.bin", "images.bin", "points3D.bin")
SEED = 20260927
MAX_SECONDS = 600
MAX_RSS_KIB = 4 * 1024**2
MAX_OUTPUT = 100 * 1024**2
MAX_LOG = 4 * 1024**2


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    left = int(pos)
    return ordered[left] * (1 - pos + left) + ordered[min(left + 1, len(ordered) - 1)] * (pos - left)


def residual_summary(values, denominator):
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {"denominator": denominator, "finite_count": len(finite),
            "invalid_count": denominator - len(finite),
            "mean_pixels": statistics.fmean(finite) if finite else None,
            "p95_pixels": percentile(finite, 0.95)}


def measured_observation(model, point_id, element, xyz):
    image = model.images[int(element.image_id)]
    index = int(element.point2D_idx)
    if not 0 <= index < len(image.points2D):
        raise ValueError("track observation index outside image")
    measured = image.points2D[index]
    if int(measured.point3D_id) != int(point_id):
        raise ValueError("track-to-image backlink differs")
    xy = np.asarray(measured.xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("nonfinite original measured observation")
    projected = image.project_point(xyz)
    if projected is None:
        residual = None
    else:
        projected = np.asarray(projected, dtype=float)
        residual = float(np.linalg.norm(projected - xy)) if (
            projected.shape == (2,) and np.isfinite(projected).all()) else None
    return {"image_id": int(element.image_id), "point2D_idx": index,
            "xy": xy.tolist(), "baseline_residual": residual}


def select_observations(records):
    """One minimum finite baseline reprojection per image; tie by 2D index."""
    groups = defaultdict(list)
    for record in records:
        groups[record["image_id"]].append(record)
    selected, redundant = [], []
    for image_id in sorted(groups):
        candidates = groups[image_id]
        finite = [item for item in candidates if item["baseline_residual"] is not None and
                  math.isfinite(item["baseline_residual"])]
        if not finite:
            return [], list(records), "no_finite_reprojection_in_view"
        winner = min(finite, key=lambda item: (item["baseline_residual"], item["point2D_idx"]))
        selected.append(winner)
        redundant.extend(item for item in candidates if item is not winner)
    if len(selected) < 2:
        return [], list(records), "fewer_than_two_distinct_views"
    return selected, redundant, None


def frozen_cameras_and_poses(original, candidate):
    if (set(original.cameras) != set(candidate.cameras) or
            set(original.images) != set(candidate.images) or
            set(original.reg_image_ids()) != set(candidate.reg_image_ids())):
        return False
    for camera_id, old in original.cameras.items():
        new = candidate.cameras[camera_id]
        if (old.model.name != new.model.name or old.width != new.width or old.height != new.height or
                not np.array_equal(old.params, new.params)):
            return False
    for image_id, old in original.images.items():
        new = candidate.images[image_id]
        if (old.name != new.name or old.camera_id != new.camera_id or
                not np.isfinite(old.cam_from_world.matrix()).all() or
                not np.isfinite(new.cam_from_world.matrix()).all() or
                not np.array_equal(old.cam_from_world.matrix(), new.cam_from_world.matrix()) or
                len(old.points2D) != len(new.points2D) or
                any(not np.array_equal(a.xy, b.xy) for a, b in zip(old.points2D, new.points2D))):
            return False
    return True


def track_inventory(model):
    repeated, observations = 0, 0
    for point in model.points3D.values():
        ids = [int(element.image_id) for element in point.track.elements]
        observations += len(ids)
        repeated += len(ids) != len(set(ids))
    return {"points": len(model.points3D), "observations": observations,
            "duplicate_same_image_tracks": repeated}


def repair_model(original, candidate, estimator, options):
    """Mutate only candidate, never original; return same-population metrics."""
    if not frozen_cameras_and_poses(original, candidate):
        raise ValueError("candidate cameras/poses/2D measurements differ before repair")
    baseline = track_inventory(original)
    before_by_point, chosen_by_point = {}, {}
    selected_manifest = []
    dropped = Counter()
    removed = 0
    unchanged_clean_points = 0
    baseline_all = []
    for point_id in sorted(original.points3D):
        point = original.points3D[point_id]
        records = [measured_observation(original, point_id, element, point.xyz)
                   for element in list(point.track.elements)]
        if len({(item["image_id"], item["point2D_idx"]) for item in records}) != len(records):
            raise ValueError("exact duplicate track observation")
        before_by_point[point_id] = records
        baseline_all.extend(item["baseline_residual"] for item in records)
        selected_keys = {(item["image_id"], item["point2D_idx"]) for item in records}
        if len({item["image_id"] for item in records}) == len(records):
            # Clean tracks are controls: leave their XYZ and 2D links unchanged.
            chosen_by_point[point_id] = selected_keys
            unchanged_clean_points += 1
            continue
        selected, redundant, reason = select_observations(records)
        if reason:
            candidate.delete_point3D(point_id)
            dropped[reason] += 1
            removed += len(records)
            continue
        # Copy IDs before mutating the track: pycolmap updates both the track
        # and point2D backlink and auto-deletes a two-element point.
        for item in sorted(redundant, key=lambda row: (row["image_id"], row["point2D_idx"])):
            if point_id not in candidate.points3D or len(candidate.points3D[point_id].track.elements) <= 2:
                raise ValueError("deletion would auto-delete a two-observation point")
            candidate.delete_observation(item["image_id"], item["point2D_idx"])
            removed += 1
        selected_keys = {(item["image_id"], item["point2D_idx"]) for item in selected}
        actual_keys = {(int(e.image_id), int(e.point2D_idx))
                       for e in candidate.points3D[point_id].track.elements}
        if actual_keys != selected_keys or len(actual_keys) != len(selected):
            raise ValueError("native deletion did not produce selected track")
        xy = np.asarray([item["xy"] for item in selected], dtype=np.float64)
        poses = [candidate.images[item["image_id"]].cam_from_world for item in selected]
        cameras = [candidate.cameras[candidate.images[item["image_id"]].camera_id] for item in selected]
        estimate = estimator(xy, poses, cameras, options)
        if estimate is None or set(estimate) != {"xyz", "inliers"}:
            candidate.delete_point3D(point_id)
            dropped["triangulation_failed"] += 1
            removed += len(selected)
            continue
        xyz = np.asarray(estimate["xyz"], dtype=np.float64)
        inliers = np.asarray(estimate["inliers"], dtype=bool)
        if (xyz.shape != (3,) or not np.isfinite(xyz).all() or
                inliers.shape != (len(selected),) or not inliers.all()):
            candidate.delete_point3D(point_id)
            dropped["triangulation_nonfinite_or_partial_inliers"] += 1
            removed += len(selected)
            continue
        depths = [float((candidate.images[item["image_id"]].cam_from_world * xyz)[2])
                  for item in selected]
        if any(not math.isfinite(depth) or depth <= 0 for depth in depths):
            candidate.delete_point3D(point_id)
            dropped["nonpositive_selected_depth"] += 1
            removed += len(selected)
            continue
        candidate.points3D[point_id].xyz = xyz
        chosen_by_point[point_id] = selected_keys
        selected_manifest.append([point_id, sorted(selected_keys)])
    candidate.update_point_3d_errors()
    if any(not math.isfinite(float(point.error)) or float(point.error) < 0
           for point in candidate.points3D.values()):
        raise ValueError("recomputed point errors are nonfinite or negative")
    candidate.check()
    if not frozen_cameras_and_poses(original, candidate):
        raise ValueError("camera/pose/2D measurements changed during repair")
    for point_id, records in before_by_point.items():
        if len({item["image_id"] for item in records}) == len(records):
            if (point_id not in candidate.points3D or
                    not np.array_equal(original.points3D[point_id].xyz, candidate.points3D[point_id].xyz) or
                    [(int(e.image_id), int(e.point2D_idx)) for e in original.points3D[point_id].track.elements] !=
                    [(int(e.image_id), int(e.point2D_idx)) for e in candidate.points3D[point_id].track.elements]):
                raise ValueError("clean control point geometry or track changed")
    after = track_inventory(candidate)
    if after["duplicate_same_image_tracks"]:
        raise ValueError("repaired model still contains repeated image tracks")
    repaired_all_common, baseline_common, repaired_selected, baseline_selected = [], [], [], []
    surviving_original = 0
    for point_id, records in before_by_point.items():
        if point_id not in candidate.points3D:
            continue
        surviving_original += len(records)
        xyz = candidate.points3D[point_id].xyz
        for item in records:
            image = candidate.images[item["image_id"]]
            projected = image.project_point(xyz)
            residual = None if projected is None else float(np.linalg.norm(
                np.asarray(projected, dtype=float) - np.asarray(item["xy"], dtype=float)))
            if residual is not None and not math.isfinite(residual):
                residual = None
            repaired_all_common.append(residual)
            baseline_common.append(item["baseline_residual"])
            if (item["image_id"], item["point2D_idx"]) in chosen_by_point[point_id]:
                repaired_selected.append(residual)
                baseline_selected.append(item["baseline_residual"])
    if after["observations"] != len(repaired_selected) or removed != baseline["observations"] - after["observations"]:
        raise ValueError("repair observation accounting mismatch")
    selection_hash = hashlib.sha256(json.dumps(selected_manifest, separators=(",", ":")).encode()).hexdigest()
    residuals = {"baseline_all_original": residual_summary(baseline_all, baseline["observations"]),
                 "baseline_common_original": residual_summary(baseline_common, surviving_original),
                 "repaired_all_original_common": residual_summary(repaired_all_common, surviving_original),
                 "baseline_selected": residual_summary(baseline_selected, after["observations"]),
                 "repaired_selected": residual_summary(repaired_selected, after["observations"])}
    gates = {"retained_points_at_least_90pct": after["points"] >= 0.9 * baseline["points"],
             "retained_selected_observations_at_least_90pct":
                 after["observations"] >= 0.9 * baseline["observations"],
             "all_registered_poses_and_intrinsics_unchanged": frozen_cameras_and_poses(original, candidate),
             "zero_repeated_image_tracks": after["duplicate_same_image_tracks"] == 0,
             "all_selected_reprojections_finite": residuals["repaired_selected"]["invalid_count"] == 0,
             "all_selected_depths_positive": True,
             "common_original_mean_within_110pct": False,
             "common_original_p95_within_110pct": False}
    for statistic, key in (("mean_pixels", "common_original_mean_within_110pct"),
                           ("p95_pixels", "common_original_p95_within_110pct")):
        before, now = (residuals["baseline_common_original"][statistic],
                       residuals["repaired_all_original_common"][statistic])
        gates[key] = (before is not None and now is not None and
                      residuals["baseline_common_original"]["invalid_count"] == 0 and
                      residuals["repaired_all_original_common"]["invalid_count"] == 0 and
                      now <= 1.10 * before)
    gates["dense_eligible_predeclared"] = all(gates.values())
    return {"baseline": baseline, "output": after,
            "removed_observations": removed, "dropped_points_by_reason": dict(dropped),
            "unchanged_clean_points": unchanged_clean_points,
            "surviving_point_original_observations": surviving_original,
            "selection_sha256": selection_hash, "residuals": residuals, "gates": gates}


def _regular(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"missing, linked, empty or oversized source: {path}")


def source_contract(args):
    run, stage_root = Path(args.run), Path(args.stage_root)
    if (run.is_symlink() or stage_root.is_symlink() or
            any(path.is_symlink() or not path.is_dir() for path in (run / "images", run / "masks"))):
        raise ValueError("linked or missing source run/photo/mask directory")
    result_path, sfm_path = run / "result.json", run / "sfm.json"
    stage_path = stage_root / "stage-report.json"
    result = support.bound_json(result_path, args.result_sha256)
    stage = support.bound_json(stage_path, args.stage_sha256)
    if (result.get("schema") != "classical_backend_v1" or result.get("status") != "sparse_complete" or
            result.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap" or
            result["sfm_source"].get("matching") != "exhaustive" or
            result["sfm_source"].get("intrinsics_policy") != "fixed-initial"):
        raise ValueError("repair source is not sealed fixed-initial exhaustive image-only SfM")
    _regular(sfm_path, 1024**2)
    sfm_sha = common.digest(sfm_path)
    sfm = json.loads(sfm_path.read_text())
    names, _ = support.load_masks(stage_root, stage)
    selected = [row for row in sfm.get("candidate_models", [])
                if row.get("index") == sfm.get("selected_model_index")]
    if (len(selected) != 1 or selected[0].get("registered_images") != len(names) or
            sfm.get("registered_images") != len(names) or
            sorted(sfm.get("registered_names", [])) != sorted(names) or
            sfm.get("sparse_points") != selected[0].get("sparse_points")):
        raise ValueError("source SfM does not register the exact TRAIN inventory")
    profile = result["sfm_source"]
    expected_profile = {"random_seed": SEED, "camera_mode": "SINGLE", "camera_model": "SIMPLE_RADIAL",
                        "sift_max_features": 1800, "sift_max_image_size": 1200,
                        "init_image_pair_names": None, "sfm_max_models": 5, "sfm_min_model_size": 10}
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise ValueError("source mapper profile differs from frozen trial")
    effective = profile.get("effective_options", {})
    if (effective.get("version") != "3.11.1" or
            effective.get("image_reader", {}).get("mask_path") != str((run / "masks").resolve())):
        raise ValueError("source PyCOLMAP version or effective mask path differs")
    inputs = result.get("inputs", [])
    if ([row.get("name") for row in inputs] != names or
            {row.get("name"): row.get("sha256") for row in inputs} != stage["train_photo_sha256"]):
        raise ValueError("source photos differ from staged TRAIN manifest")
    if {path.name for path in (run / "images").iterdir()} != set(names):
        raise ValueError("producer photo directory inventory differs")
    for row in inputs:
        path = run / "images" / row["name"]
        _regular(path, 20 * 1024**2)
        if common.digest(path) != row["sha256"]:
            raise ValueError("source photo changed")
    mask_rows = result.get("pose_masks", [])
    expected_masks = {name + ".png": stage["cleaned_masks"][name]["sha256"] for name in names}
    if (len(mask_rows) != len(names) or
            {row.get("name"): row.get("sha256") for row in mask_rows} != expected_masks):
        raise ValueError("producer pose mask inventory differs from accepted stage")
    if {path.name for path in (run / "masks").iterdir()} != set(expected_masks):
        raise ValueError("producer mask directory inventory differs")
    for mask_name, mask_sha in expected_masks.items():
        mask_path = run / "masks" / mask_name
        _regular(mask_path, 4 * 1024**2)
        if common.digest(mask_path) != mask_sha:
            raise ValueError("producer pose mask bytes differ from accepted stage")
    model_dir = run / "sparse" / "0"
    if model_dir.is_symlink() or not model_dir.is_dir() or {p.name for p in model_dir.iterdir()} != set(FILES):
        raise ValueError("accepted baseline sparse export differs")
    for name in FILES:
        _regular(model_dir / name, 256 * 1024**2)
    baseline_files = {name: common.digest(model_dir / name) for name in FILES}
    stage_masks = {name: stage["cleaned_masks"][name]["sha256"] for name in names}
    return {"result": result, "result_sha256": args.result_sha256,
            "sfm": sfm, "sfm_sha256": sfm_sha, "stage": stage,
            "stage_report_sha256": args.stage_sha256,
            "model_dir": model_dir, "model_files_sha256": baseline_files,
            "registered_names": names, "train_photo_sha256": stage["train_photo_sha256"],
            "cleaned_mask_sha256": stage_masks}


def triangulation_options(pycolmap):
    options = pycolmap.EstimateTriangulationOptions()
    expected = {"min_tri_angle": 0.0, "residual_type": "TriangulationResidualType.ANGULAR_ERROR",
                "ransac": {"max_error": 0.03490658503988659, "max_num_trials": 10000,
                           "min_num_trials": 0, "min_inlier_ratio": 0.02,
                           "confidence": 0.9999, "dyn_num_trials_multiplier": 3.0}}
    actual = options.todict()
    frozen = {**actual, "residual_type": str(actual["residual_type"])}
    if frozen != expected:
        raise ValueError("pinned PyCOLMAP triangulation defaults differ")
    serializable = {**frozen, "max_error_units": "radians angular error, not pixels"}
    return options, serializable


def output_bytes(path):
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_symlink():
                raise ValueError("linked output artifact")
            if item.is_file():
                total += item.stat().st_size
                if total > MAX_OUTPUT:
                    break
    return total


def worker(args):
    import pycolmap
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    mac.host_preflight(output)
    contract = source_contract(args)
    core_sha = common.digest(pycolmap._core.__file__)
    if (pycolmap.__version__ != "3.11.1" or
            core_sha != contract["result"]["sfm_source"]["effective_options"]["binary_sha256"]):
        raise ValueError("active PyCOLMAP build differs from producer")
    runner_sha = common.digest(__file__)
    helper_sha = common.digest(support.__file__)
    resource_helper_sha = common.digest(mac.__file__)
    output.mkdir()
    model_out = output / "model"
    model_out.mkdir()
    report = {"schema": "mustard_sparse_track_repair_v1", "status": "running",
              "scope": "TRAIN-only image-estimated sparse integrity candidate; no GT/heldout",
              "baseline": {"result_sha256": contract["result_sha256"],
                           "sfm_sha256": contract["sfm_sha256"],
                           "model_files_sha256": contract["model_files_sha256"],
                           "registered_names": contract["registered_names"]},
              "stage": {"report_sha256": contract["stage_report_sha256"],
                        "train_photo_sha256": contract["train_photo_sha256"],
                        "cleaned_mask_sha256": contract["cleaned_mask_sha256"]},
              "runner_sha256": runner_sha, "helper_sha256": helper_sha,
              "resource_helper_sha256": resource_helper_sha,
              "software": {"pycolmap_version": pycolmap.__version__,
                           "pycolmap_core_sha256": core_sha}}
    started = time.monotonic()
    try:
        baseline = pycolmap.Reconstruction(str(contract["model_dir"]))
        candidate = pycolmap.Reconstruction(str(contract["model_dir"]))
        baseline.check()
        options, option_record = triangulation_options(pycolmap)
        pycolmap.set_random_seed(SEED)
        result = repair_model(baseline, candidate, pycolmap.estimate_triangulation, options)
        if (result["baseline"]["points"] != contract["sfm"]["sparse_points"] or
                baseline.num_reg_images() != len(contract["registered_names"])):
            raise ValueError("baseline model/SfM summary mismatch")
        candidate.write_binary(str(model_out))
        if {p.name for p in model_out.iterdir()} != set(FILES):
            raise ValueError("repaired model file inventory differs")
        output_files = {name: common.digest(model_out / name) for name in FILES}
        reopened = pycolmap.Reconstruction(str(model_out))
        reopened.check()
        if (not frozen_cameras_and_poses(baseline, reopened) or
                track_inventory(reopened) != result["output"] or
                not frozen_cameras_and_poses(candidate, reopened)):
            raise ValueError("serialized repair differs from checked in-memory candidate")
        report["baseline"].update(points=result["baseline"]["points"],
                                  observations=result["baseline"]["observations"])
        report["output"] = {"model_dir": str(model_out.resolve()),
                            "model_files_sha256": output_files,
                            "registered_names": contract["registered_names"],
                            **result["output"]}
        report["repair"] = {"seed": SEED, "triangulation_options": option_record,
                            "per_image_choice_rule": "lowest finite baseline reprojection L2 pixels; tie point2D_idx",
                            "removed_observations": result["removed_observations"],
                            "unchanged_clean_points": result["unchanged_clean_points"],
                            "dropped_points_by_reason": result["dropped_points_by_reason"],
                            "surviving_point_original_observations":
                                result["surviving_point_original_observations"],
                            "selection_sha256": result["selection_sha256"],
                            "duplicate_track_count_before":
                                result["baseline"]["duplicate_same_image_tracks"],
                            "duplicate_track_count_after":
                                result["output"]["duplicate_same_image_tracks"],
                            "preserved_camera_and_pose": True, "backlink_valid": True}
        report["residuals"] = result["residuals"]
        report["gates"] = result["gates"]
        if (time.monotonic() - started > MAX_SECONDS or output_bytes(output) > MAX_OUTPUT or
                mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES or
                common.digest(__file__) != runner_sha or common.digest(support.__file__) != helper_sha or
                common.digest(mac.__file__) != resource_helper_sha or
                common.digest(pycolmap._core.__file__) != core_sha):
            raise ValueError("repair runtime/output/source integrity gate failed")
        fresh = source_contract(args)
        if (fresh["model_files_sha256"] != contract["model_files_sha256"] or
                fresh["sfm_sha256"] != contract["sfm_sha256"] or
                fresh["stage_report_sha256"] != contract["stage_report_sha256"] or
                any(common.digest(model_out / name) != sha for name, sha in output_files.items())):
            raise ValueError("source/output changed during repair")
        report["status"] = ("candidate_unreviewed" if result["gates"]["dense_eligible_predeclared"]
                            else "candidate_ineligible")
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("repair report >1 MiB")
        with (output / "report.json").open("xb") as stream:
            stream.write(payload)
    return report


def supervise(args):
    output = Path(args.output)
    sidecar = output.parent / (output.name + ".supervisor.json")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("repair output/sidecar must be fresh")
    mac.host_preflight(output)
    source_contract(args)
    command = [sys.executable, "-m", "scripts.object_motion.sparse_track_repair", "--worker"]
    for name in ("run", "result_sha256", "stage_root", "stage_sha256", "output"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryFile(dir=output.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        started, peak, reason = time.monotonic(), 0, None
        try:
            while child.poll() is None:
                peak = max(peak, mac.rss_kib(child.pid))
                if (time.monotonic() - started > MAX_SECONDS or peak > MAX_RSS_KIB or
                        mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES or
                        output_bytes(output) > MAX_OUTPUT or logfile.seek(0, os.SEEK_END) > MAX_LOG):
                    reason = "time/RSS/disk/output/log cap exceeded"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            child.wait()
            if (time.monotonic() - started > MAX_SECONDS or peak > MAX_RSS_KIB or
                    mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES or
                    output_bytes(output) > MAX_OUTPUT or logfile.seek(0, os.SEEK_END) > MAX_LOG):
                reason = "post-exit time/RSS/disk/output/log cap exceeded"
            if child.returncode != 0 and reason is None:
                reason = f"worker exit {child.returncode}"
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    report_path = output / "report.json"
    if reason is None:
        if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size > 1024**2:
            reason = "missing repair report"
        elif json.loads(report_path.read_text()).get("status") != "candidate_unreviewed":
            reason = "worker repair report not candidate_unreviewed"
    result = {"schema": "mustard_sparse_track_repair_supervisor_v1",
              "status": "candidate_unreviewed" if reason is None else "failed",
              "reason": reason, "log_excerpt": excerpt,
              "seconds": time.monotonic() - started, "peak_worker_rss_kib": peak}
    path = output / "supervisor.json" if output.is_dir() else sidecar
    with path.open("xb") as stream:
        stream.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "result-sha256", "stage-root", "stage-sha256", "output"):
        parser.add_argument("--" + name, required=True, type=Path if name in ("run", "stage-root", "output") else str)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = worker(args) if args.worker else supervise(args)
    if report["status"] != "candidate_unreviewed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
