#!/usr/bin/env python3
"""Bounded Mac CPU SAM trial driven by one reviewed, versioned TRAIN recipe."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_motion import sam_m1_board5 as legacy
from scripts.object_motion import sam_m1_parity as mac
from scripts.object_motion import sam_m1_resume as resume
from scripts.object_motion import sam_mask_cleanup as cleanup
from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import sam_point_full as full
from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import sam_prompt_recipe as recipes
from scripts.object_motion import ycb_object_masks as common


RECIPE_PATH = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_two_view_recipe.json"
PARENT_MANIFEST_SHA = "1008fc7bdbaa75ac33f17f8ddf44eaa275b3e98eb21466906060ebc3300c638c"
PARENT_SUPERVISOR_SHA = "b4485b72c539d59b6b3229917f0099ceb11359cdb8c9228bd0553208989619ce"


def predictor_factory(source, checkpoint):
    """Same pinned model settings as v1; point/box geometry comes from recipe."""
    sys.path.insert(0, str(source))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint),
                       device="cpu", apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)

    def predict(image, box, points):
        recipes.validate_geometry(box, points, 2)
        xy = np.asarray([[x, y] for x, y, _ in points], dtype=np.float32)
        labels = np.asarray([label for _, _, label in points], dtype=np.int32)
        with torch.inference_mode():
            predictor.set_image(np.array(image, copy=True))
            masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32),
                                            point_coords=xy, point_labels=labels,
                                            multimask_output=False)
        return masks
    return predict


def membership(mask, points):
    if mask.shape != (1024, 1280) or mask.dtype != np.uint8:
        raise ValueError("mask is not original-size uint8")
    expected = [label for _, _, label in points]
    found = [int(mask[y, x] != 0) for x, y, _ in points]
    if found != expected:
        raise ValueError(f"recipe point membership differs: {found}")
    return found


def parent_lineage(parent_root, recipe):
    parent_root = Path(parent_root)
    manifest, manifest_sha = resume.bound_metadata(parent_root / "manifest.json", PARENT_MANIFEST_SHA)
    supervisor, supervisor_sha = resume.bound_metadata(parent_root / "supervisor.json", PARENT_SUPERVISOR_SHA)
    if (recipe.get("parent_board5_manifest_sha256") != manifest_sha or
            recipe.get("parent_board5_supervisor_sha256") != supervisor_sha or
            manifest.get("schema") != "sam21_mustard_m1_board5_candidate_v1" or
            manifest.get("status") != "partial_unusable" or
            supervisor.get("schema") != "sam21_mustard_m1_board5_supervisor_v1" or
            supervisor.get("status") != "failed" or supervisor.get("reason") != "worker exit 1"):
        raise ValueError("partial parent outcome was mislabeled or changed")
    images = manifest.get("images")
    if not isinstance(images, list) or [r.get("name") for r in images] != list(legacy.NAMES):
        raise ValueError("parent five-view rows missing/reordered")
    raw_dir, clean_dir = parent_root / "raw_masks", parent_root / "cleaned_masks"
    if (raw_dir.is_symlink() or clean_dir.is_symlink() or
            not raw_dir.is_dir() or not clean_dir.is_dir() or
            {p.name for p in raw_dir.iterdir()} != {f"{name}.png" for name in legacy.NAMES} or
            {p.name for p in clean_dir.iterdir()} !=
            {f"{name}.png" for name in ("NP3_318.jpg", "NP3_330.jpg", "NP3_348.jpg")}):
        raise ValueError("partial parent raw/clean directories have missing/extra/linked files")
    if [r.get("status") for r in images] != ["complete_unreviewed", "complete_unreviewed",
                                                "failed_unusable", "failed_unusable",
                                                "complete_unreviewed"]:
        raise ValueError("parent per-view outcome differs")
    for row in images:
        raw = parent_root / "raw_masks" / f"{row['name']}.png"
        if (row.get("raw_mask_path") != f"raw_masks/{row['name']}.png" or
                raw.is_symlink() or not raw.is_file() or common.digest(raw) != row.get("raw_mask_sha256")):
            raise ValueError("parent raw mask changed")
        if row["status"] == "complete_unreviewed":
            clean = parent_root / "cleaned_masks" / f"{row['name']}.png"
            if (row.get("cleaned_mask_path") != f"cleaned_masks/{row['name']}.png" or
                    clean.is_symlink() or not clean.is_file() or
                    common.digest(clean) != row.get("cleaned_mask_sha256")):
                raise ValueError("parent completed mask changed")
        elif row.get("cleaned_mask_sha256") is not None:
            raise ValueError("failed parent row claims a cleaned mask")
    return {"parent_board5_manifest_sha256": manifest_sha,
            "parent_board5_supervisor_sha256": supervisor_sha,
            "parent_board5_status": "partial_unusable/failed"}


def inputs(args):
    mac.external_inputs((args.package, args.stage_root, args.stage_report,
                         args.original_prompts, args.checkpoint, args.sam_source,
                         args.base_root, args.base_inventory, args.parent_board5))
    rows, _, package_sha = full.validated_inputs(
        args.package, args.checkpoint, args.sam_source, args.original_prompts)
    stage_sha = resume.staged_train48(args.stage_root, rows, package_sha, args.stage_report)
    base_hashes = legacy.source_seal(args.base_root, args.base_inventory, rows, package_sha)
    recipe, selected = recipes.load_v2(args.recipe, args.recipe_sha256, rows)
    for key, expected in (("package_sha256", package_sha),
                          ("original_prompt_sha256", full.PROMPTS_SHA),
                          ("base_inventory_sha256", legacy.INVENTORY_SHA),
                          ("rejected_full_qa_sha256", legacy.QA_SHA),
                          ("checkpoint_sha256", point.MODEL_SHA),
                          ("source_inventory_sha256", point.SOURCE_SHA)):
        if recipe.get(key) != expected:
            raise ValueError(f"recipe header {key} differs from sealed input")
    parent = parent_lineage(args.parent_board5, recipe)
    for row in selected:
        common.verified_training_photo(args.stage_root, row)
    return rows, selected, recipe, package_sha, stage_sha, base_hashes, parent


def dependencies():
    return {"runner_sha256": common.digest(__file__),
            "recipe_helper_sha256": common.digest(recipes.__file__),
            "legacy_helper_sha256": common.digest(legacy.__file__),
            "mac_helper_sha256": common.digest(mac.__file__),
            "resume_helper_sha256": common.digest(resume.__file__),
            "full_helper_sha256": common.digest(full.__file__),
            "point_helper_sha256": common.digest(point.__file__),
            "smoke_helper_sha256": common.digest(smoke.__file__),
            "cleanup_helper_sha256": common.digest(cleanup.__file__),
            "shared_helper_sha256": common.digest(common.__file__)}


def worker(args, predictor=None):
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    mac.host_preflight(output)
    rows, selected, recipe, package_sha, stage_sha, base_hashes, parent = inputs(args)
    deps = dependencies()
    box = recipe["box_xyxy_original_pixels"]
    prompts = recipe["images"]
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    report = {"schema": "sam21_mustard_m1_recipe_trial_v2", "status": "running",
              "mask_role": "unreviewed coarse pose support; not exact silhouette",
              "recipe_sha256": args.recipe_sha256, "selected_training_names": recipe["selected_training_names"],
              "package_sha256": package_sha, "stage_report_sha256": stage_sha,
              "original_prompt_sha256": full.PROMPTS_SHA,
              "base_inventory_sha256": legacy.INVENTORY_SHA,
              "rejected_full_qa_sha256": legacy.QA_SHA,
              "checkpoint_sha256": point.MODEL_SHA,
              "source_inventory_sha256": point.SOURCE_SHA,
              "base_unchanged_mask_sha256": base_hashes, **parent, **deps, "images": []}
    started = time.monotonic()
    try:
        predict = predictor or predictor_factory(args.sam_source, args.checkpoint)
        for row, prompt in zip(selected, prompts):
            if (time.monotonic() - started > legacy.MAX_SECONDS or
                    mac.available_kib() < mac.MIN_AVAILABLE_KIB or
                    mac.rss_kib(os.getpid()) > mac.MAX_RSS_KIB or
                    mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES):
                raise RuntimeError("worker resource gate between recipe views")
            name = prompt["name"]
            points = prompt["points_xy_label"]
            entry = {**recipes.canonical_row(prompt, box),
                     "recipe_row_sha256": recipes.row_sha256(prompt, box)}
            report["images"].append(entry)
            try:
                photo = common.verified_training_photo(args.stage_root, row)
                raw = point.binary_raw(predict(photo, box, points))
                raw_path = output / "raw_masks" / f"{name}.png"
                Image.fromarray(raw, "L").save(raw_path, "PNG")
                entry.update({"raw_mask_path": f"raw_masks/{name}.png",
                              "raw_mask_sha256": common.digest(raw_path)})
                membership(raw, points)
                smoke.bounded_mask(raw)
                clean, stats = cleanup.largest_component(raw)
                membership(clean, points)
                clean_path = output / "cleaned_masks" / f"{name}.png"
                Image.fromarray(clean, "L").save(clean_path, "PNG")
                entry.update({"cleaned_mask_path": f"cleaned_masks/{name}.png",
                              "cleaned_mask_sha256": common.digest(clean_path),
                              "cleanup": stats, "status": "complete_unreviewed"})
            except ValueError as error:
                entry.update({"status": "failed_unusable", "failure": repr(error)})
            if legacy.output_bytes(output) > legacy.MAX_OUTPUT - 1024**2:
                raise ValueError("recipe output cap reached")
        sheet = Image.new("RGB", (len(selected) * 290, 3 * 370), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (row, entry) in enumerate(zip(selected, report["images"])):
            photo = common.verified_training_photo(args.stage_root, row)
            for lane, kind in enumerate(("rgb", "raw", "cleaned")):
                if kind == "rgb":
                    tile = photo.crop(smoke.BOX)
                    ImageDraw.Draw(tile).rectangle((box[0] - smoke.BOX[0], box[1] - smoke.BOX[1],
                                                    box[2] - smoke.BOX[0], box[3] - smoke.BOX[1]),
                                                   outline="red", width=2)
                else:
                    key = "raw_mask_path" if kind == "raw" else "cleaned_mask_path"
                    if key in entry:
                        with Image.open(output / entry[key]) as mask:
                            masked = Image.new("RGB", photo.size, "black")
                            masked.paste(photo, (0, 0), mask)
                        tile = masked.crop(smoke.BOX)
                    else:
                        tile = Image.new("RGB", (smoke.BOX[2] - smoke.BOX[0],
                                                  smoke.BOX[3] - smoke.BOX[1]), "black")
                tile.thumbnail((280, 340), Image.Resampling.LANCZOS)
                sheet.paste(tile, (index * 290 + (290 - tile.width) // 2, lane * 370))
                draw.text((index * 290 + 4, lane * 370 + 345),
                          f"{entry['name']} {kind}", fill="black")
        sheet_path = output / "review_sheet.jpg"
        sheet.save(sheet_path, "JPEG", quality=88)
        report["review_sheet_sha256"] = common.digest(sheet_path)
        if (common.digest(args.package) != package_sha or common.digest(args.recipe) != args.recipe_sha256 or
                common.digest(args.original_prompts) != full.PROMPTS_SHA or
                common.digest(args.stage_report) != stage_sha or
                common.digest(args.checkpoint) != point.MODEL_SHA or
                smoke.source_digest(args.sam_source) != point.SOURCE_SHA or
                dependencies() != deps):
            raise ValueError("recipe/model/photo source/code changed during trial")
        legacy.source_seal(args.base_root, args.base_inventory, rows, package_sha)
        parent_lineage(args.parent_board5, recipe)
        if legacy.output_bytes(output) > legacy.MAX_OUTPUT:
            raise ValueError("recipe output above 20 MiB")
        report["status"] = ("generated_unreviewed" if all(
            r["status"] == "complete_unreviewed" for r in report["images"])
            else "partial_unusable")
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("recipe manifest above 1 MiB")
        with (output / "manifest.json").open("xb") as file:
            file.write(payload)
    return report


def supervise(args):
    """Use the frozen Mac resource probes/cap order; run exactly one worker."""
    output = args.output
    sidecar = output.parent / f"{output.name}.supervisor.json"
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("fresh recipe output and sidecar required")
    mac.host_preflight(output, check_memory=False)
    if mac.available_kib() < legacy.MIN_LAUNCH_KIB:
        raise RuntimeError("launch below 4 GiB conservative available estimate")
    started = time.monotonic()
    inputs(args)
    if (time.monotonic() - started > legacy.TOTAL_SECONDS or
            mac.available_kib() < legacy.MIN_LAUNCH_KIB or
            mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES):
        raise RuntimeError("prelaunch deadline/memory/disk recheck failed")
    command = [sys.executable, "-m", "scripts.object_motion.sam_m1_recipe_trial", "--worker"]
    for key in ("package", "stage_root", "stage_report", "original_prompts", "checkpoint",
                "sam_source", "base_root", "base_inventory", "parent_board5", "recipe", "output"):
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    command.extend(("--recipe-sha256", args.recipe_sha256))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryFile(dir=output.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        worker_start, peak, reason = time.monotonic(), 0, None
        try:
            while child.poll() is None:
                peak = max(peak, mac.rss_kib(child.pid))
                reason = legacy.terminal_guard(time.monotonic() - worker_start,
                                               time.monotonic() - started, peak,
                                               mac.available_kib(), legacy.output_bytes(output),
                                               logfile.seek(0, os.SEEK_END),
                                               mac.disk_free_both(output.parent), 0)
                if reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            child.wait()
            reason = reason or legacy.terminal_guard(time.monotonic() - worker_start,
                                                     time.monotonic() - started, peak,
                                                     mac.available_kib(), legacy.output_bytes(output),
                                                     logfile.seek(0, os.SEEK_END),
                                                     mac.disk_free_both(output.parent), child.returncode)
            logfile.seek(0)
            excerpt = logfile.read(legacy.MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    manifest = output / "manifest.json"
    status = None
    if reason is None:
        if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1024**2:
            reason = "worker lacks valid recipe manifest"
        else:
            try:
                doc = json.loads(manifest.read_text())
                status = doc.get("status")
                if (doc.get("schema") != "sam21_mustard_m1_recipe_trial_v2" or
                        status not in ("generated_unreviewed", "partial_unusable")):
                    reason = "worker recipe manifest invalid"
            except (OSError, ValueError, AttributeError):
                reason = "worker recipe manifest unreadable"
    result = {"schema": "sam21_mustard_m1_recipe_supervisor_v2",
              "status": status if reason is None else "failed", "reason": reason,
              "log_excerpt": excerpt, "seconds_total": time.monotonic() - started,
              "seconds_worker": time.monotonic() - worker_start,
              "peak_worker_rss_kib": peak}
    target = output / "supervisor.json" if output.is_dir() else sidecar
    with target.open("xb") as file:
        file.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "stage-root", "stage-report", "original-prompts", "checkpoint",
                "sam-source", "base-root", "base-inventory", "parent-board5", "recipe", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = worker(args) if args.worker else supervise(args)
    if result["status"] != "generated_unreviewed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
