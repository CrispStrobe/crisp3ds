#!/usr/bin/env python3
"""Bounded six-by-eight SAM2 point-prompt candidate on 48 mustard TRAIN photos."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

from scripts.object_motion import sam_mask_cleanup as cleanup
from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import ycb_object_masks as common


PROMPTS_SHA = "28440795d5f38590df8c5568969970d6623fbe22ea6b3e8781cba10608149d32"
BATCH_SIZE = 8
NUM_BATCHES = 6
BATCH_SECONDS = 90
TOTAL_SECONDS = 600
RAM_WAIT_SECONDS = 60
MAX_OUTPUT = 60 * 1024**2
MAX_LOG = 1024**2


def validated_prompts(path, rows):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 or common.digest(path) != PROMPTS_SHA:
        raise ValueError("48-view prompt file differs from reviewed SHA-256")
    prompts = json.loads(path.read_text())
    if (prompts.get("schema") != "sam21_mustard_common_point_prompts_48_v1" or
            prompts.get("box_xyxy_original_pixels") != list(smoke.BOX) or
            not isinstance(prompts.get("images"), list) or len(prompts["images"]) != 48):
        raise ValueError("invalid frozen 48-view prompt schema")
    for record, row in zip(prompts["images"], rows):
        if record.get("name") != Path(row["path"]).name or record.get("source_sha256") != row["sha256"]:
            raise ValueError("prompt name/photo hash differs from training package")
        if record.get("points_xy_label") != [[600, 525, 1], [720, 620, 0], [520, 630, 0]]:
            raise ValueError("common reviewed points differ")
        point.point_arrays(record["points_xy_label"])
    return prompts["images"]


def validated_inputs(package_path, checkpoint, source, prompt_path):
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != smoke.OBJECT_ID:
        raise ValueError("mustard package required")
    rows = common.training_records(package)
    prompts = validated_prompts(prompt_path, rows)
    smoke.checkpoint_metadata(checkpoint, point.MODEL_SHA)
    if smoke.source_digest(source) != point.SOURCE_SHA:
        raise ValueError("SAM2 source differs from frozen reviewed inventory")
    return rows, prompts, package_sha


def dependencies():
    return {"runner_sha256": common.digest(__file__),
            "point_helper_sha256": common.digest(point.__file__),
            "smoke_helper_sha256": common.digest(smoke.__file__),
            "cleanup_helper_sha256": common.digest(cleanup.__file__),
            "shared_helper_sha256": common.digest(common.__file__)}


def batch_worker(package_path, dataset_root, output, checkpoint, source, prompt_path, index,
                 predictor=None):
    if index not in range(NUM_BATCHES):
        raise ValueError("batch index outside frozen six batches")
    package_path, output, checkpoint, source, prompt_path = map(
        Path, (package_path, output, checkpoint, source, prompt_path))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows, prompts, package_sha = validated_inputs(package_path, checkpoint, source, prompt_path)
    rows = rows[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
    prompts = prompts[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    deps = dependencies()
    report = {"schema": "sam21_mustard_point_batch_v1", "status": "running",
              "batch_index": index, "mask_status": "generated_unreviewed",
              "package_sha256": package_sha, "prompt_sha256": PROMPTS_SHA,
              "checkpoint_sha256": point.MODEL_SHA, "sam2_source_inventory_sha256": point.SOURCE_SHA,
              "box_xyxy_original_pixels": list(smoke.BOX), **deps, "images": []}
    used = 0
    try:
        predict = predictor or point.sam_predictor(source, checkpoint)
        for row, prompt in zip(rows, prompts):
            name = Path(row["path"]).name
            entry = {"name": name, "source_sha256": row["sha256"],
                     "points_xy_label": prompt["points_xy_label"]}
            started = time.monotonic()
            try:
                photo = common.verified_training_photo(dataset_root, row)
                raw = point.binary_raw(predict(photo, smoke.BOX, prompt["points_xy_label"]))
                raw_path = output / "raw_masks" / f"{name}.png"
                used = common.save_bounded_image(Image.fromarray(raw, "L"), raw_path, "PNG", used)
                entry["raw_mask_sha256"] = common.digest(raw_path)
                entry["raw_pixels"] = int(np.count_nonzero(raw))
                entry["point_membership"] = point.point_membership(raw, prompt["points_xy_label"])
                smoke.bounded_mask(raw)
                clean, stats = cleanup.largest_component(raw)
                point.point_membership(clean, prompt["points_xy_label"])
                clean_path = output / "cleaned_masks" / f"{name}.png"
                used = common.save_bounded_image(Image.fromarray(clean, "L"), clean_path, "PNG", used)
                entry.update({"status": "complete_unreviewed", "cleaned_mask_sha256": common.digest(clean_path),
                              **stats})
            except ValueError as error:
                entry.update({"status": "failed_unusable", "failure": repr(error)})
            entry["seconds"] = time.monotonic() - started
            report["images"].append(entry)
            if used > MAX_OUTPUT // NUM_BATCHES - 1024**2:
                raise ValueError("batch output exceeded cap")
        if (common.digest(package_path) != package_sha or common.digest(prompt_path) != PROMPTS_SHA or
                common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA or
                dependencies() != deps):
            raise ValueError("source/model/runner changed during batch")
        report["status"] = "complete_unreviewed" if all(r["status"] == "complete_unreviewed" for r in report["images"]) else "partial_unusable"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("batch report exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def output_size(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def terminal_guard(elapsed, deadline_passed, log_bytes, output_bytes, free_bytes):
    if elapsed > BATCH_SECONDS:
        return "batch above 90 seconds at exit"
    if deadline_passed:
        return "whole trial above 600 seconds at exit"
    if log_bytes > MAX_LOG:
        return "batch log above 1 MiB at exit"
    if output_bytes > MAX_OUTPUT:
        return "output above 60 MiB at exit"
    if free_bytes < common.MIN_FREE_BYTES:
        return "output filesystem below 10 GiB free at exit"
    return None


def wait_for_ram(deadline, waited):
    while smoke.available_kib() < smoke.MIN_AVAILABLE_KIB:
        if waited >= RAM_WAIT_SECONDS or time.monotonic() >= deadline:
            raise RuntimeError("2.5 GiB RAM pre-batch gate unavailable within 60-second aggregate wait")
        time.sleep(1)
        waited += 1
    return waited


def bounded_batch(args, root, index, deadline):
    command = [sys.executable, "-m", "scripts.object_motion.sam_point_full", "--worker",
               "--batch-index", str(index), *args]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryFile(dir=root.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        started = time.monotonic()
        peak = 0
        reason = None
        try:
            while child.poll() is None:
                peak = max(peak, smoke.rss_kib(child.pid))
                if peak > smoke.MAX_RSS_KIB:
                    reason = "worker RSS above 1.5 GiB"
                elif time.monotonic() - started > BATCH_SECONDS:
                    reason = "batch above 90 seconds"
                elif time.monotonic() > deadline:
                    reason = "whole trial above 600 seconds"
                elif logfile.seek(0, os.SEEK_END) > MAX_LOG:
                    reason = "batch log above 1 MiB"
                elif output_size(root) > MAX_OUTPUT:
                    reason = "output above 60 MiB"
                elif shutil.disk_usage(root.parent).free < common.MIN_FREE_BYTES:
                    reason = "output filesystem below 10 GiB free"
                if reason:
                    break
                time.sleep(0.2)
            if reason:
                os.killpg(child.pid, 15)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, 9)
            child.wait()
            final_log_bytes = logfile.seek(0, os.SEEK_END)
            final_elapsed = time.monotonic() - started
            if reason is None:
                reason = terminal_guard(final_elapsed, time.monotonic() > deadline,
                                        final_log_bytes, output_size(root),
                                        shutil.disk_usage(root.parent).free)
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
    return {"batch_index": index, "status": "complete" if child.returncode == 0 and not reason else "failed",
            "reason": reason or (excerpt if child.returncode else None),
            "returncode": child.returncode, "seconds": time.monotonic() - started,
            "peak_worker_rss_kib": peak}


def review_sheets(root, rows, entries, dataset_root):
    by_name = {entry["name"]: entry for entry in entries}
    tile = (200, 210)
    sheets = {kind: Image.new("RGB", (8 * tile[0], 6 * tile[1]), "white")
              for kind in ("rgb", "raw", "cleaned")}
    draws = {kind: ImageDraw.Draw(sheet) for kind, sheet in sheets.items()}
    for index, row in enumerate(rows):
        name = Path(row["path"]).name
        photo = common.verified_training_photo(dataset_root, row)
        entry = by_name.get(name, {})
        for kind in sheets:
            if kind == "rgb":
                display = photo.crop(smoke.BOX)
            else:
                expected = entry.get(f"{kind}_mask_sha256")
                path = root / f"batch-{index // BATCH_SIZE}" / f"{kind}_masks" / f"{name}.png"
                if expected is None:
                    display = Image.new("RGB", (280, 350), "black")
                    ImageDraw.Draw(display).text((8, 8), "FAILED / MISSING", fill="red")
                else:
                    if path.is_symlink() or not path.is_file() or common.digest(path) != expected:
                        raise ValueError("review mask hash differs from batch manifest")
                    with Image.open(path) as loaded:
                        mask = loaded.convert("L")
                    masked = Image.new("RGB", photo.size, "black")
                    masked.paste(photo, (0, 0), mask)
                    display = masked.crop(smoke.BOX)
            display.thumbnail((tile[0] - 8, tile[1] - 28), Image.Resampling.LANCZOS)
            x, y = (index % 8) * tile[0], (index // 8) * tile[1]
            sheets[kind].paste(display, (x + (tile[0] - display.width) // 2, y + 2))
            draws[kind].text((x + 4, y + tile[1] - 22), name, fill="black")
    hashes = {}
    for kind, sheet in sheets.items():
        path = root / f"training_{kind}_sheet.jpg"
        sheet.save(path, "JPEG", quality=90)
        hashes[kind] = common.digest(path)
    return hashes


def supervise(package_path, dataset_root, output, checkpoint, source, prompt_path):
    package_path, output, checkpoint, source, prompt_path = map(
        Path, (package_path, output, checkpoint, source, prompt_path))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows, _, package_sha = validated_inputs(package_path, checkpoint, source, prompt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output filesystem below 10 GiB free")
    output.mkdir()
    started = time.monotonic()
    deadline = started + TOTAL_SECONDS
    deps = dependencies()
    report = {"schema": "sam21_mustard_point_full_v1", "status": "running",
              "mask_status": "generated_unreviewed", "source_scope": "48 TRAIN RGB only; no held-out/GT/pose/depth",
              "package_sha256": package_sha, "prompt_sha256": PROMPTS_SHA,
              "checkpoint_sha256": point.MODEL_SHA, "sam2_source_inventory_sha256": point.SOURCE_SHA,
              "box_xyxy_original_pixels": list(smoke.BOX), **deps,
              "batches": [], "images": []}
    args = ["--package", str(package_path), "--dataset-root", str(dataset_root),
            "--output", str(output), "--checkpoint", str(checkpoint),
            "--sam-source", str(source), "--prompts", str(prompt_path)]
    waited = 0
    try:
        for index in range(NUM_BATCHES):
            if (time.monotonic() >= deadline or dependencies() != deps or
                    common.digest(package_path) != package_sha or common.digest(prompt_path) != PROMPTS_SHA):
                raise RuntimeError("whole-trial time or input/code freeze failed before batch")
            waited = wait_for_ram(deadline, waited)
            batch = bounded_batch(args, output, index, deadline)
            report["batches"].append(batch)
            manifest = output / f"batch-{index}" / "manifest.json"
            if manifest.is_file():
                sealed = json.loads(manifest.read_text())
                if (sealed.get("batch_index") != index or
                        any(sealed.get(key) != digest for key, digest in deps.items())):
                    raise ValueError("batch provenance differs from frozen supervisor")
                batch["manifest_sha256"] = common.digest(manifest)
                report["images"].extend(sealed["images"])
            if batch["status"] != "complete" or not manifest.is_file():
                raise RuntimeError(f"batch {index} failed: {batch['reason']}")
        if len(report["images"]) != 48 or [r["name"] for r in report["images"]] != [Path(r["path"]).name for r in rows]:
            raise ValueError("exact 48 training inventory not completed")
        report["review_sheet_sha256"] = review_sheets(output, rows, report["images"], dataset_root)
        if any(row["status"] != "complete_unreviewed" for row in report["images"]):
            raise ValueError("one or more point-prompt masks failed acceptance")
        if (time.monotonic() >= deadline or output_size(output) > MAX_OUTPUT or
                common.digest(package_path) != package_sha or common.digest(prompt_path) != PROMPTS_SHA or
                common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA or
                dependencies() != deps):
            raise ValueError("whole-trial time/bytes/model/code gate failed")
        report["status"] = "complete_unreviewed"
    except BaseException as error:
        report["status"] = "failed_or_partial_unusable"
        report["failure"] = repr(error)
        if "review_sheet_sha256" not in report:
            try:
                report["review_sheet_sha256"] = review_sheets(output, rows, report["images"], dataset_root)
            except BaseException as preview_error:
                report["review_sheet_failure"] = repr(preview_error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["ram_wait_seconds"] = waited
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("full report exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.batch_index not in range(NUM_BATCHES):
            raise ValueError("worker requires frozen batch index")
        batch_worker(args.package, args.dataset_root, args.output / f"batch-{args.batch_index}",
                     args.checkpoint, args.sam_source, args.prompts, args.batch_index)
    else:
        if args.batch_index is not None:
            raise ValueError("batch index is supervisor-internal")
        supervise(args.package, args.dataset_root, args.output, args.checkpoint,
                  args.sam_source, args.prompts)


if __name__ == "__main__":
    main()
