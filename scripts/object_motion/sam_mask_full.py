#!/usr/bin/env python3
"""Three bounded SAM2 tiny batches over frozen mustard TRAIN images only."""

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
from scripts.object_motion import ycb_object_masks as common


MODEL_SHA = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SOURCE_SHA = "c7eb4585a22dadd4f54ffd9134e3103c1951745a3b4631ceb0684b55742069e3"
SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
HF_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
BATCH_SIZE = 16
BATCH_SECONDS = 120
TOTAL_SECONDS = 420
TOTAL_OUTPUT_BYTES = 60 * 1024**2
MAX_LOG_BYTES = 1024**2


def exact_training_rows(package):
    if package.get("object_id") != smoke.OBJECT_ID:
        raise ValueError("mustard package required")
    rows = common.training_records(package)
    if len(rows) != 48 or [len(rows[i:i + BATCH_SIZE]) for i in (0, 16, 32)] != [16, 16, 16]:
        raise ValueError("exact three-by-sixteen training partition required")
    return rows


def validated_inputs(package_path, checkpoint, source):
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    rows = exact_training_rows(package)
    checkpoint_info = smoke.checkpoint_metadata(checkpoint, MODEL_SHA)
    source_sha = smoke.source_digest(source)
    if source_sha != SOURCE_SHA:
        raise ValueError("SAM2 source differs from frozen reviewed inventory")
    return rows, package_sha, checkpoint_info


def segment_one(image, predict):
    raw_image, raw_stats = smoke.bounded_mask(predict(image, smoke.BOX))
    raw = np.array(raw_image)
    cleaned, cleaned_stats = cleanup.largest_component(raw)
    return raw_image, Image.fromarray(cleaned, "L"), {**raw_stats, **cleaned_stats}


def run_batch(package_path, dataset_root, output, checkpoint, source, index,
              predictor=None):
    if index not in (0, 1, 2):
        raise ValueError("batch index must be 0, 1, or 2")
    package_path, output, checkpoint, source = map(Path, (package_path, output, checkpoint, source))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows, package_sha, checkpoint_info = validated_inputs(package_path, checkpoint, source)
    rows = rows[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    report = {"schema": "sam21_tiny_mustard_batch_v1", "status": "running",
              "batch_index": index, "mask_status": "generated_unreviewed",
              "box_xyxy_original_pixels": list(smoke.BOX), "package_sha256": package_sha,
              **checkpoint_info, "source_inventory_sha256": SOURCE_SHA,
              "source_git_commit": SOURCE_COMMIT, "checkpoint_hf_revision": HF_REVISION,
              "runner_sha256": common.digest(__file__), "smoke_helper_sha256": common.digest(smoke.__file__),
              "cleanup_sha256": common.digest(cleanup.__file__),
              "shared_helper_sha256": common.digest(common.__file__), "images": []}
    used = 0
    try:
        predict = predictor or smoke.predictor_from_checkpoint(source, checkpoint)
        for row in rows:
            name = Path(row["path"]).name
            entry = {"name": name, "source_sha256": row["sha256"]}
            start = time.monotonic()
            try:
                image = common.verified_training_photo(dataset_root, row)
                raw_image, raw_stats = smoke.bounded_mask(predict(image, smoke.BOX))
                raw_path = output / "raw_masks" / f"{name}.png"
                used = common.save_bounded_image(raw_image, raw_path, "PNG", used)
                entry["raw_mask_sha256"] = common.digest(raw_path)
                cleaned, cleaned_stats = cleanup.largest_component(np.array(raw_image))
                clean_image = Image.fromarray(cleaned, "L")
                clean_path = output / "cleaned_masks" / f"{name}.png"
                used = common.save_bounded_image(clean_image, clean_path, "PNG", used)
                entry.update({"status": "complete_unreviewed", "cleaned_mask_sha256": common.digest(clean_path),
                              **raw_stats, **cleaned_stats})
            except (ValueError, RuntimeError) as exc:
                entry.update({"status": "failed_unusable", "failure": repr(exc)})
            entry["seconds"] = time.monotonic() - start
            report["images"].append(entry)
            if used > TOTAL_OUTPUT_BYTES // 3 - 1024**2:
                raise ValueError("batch artifact cap exceeded")
        if (common.digest(package_path) != package_sha or common.digest(checkpoint) != MODEL_SHA or
                smoke.source_digest(source) != SOURCE_SHA or common.digest(__file__) != report["runner_sha256"] or
                common.digest(smoke.__file__) != report["smoke_helper_sha256"] or
                common.digest(cleanup.__file__) != report["cleanup_sha256"] or
                common.digest(common.__file__) != report["shared_helper_sha256"]):
            raise ValueError("input/model/runner changed during batch")
        report["status"] = "complete_unreviewed" if all(r["status"] == "complete_unreviewed" for r in report["images"]) else "partial_unusable"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        raise
    finally:
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("batch manifest exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def total_size(directory):
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def supervised_batch(base_args, root, index, start):
    if smoke.available_kib() < smoke.MIN_AVAILABLE_KIB:
        raise RuntimeError("below 2.5 GiB available RAM before batch")
    if shutil.disk_usage(root.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("below 10 GiB free before batch")
    command = [sys.executable, "-m", "scripts.object_motion.sam_mask_full", "--worker",
               "--batch-index", str(index), *base_args]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryFile(dir=root.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        batch_start = time.monotonic()
        peak_rss = 0
        reason = None
        try:
            while child.poll() is None:
                peak_rss = max(peak_rss, smoke.rss_kib(child.pid))
                if peak_rss > smoke.MAX_RSS_KIB:
                    reason = "worker RSS exceeded 1.5 GiB"
                elif time.monotonic() - batch_start > BATCH_SECONDS:
                    reason = "batch exceeded 120 seconds"
                elif time.monotonic() - start > TOTAL_SECONDS:
                    reason = "whole trial exceeded 420 seconds"
                elif logfile.seek(0, os.SEEK_END) > MAX_LOG_BYTES:
                    reason = "batch log exceeded 1 MiB"
                elif total_size(root) > TOTAL_OUTPUT_BYTES:
                    reason = "output exceeded 60 MiB"
                elif shutil.disk_usage(root.parent).free < common.MIN_FREE_BYTES:
                    reason = "output filesystem below 10 GiB free"
                if reason:
                    break
                time.sleep(0.25)
            if reason:
                os.killpg(child.pid, 15)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, 9)
            child.wait()
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG_BYTES).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
    return {"batch_index": index, "status": "complete" if child.returncode == 0 and not reason else "failed",
            "reason": reason or (excerpt if child.returncode else None), "returncode": child.returncode,
            "seconds": time.monotonic() - batch_start, "peak_worker_rss_kib": peak_rss}


def review_sheets(root, rows, entries, dataset_root):
    tile = (200, 210)
    sheets = {kind: Image.new("RGB", (8 * tile[0], 6 * tile[1]), "white")
              for kind in ("raw", "cleaned", "full_context")}
    draws = {key: ImageDraw.Draw(image) for key, image in sheets.items()}
    by_name = {entry["name"]: entry for entry in entries}
    for index, row in enumerate(rows):
        name = Path(row["path"]).name
        entry = by_name.get(name, {})
        photo = common.verified_training_photo(dataset_root, row)
        for kind in sheets:
            if kind == "full_context":
                display = photo.copy()
            else:
                path = root / f"batch-{index // BATCH_SIZE}" / f"{kind}_masks" / f"{name}.png"
                expected = entry.get(f"{kind}_mask_sha256")
                if expected is None:
                    display = Image.new("RGB", (smoke.BOX[2] - smoke.BOX[0],
                                                smoke.BOX[3] - smoke.BOX[1]), "black")
                    ImageDraw.Draw(display).text((8, 8), "FAILED / MISSING", fill="red")
                else:
                    if path.is_symlink() or not path.is_file() or common.digest(path) != expected:
                        raise ValueError("review mask differs from batch manifest")
                    with Image.open(path) as file:
                        mask = file.convert("L")
                    display = Image.new("RGB", photo.size, "black")
                    display.paste(photo, (0, 0), mask)
                    display = display.crop(smoke.BOX)
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


def supervise(package_path, dataset_root, output, checkpoint, source):
    package_path, dataset_root, output, checkpoint, source = map(
        Path, (package_path, dataset_root, output, checkpoint, source))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows, package_sha, _ = validated_inputs(package_path, checkpoint, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output filesystem below 10 GiB free")
    output.mkdir()
    started = time.monotonic()
    report = {"schema": "sam21_tiny_mustard_full_v1", "status": "running",
              "mask_status": "generated_unreviewed", "source_scope": "48 TRAIN RGB photos only; no held-out/GT/poses/depth",
              "package_sha256": package_sha, "model_sha256": MODEL_SHA,
              "source_inventory_sha256": SOURCE_SHA, "source_git_commit": SOURCE_COMMIT,
              "checkpoint_hf_revision": HF_REVISION, "box_xyxy_original_pixels": list(smoke.BOX),
              "cleanup_policy": "sole largest 8-connected component; no fill/erosion/dilation",
              "runner_sha256": common.digest(__file__), "smoke_helper_sha256": common.digest(smoke.__file__),
              "cleanup_sha256": common.digest(cleanup.__file__),
              "shared_helper_sha256": common.digest(common.__file__), "batches": [], "images": []}
    try:
        base_args = ["--package", str(package_path), "--dataset-root", str(dataset_root),
                     "--output", str(output), "--checkpoint", str(checkpoint),
                     "--sam-source", str(source)]
        for index in range(3):
            if (time.monotonic() - started > TOTAL_SECONDS or common.digest(__file__) != report["runner_sha256"] or
                    common.digest(smoke.__file__) != report["smoke_helper_sha256"] or
                    common.digest(cleanup.__file__) != report["cleanup_sha256"] or
                    common.digest(common.__file__) != report["shared_helper_sha256"]):
                raise RuntimeError("whole-trial time or implementation freeze failed before batch")
            batch = supervised_batch(base_args, output, index, started)
            report["batches"].append(batch)
            batch_report_path = output / f"batch-{index}" / "manifest.json"
            if batch_report_path.is_file():
                batch_report = json.loads(batch_report_path.read_text())
                if (batch_report.get("batch_index") != index or batch_report.get("runner_sha256") != report["runner_sha256"] or
                        batch_report.get("smoke_helper_sha256") != report["smoke_helper_sha256"] or
                        batch_report.get("cleanup_sha256") != report["cleanup_sha256"] or
                        batch_report.get("shared_helper_sha256") != report["shared_helper_sha256"]):
                    raise ValueError("batch implementation provenance differs from frozen supervisor")
                batch["manifest_sha256"] = common.digest(batch_report_path)
                report["images"].extend(batch_report["images"])
            if batch["status"] != "complete":
                raise RuntimeError(f"batch {index} failed: {batch['reason']}")
            if not batch_report_path.is_file():
                raise ValueError("batch manifest missing")
        if len(report["images"]) != 48 or [r["name"] for r in report["images"]] != [Path(r["path"]).name for r in rows]:
            raise ValueError("full run did not cover exact 48 training photos")
        report["review_sheet_sha256"] = review_sheets(output, rows, report["images"], dataset_root)
        if any(row["status"] != "complete_unreviewed" for row in report["images"]):
            raise ValueError("one or more masks failed or are unusable")
        if (time.monotonic() - started > TOTAL_SECONDS or total_size(output) > TOTAL_OUTPUT_BYTES or
                common.digest(package_path) != package_sha or common.digest(checkpoint) != MODEL_SHA or
                smoke.source_digest(source) != SOURCE_SHA):
            raise ValueError("whole-trial time, byte, or source-hash gate failed")
        report["status"] = "complete_unreviewed"
    except BaseException as exc:
        report["status"] = "failed_or_partial_unusable"
        report["failure"] = repr(exc)
        if "review_sheet_sha256" not in report:
            try:
                report["review_sheet_sha256"] = review_sheets(output, rows, report["images"], dataset_root)
            except BaseException as preview_error:
                report["review_sheet_failure"] = repr(preview_error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("full manifest exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.batch_index not in (0, 1, 2):
            raise ValueError("worker needs exact batch index")
        run_batch(args.package, args.dataset_root, args.output / f"batch-{args.batch_index}",
                  args.checkpoint, args.sam_source, args.batch_index)
    else:
        if args.batch_index is not None:
            raise ValueError("batch index is supervisor-internal")
        supervise(args.package, args.dataset_root, args.output, args.checkpoint, args.sam_source)


if __name__ == "__main__":
    main()
