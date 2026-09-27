#!/usr/bin/env python3
"""Resumable, hash-bound four-view SAM2 mustard TRAIN mask generation."""

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
from scripts.object_motion import sam_point_full as full
from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import ycb_object_masks as common


PRIOR_SHA = "0ff3c6203301988890ded40276c58d510582bb08a7fa2f1884d2ba792349fd09"
PRIOR_BATCH_SHA = (
    "ec40a87c334f1168da4e56c54dd7373dd8650bd4b71a4bdcf70a124b580436c8",
    "2f0c8d51ee40e14c3dd57679d868fae01167563479abde29fbfc262bce839ee0",
)
BATCH_SIZE = 4
BATCH_SECONDS = 90
INVOCATION_SECONDS = 600
RAM_WAIT_SECONDS = 60
MAX_OUTPUT = 60 * 1024**2
MAX_LOG = 1024**2


def json_bytes(value):
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > 1024**2:
        raise ValueError("JSON report exceeds 1 MiB")
    return encoded


def exclusive_json(path, value):
    with Path(path).open("xb") as output:
        output.write(json_bytes(value))


def fixed_inputs(package_path, prompt_path, checkpoint, source):
    rows, prompts, package_sha = full.validated_inputs(package_path, checkpoint, source, prompt_path)
    if len(rows) != 48 or len(prompts) != 48:
        raise ValueError("exactly 48 training rows/prompts required")
    return rows, prompts, package_sha


def prior_sealed_rows(prior_root, rows, prompts):
    prior_root = Path(prior_root)
    if prior_root.is_symlink() or not prior_root.is_dir():
        raise ValueError("prior output must be a real directory")
    parent_path = prior_root / "manifest.json"
    if parent_path.is_symlink() or not parent_path.is_file() or common.digest(parent_path) != PRIOR_SHA:
        raise ValueError("prior parent manifest differs from sealed partial run")
    parent = json.loads(parent_path.read_text())
    if (parent.get("schema") != "sam21_mustard_point_full_v1" or
            parent.get("status") != "failed_or_partial_unusable" or
            parent.get("prompt_sha256") != full.PROMPTS_SHA or
            parent.get("checkpoint_sha256") != point.MODEL_SHA or
            parent.get("sam2_source_inventory_sha256") != point.SOURCE_SHA or
            len(parent.get("images", [])) != 16):
        raise ValueError("prior parent is not exact sealed 16-view source")
    sealed = []
    for index, expected in enumerate(PRIOR_BATCH_SHA):
        manifest = prior_root / f"batch-{index}" / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file() or common.digest(manifest) != expected:
            raise ValueError("prior batch manifest differs from sealed source")
        batch = json.loads(manifest.read_text())
        if (batch.get("batch_index") != index or batch.get("status") != "complete_unreviewed" or
                len(batch.get("images", [])) != 8):
            raise ValueError("prior batch not complete 8-view source")
        sealed.extend(batch["images"])
    if parent["images"] != sealed or [r["name"] for r in sealed] != [Path(r["path"]).name for r in rows[:16]]:
        raise ValueError("prior parent/batches do not exactly match first 16 training views")
    for index, row in enumerate(sealed):
        expected_row, prompt = rows[index], prompts[index]
        if (row.get("status") != "complete_unreviewed" or
                row.get("source_sha256") != expected_row["sha256"] or
                row.get("points_xy_label") != prompt["points_xy_label"]):
            raise ValueError("prior frame provenance differs from frozen photos/prompts")
        for kind in ("raw", "cleaned"):
            path = prior_root / f"batch-{index // 8}" / f"{kind}_masks" / f"{row['name']}.png"
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2 or
                    common.digest(path) != row.get(f"{kind}_mask_sha256")):
                raise ValueError("prior frame mask hash/size differs from sealed row")
    return sealed


def dependencies():
    return {"runner_sha256": common.digest(__file__),
            "point_helper_sha256": common.digest(point.__file__),
            "full_helper_sha256": common.digest(full.__file__),
            "smoke_helper_sha256": common.digest(smoke.__file__),
            "cleanup_helper_sha256": common.digest(cleanup.__file__),
            "shared_helper_sha256": common.digest(common.__file__)}


def expected_contract(rows, package_sha):
    return {"schema": "sam21_mustard_point_resume_contract_v1",
            "mask_status": "generated_unreviewed_not_qa_accepted",
            "scope": "exact 48 mustard TRAIN RGB photos, no held-out/GT/depth/pose",
            "package_sha256": package_sha, "prompt_sha256": full.PROMPTS_SHA,
            "checkpoint_sha256": point.MODEL_SHA, "sam2_source_inventory_sha256": point.SOURCE_SHA,
            "prior_partial_manifest_sha256": PRIOR_SHA, "prior_batch_manifest_sha256": list(PRIOR_BATCH_SHA),
            "box_xyxy_original_pixels": list(smoke.BOX), "dependencies": dependencies(),
            "expected_images": [{"name": Path(row["path"]).name, "source_sha256": row["sha256"]} for row in rows]}


def prepare_root(root, contract):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("output root may not be a symlink")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root.parent).free < common.MIN_FREE_BYTES:
            raise RuntimeError("output volume below 10 GiB free")
        root.mkdir()
        for name in ("frames", "staging", "failures", "invocations", "receipts"):
            (root / name).mkdir()
        exclusive_json(root / "contract.json", contract)
    if not root.is_dir():
        raise ValueError("output root must be a real directory")
    for name in ("frames", "staging", "failures", "invocations", "receipts"):
        path = root / name
        if path.is_symlink() or not path.is_dir():
            raise ValueError("output internal directory missing or linked")
    path = root / "contract.json"
    if path.is_symlink() or not path.is_file() or json.loads(path.read_text()) != contract:
        raise ValueError("resume contract differs from frozen inputs/code")
    if shutil.disk_usage(root).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output volume below 10 GiB free")


def frame_record(name, photo_sha, prompts, raw_sha, clean_sha, producer, extra=None):
    record = {"schema": "sam21_mustard_point_frame_v1", "status": "complete_unreviewed",
              "name": name, "source_sha256": photo_sha, "points_xy_label": prompts,
              "prompt_sha256": full.PROMPTS_SHA, "checkpoint_sha256": point.MODEL_SHA,
              "sam2_source_inventory_sha256": point.SOURCE_SHA,
              "raw_mask_sha256": raw_sha, "cleaned_mask_sha256": clean_sha,
              "producer": producer}
    if extra:
        record["diagnostics"] = extra
    return record


def publish_frame(root, invocation_id, record, raw_bytes, clean_bytes):
    root = Path(root)
    name = record["name"]
    target = root / "frames" / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    stage = root / "staging" / f"{invocation_id}-{name}"
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir()
    (stage / "raw.png").write_bytes(raw_bytes)
    (stage / "clean.png").write_bytes(clean_bytes)
    if (common.digest(stage / "raw.png") != record["raw_mask_sha256"] or
            common.digest(stage / "clean.png") != record["cleaned_mask_sha256"]):
        raise ValueError("staged mask hash differs from frame record")
    exclusive_json(stage / "frame.json", record)
    receipt = {"schema": "sam21_mustard_frame_receipt_v1", "name": name,
               "stage_name": stage.name, "frame_manifest_sha256": common.digest(stage / "frame.json"),
               "raw_mask_sha256": record["raw_mask_sha256"],
               "cleaned_mask_sha256": record["cleaned_mask_sha256"]}
    exclusive_json(root / "receipts" / f"{name}.json", receipt)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    os.rename(stage, target)


def recover_receipts(root, allowed_names):
    root = Path(root)
    allowed = set(allowed_names)
    receipts = root / "receipts"
    for path in receipts.iterdir():
        if (path.is_symlink() or not path.is_file() or not path.name.endswith(".json") or
                path.name[:-5] not in allowed or path.stat().st_size > 4096):
            raise ValueError("extra/linked/oversize receipt")
        receipt = json.loads(path.read_text())
        name = path.name[:-5]
        stage_name = receipt.get("stage_name")
        if (receipt.get("schema") != "sam21_mustard_frame_receipt_v1" or receipt.get("name") != name or
                not isinstance(stage_name, str) or not stage_name.endswith("-" + name) or
                "/" in stage_name or "\\" in stage_name):
            raise ValueError("invalid frame receipt")
        target = root / "frames" / name
        if not target.exists():
            stage = root / "staging" / stage_name
            if stage.is_symlink() or not stage.is_dir():
                raise ValueError("published receipt lacks staged frame")
            if {p.name for p in stage.iterdir()} != {"raw.png", "clean.png", "frame.json"}:
                raise ValueError("receipt staged frame incomplete")
            for member in stage.iterdir():
                if member.is_symlink() or not member.is_file():
                    raise ValueError("receipt staged frame contains link")
            if (common.digest(stage / "frame.json") != receipt.get("frame_manifest_sha256") or
                    common.digest(stage / "raw.png") != receipt.get("raw_mask_sha256") or
                    common.digest(stage / "clean.png") != receipt.get("cleaned_mask_sha256")):
                raise ValueError("receipt staged frame hash mismatch")
            os.rename(stage, target)


def import_prior(root, prior_root, rows, prompts, sealed, invocation_id, dataset_root):
    existing = {path.name for path in (Path(root) / "frames").iterdir()}
    for index, source in enumerate(sealed):
        name = source["name"]
        if name in existing:
            continue
        common.verified_training_photo(dataset_root, rows[index])
        raw_path = Path(prior_root) / f"batch-{index // 8}" / "raw_masks" / f"{name}.png"
        clean_path = Path(prior_root) / f"batch-{index // 8}" / "cleaned_masks" / f"{name}.png"
        raw_bytes, clean_bytes = raw_path.read_bytes(), clean_path.read_bytes()
        record = frame_record(name, rows[index]["sha256"], prompts[index]["points_xy_label"],
                              source["raw_mask_sha256"], source["cleaned_mask_sha256"],
                              {"kind": "sealed_prior", "parent_sha256": PRIOR_SHA,
                               "batch_sha256": PRIOR_BATCH_SHA[index // 8]},
                              {"source_frame_sha256": common.digest(Path(prior_root) / f"batch-{index // 8}" / "manifest.json")})
        publish_frame(root, f"import-{invocation_id}", record, raw_bytes, clean_bytes)


def verified_frames(root, rows, prompts, sealed, expected_producer_sha=None):
    root = Path(root)
    if expected_producer_sha is None:
        expected_producer_sha = dependencies()["runner_sha256"]
    if (not isinstance(expected_producer_sha, str) or len(expected_producer_sha) != 64 or
            any(char not in "0123456789abcdef" for char in expected_producer_sha)):
        raise ValueError("expected producer SHA-256 must be a lowercase digest")
    expected = {Path(row["path"]).name: (row, prompt) for row, prompt in zip(rows, prompts)}
    result = {}
    sealed_by_name = {row["name"]: (index, row) for index, row in enumerate(sealed)}
    receipts = {p.name[:-5] for p in (root / "receipts").iterdir() if p.name.endswith(".json")}
    if receipts != {p.name for p in (root / "frames").iterdir()}:
        raise ValueError("frame/receipt inventory mismatch")
    for directory in (root / "frames").iterdir():
        if directory.is_symlink() or not directory.is_dir() or directory.name not in expected:
            raise ValueError("resume frames contain linked, extra or non-directory entry")
        if {p.name for p in directory.iterdir()} != {"raw.png", "clean.png", "frame.json"}:
            raise ValueError("resume frame has missing/extra files")
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("resume frame contains linked/non-file artifact")
        path = directory / "frame.json"
        if path.stat().st_size > 16_384:
            raise ValueError("frame manifest exceeds byte cap")
        record = json.loads(path.read_text())
        receipt = json.loads((root / "receipts" / f"{directory.name}.json").read_text())
        if (common.digest(path) != receipt.get("frame_manifest_sha256") or
                common.digest(directory / "raw.png") != receipt.get("raw_mask_sha256") or
                common.digest(directory / "clean.png") != receipt.get("cleaned_mask_sha256")):
            raise ValueError("published frame differs from append receipt")
        source, prompt = expected[directory.name]
        if (record.get("schema") != "sam21_mustard_point_frame_v1" or
                record.get("status") != "complete_unreviewed" or record.get("name") != directory.name or
                record.get("source_sha256") != source["sha256"] or
                record.get("points_xy_label") != prompt["points_xy_label"] or
                record.get("prompt_sha256") != full.PROMPTS_SHA or
                record.get("checkpoint_sha256") != point.MODEL_SHA or
                record.get("sam2_source_inventory_sha256") != point.SOURCE_SHA):
            raise ValueError("resume frame provenance differs from frozen contract")
        producer = record.get("producer")
        if not isinstance(producer, dict) or producer.get("kind") not in ("sealed_prior", "resume_inference"):
            raise ValueError("resume frame producer invalid")
        if directory.name in sealed_by_name:
            index, prior = sealed_by_name[directory.name]
            if (producer.get("kind") != "sealed_prior" or producer.get("parent_sha256") != PRIOR_SHA or
                    producer.get("batch_sha256") != PRIOR_BATCH_SHA[index // 8] or
                    record.get("raw_mask_sha256") != prior["raw_mask_sha256"] or
                    record.get("cleaned_mask_sha256") != prior["cleaned_mask_sha256"]):
                raise ValueError("first 16 must derive only from sealed prior")
        elif (producer.get("kind") != "resume_inference" or
              producer.get("runner_sha256") != expected_producer_sha or
              not isinstance(producer.get("invocation_id"), str) or
              not producer["invocation_id"].isdigit()):
            raise ValueError("later frames must derive from frozen resume inference")
        for kind, filename in (("raw", "raw.png"), ("cleaned", "clean.png")):
            mask_path = directory / filename
            if mask_path.stat().st_size > 2 * 1024**2 or common.digest(mask_path) != record.get(f"{kind}_mask_sha256"):
                raise ValueError("resume frame mask size/hash mismatch")
            with Image.open(mask_path) as image:
                if image.mode != "L" or image.size != common.IMAGE_SIZE:
                    raise ValueError("resume frame mask mode/size mismatch")
                array = np.asarray(image)
            if not np.isin(array, (0, 255)).all():
                raise ValueError("resume frame mask nonbinary")
            point.point_membership(array, prompt["points_xy_label"])
        result[directory.name] = record
    if len(result) != len(set(result)):
        raise ValueError("duplicate resume frame name")
    return result


def check_auxiliary(root, invocation_id):
    root = Path(root)
    for name in ("staging", "failures", "invocations"):
        directory = root / name
        for entry in directory.iterdir():
            if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
                raise ValueError("linked or unsupported auxiliary entry")
    failures = list((root / "failures").iterdir())
    if failures:
        raise RuntimeError("prior deterministic frame failure exists; no automatic retry")
    if (root / "invocations" / f"{invocation_id}.json").exists():
        raise FileExistsError("invocation id already used")


def frame_hash_map(root, records):
    root = Path(root)
    return {name: {"frame_manifest_sha256": common.digest(root / "frames" / name / "frame.json"),
                   "raw_mask_sha256": record["raw_mask_sha256"],
                   "cleaned_mask_sha256": record["cleaned_mask_sha256"]}
            for name, record in sorted(records.items())}


def verify_prior_invocations(root, records):
    current = frame_hash_map(root, records)
    for path in (Path(root) / "invocations").iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("linked/oversize prior invocation report")
        report = json.loads(path.read_text())
        prior = report.get("frame_sha256_map")
        if report.get("schema") != "sam21_mustard_point_resume_invocation_v1" or not isinstance(prior, dict):
            raise ValueError("prior invocation lacks frame hash seal")
        for name, hashes in prior.items():
            if current.get(name) != hashes:
                raise ValueError("published frame differs from prior invocation seal")


def total_size(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def candidate_masks(predict, photo, prompt):
    raw = point.binary_raw(predict(photo, smoke.BOX, prompt["points_xy_label"]))
    point.point_membership(raw, prompt["points_xy_label"])
    smoke.bounded_mask(raw)
    cleaned, stats = cleanup.largest_component(raw)
    point.point_membership(cleaned, prompt["points_xy_label"])
    return raw, cleaned, stats


def worker(root, invocation_id, names, prior_root, package_path, dataset_root, prompt_path, checkpoint, source):
    rows, prompts, package_sha = fixed_inputs(package_path, prompt_path, checkpoint, source)
    contract = expected_contract(rows, package_sha)
    prepare_root(root, contract)
    sealed = prior_sealed_rows(prior_root, rows, prompts)
    recover_receipts(root, [Path(row["path"]).name for row in rows])
    existing = verified_frames(root, rows, prompts, sealed)
    verify_prior_invocations(root, existing)
    expected_names = [Path(row["path"]).name for row in rows if Path(row["path"]).name not in existing][:BATCH_SIZE]
    if names != expected_names or not names:
        raise ValueError("worker names differ from next four frozen pending views")
    lookup = {Path(row["path"]).name: (row, prompt) for row, prompt in zip(rows, prompts)}
    predict = point.sam_predictor(source, checkpoint)
    for name in names:
        row, prompt = lookup[name]
        stage_failure = Path(root) / "failures" / f"{name}.json"
        try:
            photo = common.verified_training_photo(dataset_root, row)
            raw, cleaned, stats = candidate_masks(predict, photo, prompt)
            raw_image = Image.fromarray(raw, "L")
            with tempfile.TemporaryFile(dir=Path(root) / "staging") as raw_file:
                raw_image.save(raw_file, "PNG")
                raw_file.seek(0)
                raw_bytes = raw_file.read()
            with tempfile.TemporaryFile(dir=Path(root) / "staging") as clean_file:
                Image.fromarray(cleaned, "L").save(clean_file, "PNG")
                clean_file.seek(0)
                clean_bytes = clean_file.read()
            import hashlib
            record = frame_record(name, row["sha256"], prompt["points_xy_label"],
                                  hashlib.sha256(raw_bytes).hexdigest(), hashlib.sha256(clean_bytes).hexdigest(),
                                  {"kind": "resume_inference", "invocation_id": invocation_id,
                                   "runner_sha256": common.digest(__file__)}, stats)
            publish_frame(root, invocation_id, record, raw_bytes, clean_bytes)
        except ValueError as error:
            exclusive_json(stage_failure, {"schema": "sam21_mustard_frame_failure_v1", "name": name,
                                           "invocation_id": invocation_id, "failure": repr(error),
                                           "source_sha256": row["sha256"], "prompt_sha256": full.PROMPTS_SHA})
            raise
    if (common.digest(package_path) != package_sha or common.digest(prompt_path) != full.PROMPTS_SHA or
            common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA or
            dependencies() != contract["dependencies"]):
        raise ValueError("source/model/code changed during batch")


def wait_for_ram(deadline, waited):
    while smoke.available_kib() < smoke.MIN_AVAILABLE_KIB:
        if waited >= RAM_WAIT_SECONDS or time.monotonic() >= deadline:
            raise RuntimeError("RAM pre-batch gate unavailable within aggregate 60 seconds")
        time.sleep(1)
        waited += 1
    return waited


def bounded_batch(args, root, deadline):
    command = [sys.executable, "-m", "scripts.object_motion.sam_point_resume", "--worker", *args]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryFile(dir=Path(root) / "staging") as logfile:
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
                    reason = "invocation above 600 seconds"
                elif logfile.seek(0, os.SEEK_END) > MAX_LOG:
                    reason = "log above 1 MiB"
                elif total_size(root) > MAX_OUTPUT:
                    reason = "output above 60 MiB"
                elif shutil.disk_usage(root).free < common.MIN_FREE_BYTES:
                    reason = "output volume below 10 GiB free"
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
            final_log = logfile.seek(0, os.SEEK_END)
            if reason is None:
                reason = full.terminal_guard(time.monotonic() - started, time.monotonic() > deadline,
                                             final_log, total_size(root), shutil.disk_usage(root).free)
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
    return {"status": "complete" if child.returncode == 0 and not reason else "failed",
            "reason": reason or (excerpt if child.returncode else None),
            "returncode": child.returncode, "seconds": time.monotonic() - started,
            "peak_worker_rss_kib": peak}


def review_sheets(root, rows, records, dataset_root, invocation_id):
    tile = (200, 210)
    sheets = {kind: Image.new("RGB", (8 * tile[0], 6 * tile[1]), "white") for kind in ("rgb", "raw", "cleaned")}
    draws = {kind: ImageDraw.Draw(sheet) for kind, sheet in sheets.items()}
    for index, row in enumerate(rows):
        name = Path(row["path"]).name
        photo = common.verified_training_photo(dataset_root, row)
        record = records.get(name)
        for kind in sheets:
            if kind == "rgb":
                display = photo.crop(smoke.BOX)
            elif record is None:
                display = Image.new("RGB", (280, 350), "black")
                ImageDraw.Draw(display).text((8, 8), "MISSING", fill="red")
            else:
                path = Path(root) / "frames" / name / ("raw.png" if kind == "raw" else "clean.png")
                if common.digest(path) != record[f"{kind}_mask_sha256"]:
                    raise ValueError("review sheet input mask hash changed")
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
        path = Path(root) / f"review_{kind}_{len(records):02}_frames_{invocation_id}.jpg"
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        sheet.save(path, "JPEG", quality=90)
        hashes[kind] = common.digest(path)
    return hashes


def complete_inventory(root, rows, records, contract):
    if len(records) != 48:
        return None
    inventory = {"schema": "sam21_mustard_point_mask_inventory_v1",
                 "status": "generated_unreviewed", "full_training_package_ready": False,
                 "mask_role": "coarse pose support, not exact silhouette",
                 "contract_sha256": common.digest(Path(root) / "contract.json"),
                 "package_sha256": contract["package_sha256"], "prompt_sha256": full.PROMPTS_SHA,
                 "checkpoint_sha256": point.MODEL_SHA, "sam2_source_inventory_sha256": point.SOURCE_SHA,
                 "images": [{"name": Path(row["path"]).name, "source_sha256": row["sha256"],
                             "cleaned_mask_sha256": records[Path(row["path"]).name]["cleaned_mask_sha256"],
                             "cleaned_mask_path": f"frames/{Path(row['path']).name}/clean.png",
                             "frame_manifest_sha256": common.digest(Path(root) / "frames" / Path(row["path"]).name / "frame.json")}
                            for row in rows]}
    path = Path(root) / "complete_inventory.json"
    if path.exists():
        if json.loads(path.read_text()) != inventory:
            raise ValueError("existing complete inventory differs from validated frames")
    else:
        exclusive_json(path, inventory)
    return common.digest(path)


def supervise(root, invocation_id, prior_root, package_path, dataset_root, prompt_path, checkpoint, source):
    root, prior_root, package_path, prompt_path, checkpoint, source = map(
        Path, (root, prior_root, package_path, prompt_path, checkpoint, source))
    if not invocation_id.isdigit() or not 1 <= len(invocation_id) <= 6:
        raise ValueError("invocation id must be 1-6 digits")
    rows, prompts, package_sha = fixed_inputs(package_path, prompt_path, checkpoint, source)
    sealed = prior_sealed_rows(prior_root, rows, prompts)
    contract = expected_contract(rows, package_sha)
    prepare_root(root, contract)
    check_auxiliary(root, invocation_id)
    recover_receipts(root, [Path(row["path"]).name for row in rows])
    records = verified_frames(root, rows, prompts, sealed)
    verify_prior_invocations(root, records)
    if any(name not in records for name in [r["name"] for r in sealed]):
        import_prior(root, prior_root, rows, prompts, sealed, invocation_id, dataset_root)
        records = verified_frames(root, rows, prompts, sealed)
    if len(records) < 16:
        raise ValueError("sealed first 16 not imported")
    started = time.monotonic()
    deadline = started + INVOCATION_SECONDS
    waited = 0
    report = {"schema": "sam21_mustard_point_resume_invocation_v1", "status": "running",
              "invocation_id": invocation_id, "contract_sha256": common.digest(root / "contract.json"),
              "prior_partial_sha256": PRIOR_SHA, "initial_frames": len(records), "batches": []}
    try:
        while len(records) < 48:
            if (time.monotonic() >= deadline or dependencies() != contract["dependencies"] or
                    common.digest(package_path) != package_sha or common.digest(prompt_path) != full.PROMPTS_SHA or
                    common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA):
                raise RuntimeError("invocation time or input/code hash gate failed")
            waited = wait_for_ram(deadline, waited)
            names = [Path(row["path"]).name for row in rows if Path(row["path"]).name not in records][:BATCH_SIZE]
            args = ["--root", str(root), "--invocation-id", invocation_id,
                    "--prior-root", str(prior_root), "--package", str(package_path),
                    "--dataset-root", str(dataset_root), "--prompts", str(prompt_path),
                    "--checkpoint", str(checkpoint), "--sam-source", str(source),
                    "--names", ",".join(names)]
            batch = bounded_batch(args, root, deadline)
            batch["names"] = names
            report["batches"].append(batch)
            records = verified_frames(root, rows, prompts, sealed)
            batch["frames_after"] = len(records)
            if batch["status"] != "complete":
                raise RuntimeError(f"batch failed: {batch['reason']}")
            if any(name not in records for name in names):
                raise ValueError("batch reported success without all four atomic frame records")
        report["inventory_sha256"] = complete_inventory(root, rows, records, contract)
        report["review_sheet_sha256"] = review_sheets(root, rows, records, dataset_root, invocation_id)
        if (time.monotonic() >= deadline or total_size(root) > MAX_OUTPUT or
                common.digest(package_path) != package_sha or common.digest(prompt_path) != full.PROMPTS_SHA or
                common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA or
                dependencies() != contract["dependencies"]):
            raise ValueError("final invocation resource/input/code gate failed")
        report["status"] = "complete_unreviewed"
    except BaseException as error:
        report["status"] = "failed_or_partial"
        report["failure"] = repr(error)
        try:
            records = verified_frames(root, rows, prompts, sealed)
            report["frames_after"] = len(records)
            report["review_sheet_sha256"] = review_sheets(root, rows, records, dataset_root, invocation_id)
        except BaseException as preview_error:
            report["review_sheet_failure"] = repr(preview_error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["ram_wait_seconds"] = waited
        report["frames_after"] = len(records)
        report["frame_sha256_map"] = frame_hash_map(root, records)
        exclusive_json(root / "invocations" / f"{invocation_id}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--names", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.names:
            raise ValueError("worker requires exact names")
        worker(args.root, args.invocation_id, args.names.split(","), args.prior_root, args.package,
               args.dataset_root, args.prompts, args.checkpoint, args.sam_source)
    else:
        if args.names is not None:
            raise ValueError("names are supervisor-internal")
        supervise(args.root, args.invocation_id, args.prior_root, args.package,
                  args.dataset_root, args.prompts, args.checkpoint, args.sam_source)


if __name__ == "__main__":
    main()
