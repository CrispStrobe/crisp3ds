#!/usr/bin/env python3
"""Bounded Mac CPU continuation for 32 remaining mustard TRAIN masks."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from PIL import Image

from scripts.object_motion import sam_m1_parity as mac
from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import sam_point_full as full
from scripts.object_motion import sam_point_resume as core
from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import ycb_object_masks as common


QA_SHA = "b7e5457c567c33f5f3626a16dc6cd1645ed022d5db477dba33ec9fde46eab010"
QA_PATH = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_parity_review.json"
SETUP_SHA = "2c520f3e07729ea881707b13fff234fd5400e851acbe1390b743abcec07fc874"
BATCH_SIZE = 4
BATCH_SECONDS = 90
TOTAL_SECONDS = 600
RAM_WAIT_SECONDS = 60
MAX_OUTPUT = 60 * 1024**2
MAX_LOG = 1024**2


def bound_metadata(path, expected_sha=None, limit=1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing, linked or oversize metadata: {path}")
    sha = common.digest(path)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(f"metadata SHA-256 mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return value, sha


def accepted_parity(qa_path, setup_report, parity_root):
    qa, qa_sha = bound_metadata(qa_path, QA_SHA, 16_384)
    setup, setup_sha = bound_metadata(setup_report, SETUP_SHA)
    parity_root = Path(parity_root)
    parity, parity_sha = bound_metadata(parity_root / "manifest.json", qa.get("parity_manifest_sha256"))
    supervisor, supervisor_sha = bound_metadata(parity_root / "supervisor.json", qa.get("supervisor_sha256"))
    if (qa.get("schema") != "sam21_m1_parity_visual_qa_v1" or qa.get("status") != "accepted" or
            qa.get("decision") != "accepted_three_view_m1_cpu_parity_only" or
            qa.get("names") != list(mac.NAMES) or qa.get("full_training_package_ready") is not False or
            qa.get("heldout_or_reference_geometry_used") is not False or
            qa.get("remaining_training_masks_generated") is not False or
            qa.get("setup_report_sha256") != setup_sha or
            parity.get("schema") != "sam21_m1_cpu_parity_v1" or
            parity.get("status") != "parity_observed_pending_visual_qa" or
            supervisor.get("status") != "parity_observed_pending_visual_qa" or
            setup.get("status") != "complete"):
        raise ValueError("accepted narrow M1 parity QA/setup gate failed")
    sheet = Path(qa["review_sheet"])
    if sheet.is_symlink() or not sheet.is_file() or common.digest(sheet) != qa.get("review_sheet_sha256"):
        raise ValueError("reviewed parity sheet changed")
    return {"qa_sha256": qa_sha, "setup_report_sha256": setup_sha,
            "parity_manifest_sha256": parity_sha, "parity_supervisor_sha256": supervisor_sha,
            "parity_review_sheet_sha256": qa["review_sheet_sha256"]}


def staged_train48(stage_root, rows, package_sha, stage_report):
    stage_root = Path(stage_root)
    stage_report = Path(stage_report)
    report, report_sha = bound_metadata(stage_report, limit=256 * 1024)
    if (report.get("schema") != "sam21_m1_train48_photos_v1" or report.get("status") != "complete" or
            report.get("package_sha256") != package_sha or
            stage_report.resolve() != stage_root.resolve() / "stage-report.json" or
            not isinstance(report.get("output"), str) or Path(report["output"]).resolve() != stage_root.resolve() or
            report.get("contains_masks_or_reference_geometry") is not False):
        raise ValueError("48-photo stage report not complete/bound to package")
    expected = [Path(row["path"]).name for row in rows]
    expected_hashes = {Path(row["path"]).name: row["sha256"] for row in rows}
    if report.get("train_names") != expected or report.get("photo_sha256") != expected_hashes:
        raise ValueError("48-photo stage report name/hash map differs from frozen training package")
    photos = stage_root / "photos"
    if photos.is_symlink() or not photos.is_dir():
        raise ValueError("staged photos directory missing or linked")
    if {p.name for p in photos.iterdir()} != set(expected):
        raise ValueError("staged photos contain missing/extra/held-out filenames")
    for row in rows:
        common.verified_training_photo(stage_root, row)
    return report_sha


def inputs(package, dataset_root, prompts, checkpoint, source, prior_root, setup_report,
           parity_root, stage_report, qa_path):
    mac.external_inputs((package, dataset_root, prompts, checkpoint, source, prior_root,
                         setup_report, parity_root, stage_report))
    rows, prompt_rows, package_sha = full.validated_inputs(package, checkpoint, source, prompts)
    sealed = core.prior_sealed_rows(prior_root, rows, prompt_rows)
    stage_sha = staged_train48(dataset_root, rows, package_sha, stage_report)
    qa = accepted_parity(qa_path, setup_report, parity_root)
    return rows, prompt_rows, sealed, package_sha, stage_sha, qa


def contract(rows, package_sha, stage_sha, qa):
    return {"schema": "sam21_mustard_m1_resume_contract_v1",
            "mask_status": "generated_unreviewed_not_qa_accepted",
            "scope": "48 mustard TRAIN RGB only; first 16 sealed prior plus Mac CPU remaining 32",
            "package_sha256": package_sha, "prompt_sha256": full.PROMPTS_SHA,
            "checkpoint_sha256": point.MODEL_SHA, "sam2_source_inventory_sha256": point.SOURCE_SHA,
            "prior_partial_manifest_sha256": core.PRIOR_SHA,
            "prior_batch_manifest_sha256": list(core.PRIOR_BATCH_SHA),
            "stage_report_sha256": stage_sha, "accepted_parity": qa,
            "box_xyxy_original_pixels": list(smoke.BOX),
            "host": {"platform": "darwin_arm64_cpu", "m1_resource_adapter_sha256": common.digest(mac.__file__)},
            "dependencies": {**core.dependencies(), "m1_runner_sha256": common.digest(__file__)},
            "expected_images": [{"name": Path(row["path"]).name, "source_sha256": row["sha256"]}
                                for row in rows]}


def verified_mac_frames(root, rows, prompts, sealed, expected_contract):
    records = core.verified_frames(root, rows, prompts, sealed,
                                   expected_producer_sha=expected_contract["dependencies"]["m1_runner_sha256"])
    prior_names = {row["name"] for row in sealed}
    contract_sha = common.digest(Path(root) / "contract.json")
    for name, record in records.items():
        if name in prior_names:
            continue
        producer = record["producer"]
        if (producer.get("host_backend") != "darwin_arm64_cpu" or
                producer.get("resource_adapter_sha256") != expected_contract["host"]["m1_resource_adapter_sha256"] or
                producer.get("contract_sha256") != contract_sha):
            raise ValueError("Mac frame producer/host contract mismatch")
    return records


def jpeg_bytes(mask):
    with io.BytesIO() as stream:
        Image.fromarray(mask, "L").save(stream, "PNG")
        return stream.getvalue()


def infer_and_publish(root, invocation_id, row, prompt, predict, dataset_root):
    root = Path(root)
    name = Path(row["path"]).name
    photo = common.verified_training_photo(dataset_root, row)
    raw = point.binary_raw(predict(photo, smoke.BOX, prompt["points_xy_label"]))
    raw_bytes = jpeg_bytes(raw)
    try:
        point.point_membership(raw, prompt["points_xy_label"])
        smoke.bounded_mask(raw)
        from scripts.object_motion import sam_mask_cleanup as cleanup
        clean, stats = cleanup.largest_component(raw)
        point.point_membership(clean, prompt["points_xy_label"])
        clean_bytes = jpeg_bytes(clean)
        producer = {"kind": "resume_inference", "invocation_id": invocation_id,
                    "runner_sha256": common.digest(__file__), "host_backend": "darwin_arm64_cpu",
                    "resource_adapter_sha256": common.digest(mac.__file__),
                    "contract_sha256": common.digest(root / "contract.json")}
        record = core.frame_record(name, row["sha256"], prompt["points_xy_label"],
                                   hashlib.sha256(raw_bytes).hexdigest(),
                                   hashlib.sha256(clean_bytes).hexdigest(), producer, stats)
        core.publish_frame(root, invocation_id, record, raw_bytes, clean_bytes)
    except ValueError as error:
        failure = root / "failures" / f"{name}.json"
        raw_path = root / "failures" / f"{name}.raw.png"
        with raw_path.open("xb") as out:
            out.write(raw_bytes)
        core.exclusive_json(failure, {"schema": "sam21_m1_resume_frame_failure_v1", "name": name,
                                      "source_sha256": row["sha256"], "raw_mask_sha256": common.digest(raw_path),
                                      "failure": repr(error), "invocation_id": invocation_id})
        raise
    return record


def worker(root, invocation_id, names, package, dataset_root, prompts, checkpoint, source,
           prior_root, setup_report, parity_root, stage_report, qa_path):
    root = Path(root)
    mac.host_preflight(root)
    rows, prompt_rows, sealed, package_sha, stage_sha, qa = inputs(
        package, dataset_root, prompts, checkpoint, source, prior_root, setup_report,
        parity_root, stage_report, qa_path)
    expected = contract(rows, package_sha, stage_sha, qa)
    core.prepare_root(root, expected)
    core.recover_receipts(root, [Path(row["path"]).name for row in rows])
    records = verified_mac_frames(root, rows, prompt_rows, sealed, expected)
    core.verify_prior_invocations(root, records)
    pending = [Path(row["path"]).name for row in rows if Path(row["path"]).name not in records][:BATCH_SIZE]
    if not pending or names != pending:
        raise ValueError("worker names differ from next frozen four pending training photos")
    lookup = {Path(row["path"]).name: (row, prompt) for row, prompt in zip(rows, prompt_rows)}
    predict = point.sam_predictor(source, checkpoint)
    started = time.monotonic()
    for name in names:
        if (time.monotonic() - started > BATCH_SECONDS or mac.available_kib() < mac.MIN_AVAILABLE_KIB or
                mac.rss_kib(os.getpid()) > mac.MAX_RSS_KIB or
                mac.disk_free_both(root) < common.MIN_FREE_BYTES or core.total_size(root) > MAX_OUTPUT):
            raise RuntimeError("Mac worker resource gate failed between frames")
        row, prompt = lookup[name]
        infer_and_publish(root, invocation_id, row, prompt, predict, dataset_root)
    if (common.digest(package) != package_sha or common.digest(prompts) != full.PROMPTS_SHA or
            common.digest(checkpoint) != point.MODEL_SHA or smoke.source_digest(source) != point.SOURCE_SHA or
            bound_metadata(stage_report)[1] != stage_sha or accepted_parity(qa_path, setup_report, parity_root) != qa or
            contract(rows, package_sha, stage_sha, qa) != expected):
        raise ValueError("Mac worker input/code/QA changed during batch")


def wait_for_ram(deadline, waited):
    while mac.available_kib() < mac.MIN_AVAILABLE_KIB:
        if waited >= RAM_WAIT_SECONDS or time.monotonic() >= deadline:
            raise RuntimeError("Mac RAM gate unavailable within aggregate 60 seconds")
        time.sleep(1)
        waited += 1
    return waited


def terminal_guard(seconds, deadline, rss, ram, output_bytes, log_bytes, disk_bytes, returncode):
    if seconds > BATCH_SECONDS:
        return "batch above 90 seconds"
    if deadline:
        return "invocation above 600 seconds"
    if rss > mac.MAX_RSS_KIB:
        return "worker RSS above 1.5 GiB"
    if ram < mac.MIN_AVAILABLE_KIB:
        return "Mac memory estimate below 2.5 GiB"
    if output_bytes > MAX_OUTPUT:
        return "output above 60 MiB"
    if log_bytes > MAX_LOG:
        return "log above 1 MiB"
    if disk_bytes < common.MIN_FREE_BYTES:
        return "internal/external disk below 10 GiB"
    if returncode != 0:
        return f"worker exit {returncode}"
    return None


def bounded_batch(args, root, deadline):
    command = [sys.executable, "-m", "scripts.object_motion.sam_m1_resume", "--worker", *args]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryFile(dir=Path(root) / "staging") as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        started, peak, reason = time.monotonic(), 0, None
        try:
            while child.poll() is None:
                peak = max(peak, mac.rss_kib(child.pid))
                reason = terminal_guard(time.monotonic() - started, time.monotonic() > deadline,
                                        peak, mac.available_kib(), core.total_size(root),
                                        logfile.seek(0, os.SEEK_END), mac.disk_free_both(root), 0)
                if reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            child.wait()
            reason = reason or terminal_guard(time.monotonic() - started, time.monotonic() > deadline,
                                              peak, mac.available_kib(), core.total_size(root),
                                              logfile.seek(0, os.SEEK_END), mac.disk_free_both(root), child.returncode)
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return {"status": "complete" if reason is None else "failed", "reason": reason or
            (excerpt if child.returncode else None), "returncode": child.returncode,
            "seconds": time.monotonic() - started, "peak_worker_rss_kib": peak}


def supervise(root, invocation_id, package, dataset_root, prompts, checkpoint, source,
              prior_root, setup_report, parity_root, stage_report, qa_path):
    root = Path(root)
    if not invocation_id.isdigit() or not 1 <= len(invocation_id) <= 6:
        raise ValueError("invocation id must be 1-6 digits")
    mac.host_preflight(root)
    rows, prompt_rows, sealed, package_sha, stage_sha, qa = inputs(
        package, dataset_root, prompts, checkpoint, source, prior_root, setup_report,
        parity_root, stage_report, qa_path)
    expected = contract(rows, package_sha, stage_sha, qa)
    core.prepare_root(root, expected)
    core.check_auxiliary(root, invocation_id)
    core.recover_receipts(root, [Path(row["path"]).name for row in rows])
    records = verified_mac_frames(root, rows, prompt_rows, sealed, expected)
    core.verify_prior_invocations(root, records)
    if any(row["name"] not in records for row in sealed):
        core.import_prior(root, prior_root, rows, prompt_rows, sealed, invocation_id, dataset_root)
        records = verified_mac_frames(root, rows, prompt_rows, sealed, expected)
    if len(records) < 16:
        raise ValueError("sealed prior 16 incomplete")
    started, waited = time.monotonic(), 0
    deadline = started + TOTAL_SECONDS
    report = {"schema": "sam21_mustard_point_resume_invocation_v1", "status": "running",
              "host_backend": "darwin_arm64_cpu", "invocation_id": invocation_id,
              "contract_sha256": common.digest(root / "contract.json"), "initial_frames": len(records),
              "batches": []}
    try:
        while len(records) < 48:
            if (time.monotonic() > deadline or mac.disk_free_both(root) < common.MIN_FREE_BYTES or
                    core.total_size(root) > MAX_OUTPUT or contract(rows, package_sha, stage_sha, qa) != expected or
                    accepted_parity(qa_path, setup_report, parity_root) != qa):
                raise RuntimeError("Mac continuation time/disk/input gate failed")
            waited = wait_for_ram(deadline, waited)
            names = [Path(row["path"]).name for row in rows if Path(row["path"]).name not in records][:BATCH_SIZE]
            arguments = ["--root", str(root), "--invocation-id", invocation_id,
                         "--package", str(package), "--dataset-root", str(dataset_root),
                         "--prompts", str(prompts), "--checkpoint", str(checkpoint),
                         "--sam-source", str(source), "--prior-root", str(prior_root),
                         "--setup-report", str(setup_report), "--parity-root", str(parity_root),
                         "--stage-report", str(stage_report), "--qa", str(qa_path),
                         "--names", ",".join(names)]
            batch = bounded_batch(arguments, root, deadline)
            batch["names"] = names
            report["batches"].append(batch)
            records = verified_mac_frames(root, rows, prompt_rows, sealed, expected)
            batch["frames_after"] = len(records)
            if batch["status"] != "complete" or any(name not in records for name in names):
                raise RuntimeError(f"Mac continuation batch failed/incomplete: {batch['reason']}")
        report["inventory_sha256"] = core.complete_inventory(root, rows, records, expected)
        report["review_sheet_sha256"] = core.review_sheets(root, rows, records, dataset_root, invocation_id)
        if (time.monotonic() > deadline or core.total_size(root) > MAX_OUTPUT or
                mac.disk_free_both(root) < common.MIN_FREE_BYTES or contract(rows, package_sha, stage_sha, qa) != expected):
            raise RuntimeError("Mac continuation final resource/input gate failed")
        report["status"] = "complete_unreviewed"
    except BaseException as error:
        report["status"] = "failed_or_partial"
        report["failure"] = repr(error)
        try:
            records = verified_mac_frames(root, rows, prompt_rows, sealed, expected)
            report["review_sheet_sha256"] = core.review_sheets(root, rows, records, dataset_root, invocation_id)
        except BaseException as preview_error:
            report["review_sheet_failure"] = repr(preview_error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["ram_wait_seconds"] = waited
        report["frames_after"] = len(records)
        report["frame_sha256_map"] = core.frame_hash_map(root, records)
        core.exclusive_json(root / "invocations" / f"{invocation_id}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "package", "dataset-root", "prompts", "checkpoint", "sam-source",
                 "prior-root", "setup-report", "parity-root", "stage-report", "qa"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--names", help=argparse.SUPPRESS)
    args = parser.parse_args()
    values = (args.root, args.invocation_id, args.package, args.dataset_root, args.prompts,
              args.checkpoint, args.sam_source, args.prior_root, args.setup_report,
              args.parity_root, args.stage_report, args.qa)
    if args.worker:
        if not args.names:
            raise ValueError("worker requires supervisor-selected names")
        worker(args.root, args.invocation_id, args.names.split(","), *values[2:])
    else:
        if args.names:
            raise ValueError("names are supervisor-internal")
        supervise(*values)


if __name__ == "__main__":
    main()
