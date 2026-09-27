#!/usr/bin/env python3
"""One bounded, unreviewed five-view SAM board-negative candidate on Mac CPU."""

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

from scripts.object_motion import sam_m1_parity as mac
from scripts.object_motion import sam_m1_resume as resume
from scripts.object_motion import sam_mask_cleanup as cleanup
from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import sam_point_full as full
from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import ycb_object_masks as common


NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in (318, 330, 336, 342, 348))
PROMPT_PATH = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_board5_prompts.json"
PROMPT_SHA = "8cc0b6fa49d977194870ef91f10b83a793294d6b3903cb1e08643016bdc9b4f8"
INVENTORY_SHA = "b47e8caec7d05fae7bb84ae9b3eec7d4a35b1457ca9df67a3b984cbaaeed8a98"
QA_PATH = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_full_review.json"
QA_SHA = "4084e8e458b55c566dd138f9d4b8b853ab2c68bfc20f5f831447a0cc99e9528a"
MAX_SECONDS = 90
TOTAL_SECONDS = 240
MAX_OUTPUT = 20 * 1024**2
MAX_LOG = 1024**2
MIN_LAUNCH_KIB = 4 * 1024**2


def four_points(points):
    expected = [[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, 0]]
    if points != expected or any(type(v) is not int for p in points for v in p):
        raise ValueError("frozen four original-pixel point prompts differ")
    if any(not (smoke.BOX[0] <= x < smoke.BOX[2] and smoke.BOX[1] <= y < smoke.BOX[3])
           for x, y, _ in points):
        raise ValueError("point outside frozen box")
    return (np.asarray([[x, y] for x, y, _ in points], dtype=np.float32),
            np.asarray([label for _, _, label in points], dtype=np.int32))


def membership(mask, points):
    xy, labels = four_points(points)
    if mask.shape != (1024, 1280) or mask.dtype != np.uint8:
        raise ValueError("membership needs original-size uint8 mask")
    found = [int(mask[int(y), int(x)] != 0) for x, y in xy]
    if found != labels.tolist():
        raise ValueError(f"four-point foreground/background mismatch: {found}")
    return found


def sam_predictor_four(source, checkpoint):
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
        xy, labels = four_points(points)
        with torch.inference_mode():
            predictor.set_image(np.array(image, copy=True))
            masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32),
                                            point_coords=xy, point_labels=labels,
                                            multimask_output=False)
        return masks
    return predict


def source_seal(base_root, inventory_path, rows, package_sha):
    base_root, inventory_path = Path(base_root), Path(inventory_path)
    if inventory_path.resolve() != base_root.resolve() / "complete_inventory.json":
        raise ValueError("inventory must be the base output's complete_inventory.json")
    inventory, _ = resume.bound_metadata(inventory_path, INVENTORY_SHA)
    qa, _ = resume.bound_metadata(QA_PATH, QA_SHA, 64 * 1024)
    if (inventory.get("schema") != "sam21_mustard_point_mask_inventory_v1" or
            inventory.get("status") != "generated_unreviewed" or
            inventory.get("package_sha256") != package_sha or
            inventory.get("prompt_sha256") != full.PROMPTS_SHA or
            qa.get("schema") != "mustard_sam_mask_qa_v1" or qa.get("status") != "rejected" or
            qa.get("decision") != "rejected_checkerboard_leakage" or
            qa.get("inventory_sha256") != INVENTORY_SHA or
            qa.get("rejected_names") != list(NAMES)):
        raise ValueError("base inventory/rejection QA lineage differs")
    images = inventory.get("images")
    expected = [Path(row["path"]).name for row in rows]
    if not isinstance(images, list) or [item.get("name") for item in images] != expected:
        raise ValueError("base inventory missing/extra/reordered training photos")
    if set(qa.get("cleaned_mask_sha256", {})) != set(expected):
        raise ValueError("QA cleaned mask map incomplete")
    frames = base_root / "frames"
    if frames.is_symlink() or not frames.is_dir() or {p.name for p in frames.iterdir()} != set(expected):
        raise ValueError("base frames contain missing/extra/linked names")
    base_hashes = {}
    for item, row in zip(images, rows):
        name = item["name"]
        path = base_root / "frames" / name / "clean.png"
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise ValueError(f"base frame directory linked or missing: {name}")
        if (item.get("cleaned_mask_path") != f"frames/{name}/clean.png" or
                item.get("source_sha256") != row["sha256"] or
                item.get("cleaned_mask_sha256") != qa["cleaned_mask_sha256"][name] or
                path.is_symlink() or not path.is_file() or common.digest(path) != item["cleaned_mask_sha256"]):
            raise ValueError(f"base cleaned mask differs from sealed QA: {name}")
        base_hashes[name] = item["cleaned_mask_sha256"]
    return base_hashes


def validated_inputs(package, stage_root, stage_report, original_prompts, checkpoint,
                     source, base_root, inventory_path, prompt_path):
    mac.external_inputs((package, stage_root, stage_report, original_prompts, checkpoint,
                         source, base_root, inventory_path))
    rows, _, package_sha = full.validated_inputs(package, checkpoint, source, original_prompts)
    stage_sha = resume.staged_train48(stage_root, rows, package_sha, stage_report)
    base_hashes = source_seal(base_root, inventory_path, rows, package_sha)
    prompts, prompt_sha = resume.bound_metadata(prompt_path, PROMPT_SHA, 16 * 1024)
    if (prompts.get("schema") != "sam21_mustard_m1_board_negative5_prompts_v1" or
            prompts.get("box_xyxy_original_pixels") != list(smoke.BOX) or
            prompts.get("original_prompt_sha256") != full.PROMPTS_SHA or
            prompts.get("base_inventory_sha256") != INVENTORY_SHA or
            prompts.get("rejected_full_qa_sha256") != QA_SHA):
        raise ValueError("correction prompt lineage differs")
    selected = [row for row in rows if Path(row["path"]).name in NAMES]
    entries = prompts.get("images")
    if (tuple(Path(row["path"]).name for row in selected) != NAMES or
            not isinstance(entries, list) or [entry.get("name") for entry in entries] != list(NAMES)):
        raise ValueError("correction prompts must contain exactly the five rejected TRAIN names")
    for row, entry in zip(selected, entries):
        if entry.get("source_sha256") != row["sha256"]:
            raise ValueError("correction prompt photo SHA differs")
        four_points(entry.get("points_xy_label"))
        common.verified_training_photo(stage_root, row)
    return selected, entries, package_sha, stage_sha, prompt_sha, base_hashes


def dependency_hashes():
    return {"runner_sha256": common.digest(__file__),
            "mac_helper_sha256": common.digest(mac.__file__),
            "resume_helper_sha256": common.digest(resume.__file__),
            "full_helper_sha256": common.digest(full.__file__),
            "point_helper_sha256": common.digest(point.__file__),
            "smoke_helper_sha256": common.digest(smoke.__file__),
            "cleanup_helper_sha256": common.digest(cleanup.__file__),
            "shared_helper_sha256": common.digest(common.__file__)}


def output_bytes(path):
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("linked output path")
        if item.is_file():
            total += item.stat().st_size
            if total > MAX_OUTPUT:
                return total
    return total


def worker(args, predictor=None):
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    mac.host_preflight(output)
    selected, prompts, package_sha, stage_sha, prompt_sha, base_hashes = validated_inputs(
        args.package, args.stage_root, args.stage_report, args.original_prompts,
        args.checkpoint, args.sam_source, args.base_root, args.base_inventory, args.correction_prompts)
    deps = dependency_hashes()
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    report = {"schema": "sam21_mustard_m1_board5_candidate_v1", "status": "running",
              "mask_role": "unreviewed coarse pose support; not exact silhouette",
              "names": list(NAMES), "package_sha256": package_sha,
              "stage_report_sha256": stage_sha,
              "original_prompt_sha256": full.PROMPTS_SHA,
              "correction_prompt_sha256": prompt_sha,
              "base_inventory_sha256": INVENTORY_SHA, "rejected_full_qa_sha256": QA_SHA,
              "checkpoint_sha256": point.MODEL_SHA,
              "source_inventory_sha256": point.SOURCE_SHA,
              "base_unchanged_43_sha256": {k: v for k, v in base_hashes.items() if k not in NAMES},
              **deps, "images": []}
    started = time.monotonic()
    try:
        predict = predictor or sam_predictor_four(args.sam_source, args.checkpoint)
        for row, prompt in zip(selected, prompts):
            if (time.monotonic() - started > MAX_SECONDS or mac.available_kib() < mac.MIN_AVAILABLE_KIB or
                    mac.rss_kib(os.getpid()) > mac.MAX_RSS_KIB or
                    mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES):
                raise RuntimeError("worker resource gate between views")
            name = Path(row["path"]).name
            entry = {"name": name, "source_sha256": row["sha256"],
                     "points_xy_label": prompt["points_xy_label"]}
            report["images"].append(entry)
            try:
                photo = common.verified_training_photo(args.stage_root, row)
                raw = point.binary_raw(predict(photo, smoke.BOX, entry["points_xy_label"]))
                raw_path = output / "raw_masks" / f"{name}.png"
                Image.fromarray(raw, "L").save(raw_path, "PNG")
                entry["raw_mask_path"] = f"raw_masks/{name}.png"
                entry["raw_mask_sha256"] = common.digest(raw_path)
                membership(raw, entry["points_xy_label"])
                smoke.bounded_mask(raw)
                clean, stats = cleanup.largest_component(raw)
                membership(clean, entry["points_xy_label"])
                clean_path = output / "cleaned_masks" / f"{name}.png"
                Image.fromarray(clean, "L").save(clean_path, "PNG")
                entry.update({"cleaned_mask_path": f"cleaned_masks/{name}.png",
                              "cleaned_mask_sha256": common.digest(clean_path),
                              "support_pixels": stats["cleaned_pixels"], "cleanup": stats,
                              "status": "complete_unreviewed"})
            except ValueError as error:
                entry.update({"status": "failed_unusable", "failure": repr(error)})
            if output_bytes(output) > MAX_OUTPUT - 1024**2:
                raise ValueError("candidate output cap reached")
        # A small crop review sheet is evidence only; no mask is QA-accepted here.
        sheet = Image.new("RGB", (5 * 280, 3 * 370), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (row, entry) in enumerate(zip(selected, report["images"])):
            photo = common.verified_training_photo(args.stage_root, row)
            for lane, kind in enumerate(("rgb", "raw", "cleaned")):
                if kind == "rgb":
                    tile = photo.crop(smoke.BOX)
                else:
                    path_key = "raw_mask_path" if kind == "raw" else "cleaned_mask_path"
                    if path_key in entry:
                        with Image.open(output / entry[path_key]) as mask:
                            masked = Image.new("RGB", photo.size, "black")
                            masked.paste(photo, (0, 0), mask)
                        tile = masked.crop(smoke.BOX)
                    else:
                        tile = Image.new("RGB", (280, 350), "black")
                sheet.paste(tile, (index * 280, lane * 370))
                draw.text((index * 280 + 4, lane * 370 + 352),
                          f"{entry['name']} {kind}", fill="black")
        sheet_path = output / "review_sheet.jpg"
        sheet.save(sheet_path, "JPEG", quality=88)
        report["review_sheet_sha256"] = common.digest(sheet_path)
        if (common.digest(args.package) != package_sha or
                common.digest(args.correction_prompts) != prompt_sha or
                common.digest(args.original_prompts) != full.PROMPTS_SHA or
                common.digest(args.stage_report) != stage_sha or
                common.digest(args.checkpoint) != point.MODEL_SHA or
                smoke.source_digest(args.sam_source) != point.SOURCE_SHA or
                dependency_hashes() != deps):
            raise ValueError("model, package, prompts or code changed during trial")
        source_seal(args.base_root, args.base_inventory,
                    common.training_records(common.load_package(args.package)), package_sha)
        if output_bytes(output) > MAX_OUTPUT:
            raise ValueError("candidate output exceeded cap")
        report["status"] = ("generated_unreviewed" if all(
            row["status"] == "complete_unreviewed" for row in report["images"])
            else "partial_unusable")
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("manifest above 1 MiB")
        with (output / "manifest.json").open("xb") as file:
            file.write(payload)
    return report


def terminal_guard(seconds, total_seconds, rss, available, output_size, log_size, free, code):
    if seconds > MAX_SECONDS or total_seconds > TOTAL_SECONDS:
        return "90-second worker or 240-second total cap exceeded"
    if rss > mac.MAX_RSS_KIB:
        return "worker RSS above 1.5 GiB"
    if available < mac.MIN_AVAILABLE_KIB:
        return "Mac available estimate below 2.5 GiB"
    if output_size > MAX_OUTPUT or log_size > MAX_LOG:
        return "candidate output/log cap exceeded"
    if free < common.MIN_FREE_BYTES:
        return "internal/external free space below 10 GiB"
    if code != 0:
        return f"worker exit {code}"
    return None


def supervise(args):
    output = args.output
    sidecar = output.parent / f"{output.name}.supervisor.json"
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("fresh correction output and failure sidecar required")
    mac.host_preflight(output, check_memory=False)
    if mac.available_kib() < MIN_LAUNCH_KIB:
        raise RuntimeError("launch requires 4 GiB conservative Mac available estimate")
    mac.external_inputs((args.package, args.stage_root, args.stage_report, args.original_prompts,
                         args.checkpoint, args.sam_source, args.base_root, args.base_inventory))
    start = time.monotonic()
    # Complete input/QA verification before any inference child is spawned.
    validated_inputs(args.package, args.stage_root, args.stage_report, args.original_prompts,
                     args.checkpoint, args.sam_source, args.base_root, args.base_inventory,
                     args.correction_prompts)
    if time.monotonic() - start > TOTAL_SECONDS:
        raise RuntimeError("240-second total deadline reached before worker launch")
    if mac.available_kib() < MIN_LAUNCH_KIB:
        raise RuntimeError("prelaunch recheck below 4 GiB conservative available estimate")
    if mac.disk_free_both(output.parent) < common.MIN_FREE_BYTES:
        raise RuntimeError("prelaunch internal/external disk below 10 GiB")
    payload = [sys.executable, "-m", "scripts.object_motion.sam_m1_board5", "--worker"]
    for key in ("package", "stage_root", "stage_report", "original_prompts", "checkpoint",
                "sam_source", "base_root", "base_inventory", "correction_prompts", "output"):
        payload.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryFile(dir=output.parent) as logfile:
        child = subprocess.Popen(payload, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        worker_started, peak, reason = time.monotonic(), 0, None
        try:
            while child.poll() is None:
                peak = max(peak, mac.rss_kib(child.pid))
                reason = terminal_guard(time.monotonic() - worker_started, time.monotonic() - start,
                                        peak, mac.available_kib(), output_bytes(output),
                                        logfile.seek(0, os.SEEK_END), mac.disk_free_both(output.parent), 0)
                if reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            child.wait()
            reason = reason or terminal_guard(time.monotonic() - worker_started, time.monotonic() - start,
                                              peak, mac.available_kib(), output_bytes(output),
                                              logfile.seek(0, os.SEEK_END), mac.disk_free_both(output.parent),
                                              child.returncode)
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    manifest = output / "manifest.json"
    status = None
    if reason is None:
        if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1024**2:
            reason = "worker lacks valid manifest"
        else:
            try:
                record = json.loads(manifest.read_text())
                status = record.get("status")
                if (record.get("schema") != "sam21_mustard_m1_board5_candidate_v1" or
                        status not in ("generated_unreviewed", "partial_unusable")):
                    reason = "worker manifest invalid/incomplete"
            except (OSError, ValueError, AttributeError):
                reason = "worker manifest unreadable"
    result = {"schema": "sam21_mustard_m1_board5_supervisor_v1",
              "status": status if reason is None else "failed", "reason": reason,
              "log_excerpt": excerpt,
              "seconds_total": time.monotonic() - start,
              "seconds_worker": time.monotonic() - worker_started,
              "peak_worker_rss_kib": peak}
    target = output / "supervisor.json" if output.is_dir() else sidecar
    with target.open("xb") as file:
        file.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "stage-root", "stage-report", "original-prompts", "checkpoint",
                "sam-source", "base-root", "base-inventory", "correction-prompts", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = worker(args) if args.worker else supervise(args)
    if result["status"] != "generated_unreviewed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
