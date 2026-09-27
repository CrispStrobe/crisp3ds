#!/usr/bin/env python3
"""Frozen three-training-view SAM2 box+point prompt candidate, never QA approval."""

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
from scripts.object_motion import sam_mask_trial as prior
from scripts.object_motion import ycb_object_masks as common


NAMES = ("NP3_000.jpg", "NP3_066.jpg", "NP3_180.jpg")
PROMPT_SHA = "d4c8c913e831f1b78e21143d789060c0626f589baeb3f180ef2ec0526ab0c4ca"
MODEL_SHA = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SOURCE_SHA = "c7eb4585a22dadd4f54ffd9134e3103c1951745a3b4631ceb0684b55742069e3"
MAX_SECONDS = 120
MAX_OUTPUT = 20 * 1024**2
MAX_LOG = 1024**2


def validated_prompts(path, package_rows):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_384 or common.digest(path) != PROMPT_SHA:
        raise ValueError("prompt file differs from approved bounded manifest")
    document = json.loads(path.read_text())
    if (document.get("schema") != "sam21_mustard_three_point_prompts_v1" or
            document.get("box_xyxy_original_pixels") != list(prior.BOX)):
        raise ValueError("prompt schema or frozen box differs")
    prompts = document.get("images")
    if not isinstance(prompts, list) or [item.get("name") for item in prompts] != list(NAMES):
        raise ValueError("prompt images must be exactly frozen three training names")
    lookup = {Path(row["path"]).name: row for row in package_rows}
    for item in prompts:
        name = item["name"]
        if lookup[name]["sha256"] != item.get("source_sha256"):
            raise ValueError("prompt source photo SHA differs from frozen package")
        point_arrays(item["points_xy_label"])
    return prompts


def point_arrays(points):
    if not isinstance(points, list) or len(points) != 3:
        raise ValueError("exactly one positive and two negatives required")
    seen = set()
    for point in points:
        if (not isinstance(point, list) or len(point) != 3 or
                any(not isinstance(v, int) or isinstance(v, bool) for v in point)):
            raise ValueError("point must be integer [x,y,label]")
        x, y, label = point
        if not prior.BOX[0] <= x < prior.BOX[2] or not prior.BOX[1] <= y < prior.BOX[3]:
            raise ValueError("point outside frozen box in original image coordinates")
        if label not in (0, 1) or (x, y) in seen:
            raise ValueError("point label or duplicate location invalid")
        seen.add((x, y))
    if [point[2] for point in points] != [1, 0, 0]:
        raise ValueError("points must be ordered positive, negative, negative")
    xy = np.asarray([[p[0], p[1]] for p in points], dtype=np.float32)
    labels = np.asarray([p[2] for p in points], dtype=np.int32)
    return xy, labels


def binary_raw(predicted):
    array = np.asarray(predicted)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (1024, 1280) or array.dtype not in (np.dtype(bool), np.dtype("uint8"), np.dtype("float32")):
        raise ValueError("predictor must return one full-resolution binary mask")
    allowed = (0, 1, 255) if array.dtype == np.uint8 else (0, 1)
    if not np.isin(array, allowed).all():
        raise ValueError("predictor returned nonbinary/nonfinite values")
    return np.where(array != 0, 255, 0).astype(np.uint8)


def point_membership(mask, points):
    if mask.shape != (1024, 1280) or mask.dtype != np.uint8:
        raise ValueError("membership requires full-resolution uint8 mask")
    xy, labels = point_arrays(points)
    observed = [int(mask[int(y), int(x)] != 0) for x, y in xy]
    if observed != labels.tolist():
        raise ValueError(f"mask violates frozen point labels: observed={observed}")
    return observed


def sam_predictor(source, checkpoint):
    source = Path(source)
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
        xy, labels = point_arrays(points)
        with torch.inference_mode():
            predictor.set_image(np.array(image, copy=True))
            masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32),
                                            point_coords=xy, point_labels=labels,
                                            multimask_output=False)
        return masks
    return predict


def candidate(package_path, dataset_root, output, checkpoint, source, prompt_path, predictor=None):
    package_path, output, checkpoint, source, prompt_path = map(
        Path, (package_path, output, checkpoint, source, prompt_path))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    package_sha = common.digest(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != prior.OBJECT_ID:
        raise ValueError("mustard training package required")
    rows = common.training_records(package)
    prompts = validated_prompts(prompt_path, rows)
    prompt_lookup = {item["name"]: item for item in prompts}
    selected = [row for row in rows if Path(row["path"]).name in NAMES]
    if tuple(Path(row["path"]).name for row in selected) != NAMES:
        raise ValueError("exact three names absent from training split")
    prior.checkpoint_metadata(checkpoint, MODEL_SHA)
    if prior.source_digest(source) != SOURCE_SHA:
        raise ValueError("SAM2 source differs from reviewed inventory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output filesystem below 10 GiB free")
    dependency_shas = {"runner_sha256": common.digest(__file__),
                       "prior_helper_sha256": common.digest(prior.__file__),
                       "cleanup_helper_sha256": common.digest(cleanup.__file__),
                       "shared_helper_sha256": common.digest(common.__file__)}
    output.mkdir()
    (output / "raw_masks").mkdir()
    (output / "cleaned_masks").mkdir()
    report = {"schema": "sam21_tiny_mustard_point_smoke_v1", "status": "running",
              "mask_status": "generated_unreviewed", "source_scope": "three TRAIN RGB photos only; no held-out/GT/pose/depth",
              "box_xyxy_original_pixels": list(prior.BOX), "prompt_sha256": PROMPT_SHA,
              "package_sha256": package_sha, "checkpoint_sha256": MODEL_SHA,
              "sam2_source_inventory_sha256": SOURCE_SHA, **dependency_shas,
              "images": []}
    used = 0
    try:
        predict = predictor or sam_predictor(source, checkpoint)
        sheet = Image.new("RGB", (3 * 320, 3 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            name = Path(row["path"]).name
            points = prompt_lookup[name]["points_xy_label"]
            image = common.verified_training_photo(dataset_root, row)
            entry = {"name": name, "source_sha256": row["sha256"],
                     "points_xy_label": points}
            start = time.monotonic()
            try:
                raw = binary_raw(predict(image, prior.BOX, points))
                raw_path = output / "raw_masks" / f"{name}.png"
                used = common.save_bounded_image(Image.fromarray(raw, "L"), raw_path, "PNG", used)
                entry["raw_mask_sha256"] = common.digest(raw_path)
                entry["raw_pixels"] = int(np.count_nonzero(raw))
                entry["point_membership"] = point_membership(raw, points)
                prior.bounded_mask(raw)  # preserve old area and image-boundary guards
                clean, stats = cleanup.largest_component(raw)
                point_membership(clean, points)
                clean_path = output / "cleaned_masks" / f"{name}.png"
                used = common.save_bounded_image(Image.fromarray(clean, "L"), clean_path, "PNG", used)
                entry.update({"status": "complete_unreviewed", "cleaned_mask_sha256": common.digest(clean_path),
                              **stats})
            except ValueError as error:
                entry.update({"status": "failed_unusable", "failure": repr(error)})
            entry["seconds"] = time.monotonic() - start
            report["images"].append(entry)
            for lane, pixels in enumerate((None, raw if "raw_mask_sha256" in entry else None,
                                            clean if "cleaned_mask_sha256" in entry else None)):
                if lane == 0:
                    display = image.crop(prior.BOX)
                elif pixels is None:
                    display = Image.new("RGB", (prior.BOX[2] - prior.BOX[0], prior.BOX[3] - prior.BOX[1]), "black")
                    ImageDraw.Draw(display).text((8, 8), "FAILED / MISSING", fill="red")
                else:
                    masked = Image.new("RGB", image.size, "black")
                    masked.paste(image, (0, 0), Image.fromarray(pixels, "L"))
                    display = masked.crop(prior.BOX)
                display.thumbnail((310, 245), Image.Resampling.LANCZOS)
                sheet.paste(display, (index * 320 + (320 - display.width) // 2, lane * 280))
                draw.text((index * 320 + 8, lane * 280 + 253), name + (" RGB" if lane == 0 else " raw" if lane == 1 else " clean"), fill="black")
            if used > MAX_OUTPUT - 1024**2:
                raise ValueError("output cap exceeded")
        sheet_path = output / "point_review_sheet.jpg"
        used = common.save_bounded_image(sheet, sheet_path, "JPEG", used, quality=90)
        report["review_sheet_sha256"] = common.digest(sheet_path)
        if (common.digest(package_path) != package_sha or common.digest(prompt_path) != PROMPT_SHA or
                common.digest(checkpoint) != MODEL_SHA or prior.source_digest(source) != SOURCE_SHA or
                any(common.digest(path) != digest for path, digest in
                    ((Path(__file__), dependency_shas["runner_sha256"]),
                     (Path(prior.__file__), dependency_shas["prior_helper_sha256"]),
                     (Path(cleanup.__file__), dependency_shas["cleanup_helper_sha256"]),
                     (Path(common.__file__), dependency_shas["shared_helper_sha256"])))):
            raise ValueError("input/model/runner changed during point trial")
        report["status"] = "complete_unreviewed" if all(item["status"] == "complete_unreviewed" for item in report["images"]) else "partial_unusable"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("manifest exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def supervise(arguments, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES or prior.available_kib() < prior.MIN_AVAILABLE_KIB:
        raise RuntimeError("disk below 10 GiB or RAM below 2.5 GiB before inference")
    command = [sys.executable, "-m", "scripts.object_motion.sam_point_trial", "--worker", *arguments]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryFile(dir=output.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        start = time.monotonic()
        peak = 0
        reason = None
        try:
            while child.poll() is None:
                peak = max(peak, prior.rss_kib(child.pid))
                if peak > prior.MAX_RSS_KIB:
                    reason = "worker RSS exceeded 1.5 GiB"
                elif time.monotonic() - start > MAX_SECONDS:
                    reason = "point trial exceeded 120 seconds"
                elif logfile.seek(0, os.SEEK_END) > MAX_LOG:
                    reason = "log exceeded 1 MiB"
                elif output.exists() and sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_OUTPUT:
                    reason = "output exceeded 20 MiB"
                elif shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
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
            logfile.seek(0)
            excerpt = logfile.read(MAX_LOG).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
    worker_status = None
    manifest_path = output / "manifest.json"
    if child.returncode == 0 and manifest_path.is_file():
        worker_status = json.loads(manifest_path.read_text()).get("status")
    result = {"status": worker_status if child.returncode == 0 and not reason else "failed",
              "returncode": child.returncode,
              "reason": reason or (excerpt if child.returncode else
                                   "one or more point-prompt masks failed acceptance" if worker_status != "complete_unreviewed" else None),
              "seconds": time.monotonic() - start, "peak_worker_rss_kib": peak}
    output.mkdir(parents=True, exist_ok=True)
    (output / "supervisor.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "complete_unreviewed":
        raise RuntimeError(result["reason"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        candidate(args.package, args.dataset_root, args.output, args.checkpoint,
                  args.sam_source, args.prompts)
    else:
        parameters = ["--package", str(args.package), "--dataset-root", str(args.dataset_root),
                      "--output", str(args.output), "--checkpoint", str(args.checkpoint),
                      "--sam-source", str(args.sam_source), "--prompts", str(args.prompts)]
        supervise(parameters, args.output)


if __name__ == "__main__":
    main()
