#!/usr/bin/env python3
"""Bounded, train-only SAM 2.1 tiny mustard mask candidate; never QA approval."""

import argparse
import hashlib
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

from scripts.object_motion import ycb_object_masks as common


OBJECT_ID = "006_mustard_bottle"
BOX = (480, 300, 760, 650)
SMOKE_NAMES = ("NP3_000.jpg", "NP3_006.jpg", "NP3_012.jpg")
MAX_OUTPUT = 20 * 1024**2
MAX_RSS_KIB = 1536 * 1024
MAX_SECONDS = 300
MIN_AVAILABLE_KIB = 2560 * 1024
MAX_LOG_BYTES = 1024**2


def sha256(path):
    return common.digest(path)


def source_digest(source):
    """Bind the actual Python/config/license bytes used, not just a git label."""
    source = Path(source)
    files = sorted((source / "sam2").rglob("*.py")) + sorted((source / "sam2" / "configs").rglob("*.yaml"))
    files += [source / "LICENSE"]
    if not 30 <= len(files) <= 500:
        raise ValueError("unexpected SAM2 source file inventory")
    total = 0
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError("SAM2 source contains missing or linked code/config/license")
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        total += size
        if size > 2 * 1024**2 or total > 15 * 1024**2:
            raise ValueError("SAM2 source inventory exceeds byte cap")
        digest.update(relative.encode() + b"\0" + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def checkpoint_metadata(checkpoint, expected_sha):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file() or not 100_000_000 <= checkpoint.stat().st_size <= 180_000_000:
        raise ValueError("checkpoint must be a regular 100-180 MB file")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    actual = sha256(checkpoint)
    if actual != expected_sha:
        raise ValueError("checkpoint differs from reviewed SHA-256")
    return {"checkpoint_sha256": actual, "checkpoint_bytes": checkpoint.stat().st_size}


def bounded_mask(array, size=common.IMAGE_SIZE):
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (size[1], size[0]) or arr.dtype not in (np.dtype(bool), np.dtype("uint8"), np.dtype("float32")):
        raise ValueError("SAM mask must be one full-resolution binary image")
    allowed = (0, 1, 255) if arr.dtype == np.uint8 else (0, 1)
    if not np.isin(arr, allowed).all():
        raise ValueError("SAM mask is not binary")
    result = np.where(arr != 0, 255, 0).astype(np.uint8)
    pixels = int(np.count_nonzero(result))
    if not 1_000 <= pixels <= 150_000:
        raise ValueError(f"implausible support area: {pixels}")
    ys, xs = np.nonzero(result)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if (bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= size[0] or bbox[3] >= size[1]):
        raise ValueError("support touches image boundary")
    return Image.fromarray(result, "L"), {"support_pixels": pixels, "bbox_xyxy_exclusive": bbox}


def available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def predictor_from_checkpoint(source, checkpoint):
    source = Path(source)
    if not (source / "sam2" / "build_sam.py").is_file():
        raise ValueError("not an unpacked SAM2 source tree")
    sys.path.insert(0, str(source))
    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device="cpu", apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)

    def predict(image, box):
        import torch
        with torch.inference_mode():
            predictor.set_image(np.array(image, copy=True))
            masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
        return masks
    return predict


def run_candidate(package_path, dataset_root, output, checkpoint, source,
                  expected_checkpoint_sha, expected_source_sha, predictor=None):
    package_path, dataset_root, output = Path(package_path), Path(dataset_root), Path(output)
    checkpoint, source = Path(checkpoint), Path(source)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    checkpoint_info = checkpoint_metadata(checkpoint, expected_checkpoint_sha)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("SAM2 source must be a real directory")
    source_hash = source_digest(source)
    if source_hash != expected_source_sha:
        raise ValueError("SAM2 source differs from reviewed digest")
    package_hash = sha256(package_path)
    package = common.load_package(package_path)
    if package["object_id"] != OBJECT_ID:
        raise ValueError("mustard package required")
    rows = [r for r in common.training_records(package) if Path(r["path"]).name in SMOKE_NAMES]
    if tuple(Path(r["path"]).name for r in rows) != SMOKE_NAMES:
        raise ValueError("smoke names not present in frozen split")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output volume below 10 GiB free")
    runner_hash = sha256(__file__)
    helper_hash = sha256(common.__file__)
    checkpoint_hash = checkpoint_info["checkpoint_sha256"]
    output.mkdir()
    masks_dir = output / "masks"
    masks_dir.mkdir()
    report = {"schema": "sam21_tiny_mustard_smoke_v1", "status": "running",
              "mask_status": "generated_unreviewed", "mask_role": "candidate coarse pose support, not approved mesh silhouette",
              "source_scope": "three frozen mustard training RGB photos only; no held-out/GT/poses/depth",
              "box_xyxy_original_pixels": list(BOX), "package_sha256": package_hash,
              "runner_sha256": runner_hash, "shared_helper_sha256": helper_hash,
              **checkpoint_info, "sam2_source_sha256": source_hash,
              "images": []}
    used = 0
    try:
        predict = predictor or predictor_from_checkpoint(source, checkpoint)
        sheet = Image.new("RGB", (3 * 320, 2 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            name = Path(row["path"]).name
            image = common.verified_training_photo(dataset_root, row)
            started = time.monotonic()
            mask, stats = bounded_mask(predict(image, BOX))
            mask_path = masks_dir / f"{name}.png"
            used = common.save_bounded_image(mask, mask_path, "PNG", used)
            overlay = image.copy()
            overlay.paste(Image.new("RGB", image.size, (255, 0, 0)), (0, 0), mask.point(lambda v: v // 3))
            crop = overlay.crop(BOX)
            crop.thumbnail((310, 245), Image.Resampling.LANCZOS)
            sheet.paste(crop, (index * 320 + (320 - crop.width) // 2, 0))
            masked = Image.new("RGB", image.size, "black")
            masked.paste(image, (0, 0), mask)
            masked_crop = masked.crop(BOX)
            masked_crop.thumbnail((310, 245), Image.Resampling.LANCZOS)
            sheet.paste(masked_crop, (index * 320 + (320 - masked_crop.width) // 2, 280))
            draw.text((index * 320 + 8, 253), name + " overlay", fill="black")
            draw.text((index * 320 + 8, 533), name + " masked", fill="black")
            report["images"].append({"name": name, "source_sha256": row["sha256"],
                                     "mask_sha256": sha256(mask_path), "inference_seconds": time.monotonic() - started,
                                     **stats})
        sheet_path = output / "smoke_overlay.jpg"
        used = common.save_bounded_image(sheet, sheet_path, "JPEG", used, quality=90)
        report["overlay_sha256"] = sha256(sheet_path)
        if used > MAX_OUTPUT - 1024**2:
            raise ValueError("output cap exceeded")
        if any((sha256(path) != expected) for path, expected in
               ((package_path, package_hash), (__file__, runner_hash), (common.__file__, helper_hash),
                (checkpoint, checkpoint_hash))):
            raise ValueError("source changed during trial")
        if source_digest(source) != source_hash:
            raise ValueError("SAM2 source changed during trial")
        report["status"] = "complete_unreviewed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        raise
    finally:
        payload = (json.dumps(report, indent=2) + "\n").encode()
        if len(payload) > 1024**2:
            raise ValueError("report exceeds 1 MiB")
        (output / "manifest.json").write_bytes(payload)
    return report


def supervise(argv, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
        raise RuntimeError("output volume below 10 GiB free")
    if available_kib() < MIN_AVAILABLE_KIB:
        raise RuntimeError("less than 2.5 GiB RAM available before inference")
    command = [sys.executable, "-m", "scripts.object_motion.sam_mask_trial", "--worker", *argv]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryFile(dir=output.parent) as logfile:
        child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        start = time.monotonic()
        max_rss = 0
        reason = None
        try:
            while child.poll() is None:
                max_rss = max(max_rss, rss_kib(child.pid))
                if max_rss > MAX_RSS_KIB:
                    reason = "RSS above 1.5 GiB"
                    break
                if time.monotonic() - start > MAX_SECONDS:
                    reason = "wall time above 300 seconds"
                    break
                if logfile.seek(0, os.SEEK_END) > MAX_LOG_BYTES:
                    reason = "log above 1 MiB"
                    break
                if output.exists() and sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_OUTPUT:
                    reason = "output above 20 MiB"
                    break
                if shutil.disk_usage(output.parent).free < common.MIN_FREE_BYTES:
                    reason = "output volume below 10 GiB free"
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
            stderr = logfile.read(MAX_LOG_BYTES).decode(errors="replace")[-4096:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid, 9)
                child.wait()
    supervisor = {"status": "failed" if reason or child.returncode else "complete_unreviewed",
                  "failure": reason or (stderr if child.returncode else None),
                  "returncode": child.returncode, "elapsed_seconds": time.monotonic() - start,
                  "peak_worker_rss_kib": max_rss}
    output.mkdir(parents=True, exist_ok=True)
    (output / "supervisor.json").write_text(json.dumps(supervisor, indent=2) + "\n")
    if supervisor["status"] != "complete_unreviewed":
        raise RuntimeError(supervisor["failure"])
    return supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_candidate(args.package, args.dataset_root, args.output, args.checkpoint, args.sam_source,
                      args.checkpoint_sha256, args.source_sha256)
    else:
        parameters = ["--package", str(args.package), "--dataset-root", str(args.dataset_root),
                      "--output", str(args.output), "--checkpoint", str(args.checkpoint),
                      "--sam-source", str(args.sam_source), "--checkpoint-sha256", args.checkpoint_sha256,
                      "--source-sha256", args.source_sha256]
        supervise(parameters, args.output)


if __name__ == "__main__":
    main()
