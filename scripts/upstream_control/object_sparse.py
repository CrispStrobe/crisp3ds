#!/usr/bin/env python3
"""Bounded stock PyCOLMAP sparse run on 60 original YCB cracker-box NP3 JPEGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import time

from scripts.classical_backend.run import RESERVE, StageError, digest, folder_bytes, stage
from scripts.upstream_control import image_only as stock

ROOT = stock.ROOT
SOURCE = ROOT / ".local-tools/test-data/ycb-cracker-box"
MASK_MANIFEST = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
PYTHON = stock.PYTHON
PHOTO_MANIFEST_SHA256 = "d3d1945828a34413e2205ff7617850a36a99b4db462585087a8cecade5de9c64"
MASK_MANIFEST_SHA256 = "0b78469039d11516d0ac96b642553099267cd97c9a6439a1ec4b2fcb133615bd"
NAMES = tuple(f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6))
STRUCTURAL_KEYS = ("duplicate_exact_observations", "points_with_fewer_than_two_distinct_views",
                   "invalid_track_links", "orphan_point2d_links", "nonfinite_observations",
                   "nonfinite_points", "nonfinite_cameras", "nonfinite_poses")


def regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular input: {path}")


def inventory(source: Path, arm: str, mask_manifest: Path) -> tuple[list[dict], list[dict], dict]:
    if arm not in ("raw", "masked"):
        raise ValueError("arm must be raw or masked")
    manifest_path = source / "manifest.json"
    regular(manifest_path)
    if digest(manifest_path) != PHOTO_MANIFEST_SHA256:
        raise ValueError("photo manifest differs from frozen SHA-256")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("photos", [])
    if (manifest.get("object_id") != "003_cracker_box" or len(rows) != 60 or
            [r.get("path") for r in rows] != [f"photos/{name}" for name in NAMES] or
            [r.get("turntable_angle_degrees") for r in rows] != list(range(0, 360, 6))):
        raise ValueError("photo manifest does not select exactly the 60 NP3 views")
    if (source / "photos").is_symlink():
        raise ValueError("photo directory is a symlink")
    for row in rows:
        path = source / row["path"]
        regular(path)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"photo differs from manifest: {path.name}")
    masks = []
    meta = {"photo_manifest_sha256": PHOTO_MANIFEST_SHA256, "mask_manifest_sha256": None}
    if arm == "masked":
        regular(mask_manifest)
        if digest(mask_manifest) != MASK_MANIFEST_SHA256:
            raise ValueError("mask manifest differs from frozen SHA-256")
        prepared = json.loads(mask_manifest.read_text())
        masks = prepared.get("images", [])
        if (prepared.get("schema") != "object_motion_prepare_v1" or
                prepared.get("source_manifest_sha256") != PHOTO_MANIFEST_SHA256 or
                len(masks) != 60 or [r.get("name") for r in masks] != list(NAMES)):
            raise ValueError("mask manifest inventory differs from 60 original photos")
        mask_dir = mask_manifest.parent / "masks"
        if mask_dir.is_symlink():
            raise ValueError("mask directory is a symlink")
        for row, photo in zip(masks, rows):
            path = mask_dir / f"{row['name']}.png"
            if (Path(row["pose_support_mask"]) != path or row["sha256"] != photo["sha256"] or
                    Path(row["path"]) != source / photo["path"]):
                raise ValueError(f"mask manifest binding differs: {row['name']}")
            regular(path)
            if digest(path) != row["mask_sha256"]:
                raise ValueError(f"mask differs from manifest: {row['name']}")
        meta["mask_manifest_sha256"] = MASK_MANIFEST_SHA256
    return rows, masks, meta


def run(source: Path, output: Path, arm: str, mask_manifest: Path = MASK_MANIFEST,
        python: Path = PYTHON) -> dict:
    source, output, mask_manifest, python = map(Path, (source, output, mask_manifest, python))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output must be fresh: {output}")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    if not python.is_file():
        raise ValueError(f"pinned PyCOLMAP interpreter unavailable: {python}")
    rows, masks, meta = inventory(source, arm, mask_manifest)
    input_bytes = sum(r["bytes"] for r in rows)
    if masks:
        input_bytes += sum((mask_manifest.parent / "masks" / f"{r['name']}.png").stat().st_size for r in masks)
    if input_bytes >= stock.MAX_BYTES:
        raise ValueError("inputs exceed output budget")
    if (shutil.disk_usage(output.parent).free < RESERVE + stock.MAX_BYTES or
            shutil.disk_usage(ROOT).free < RESERVE):
        raise ValueError("10 GiB volume reserve or output budget unavailable")
    software = stock.toolchain(python)
    software["object_runner_sha256"] = digest(Path(__file__))
    output = output.absolute()
    output.mkdir()
    (output / "images").mkdir()
    if masks:
        (output / "masks").mkdir()
    deadline = time.monotonic() + stock.MAX_SECONDS
    report = {"schema": "ycb_object_stock_pycolmap_sparse_v1", "status": "running", "arm": arm,
              "source": "original Berkeley NP3 JPEGs; no supplied calibration, poses, depth, or scanner geometry",
              "camera_policy": "one shared SIMPLE_RADIAL camera; native self-calibration",
              "mask_role": ("coarse photo-derived feature support; four sample angles received human QA, not all 60"
                            if masks else None), **meta, "software": software,
              "platform": platform.platform(), "machine": platform.machine(),
              "limits": {"max_output_bytes": stock.MAX_BYTES, "max_child_rss_bytes": stock.MAX_RSS,
                         "max_log_bytes_per_stage": stock.MAX_LOG, "max_seconds_total": stock.MAX_SECONDS,
                         "min_free_bytes_each_volume": RESERVE},
              "inputs": [], "masks": [], "stages": [], "models": [], "quality_accepted": False,
              "independent_camera_review_complete": False}

    def save() -> None:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for row in rows:
            name = Path(row["path"]).name
            source_file = source / row["path"]
            target = output / "images" / name
            if (shutil.disk_usage(output).free < RESERVE + row["bytes"] or
                    shutil.disk_usage(ROOT).free < RESERVE or
                    folder_bytes(output) + row["bytes"] > stock.MAX_BYTES):
                raise ValueError("photo copy exceeds reserve or output cap")
            shutil.copyfile(source_file, target)
            if digest(target) != row["sha256"]:
                raise ValueError(f"copied photo hash differs: {name}")
            report["inputs"].append({"name": name, "sha256": row["sha256"], "bytes": row["bytes"]})
        for row in masks:
            name = f"{row['name']}.png"
            source_file = mask_manifest.parent / "masks" / name
            target = output / "masks" / name
            size = source_file.stat().st_size
            if (shutil.disk_usage(output).free < RESERVE + size or
                    shutil.disk_usage(ROOT).free < RESERVE or
                    folder_bytes(output) + size > stock.MAX_BYTES):
                raise ValueError("mask copy exceeds reserve or output cap")
            shutil.copyfile(source_file, target)
            if digest(target) != row["mask_sha256"]:
                raise ValueError(f"copied mask hash differs: {name}")
            report["masks"].append({"name": name, "sha256": row["mask_sha256"], "bytes": size})
        (output / "worker-config.json").write_text(json.dumps({
            "image_names": list(NAMES), "camera_mode": "SINGLE", "use_masks": bool(masks)}) + "\n")
        report["worker_config_sha256"] = digest(output / "worker-config.json")
        save()
        for kind in ("features", "matching", "mapping"):
            command = [str(python), "-m", "scripts.upstream_control.object_sparse",
                       "--worker", kind, "--output", str(output)]
            try:
                result = stage(output, kind, command, deadline, stock.MAX_BYTES, stock.MAX_LOG,
                               stock.MAX_RSS, extra_reserve_paths=(ROOT,))
            except StageError as error:
                report["stages"].append(error.result)
                raise
            report["stages"].append(result)
            if kind == "features":
                report["effective_options"] = json.loads((output / "effective-options.json").read_text())
            if kind in ("features", "matching"):
                report[f"database_after_{kind}"] = {"sha256": digest(output / "database.db"),
                                                      "counts": stock.database_counts(output / "database.db")}
            save()
        report["models"] = json.loads((output / "models.json").read_text())
        report["database_after_mapping"] = {"sha256": digest(output / "database.db"),
                                            "counts": stock.database_counts(output / "database.db")}
        best = max(report["models"], key=lambda m: (m["registered_images"], m["sparse_points"], -m["index"]),
                   default=None)
        report["best_model_index"] = best["index"] if best else None
        report["structural_sparse_eligible_pending_camera_review"] = bool(
            best and best["registered_images"] >= 54 and best["sparse_points"] > 0 and
            all(best[key] == 0 for key in STRUCTURAL_KEYS))
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", failure=str(error),
                      structural_sparse_eligible_pending_camera_review=False)
    finally:
        report["output_bytes"] = folder_bytes(output)
        report["source_images_unchanged"] = all(
            stock.same_digest(source / "photos" / item["name"], item["sha256"]) for item in report["inputs"])
        report["photo_manifest_unchanged"] = stock.same_digest(
            source / "manifest.json", PHOTO_MANIFEST_SHA256)
        report["copied_images_unchanged"] = all(
            stock.same_digest(output / "images" / item["name"], item["sha256"]) for item in report["inputs"])
        report["source_masks_unchanged"] = all(
            stock.same_digest(mask_manifest.parent / "masks" / item["name"], item["sha256"])
            for item in report["masks"])
        report["mask_manifest_unchanged"] = (
            not masks or stock.same_digest(mask_manifest, MASK_MANIFEST_SHA256))
        report["worker_config_unchanged"] = (
            "worker_config_sha256" not in report or stock.same_digest(
                output / "worker-config.json", report["worker_config_sha256"]))
        report["copied_masks_unchanged"] = all(
            stock.same_digest(output / "masks" / item["name"], item["sha256"])
            for item in report["masks"])
        report["free_bytes_after"] = {"output": shutil.disk_usage(output).free,
                                      "workspace": shutil.disk_usage(ROOT).free}
        try:
            current = stock.toolchain(python)
            current["object_runner_sha256"] = digest(Path(__file__))
            report["software_unchanged"] = current == software
        except Exception:
            report["software_unchanged"] = False
        report["partial_model_files_sha256"] = stock.partial_model_hashes(output)
        report["elapsed_seconds_total"] = round(stock.MAX_SECONDS - (deadline - time.monotonic()), 3)
        violations = []
        if not all(report[key] for key in ("source_images_unchanged", "copied_images_unchanged",
                                            "source_masks_unchanged", "copied_masks_unchanged",
                                            "photo_manifest_unchanged", "mask_manifest_unchanged",
                                            "worker_config_unchanged")):
            violations.append("input hash changed")
        if not report["software_unchanged"]:
            violations.append("software changed")
        if report["output_bytes"] > stock.MAX_BYTES:
            violations.append("output byte limit exceeded")
        if any(free < RESERVE for free in report["free_bytes_after"].values()):
            violations.append("10 GiB disk reserve reached")
        if time.monotonic() > deadline:
            violations.append("total deadline exceeded")
        if violations:
            report.update(status="failed", failure="; ".join(violations),
                          structural_sparse_eligible_pending_camera_review=False)
        save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw", "masked"), default="raw")
    parser.add_argument("--mask-manifest", type=Path, default=MASK_MANIFEST)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--worker", choices=("features", "matching", "mapping"))
    args = parser.parse_args()
    if args.worker:
        stock.worker(args.worker, args.output)
        return 0
    report = run(args.source, args.output, args.arm, args.mask_manifest, args.python)
    print(json.dumps({"status": report["status"], "arm": args.arm,
                      "structural_sparse_eligible_pending_camera_review": report.get(
                          "structural_sparse_eligible_pending_camera_review", False),
                      "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
