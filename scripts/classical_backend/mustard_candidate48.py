"""Compose a new mustard mask candidate from 43 sealed base + 5 reviewed corrections.

This creates an unreviewed 48-view candidate, never an accepted SfM input.
The rejected base inventory and its QA decision remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from PIL import Image

from scripts.classical_backend import mustard_stage as base
from scripts.object_motion import sam_point_trial as point

ROOT = Path(__file__).resolve().parents[2]
MOUNT = Path("/Volumes/backups")
BASE_ROOT = MOUNT / "code/crisp3ds-data/sam21-mustard-m1-resume-001"
OUTPUT = MOUNT / "code/crisp3ds-data/sam21-mustard-m1-candidate48-001"
REJECTED_QA = ROOT / "tests/datasets/sam21_mustard_m1_full_review.json"
BASE_SHA = "b47e8caec7d05fae7bb84ae9b3eec7d4a35b1457ca9df67a3b984cbaaeed8a98"
REJECTED_QA_SHA = "4084e8e458b55c566dd138f9d4b8b853ab2c68bfc20f5f831447a0cc99e9528a"
BOARD5_RUNNER_SHA = "57d6248e1c234a4ac1b5d00d2a8645594596e9a68ca104c3cad9ea61269a5c11"
BOARD5_PROMPTS_SHA = "8cc0b6fa49d977194870ef91f10b83a793294d6b3903cb1e08643016bdc9b4f8"
CORRECTED = tuple(f"NP3_{angle:03d}.jpg" for angle in (318, 330, 336, 342, 348))
MAX_BYTES = 20 * 1024**2
MIN_FREE = 10 * 1024**3
BUFFER = 64 * 1024**2
MAX_SECONDS = 180


def digest_object(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mask_info(path: Path, expected_hash: str, size: tuple[int, int]) -> int:
    if path.is_symlink() or not path.is_file() or base.sha256(path) != expected_hash:
        raise ValueError(f"source mask hash/path differs: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "L" or image.size != size:
            raise ValueError(f"mask format/dimensions differ: {path}")
        counts = image.histogram()
        if sum(counts[1:255]) or counts[0] == 0 or counts[255] == 0:
            raise ValueError(f"mask is not binary with both classes: {path}")
        return counts[255]


def safe_relative(root: Path, relative: str, expected_prefix: str, name: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("missing corrected relative mask path")
    rel = Path(relative)
    if rel.is_absolute() or rel.parts != (expected_prefix, name + ".png"):
        raise ValueError("corrected mask path is not exact expected relative path")
    target = root / rel
    if root.is_symlink() or (root / expected_prefix).is_symlink() or target.is_symlink():
        raise ValueError("corrected mask path contains symlink")
    return target


def preflight(base_root: Path, correction_root: Path, correction_manifest_sha: str,
              correction_prompts: Path, correction_prompts_sha: str,
              correction_qa: Path, correction_qa_sha: str, *,
              output: Path = OUTPUT, mount: Path = MOUNT,
              package_path: Path = base.PACKAGE, rejected_qa: Path = REJECTED_QA,
              expected_size: tuple[int, int] = (1280, 1024),
              require_mount: bool = True) -> dict:
    if (base_root.is_symlink() or correction_root.is_symlink() or
            (base_root / "frames").is_symlink() or
            output.exists() or output.is_symlink() or output.parent.is_symlink() or
            not output.parent.is_dir()):
        raise ValueError("candidate output must be a fresh real external directory")
    if require_mount and (not mount.is_mount() or mount.stat().st_dev == ROOT.stat().st_dev or
                          output.parent.stat().st_dev != mount.stat().st_dev or
                          not output.resolve().is_relative_to(mount.resolve())):
        raise ValueError("candidate output is not on a distinct mounted external volume")
    if (shutil.disk_usage(mount).free < MIN_FREE + MAX_BYTES + BUFFER or
            shutil.disk_usage(ROOT).free < MIN_FREE):
        raise ValueError("candidate disk cap or dual-disk floor unavailable")
    package = base.sealed_json(package_path, base.PACKAGE_SHA256)
    training = package.get("training_inputs", [])
    if [row.get("path") for row in training] != ["photos/" + n for n in base.TRAIN_NAMES]:
        raise ValueError("candidate source is not exact 48 TRAIN split")
    photo_hashes = {name: row["sha256"] for name, row in zip(base.TRAIN_NAMES, training)}
    base_inventory = base.sealed_json(base_root / "complete_inventory.json", BASE_SHA)
    frames_root = base_root / "frames"
    if ({item.name for item in frames_root.iterdir()} != set(base.TRAIN_NAMES) or
            any(item.is_symlink() or not item.is_dir() for item in frames_root.iterdir())):
        raise ValueError("base frames are not exactly 48 real directories")
    old_rows = base_inventory.get("images", [])
    if (base_inventory.get("schema") != "sam21_mustard_point_mask_inventory_v1" or
            base_inventory.get("status") != "generated_unreviewed" or
            base_inventory.get("package_sha256") != base.PACKAGE_SHA256 or
            base_inventory.get("prompt_sha256") != base.PROMPTS_SHA256 or
            [row.get("name") for row in old_rows] != list(base.TRAIN_NAMES)):
        raise ValueError("base 48 inventory seal or lineage differs")
    rejection = base.sealed_json(rejected_qa, REJECTED_QA_SHA)
    if (rejection.get("status") != "rejected" or rejection.get("inventory_sha256") != BASE_SHA or
            rejection.get("rejected_names") != list(CORRECTED) or
            rejection.get("reviewed_names") != list(base.TRAIN_NAMES) or
            rejection.get("approved_for_sfm") is not False):
        raise ValueError("base full-set rejection does not bind exact five failed views")
    correction = base.sealed_json(correction_root / "manifest.json", correction_manifest_sha)
    prompts = base.sealed_json(correction_prompts, correction_prompts_sha)
    corrected_rows = correction.get("images", [])
    prompt_rows = prompts.get("images", [])
    if (correction.get("schema") != "sam21_mustard_m1_board5_candidate_v1" or
            correction.get("status") != "generated_unreviewed" or
            correction.get("package_sha256") != base.PACKAGE_SHA256 or
            correction.get("original_prompt_sha256") != base.PROMPTS_SHA256 or
            correction.get("correction_prompt_sha256") != correction_prompts_sha or
            correction_prompts_sha != BOARD5_PROMPTS_SHA or
            correction_prompts_sha == base.PROMPTS_SHA256 or
            correction.get("base_inventory_sha256") != BASE_SHA or
            correction.get("rejected_full_qa_sha256") != REJECTED_QA_SHA or
            correction.get("checkpoint_sha256") != point.MODEL_SHA or
            correction.get("source_inventory_sha256") != point.SOURCE_SHA or
            correction.get("runner_sha256") != BOARD5_RUNNER_SHA or
            base.sha256(ROOT / "scripts/object_motion/sam_m1_board5.py") != BOARD5_RUNNER_SHA or
            prompts.get("schema") != "sam21_mustard_m1_board_negative5_prompts_v1" or
            prompts.get("original_prompt_sha256") != base.PROMPTS_SHA256 or
            prompts.get("base_inventory_sha256") != BASE_SHA or
            prompts.get("rejected_full_qa_sha256") != REJECTED_QA_SHA or
            [row.get("name") for row in corrected_rows] != list(CORRECTED) or
            [row.get("name") for row in prompt_rows] != list(CORRECTED)):
        raise ValueError("corrected five-view manifest/prompt lineage differs")
    supervisor_path = correction_root / "supervisor.json"
    if supervisor_path.is_symlink() or not supervisor_path.is_file():
        raise ValueError("corrected five-view supervisor evidence missing")
    supervisor_sha = base.sha256(supervisor_path)
    supervisor = base.sealed_json(supervisor_path, supervisor_sha)
    if supervisor.get("schema") != "sam21_mustard_m1_board5_supervisor_v1" or supervisor.get("status") != "generated_unreviewed" or supervisor.get("reason"):
        raise ValueError("corrected five-view supervisor did not finish successfully")
    for directory, expected in (("raw_masks", {n + ".png" for n in CORRECTED}),
                                ("cleaned_masks", {n + ".png" for n in CORRECTED})):
        folder = correction_root / directory
        if (folder.is_symlink() or not folder.is_dir() or
                {item.name for item in folder.iterdir()} != expected or
                any(item.is_symlink() or not item.is_file() for item in folder.iterdir())):
            raise ValueError("corrected mask directory contains extra, missing, or linked files")
    correction_review = base.sealed_json(correction_qa, correction_qa_sha)
    if (correction_review.get("schema") != "sam21_mustard_m1_board5_review_v1" or
            correction_review.get("status") != "accepted" or
            correction_review.get("decision") != "accepted_for_coarse_pose_support" or
            correction_review.get("manifest_sha256") != correction_manifest_sha or
            correction_review.get("correction_prompt_sha256") != correction_prompts_sha or
            correction_review.get("reviewed_names") != list(CORRECTED) or
            not correction_review.get("reviewer") or not correction_review.get("reviewed_at")):
        raise ValueError("independent corrected-five visual QA acceptance missing")
    accepted_hashes = correction_review.get("cleaned_mask_sha256")
    if not isinstance(accepted_hashes, dict) or set(accepted_hashes) != set(CORRECTED):
        raise ValueError("corrected-five QA must bind all cleaned mask hashes")
    selected = []
    source_audit = [(str(path), hash_value) for path, hash_value in (
        (package_path, base.PACKAGE_SHA256), (base_root / "complete_inventory.json", BASE_SHA),
        (rejected_qa, REJECTED_QA_SHA), (correction_root / "manifest.json", correction_manifest_sha),
        (correction_prompts, correction_prompts_sha), (correction_qa, correction_qa_sha),
        (supervisor_path, supervisor_sha))]
    for old, name in zip(old_rows, base.TRAIN_NAMES):
        frame = base_root / "frames" / name
        if frame.is_symlink() or {item.name for item in frame.iterdir()} != {"frame.json", "raw.png", "clean.png"}:
            raise ValueError(f"base frame has extra, missing, or linked entries: {name}")
        record_path = frame / "frame.json"
        record = base.sealed_json(record_path, old["frame_manifest_sha256"])
        if (old.get("source_sha256") != photo_hashes[name] or
                record.get("name") != name or record.get("source_sha256") != photo_hashes[name] or
                record.get("prompt_sha256") != base.PROMPTS_SHA256 or
                record.get("cleaned_mask_sha256") != old.get("cleaned_mask_sha256") or
                rejection["cleaned_mask_sha256"].get(name) != old.get("cleaned_mask_sha256")):
            raise ValueError(f"base frame/photo/prompt/hash differs: {name}")
        raw = frame / "raw.png"
        clean = frame / "clean.png"
        mask_info(raw, record["raw_mask_sha256"], expected_size)
        mask_info(clean, old["cleaned_mask_sha256"], expected_size)
        source_audit.extend(((str(record_path), old["frame_manifest_sha256"]),
                             (str(raw), record["raw_mask_sha256"]),
                             (str(clean), old["cleaned_mask_sha256"])))
        if name in CORRECTED:
            index = CORRECTED.index(name)
            row, prompt = corrected_rows[index], prompt_rows[index]
            expected_points = record.get("points_xy_label", []) + [[620, 330, 0]]
            if (row.get("source_sha256") != photo_hashes[name] or
                    prompt.get("source_sha256") != photo_hashes[name] or
                    row.get("points_xy_label") != prompt.get("points_xy_label") or
                    row.get("points_xy_label") != expected_points or
                    row.get("cleaned_mask_sha256") != accepted_hashes[name]):
                raise ValueError(f"corrected frame photo/prompt/QA differs: {name}")
            raw = safe_relative(correction_root, row.get("raw_mask_path"), "raw_masks", name)
            clean = safe_relative(correction_root, row.get("cleaned_mask_path"), "cleaned_masks", name)
            mask_info(raw, row["raw_mask_sha256"], expected_size)
            mask_info(clean, row["cleaned_mask_sha256"], expected_size)
            source_audit.extend(((str(raw), row["raw_mask_sha256"]),
                                 (str(clean), row["cleaned_mask_sha256"])))
            origin = "board5"
            frame_sha = correction_manifest_sha
            prompt_sha = correction_prompts_sha
            points = row["points_xy_label"]
            cleaned_sha = row["cleaned_mask_sha256"]
        else:
            origin = "base43"
            frame_sha = old["frame_manifest_sha256"]
            prompt_sha = base.PROMPTS_SHA256
            points = record["points_xy_label"]
            cleaned_sha = old["cleaned_mask_sha256"]
        selected.append({"name": name, "source_sha256": photo_hashes[name],
                         "source_clean_path": str(clean), "cleaned_mask_sha256": cleaned_sha,
                         "origin": origin, "source_frame_manifest_sha256": frame_sha,
                         "prompt_sha256": prompt_sha, "prompt_points_sha256": digest_object(points)})
    if [row["origin"] for row in selected].count("base43") != 43:
        raise ValueError("candidate must preserve exactly 43 original masks")
    return {"schema": "sam21_mustard_m1_composition_report_v1", "status": "preflight",
            "base_inventory_sha256": BASE_SHA, "base_rejected_qa_sha256": REJECTED_QA_SHA,
            "correction_manifest_sha256": correction_manifest_sha,
            "correction_prompt_sha256": correction_prompts_sha,
            "correction_qa_sha256": correction_qa_sha,
            "correction_supervisor_sha256": supervisor_sha,
            "package_sha256": base.PACKAGE_SHA256, "runner_sha256": base.sha256(Path(__file__)),
            "images": selected, "source_audit": source_audit, "output": str(output),
            "mask_role": "coarse pose support, not exact silhouette"}


def compose(*args, output: Path = OUTPUT, **kwargs) -> dict:
    plan = preflight(*args, output=output, **kwargs)
    output.mkdir()
    (output / "frames").mkdir()
    started = time.monotonic()
    copied = 0
    try:
        for row in plan["images"]:
            source = Path(row["source_clean_path"])
            target_dir = output / "frames" / row["name"]
            target_dir.mkdir()
            target = target_dir / "clean.png"
            size = source.stat().st_size
            if (time.monotonic() - started > MAX_SECONDS or copied + size > MAX_BYTES or
                    shutil.disk_usage(output).free < MIN_FREE + size + BUFFER or
                    shutil.disk_usage(ROOT).free < MIN_FREE):
                raise ValueError("composition deadline, cap, or dual-disk floor")
            shutil.copy2(source, target)
            copied += target.stat().st_size
            if base.sha256(source) != row["cleaned_mask_sha256"] or base.sha256(target) != row["cleaned_mask_sha256"]:
                raise ValueError("source/copy hash changed during composition")
        for source, expected in plan["source_audit"]:
            path = Path(source)
            if path.is_symlink() or not path.is_file() or base.sha256(path) != expected:
                raise ValueError("base/correction source or metadata changed after composition")
        for row in plan["images"]:
            if (base.sha256(Path(row["source_clean_path"])) != row["cleaned_mask_sha256"] or
                    base.sha256(output / "frames" / row["name"] / "clean.png") != row["cleaned_mask_sha256"]):
                raise ValueError("source/copy hash changed after composition")
        if (time.monotonic() - started > MAX_SECONDS or copied > MAX_BYTES or
                shutil.disk_usage(output).free < MIN_FREE or shutil.disk_usage(ROOT).free < MIN_FREE or
                base.sha256(Path(__file__)) != plan["runner_sha256"]):
            raise ValueError("composition postflight resource or runner seal failed")
        plan["status"] = "complete_unreviewed"
        plan["copied_mask_bytes"] = copied
        report_path = output / "composition-report.json"
        report_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        inventory = {"schema": "sam21_mustard_m1_composed_candidate_v1",
                     "status": "generated_unreviewed", "full_training_package_ready": False,
                     "package_sha256": base.PACKAGE_SHA256, "base_inventory_sha256": BASE_SHA,
                     "base_rejected_qa_sha256": REJECTED_QA_SHA,
                     "correction_manifest_sha256": plan["correction_manifest_sha256"],
                     "correction_prompt_sha256": plan["correction_prompt_sha256"],
                     "correction_qa_sha256": plan["correction_qa_sha256"],
                     "correction_supervisor_sha256": plan["correction_supervisor_sha256"],
                     "composition_report_sha256": base.sha256(report_path),
                     "images": [{"name": row["name"], "source_sha256": row["source_sha256"],
                                 "cleaned_mask_path": f"frames/{row['name']}/clean.png",
                                 "cleaned_mask_sha256": row["cleaned_mask_sha256"],
                                 "origin": row["origin"],
                                 "source_frame_manifest_sha256": row["source_frame_manifest_sha256"],
                                 "prompt_sha256": row["prompt_sha256"],
                                 "prompt_points_sha256": row["prompt_points_sha256"]}
                                for row in plan["images"]]}
        (output / "complete_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        output_bytes = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
        if (output_bytes > MAX_BYTES or time.monotonic() - started > MAX_SECONDS or
                shutil.disk_usage(output).free < MIN_FREE or shutil.disk_usage(ROOT).free < MIN_FREE):
            raise ValueError("composition total output, deadline, or dual-disk postflight cap")
    except BaseException as error:
        plan.update(status="failed", failure=f"{type(error).__name__}: {error}", copied_mask_bytes=copied)
        (output / "composition-report.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        raise
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--correction-manifest-sha256", required=True)
    parser.add_argument("--correction-prompts", type=Path, required=True)
    parser.add_argument("--correction-prompts-sha256", required=True)
    parser.add_argument("--correction-qa", type=Path, required=True)
    parser.add_argument("--correction-qa-sha256", required=True)
    parser.add_argument("--compose", action="store_true", help="copy only after independent board5 QA")
    args = parser.parse_args()
    params = (BASE_ROOT, args.correction_root, args.correction_manifest_sha256,
              args.correction_prompts, args.correction_prompts_sha256,
              args.correction_qa, args.correction_qa_sha256)
    result = compose(*params) if args.compose else preflight(*params)
    print(json.dumps({key: result[key] for key in ("schema", "status", "package_sha256")}, indent=2))


if __name__ == "__main__":
    main()
