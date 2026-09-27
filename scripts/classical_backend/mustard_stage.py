"""Hash-bound 48-train mustard staging gate; no SfM is launched here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MOUNT = Path("/Volumes/backups")
OUTPUT = MOUNT / "code/crisp3ds-data/mustard-sfm-train-001"
PACKAGE = ROOT / "build-opencv/ycb-evaluation-vps-001/mustard-package.json"
PACKAGE_SHA256 = "45b6ed1430627a747eaeb8d4dff7d2c49502083cca0a7914e25aacd45b594425"
ACQUISITION_SHA256 = "c48998dffc7798e7d21f275383b1c67dc2e5e9313a8bfe699f72303f61583b52"
PROMPTS_SHA256 = "28440795d5f38590df8c5568969970d6623fbe22ea6b3e8781cba10608149d32"
TRAIN_NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in range(0, 360, 6) if angle % 30 != 24)
HELDOUT_NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in range(24, 360, 30))
MAX_BYTES = 150 * 1024**2
MIN_FREE = 10 * 1024**3
BUFFER = 256 * 1024**2
MAX_SECONDS = 240


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sealed_json(path: Path, digest: str) -> dict:
    if (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) or
            path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2 or
            sha256(path) != digest):
        raise ValueError(f"missing, oversized, or changed sealed JSON: {path}")
    return json.loads(path.read_text())


def exact_files(directory: Path, expected: set[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"missing/linked train-only directory: {directory}")
    items = list(directory.iterdir())
    if {item.name for item in items} != expected or any(item.is_symlink() or not item.is_file() for item in items):
        raise ValueError("train-only directory has missing, extra, held-out, or linked files")


def mask_pixels(mask: Path, photo: Path, expected_size: tuple[int, int]) -> dict:
    with Image.open(photo) as image, Image.open(mask) as alpha:
        image.load()
        alpha.load()
        if (image.format != "JPEG" or image.mode != "RGB" or image.size != expected_size or
                alpha.format != "PNG" or alpha.mode != "L" or alpha.size != expected_size):
            raise ValueError("mustard photo/mask format or dimensions differ")
        histogram = alpha.histogram()
        if sum(histogram[1:255]) or not histogram[0] or not histogram[255]:
            raise ValueError("cleaned mask must contain both binary 0 and 255 only")
        return {"kept_pixels": histogram[255], "ignored_pixels": histogram[0]}


def preflight(package_path: Path, train_photos: Path, inventory_path: Path, inventory_root: Path,
              inventory_sha256: str, qa_path: Path, qa_sha256: str, *,
              output: Path = OUTPUT, mount: Path = MOUNT, require_mount: bool = True,
              package_sha256: str = PACKAGE_SHA256,
              expected_size: tuple[int, int] = (1280, 1024)) -> dict:
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise ValueError("staging output must be fresh")
    if require_mount:
        if (not mount.is_mount() or mount.stat().st_dev == ROOT.stat().st_dev or
                not output.resolve().is_relative_to(mount.resolve())):
            raise ValueError("staging destination must be external mounted volume")
    package = sealed_json(package_path, package_sha256)
    if (package.get("schema") != "ycb_object_evaluation_package_v1" or
            package.get("object_id") != "006_mustard_bottle" or
            package.get("manifest_sha256") != ACQUISITION_SHA256):
        raise ValueError("not the frozen mustard evaluation package")
    training = package.get("training_inputs", [])
    heldout = package.get("heldout_photos", [])
    train_names = [Path(row["path"]).name for row in training]
    heldout_names = [Path(row["path"]).name for row in heldout]
    if (train_names != list(TRAIN_NAMES) or heldout_names != list(HELDOUT_NAMES) or
            len(set(train_names + heldout_names)) != 60 or
            any(row["path"] != "photos/" + name for row, name in zip(training, TRAIN_NAMES))):
        raise ValueError("package is not exact 48-train/12-heldout split")
    exact_files(train_photos, set(TRAIN_NAMES))
    photo_hashes = {}
    for row, name in zip(training, TRAIN_NAMES):
        photo = train_photos / name
        if photo.stat().st_size != row["bytes"] or sha256(photo) != row["sha256"]:
            raise ValueError(f"train photo hash differs: {name}")
        photo_hashes[name] = row["sha256"]
    inventory = sealed_json(inventory_path, inventory_sha256)
    rows = inventory.get("images", [])
    if (inventory.get("schema") != "sam21_mustard_point_mask_inventory_v1" or
            inventory.get("status") != "generated_unreviewed" or
            inventory.get("package_sha256") != package_sha256 or
            inventory.get("prompt_sha256") != PROMPTS_SHA256 or
            [row.get("name") for row in rows] != list(TRAIN_NAMES)):
        raise ValueError("full 48-view SAM inventory is incomplete or wrong source")
    qa = sealed_json(qa_path, qa_sha256)
    if (qa.get("schema") != "mustard_sam_mask_qa_v1" or qa.get("status") != "accepted" or
            qa.get("decision") != "accepted_for_coarse_pose_support" or
            qa.get("inventory_sha256") != inventory_sha256 or
            qa.get("reviewed_names") != list(TRAIN_NAMES) or
            not qa.get("reviewer") or not qa.get("reviewed_at")):
        raise ValueError("full independent visual QA acceptance missing")
    qa_mask_hashes = qa.get("cleaned_mask_sha256")
    if not isinstance(qa_mask_hashes, dict) or set(qa_mask_hashes) != set(TRAIN_NAMES):
        raise ValueError("QA must bind all 48 cleaned mask hashes")
    frames = inventory_root / "frames"
    if frames.is_symlink() or not frames.is_dir() or {item.name for item in frames.iterdir()} != set(TRAIN_NAMES):
        raise ValueError("SAM frame directory is not exact 48-train set")
    masks = {}
    for row, name in zip(rows, TRAIN_NAMES):
        if row.get("source_sha256") != photo_hashes[name]:
            raise ValueError(f"SAM source photo differs: {name}")
        if row.get("cleaned_mask_path") != f"frames/{name}/clean.png":
            raise ValueError(f"SAM cleaned-mask relative path differs: {name}")
        path = frames / name / "clean.png"
        if (path.parent.is_symlink() or path.is_symlink() or not path.is_file() or
                row.get("cleaned_mask_sha256") != qa_mask_hashes[name] or
                sha256(path) != qa_mask_hashes[name]):
            raise ValueError(f"cleaned SAM mask hash/path differs: {name}")
        masks[name] = {"source": str(path), "sha256": qa_mask_hashes[name],
                       **mask_pixels(path, train_photos / name, expected_size)}
    names_bytes = ("\n".join(TRAIN_NAMES) + "\n").encode("ascii")
    total = sum((train_photos / name).stat().st_size for name in TRAIN_NAMES)
    total += sum(Path(row["source"]).stat().st_size for row in masks.values()) + len(names_bytes)
    if total > MAX_BYTES or shutil.disk_usage(mount).free < MIN_FREE + MAX_BYTES + BUFFER:
        raise ValueError("150 MiB stage cap or external 10 GiB reserve not met")
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        raise ValueError("internal 10 GiB reserve not met")
    return {"schema": "mustard_train_only_stage_v1", "status": "preflight",
            "package_sha256": package_sha256, "acquisition_manifest_sha256": ACQUISITION_SHA256,
            "inventory_sha256": inventory_sha256, "qa_sha256": qa_sha256,
            "train_photo_sha256": photo_hashes, "cleaned_masks": masks,
            "train_names_sha256": hashlib.sha256(names_bytes).hexdigest(),
            "train_names": list(TRAIN_NAMES), "heldout_names_excluded": list(HELDOUT_NAMES),
            "output": str(output), "copy_bytes": total,
            "mask_role": "accepted coarse pose support, not object silhouette or ground truth"}


def prepare(*args, output: Path = OUTPUT, mount: Path = MOUNT, **kwargs) -> dict:
    plan = preflight(*args, output=output, mount=mount, **kwargs)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    started = time.monotonic()
    copied = 0
    try:
        for name in TRAIN_NAMES:
            for source, target, digest in (
                (Path(args[1]) / name, output / "images" / name, plan["train_photo_sha256"][name]),
                (Path(plan["cleaned_masks"][name]["source"]), output / "masks" / (name + ".png"),
                 plan["cleaned_masks"][name]["sha256"]),
            ):
                size = source.stat().st_size
                if (time.monotonic() - started > MAX_SECONDS or copied + size > MAX_BYTES or
                        shutil.disk_usage(mount).free < MIN_FREE + size + BUFFER or
                        shutil.disk_usage(ROOT).free < MIN_FREE):
                    raise ValueError("stage deadline, byte cap, or dual-disk reserve")
                shutil.copy2(source, target)
                copied += target.stat().st_size
                if sha256(source) != digest or sha256(target) != digest:
                    raise ValueError(f"stage copy/source hash differs: {name}")
        names = ("\n".join(TRAIN_NAMES) + "\n").encode("ascii")
        (output / "train-names.txt").write_bytes(names)
        copied += len(names)
        if hashlib.sha256(names).hexdigest() != plan["train_names_sha256"]:
            raise ValueError("train-name list changed")
        # Late-source and staged-copy rehash, plus no extra files.
        if ({p.name for p in (output / "images").iterdir()} != set(TRAIN_NAMES) or
                {p.name for p in (output / "masks").iterdir()} != {n + ".png" for n in TRAIN_NAMES}):
            raise ValueError("staged train file set changed")
        for name in TRAIN_NAMES:
            for source, target, digest in (
                (Path(args[1]) / name, output / "images" / name, plan["train_photo_sha256"][name]),
                (Path(plan["cleaned_masks"][name]["source"]), output / "masks" / (name + ".png"),
                 plan["cleaned_masks"][name]["sha256"]),
            ):
                if source.is_symlink() or target.is_symlink() or sha256(source) != digest or sha256(target) != digest:
                    raise ValueError(f"stage source/copy changed after copy: {name}")
        if (copied != plan["copy_bytes"] or copied > MAX_BYTES or
                time.monotonic() - started > MAX_SECONDS or
                shutil.disk_usage(mount).free < MIN_FREE or shutil.disk_usage(ROOT).free < MIN_FREE or
                sha256(Path(args[0])) != plan["package_sha256"] or
                sha256(Path(args[2])) != plan["inventory_sha256"] or
                sha256(Path(args[5])) != plan["qa_sha256"]):
            raise ValueError("stage postflight hashes, limits, or dual-disk reserve failed")
        plan.update(status="complete", copied_bytes=copied)
    except Exception as error:
        plan.update(status="failed", failure=f"{type(error).__name__}: {error}", copied_bytes=copied)
        (output / "stage-report.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        raise
    (output / "stage-report.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-photos", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--prepare", action="store_true", help="copy only after visual QA and root approval")
    args = parser.parse_args()
    params = (PACKAGE, args.train_photos, args.inventory, args.inventory_root,
              args.inventory_sha256, args.qa, args.qa_sha256)
    report = prepare(*params) if args.prepare else preflight(*params)
    print(json.dumps({key: report[key] for key in ("status", "output", "copy_bytes", "qa_sha256")}, indent=2))


if __name__ == "__main__":
    main()
