"""Read-only checks for a frozen COLMAP-to-Brush experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

SOURCE_COMMIT = "6378a76add3b93501abb55c2dc08d71688537679"
RELEASE_COMMIT = "3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486"
RELEASE_SHA256 = "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c"
RELEASE_URL = "https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz"
MIN_FREE = 10 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images_from_binary(path: Path) -> list[str]:
    """Read only the registered image names from COLMAP images.bin."""
    names: list[str] = []
    with path.open("rb") as stream:
        count_data = stream.read(8)
        if len(count_data) != 8:
            raise ValueError("truncated images.bin")
        count = struct.unpack("<Q", count_data)[0]
        if not 1 <= count <= 100_000:
            raise ValueError("implausible registered image count")
        for _ in range(count):
            if len(stream.read(64)) != 64:  # id, quaternion, translation, camera id
                raise ValueError("truncated image pose")
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char or len(name) > 4096:
                    raise ValueError("truncated or oversized image name")
                if char == b"\0":
                    break
                name.extend(char)
            decoded = name.decode("utf-8")
            if (Path(decoded).is_absolute() or ".." in Path(decoded).parts or
                    "\\" in decoded or ":" in decoded or not decoded):
                raise ValueError("unsafe registered image name")
            names.append(decoded)
            npoints_data = stream.read(8)
            if len(npoints_data) != 8:
                raise ValueError("truncated points2D count")
            npoints = struct.unpack("<Q", npoints_data)[0]
            if npoints > 10_000_000:
                raise ValueError("implausible points2D count")
            position = stream.tell()
            end = stream.seek(0, 2)
            if npoints * 24 > end - position:
                raise ValueError("truncated points2D payload")
            stream.seek(position + npoints * 24)
        if stream.read(1):
            raise ValueError("trailing images.bin payload")
    if len(set(names)) != len(names):
        raise ValueError("duplicate COLMAP image names")
    return names


def validate_dataset(
    dataset: Path,
    output: Path,
    *,
    expected_result: Path | None = None,
    split_manifest: Path | None = None,
) -> dict:
    if dataset.is_symlink() or output.is_symlink() or output.exists():
        raise ValueError("dataset cannot be a symlink; output must be a fresh path")
    if output.parent.is_symlink():
        raise ValueError("output parent cannot be a symlink")
    dataset = dataset.resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError("dataset is not a directory")
    model = dataset / "sparse"
    if model.is_symlink():
        raise ValueError("symlink in COLMAP model path")
    if not (model / "cameras.bin").is_file():
        model = model / "0"
    if model.is_symlink() or (dataset / "images").is_symlink():
        raise ValueError("symlink in COLMAP dataset path")
    files = {name: model / name for name in ("cameras.bin", "images.bin", "points3D.bin")}
    for path in files.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/unsafe COLMAP file: {path}")
    names = _images_from_binary(files["images.bin"])
    image_hashes = {}
    for name in names:
        image = dataset / "images" / name
        if any(parent.is_symlink() for parent in image.parents if parent != dataset.parent):
            raise ValueError(f"symlinked image parent: {name}")
        if image.is_symlink() or not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"missing/unsafe registered image: {name}")
        image_hashes[name] = sha256(image)
    # A hidden eval split or unreviewed masks would change the experiment.
    if (dataset / "masks").exists():
        raise ValueError("unreviewed masks directory present")
    if expected_result is not None:
        if expected_result.is_symlink() or not expected_result.is_file():
            raise ValueError("missing/unsafe producer result")
        producer = json.loads(expected_result.read_text())
        if producer.get("status") != "complete" or producer.get("quality_accepted") is True:
            raise ValueError("producer must be completed, without inferred quality acceptance")
        result_hash = sha256(expected_result)
    else:
        result_hash = None
    split_hash = None
    split_status = "unverified"
    if split_manifest is not None:
        if split_manifest.is_symlink() or not split_manifest.is_file():
            raise ValueError("missing/unsafe split manifest")
        split = json.loads(split_manifest.read_text())
        if split.get("schema") != "brush_train_split_v1":
            raise ValueError("incorrect split manifest schema")
        train = split.get("train_image_sha256")
        heldout = split.get("heldout_image_sha256")
        if not isinstance(train, dict) or not isinstance(heldout, dict):
            raise ValueError("split manifest requires train and heldout hash maps")
        if train != image_hashes or set(train) & set(heldout):
            raise ValueError("train hashes must match all registered images; heldout must be disjoint")
        if not all(isinstance(x, str) and len(x) == 64 and set(x) <= set("0123456789abcdef") for x in heldout.values()):
            raise ValueError("invalid heldout digest")
        split_hash = sha256(split_manifest)
        split_status = "explicit all-registered-train manifest verified; external inventory not proved"
    return {
        "schema": "brush_colmap_preflight_v1",
        "source_commit": SOURCE_COMMIT,
        "release_commit": RELEASE_COMMIT,
        "release_sha256": RELEASE_SHA256,
        "dataset": str(dataset),
        "model_file_sha256": {key: sha256(path) for key, path in files.items()},
        "registered_images": len(names),
        "image_sha256": image_hashes,
        "producer_result_sha256": result_hash,
        "output": str(output.absolute()),
        "eval_split_option": "omitted; Brush source default None => all loaded views train",
        "external_heldout_inventory": split_status,
        "split_manifest_sha256": split_hash,
        "input_split_verified": split_hash is not None,
        "training_authorized": False,
        "producer_association": "unverified; result status alone does not bind this model and images",
        "mask_policy": "none; dataset/masks rejected",
        "artifact_type": "Gaussian splat PLY, not mesh",
    }


def toolchain() -> dict:
    root = Path(__file__).resolve().parents[2]
    rust_bin = root / ".local-tools/rustup/toolchains/stable-aarch64-apple-darwin/bin"
    return {
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
        "cargo": str(rust_bin / "cargo") if (rust_bin / "cargo").is_file() else shutil.which("cargo"),
        "rustc": str(rust_bin / "rustc") if (rust_bin / "rustc").is_file() else shutil.which("rustc"),
        "metal_compiler": shutil.which("xcrun") is not None and subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, timeout=5, check=False
        ).returncode == 0,
        "free_bytes": shutil.disk_usage(root).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-result", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset, args.output, expected_result=args.producer_result, split_manifest=args.split_manifest)
    report["toolchain"] = toolchain()
    report["disk_floor_met"] = report["toolchain"]["free_bytes"] >= MIN_FREE
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
