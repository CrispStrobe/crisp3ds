#!/usr/bin/env python3
"""Immutable input inventory and pre-reconstruction YCB train/held-out split."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "006_mustard_bottle": (ROOT / "build-opencv/ycb-vps-acquisition-001/mustard-manifest.json",
                           "c48998dffc7798e7d21f275383b1c67dc2e5e9313a8bfe699f72303f61583b52"),
    "035_power_drill": (ROOT / "build-opencv/ycb-vps-acquisition-001/drill-manifest.json",
                        "aeef80b879676092047d54089905dfdc29859a56b1ad872d17f800d18cc81594"),
}
ANGLES = tuple(range(0, 360, 6))
HELDOUT = tuple(angle for index, angle in enumerate(ANGLES) if index % 5 == 4)
MAX_MANIFEST_BYTES = 100_000
MAX_OUTPUT_BYTES = 100_000
MIN_FREE_BYTES = 10 * 1024**3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative asset path")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("asset path escapes or ambiguously addresses dataset root")
    return Path(*parts)


def validate_manifest(manifest, object_id):
    if object_id not in MANIFESTS:
        raise ValueError("unsupported object")
    if (manifest.get("schema") != "ycb_more_prepared_v1" or manifest.get("status") != "complete"
            or manifest.get("object_id") != object_id
            or manifest.get("source_authentication") != "observed_sha256_only_unverified_upstream_identity"):
        raise ValueError("unexpected acquisition manifest status, object or source authentication")
    selection = manifest.get("selection", {})
    if (selection.get("camera") != "Berkeley NP3" or selection.get("views") != 60
            or selection.get("angles") != list(ANGLES)
            or "Google scanner reference" not in selection.get("excluded_from_reconstruction", [])):
        raise ValueError("manifest does not preserve the 60-view NP3 selection/reference exclusion")
    archives = manifest.get("archives", {})
    if set(archives) != {"berkeley_rgbd", "google"} or any(
            not _digest(item.get("observed_sha256")) or item.get("independently_pinned") is not False
            for item in archives.values()):
        raise ValueError("archive observation provenance incomplete")
    photos = manifest.get("photos")
    if not isinstance(photos, list) or len(photos) != 60:
        raise ValueError("expected exactly 60 selected photos")
    by_angle, paths, source_members, photo_digests = {}, set(), set(), set()
    for record in photos:
        angle = record.get("turntable_angle_degrees")
        path = record.get("path")
        source = record.get("source_member")
        if (type(angle) is not int or angle not in ANGLES or angle in by_angle
                or path != f"photos/NP3_{angle:03}.jpg"
                or source != f"{object_id}/NP3_{angle}.jpg"
                or record.get("role") != "reconstruction-input-real-photograph"
                or not _digest(record.get("sha256"))
                or type(record.get("bytes")) is not int or not 0 < record["bytes"] <= 32 * 1024**2):
            raise ValueError("duplicate, missing or invalid NP3 photo record")
        _relative_path(path)
        by_angle[angle] = record
        paths.add(path)
        source_members.add(source)
        photo_digests.add(record["sha256"])
    if (set(by_angle) != set(ANGLES) or len(paths) != 60 or len(source_members) != 60
            or len(photo_digests) != 60):
        raise ValueError("photo angle/path/source/digest inventory is incomplete or duplicated")
    original = manifest.get("reference_original", {})
    converted = manifest.get("reference_binary_geometry", {})
    resolution = "16k" if object_id == "006_mustard_bottle" else "64k"
    if (original.get("path") != "reference/google_original_ascii.ply"
            or original.get("source_member") != f"{object_id}/google_{resolution}/nontextured.ply"
            or original.get("role") != "Google scanner reference only"
            or converted.get("path") != "reference/google_geometry_f64.ply"
            or converted.get("role") != "converted Google scanner reference only"):
        raise ValueError("reference provenance or path differs from acquisition contract")
    for record in (original, converted):
        _relative_path(record["path"])
        if (record["path"] in paths or record.get("sha256") in photo_digests
                or not _digest(record.get("sha256"))
                or type(record.get("bytes")) is not int or not 0 < record["bytes"] <= 32 * 1024**2):
            raise ValueError("invalid or overlapping evaluation-only reference")
    if original["sha256"] == converted["sha256"]:
        raise ValueError("evaluation-only reference records share content digest")
    if (converted.get("vertices"), converted.get("faces")) != (
            (8194, 16384) if object_id == "006_mustard_bottle" else (32770, 65536)):
        raise ValueError("reference geometry inventory differs from acquired mesh")
    return by_angle, (original, converted)


def _asset(record):
    return {"path": record["path"], "bytes": record["bytes"], "sha256": record["sha256"]}


def build_protocol(manifest, object_id, manifest_sha256):
    by_angle, references = validate_manifest(manifest, object_id)
    training = [{"angle_degrees": angle, **_asset(by_angle[angle])}
                for angle in ANGLES if angle not in HELDOUT]
    heldout = [{"angle_degrees": angle, **_asset(by_angle[angle])} for angle in HELDOUT]
    if len(training) != 48 or len(heldout) != 12 or set(x["path"] for x in training).intersection(
            x["path"] for x in heldout):
        raise AssertionError("frozen train/held-out partition failed")
    return {"schema": "ycb_object_evaluation_package_v1", "object_id": object_id,
            "status": "manifest_only_split_plan", "manifest_sha256": manifest_sha256,
            "source_authentication": manifest["source_authentication"],
            "split_rule": "sorted NP3 angle index modulo 5 equals 4 => held out",
            "heldout_angles_degrees": list(HELDOUT),
            "angle_labels_role": "split only; forbidden as SfM seed or pose constraint",
            "training_inputs": training, "heldout_photos": heldout,
            "evaluation_only": {"google_scanner_original": _asset(references[0]),
                                "google_scanner_geometry": _asset(references[1])},
            "mask_status": "pending_photo_derived_review",
            "runnable_image_only_training_package": False,
            "heldout_caveat": "all-image SfM would leak held-out photos into camera estimation",
            "coverage_caveat": "one turntable elevation; no underside or complete concavity coverage guarantee"}


def verify_files(dataset_root, protocol):
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset root is not a directory")
    files = protocol["training_inputs"] + protocol["heldout_photos"] + list(protocol["evaluation_only"].values())
    for record in files:
        relative = _relative_path(record["path"])
        path = root / relative
        component = root
        for part in relative.parts:
            component /= part
            if component.is_symlink():
                raise ValueError(f"dataset asset contains symlink: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"dataset path escapes root or is not a regular file: {relative}")
        if resolved.stat().st_size != record["bytes"] or sha256_file(resolved) != record["sha256"]:
            raise ValueError(f"dataset asset bytes or SHA-256 differ: {relative}")
    protocol["status"] = "photos_and_references_rehashed_masks_pending"
    protocol["dataset_root"] = str(root)
    protocol["verified_file_count"] = len(files)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=MANIFESTS, required=True)
    parser.add_argument("--manifest", type=Path, help="defaults to pinned copied manifest")
    parser.add_argument("--dataset-root", type=Path, help="rehash all assets at VPS acquisition root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    manifest_path, expected = MANIFESTS[args.object]
    if args.manifest is not None:
        manifest_path = args.manifest
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES or sha256_file(manifest_path) != expected:
        raise ValueError("manifest differs from pinned copied acquisition record")
    manifest = json.loads(manifest_path.read_text())
    protocol = build_protocol(manifest, args.object, expected)
    if args.dataset_root is not None:
        verify_files(args.dataset_root, protocol)
    protocol["validator_sha256"] = sha256_file(Path(__file__))
    if sha256_file(manifest_path) != expected:
        raise ValueError("manifest changed during package generation")
    encoded = (json.dumps(protocol, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("protocol output exceeds 100 KB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < MIN_FREE_BYTES:
        raise OSError("cannot preserve 10 GiB free disk")
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"object": args.object, "status": protocol["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
