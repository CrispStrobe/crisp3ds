#!/usr/bin/env python3
"""Exercise mask validation, all-valid parity, and outside-mask mutation invariance."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mve_spike"))
from convert_scene import convert as convert_sparse  # noqa: E402
from evaluate_depth import read_mvei_float  # noqa: E402
from convert_scene import RESERVE  # noqa: E402

spec = importlib.util.spec_from_file_location("masked_converter", Path(__file__).with_name("convert_scene.py"))
masked_converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masked_converter)
convert_masked = masked_converter.convert


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(fixture, output, binary, baseline_depth, reference_masked_depth):
    fixture = fixture.resolve(strict=True)
    binary = binary.resolve(strict=True)
    baseline_depth = baseline_depth.resolve(strict=True)
    reference_masked_depth = reference_masked_depth.resolve(strict=True)
    output = output.resolve(strict=False)
    if not fixture.is_dir() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("fixture must be a directory and binary must be executable")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    project_path = fixture / "project.json"
    sparse_path = fixture / "sparse-report.json"
    malformed_paths = [fixture / name for name in (
        "wrong-size-mask.json", "nonbinary-mask.json",
        "wrong-size-mask.png", "nonbinary-mask.png")]
    for path in (project_path, sparse_path, *malformed_paths):
        if not path.is_file():
            raise ValueError(f"missing fixture input: {path}")
    project = json.loads(project_path.read_text())
    sparse = json.loads(sparse_path.read_text())
    width, height = project["calibration"]["width"], project["calibration"]["height"]
    if not project["images"] or len(project["images"]) != len(sparse["views"]):
        raise ValueError("fixture view count mismatch")
    image_ids = set()
    source_paths = [project_path, sparse_path, *malformed_paths]
    for item in project["images"]:
        if item["id"] in image_ids:
            raise ValueError("duplicate image ID")
        image_ids.add(item["id"])
        for key in ("path", "maskPath"):
            name = item[key]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"invalid {key}: {name}")
            path = (fixture / name).resolve(strict=True)
            if path.parent != fixture or not path.is_file():
                raise ValueError(f"{key} escapes fixture: {name}")
            source_paths.append(path)
        with Image.open(fixture / item["path"]) as image:
            if image.mode != "L" or image.size != (width, height):
                raise ValueError("mutation requires grayscale source images at calibration size")
        with Image.open(fixture / item["maskPath"]) as mask:
            if mask.mode != "L" or mask.size != (width, height):
                raise ValueError("mask size or format invalid")
            histogram = mask.histogram()
            if not histogram[255] or any(histogram[value] for value in range(1, 255)):
                raise ValueError("mask must be nonempty and binary")
    if image_ids != {view["imageId"] for view in sparse["views"]}:
        raise ValueError("fixture images and sparse views differ")
    expected_size = ((width + 3) // 4, (height + 3) // 4)
    if read_mvei_float(baseline_depth)[:2] != expected_size:
        raise ValueError("baseline depth is not fixture scale 2")
    if read_mvei_float(reference_masked_depth)[:2] != expected_size:
        raise ValueError("masked reference depth is not fixture scale 2")
    if shutil.disk_usage(output.parent).free < RESERVE + 100 * 1024 * 1024:
        raise RuntimeError("10 GiB free-space reserve")
    provenance = {
        "binary": {"path": str(binary), "sha256": sha256(binary)},
        "baseline_depth": {"path": str(baseline_depth), "sha256": sha256(baseline_depth)},
        "reference_masked_depth": {"path": str(reference_masked_depth),
                                   "sha256": sha256(reference_masked_depth)},
        "fixture_sha256": {path.name: sha256(path) for path in source_paths},
    }
    return fixture, output, binary, baseline_depth, reference_masked_depth, project, provenance


def run(binary, scene):
    subprocess.run([str(binary), "--master-view=1", "--scale=2", "--neighbors=2",
                    "--local-neighbors=2", "--progress=simple", str(scene)], check=True,
                   stdout=subprocess.DEVNULL, timeout=30)
    return scene / "views/view_0001.mve/depth-L2.mvei"


def compare(reference, candidate):
    rw, rh, rv = read_mvei_float(reference)
    cw, ch, cv = read_mvei_float(candidate)
    if (rw, rh) != (cw, ch):
        raise AssertionError("depth dimensions differ")
    mismatches = sum(a != b for a, b in zip(rv, cv))
    return {"dimensions": [rw, rh], "pixels": len(rv), "mismatches": mismatches}


def verify(fixture, output, masked_binary, baseline_depth, reference_masked_depth):
    (fixture, output, masked_binary, baseline_depth, reference_masked_depth,
     project, provenance) = validate_inputs(
        fixture, output, masked_binary, baseline_depth, reference_masked_depth)
    output.mkdir()
    malformed = {}
    for filename in ("wrong-size-mask.json", "nonbinary-mask.json"):
        test_fixture = output / filename.removesuffix(".json")
        shutil.copytree(fixture, test_fixture)
        shutil.copyfile(fixture / filename, test_fixture / "project.json")
        scene = output / (filename + "-scene")
        try:
            convert_masked(test_fixture, scene)
        except ValueError as error:
            malformed[filename] = str(error)
        else:
            raise AssertionError(f"accepted malformed mask: {filename}")
        if scene.exists():
            raise AssertionError(f"created scene for malformed mask: {filename}")

    reordered_fixture = output / "reordered-project"
    shutil.copytree(fixture, reordered_fixture)
    reordered_project = json.loads((fixture / "project.json").read_text())
    reordered_project["images"].reverse()
    (reordered_fixture / "project.json").write_text(json.dumps(reordered_project))
    reordered_scene = output / "reordered-scene"
    convert_masked(reordered_fixture, reordered_scene)
    for index, view in enumerate(json.loads((fixture / "sparse-report.json").read_text())["views"]):
        image = next(item for item in reordered_project["images"] if item["id"] == view["imageId"])
        expected = (fixture / image["maskPath"]).read_bytes()
        actual = (reordered_scene / "views" / f"view_{index:04d}.mve" / "object-mask.png").read_bytes()
        if actual != expected:
            raise AssertionError("mask/view association depends on project image order")

    linked_fixture = output / "escaped-symlink"
    shutil.copytree(fixture, linked_fixture)
    linked_project = json.loads((fixture / "project.json").read_text())
    linked_project["images"][0]["maskPath"] = "linked-mask.png"
    (linked_fixture / "project.json").write_text(json.dumps(linked_project))
    (linked_fixture / "linked-mask.png").symlink_to((fixture / "view-1-object-mask.png").resolve())
    try:
        convert_masked(linked_fixture, output / "linked-scene")
    except ValueError as error:
        malformed["escaped-symlink"] = str(error)
    else:
        raise AssertionError("accepted symlink mask escaping fixture")

    all_valid = output / "all-valid"
    convert_sparse(fixture, all_valid)
    size = (project["calibration"]["width"], project["calibration"]["height"])
    for index in range(len(project["images"])):
        Image.new("L", size, 255).save(
            all_valid / "views" / f"view_{index:04d}.mve" / "object-mask.png")
    all_valid_depth = run(masked_binary, all_valid)
    all_valid_result = compare(baseline_depth, all_valid_depth)
    all_valid_result["output_depth_sha256"] = sha256(all_valid_depth)
    if all_valid_result["mismatches"]:
        raise AssertionError(f"all-valid baseline mismatch: {all_valid_result}")

    changed_fixture = output / "outside-mutated-fixture"
    shutil.copytree(fixture, changed_fixture)
    altered = 0
    for image in project["images"]:
        original = Image.open(fixture / image["path"]).convert("L")
        mask = Image.open(fixture / image["maskPath"]).convert("L")
        pixels = bytearray(original.tobytes())
        for index, valid in enumerate(mask.tobytes()):
            if valid == 0:
                pixels[index] = 255 - pixels[index]
                altered += 1
        Image.frombytes("L", original.size, bytes(pixels)).save(changed_fixture / image["path"])
    mutated_scene = output / "outside-mutated-scene"
    convert_masked(changed_fixture, mutated_scene)
    mutated_depth = run(masked_binary, mutated_scene)
    mutation_result = compare(reference_masked_depth, mutated_depth)
    mutation_result["output_depth_sha256"] = sha256(mutated_depth)
    if mutation_result["mismatches"]:
        raise AssertionError(f"outside-mask mutation changed depth: {mutation_result}")

    provenance["mutated_image_sha256"] = {
        image["path"]: sha256(changed_fixture / image["path"])
        for image in project["images"]}
    result = {"malformed_rejected": malformed, "reordered_images": "mask IDs preserved",
              "all_valid_parity": all_valid_result,
              "outside_mask_mutation": mutation_result, "altered_source_pixels": altered,
              "provenance": provenance, "reconstruction_timeout_seconds": 30}
    (output / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-depth", type=Path, required=True)
    parser.add_argument("--reference-masked-depth", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.fixture, args.output, args.binary, args.baseline_depth,
                            args.reference_masked_depth), indent=2))
