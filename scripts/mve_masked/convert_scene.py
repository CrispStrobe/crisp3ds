#!/usr/bin/env python3
"""Convert the sparse fixture to MVE with validated per-view object masks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mve_spike"))
from convert_scene import convert as convert_sparse  # noqa: E402


def convert(fixture: Path, scene: Path):
    fixture = fixture.resolve(strict=True)
    project = json.loads((fixture / "project.json").read_text())
    sparse = json.loads((fixture / "sparse-report.json").read_text())
    masks = {}
    width, height = project["calibration"]["width"], project["calibration"]["height"]
    for image in project["images"]:
        name = image.get("maskPath")
        if not name or Path(name).name != name or not name.endswith(".png"):
            raise ValueError("each image requires a local PNG maskPath")
        path = (fixture / name).resolve(strict=True)
        if path.parent != fixture:
            raise ValueError("maskPath escapes fixture directory")
        with Image.open(path) as mask:
            if mask.mode != "L" or mask.size != (width, height):
                raise ValueError(f"mask must be 8-bit grayscale and {width}x{height}: {name}")
            histogram = mask.histogram()
            if any(histogram[value] for value in range(1, 255)):
                raise ValueError(f"mask must be binary: {name}")
            if not histogram[255]:
                raise ValueError(f"mask must contain object pixels: {name}")
        if image["id"] in masks:
            raise ValueError("duplicate image id")
        masks[image["id"]] = path
    if set(masks) != {view["imageId"] for view in sparse["views"]}:
        raise ValueError("mask/view image IDs differ")
    if scene.exists():
        raise FileExistsError(scene)
    summary = convert_sparse(fixture, scene)
    for index, view in enumerate(sparse["views"]):
        path = masks[view["imageId"]]
        shutil.copyfile(path, scene / "views" / f"view_{index:04d}.mve" / "object-mask.png")
    summary["mask_enforcement"] = "MVE object-mask embedding, conservative pyramid and patch gate"
    summary["mask_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in masks.values()}
    (scene / "conversion.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.fixture, args.scene), indent=2))
