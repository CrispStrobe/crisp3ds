#!/usr/bin/env python3
"""Hash-bound, small three-view RGB/raw/clean review sheet; no inference."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.object_motion import sam_m1_parity as parity
from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import ycb_object_masks as common


def create(package_path, dataset_root, parity_root, output):
    package_path, dataset_root, parity_root, output = map(
        Path, (package_path, dataset_root, parity_root, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest_path = parity_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 1024**2:
        raise ValueError("missing/linked/oversize parity manifest")
    report = json.loads(manifest_path.read_text())
    if (report.get("schema") != "sam21_m1_cpu_parity_v1" or
            report.get("status") != "parity_observed_pending_visual_qa" or
            report.get("names") != list(parity.NAMES) or len(report.get("images", [])) != 3 or
            report.get("inputs", {}).get("package_sha256") != common.digest(package_path)):
        raise ValueError("parity manifest not complete and hash-bound")
    rows = {Path(row["path"]).name: row for row in
            common.training_records(common.load_package(package_path))}
    sheet = Image.new("RGB", (3 * 360, 3 * 290), "white")
    draw = ImageDraw.Draw(sheet)
    for column, entry in enumerate(report["images"]):
        name = entry["name"]
        if name != parity.NAMES[column] or name not in rows or entry["source_sha256"] != rows[name]["sha256"]:
            raise ValueError("parity photo name/hash differs from frozen training package")
        photo = common.verified_training_photo(dataset_root, rows[name])
        for lane, kind in enumerate(("rgb", "raw", "cleaned")):
            if kind == "rgb":
                crop = photo.crop(smoke.BOX)
            else:
                directory = "raw_masks" if kind == "raw" else "cleaned_masks"
                path = parity_root / directory / f"{name}.png"
                if path.is_symlink() or not path.is_file() or common.digest(path) != entry[kind]["candidate_sha256"]:
                    raise ValueError("parity mask differs from manifest")
                mask = Image.fromarray(parity.geometry(path), "L")
                masked = Image.new("RGB", photo.size, "black")
                masked.paste(photo, (0, 0), mask)
                crop = masked.crop(smoke.BOX)
            crop.thumbnail((340, 255), Image.Resampling.LANCZOS)
            x, y = column * 360, lane * 290
            sheet.paste(crop, (x + (360 - crop.width) // 2, y + 2))
            draw.text((x + 8, y + 265), f"{name}  {kind}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or output.exists():
        raise ValueError("review output parent linked or target exists")
    sheet.save(output, "JPEG", quality=86)
    if output.stat().st_size > 2 * 1024**2:
        raise ValueError("review sheet exceeds 2 MiB")
    return {"sheet_sha256": common.digest(output), "parity_manifest_sha256": common.digest(manifest_path),
            "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "dataset-root", "parity-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.package, args.dataset_root, args.parity_root, args.output), sort_keys=True))
