#!/usr/bin/env python3
"""Validate manual polygons and render overlays before any support metrics."""

import argparse
import json
from pathlib import Path
import shutil

from scripts.foreground_roi.polygon import load_contract
from scripts.foreground_roi.run import FROZEN, overlay_html
from scripts.sparse_verify.verify import sha256


def review(polygons_path, output_dir):
    polygons_path = Path(polygons_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("review output directory must be fresh")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < (10 << 30)+(50 << 20):
        raise ValueError("need 10 GiB reserve plus 50 MiB output cap")
    manifest_path = FROZEN/"holdout.json"
    manifest = json.loads(manifest_path.read_text())
    image_manifest = manifest["images"]
    polygons = load_contract(polygons_path, {im["name"]: im for im in image_manifest})
    inputs = [polygons_path, manifest_path, *(Path(im["path"]) for im in image_manifest)]
    before = {str(path): sha256(path) for path in inputs}
    for im in image_manifest:
        if before[im["path"]] != im["sha256"]:
            raise ValueError(f"PNG hash mismatch: {im['name']}")
    page = overlay_html(polygons, image_manifest, output_dir, None)
    after = {str(path): sha256(path) for path in inputs}
    if before != after:
        raise ValueError("polygon input or PNG changed during overlay review")
    code_paths = [Path(__file__), Path(__file__).with_name("run.py"),
                  Path(__file__).with_name("polygon.py")]
    report = dict(schema="foreground_roi_overlay_review_v1",
                  validation="polygons and source PNG hashes passed; no support metrics computed",
                  image_count=len(polygons), polygon_count=sum(len(p) for p in polygons.values()),
                  input_hashes_before=before, input_hashes_after=after,
                  code_hashes={path.name: sha256(path) for path in code_paths})
    report_text = json.dumps(report, indent=2)+"\n"
    if len(page.encode())+len(report_text.encode()) > (50 << 20):
        raise ValueError("overlay review output exceeds 50 MiB")
    output_dir.mkdir()
    (output_dir/"overlays.html").write_text(page)
    (output_dir/"review.json").write_text(report_text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.polygons, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("input_hashes_before", "input_hashes_after")}, indent=2))


if __name__ == "__main__":
    main()
