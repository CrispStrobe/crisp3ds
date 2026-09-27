"""Read OpenMVS v2.4.0 raw DMAPs and prove native ignore-mask-label effect.

The DMAP header/layout follows libs/MVS/Interface.h HeaderDepthDataRaw. This
is diagnostic only: it never edits depth maps or reconstruction inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from scripts.classical_backend.dense_masks import cv2, digest, native_mask_name


HEADER = struct.Struct("<HBBIIIIff")


def load_depth(path: Path) -> tuple[str, np.ndarray]:
    data = path.read_bytes()
    if len(data) < HEADER.size + 2:
        raise ValueError(f"truncated DMAP header: {path}")
    magic, flags, _, image_width, image_height, width, height, dmin, dmax = HEADER.unpack_from(data)
    if (magic != 0x5244 or not flags & 1 or not 0 < width <= 8192 or not 0 < height <= 8192 or
            not 0 < image_width <= 8192 or not 0 < image_height <= 8192 or
            not np.isfinite(dmin) or not np.isfinite(dmax)):
        raise ValueError(f"invalid DMAP header: {path}")
    offset = HEADER.size
    name_length = struct.unpack_from("<H", data, offset)[0]
    offset += 2
    if not 0 < name_length < 4096 or offset + name_length + 4 > len(data):
        raise ValueError(f"invalid DMAP image name: {path}")
    name = data[offset:offset + name_length].decode("utf-8")
    offset += name_length
    id_count = struct.unpack_from("<I", data, offset)[0]
    if not 1 <= id_count <= 200:
        raise ValueError(f"invalid DMAP view count: {path}")
    offset += 4 + id_count * 4 + 21 * 8  # view IDs; K, R, C are 21 float64s.
    if offset + width * height * 4 > len(data):
        raise ValueError(f"truncated DMAP depth payload: {path}")
    depth = np.frombuffer(data, dtype="<f4", count=width * height, offset=offset).reshape(height, width)
    if not np.all(np.isfinite(depth)) or np.any(depth < 0):
        raise ValueError(f"nonfinite or negative DMAP depth: {path}")
    return name, depth


def depth_files(folder: Path) -> dict[str, tuple[Path, np.ndarray]]:
    result = {}
    for path in sorted(folder.glob("depth[0-9][0-9][0-9][0-9].dmap")):
        image_path, depth = load_depth(path)
        name = Path(image_path).name
        if name in result or not image_path.endswith("/" + name):
            raise ValueError(f"duplicate or invalid DMAP image name: {path}")
        result[name] = path, depth
    return result


def verify(baseline: Path, masked: Path, masks: Path) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV required to reproduce OpenMVS nearest-neighbor mask scaling")
    mask_report_path = masks / "report.json"
    mask_report = json.loads(mask_report_path.read_text())
    if mask_report.get("status") != "complete" or mask_report.get("native_filename_basis") != "image stem":
        raise ValueError("mask report is incomplete or uses wrong native filename convention")
    old = depth_files(baseline)
    new = depth_files(masked)
    names = {item["name"] for item in mask_report["images"]}
    if old.keys() != new.keys() or old.keys() != names:
        raise ValueError("baseline/masked DMAP views and native mask report must match exactly")
    rows = []
    for item in mask_report["images"]:
        name = item["name"]
        old_path, old_depth = old[name]
        new_path, new_depth = new[name]
        if old_depth.shape != new_depth.shape:
            raise ValueError(f"baseline/masked DMAP dimensions differ: {name}")
        mask_path = masks / native_mask_name(name)
        if digest(mask_path) != item["mask_sha256"]:
            raise ValueError(f"native mask changed: {name}")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        if mask.shape[::-1] != tuple(item["undistorted_size"]):
            raise ValueError(f"mask dimensions differ from undistorted camera: {name}")
        scaled = cv2.resize(mask, (new_depth.shape[1], new_depth.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        old_valid = old_depth > 0
        new_valid = new_depth > 0
        rows.append({"name": name,
                     "baseline_dmap_sha256": digest(old_path), "masked_dmap_sha256": digest(new_path),
                     "depth_size": [new_depth.shape[1], new_depth.shape[0]],
                     "baseline_positive_inside": int(np.count_nonzero(old_valid & (scaled != 0))),
                     "baseline_positive_outside": int(np.count_nonzero(old_valid & (scaled == 0))),
                     "masked_positive_inside": int(np.count_nonzero(new_valid & (scaled != 0))),
                     "masked_positive_outside": int(np.count_nonzero(new_valid & (scaled == 0)))})
    totals = {key: sum(row[key] for row in rows) for key in
              ("baseline_positive_inside", "baseline_positive_outside",
               "masked_positive_inside", "masked_positive_outside")}
    return {"schema": "classical_native_mask_effect_v1", "status": "complete",
            "baseline": str(baseline.resolve()), "masked": str(masked.resolve()),
            "mask_report_sha256": digest(mask_report_path), "views": len(rows),
            "totals": totals, "all_masked_depth_within_mask": totals["masked_positive_outside"] == 0,
            "images": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--masked", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("proof report output must be fresh")
    report = verify(args.baseline, args.masked, args.masks)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"views": report["views"], "totals": report["totals"],
                      "all_masked_depth_within_mask": report["all_masked_depth_within_mask"]}))
    return 0 if report["all_masked_depth_within_mask"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
