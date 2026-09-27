"""Read-only photo-mask QA for one future 60-view cracker silhouette set.

The command reads RGBs, the sealed coarse masks, independently annotated PNGs,
and 12 repeat annotations. It writes one JSON report to stdout only. It never
opens cameras, scanner geometry, depth, or reconstruction outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in range(0, 360, 6))
REPEAT_NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in range(0, 360, 30))
SEALED_REPORT_SHA256 = "dc5ff13e84f66b810709a85aef41b6459abb6b9f78f99996e632862572923fef"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_name(name: str) -> str:
    if not name.endswith(".jpg") or Path(name).name != name:
        raise ValueError(f"invalid image name: {name}")
    return name[:-4] + ".mask.png"


def exact_files(directory: Path, names: set[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"missing or linked directory: {directory}")
    paths = list(directory.iterdir())
    if {p.name for p in paths} != names or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError(f"unexpected, missing, or linked file in {directory}")


def read_rgb(path: Path, expected_size: tuple[int, int], expected_hash: str) -> None:
    if digest(path) != expected_hash:
        raise ValueError(f"sealed RGB hash differs: {path.name}")
    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != expected_size:
            raise ValueError(f"sealed RGB format or dimensions differ: {path.name}")
        image.load()


def read_mask(path: Path, size: tuple[int, int], expected_hash: str | None = None) -> tuple[np.ndarray, str]:
    sha = digest(path)
    if expected_hash is not None and sha != expected_hash:
        raise ValueError(f"sealed mask hash differs: {path.name}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != size:
            raise ValueError(f"mask must be native-size 8-bit grayscale PNG: {path.name}")
        pixels = np.asarray(image).copy()
    if not np.all((pixels == 0) | (pixels == 255)):
        raise ValueError(f"mask must contain only 0 and 255: {path.name}")
    return pixels == 255, sha


def mask_sanity(mask: np.ndarray) -> dict:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask_sanity requires a 2D boolean array")
    area = int(mask.sum())
    border = int(mask[0, :].sum() + mask[-1, :].sum() +
                 mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    background_count, background_labels = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=8)
    edge_labels = np.unique(np.concatenate((background_labels[0], background_labels[-1],
                                            background_labels[:, 0], background_labels[:, -1])))
    holes = int(len(set(range(1, background_count)) - set(map(int, edge_labels))))
    return {"foreground_pixels": area, "components": int(count - 1),
            "border_pixels": border, "holes": holes,
            "passes": area > 0 and count == 2 and border == 0 and holes == 0}


def boundary(mask: np.ndarray) -> np.ndarray:
    source = mask.astype(np.uint8)
    eroded = cv2.erode(source, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT,
                       borderValue=0)
    return (source != 0) & (eroded == 0)


def boundary_distances(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact Euclidean pixel-center distance, directed a->b and b->a."""
    if a.shape != b.shape or not a.any() or not b.any():
        raise ValueError("boundary comparison requires two nonempty equal-size masks")
    first, second = boundary(a), boundary(b)

    def directed(source: np.ndarray, target: np.ndarray) -> dict:
        field = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        values = field[source]
        return {"median_px": float(np.median(values)), "p95_px": float(np.percentile(values, 95)),
                "max_px": float(values.max())}

    return {"a_to_b": directed(first, second), "b_to_a": directed(second, first)}


def compare(old: np.ndarray, reviewed: np.ndarray) -> dict:
    if old.shape != reviewed.shape or not old.any() or not reviewed.any():
        raise ValueError("comparison requires two nonempty equal-size masks")
    area = int(reviewed.sum())
    intersection = int(np.count_nonzero(old & reviewed))
    union = int(np.count_nonzero(old | reviewed))
    return {"old_excess_fraction_of_reviewed": int(np.count_nonzero(old & ~reviewed)) / area,
            "old_omission_fraction_of_reviewed": int(np.count_nonzero(reviewed & ~old)) / area,
            "iou": intersection / union,
            "boundary_px": boundary_distances(old, reviewed)}


def distribution(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(data)), "p95": float(np.percentile(data, 95)),
            "worst": float(data.max())}


def validate_report(report: dict, expected_names: tuple[str, ...]) -> dict[str, dict]:
    entries = report.get("images")
    if (report.get("schema") != "classical_dense_masks_v1" or report.get("status") != "complete" or
            report.get("semantics") != "photo-derived coarse pose support, not a silhouette or ground truth" or
            not isinstance(entries, list) or len(entries) != len(expected_names)):
        raise ValueError("malformed coarse-mask report")
    rows = {row.get("name"): row for row in entries if isinstance(row, dict)}
    if len(rows) != len(expected_names) or set(rows) != set(expected_names):
        raise ValueError("coarse-mask report name inventory differs")
    for name, row in rows.items():
        size = row.get("undistorted_size")
        if (row.get("native_mask_name") != mask_name(name) or
                not isinstance(size, list) or len(size) != 2 or
                any(type(x) is not int or x <= 0 for x in size) or
                any(not isinstance(row.get(k), str) or len(row[k]) != 64 or
                    any(c not in "0123456789abcdef" for c in row[k])
                    for k in ("undistorted_image_sha256", "mask_sha256"))):
            raise ValueError(f"malformed coarse-mask report entry: {name}")
    return rows


def audit(source: Path, reviewed_dir: Path, repeat_dir: Path,
          *, expected_names: tuple[str, ...] = NAMES,
          repeat_names: tuple[str, ...] = REPEAT_NAMES,
          expected_report_hash: str | None = SEALED_REPORT_SHA256) -> dict:
    """Audit exact named inputs without writing; injectable inventories support synthetic tests."""
    report_path = source / "masks" / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("missing or linked coarse-mask report")
    report_hash = digest(report_path)
    if expected_report_hash is not None and report_hash != expected_report_hash:
        raise ValueError("sealed coarse-mask report hash differs")
    rows = validate_report(json.loads(report_path.read_text()), expected_names)
    if (not repeat_names or len(set(repeat_names)) != len(repeat_names) or
            not set(repeat_names) <= set(expected_names)):
        raise ValueError("repeat names must be a distinct subset of image names")
    rgb_dir, old_dir = source / "dense" / "images", source / "masks"
    exact_files(rgb_dir, set(expected_names))
    exact_files(old_dir, {"report.json"} | {mask_name(n) for n in expected_names})
    exact_files(reviewed_dir, {mask_name(n) for n in expected_names})
    exact_files(repeat_dir, {mask_name(n) for n in repeat_names})
    result = {"schema": "cracker_mask_qa_v1", "source_report_sha256": report_hash,
              "image_count": len(expected_names), "repeat_count": len(repeat_names), "images": []}
    initial_hashes = {report_path: report_hash}
    for name in expected_names:
        row = rows[name]
        size = tuple(row["undistorted_size"])
        rgb_path, old_path = rgb_dir / name, old_dir / mask_name(name)
        reviewed_path = reviewed_dir / mask_name(name)
        read_rgb(rgb_path, size, row["undistorted_image_sha256"])
        old, old_sha = read_mask(old_path, size, row["mask_sha256"])
        reviewed, reviewed_sha = read_mask(reviewed_path, size)
        initial_hashes.update({rgb_path: row["undistorted_image_sha256"], old_path: old_sha,
                               reviewed_path: reviewed_sha})
        sanity = mask_sanity(reviewed)
        entry = {"name": name, "rgb_sha256": row["undistorted_image_sha256"],
                 "reviewed_mask_sha256": reviewed_sha, "reviewed_sanity": sanity,
                 "old_vs_reviewed": compare(old, reviewed)}
        if name in repeat_names:
            repeat_path = repeat_dir / mask_name(name)
            repeat, repeat_sha = read_mask(repeat_path, size)
            initial_hashes[repeat_path] = repeat_sha
            distances = boundary_distances(reviewed, repeat)
            entry["repeat"] = {"mask_sha256": repeat_sha, "sanity": mask_sanity(repeat),
                               "boundary_px": distances,
                               "passes_3px_p95": max(v["p95_px"] for v in distances.values()) <= 3.0}
        result["images"].append(entry)
    for path, sha in initial_hashes.items():
        if digest(path) != sha:
            raise ValueError(f"input changed during audit: {path.name}")
    entries = result["images"]
    repeats = [e for e in entries if "repeat" in e]
    result["summary"] = {
        "old_excess_fraction_of_reviewed": distribution([
            e["old_vs_reviewed"]["old_excess_fraction_of_reviewed"] for e in entries]),
        "old_omission_fraction_of_reviewed": distribution([
            e["old_vs_reviewed"]["old_omission_fraction_of_reviewed"] for e in entries]),
        "old_new_iou": distribution([e["old_vs_reviewed"]["iou"] for e in entries]),
        "old_to_reviewed_boundary_p95_px": distribution([
            e["old_vs_reviewed"]["boundary_px"]["a_to_b"]["p95_px"] for e in entries]),
        "reviewed_to_old_boundary_p95_px": distribution([
            e["old_vs_reviewed"]["boundary_px"]["b_to_a"]["p95_px"] for e in entries]),
        "repeat_worst_direction_p95_px": distribution([
            max(v["p95_px"] for v in e["repeat"]["boundary_px"].values()) for e in repeats]),
        "failed_view_names": [e["name"] for e in entries if not e["reviewed_sanity"]["passes"] or
                              ("repeat" in e and not (e["repeat"]["sanity"]["passes"] and
                                                      e["repeat"]["passes_3px_p95"]))],
    }
    result["status"] = ("pass" if all(e["reviewed_sanity"]["passes"] and
                             ("repeat" not in e or
                              (e["repeat"]["sanity"]["passes"] and e["repeat"]["passes_3px_p95"]))
                             for e in result["images"]) else "abstain")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="sealed producer root")
    parser.add_argument("--reviewed", type=Path, required=True, help="exact 60 reviewed PNGs")
    parser.add_argument("--repeat", type=Path, required=True, help="exact 12 independent PNGs")
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.reviewed, args.repeat), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
