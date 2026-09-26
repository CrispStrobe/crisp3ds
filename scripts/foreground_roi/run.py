#!/usr/bin/env python3
"""Read-only post hoc support audit for manual tree-envelope polygons."""

import argparse
import collections
import html
import json
import math
import os
from pathlib import Path
import shutil

import scripts.sparse_verify.verify as verify_math
from scripts.foreground_roi.polygon import in_roi, load_contract
from scripts.sparse_verify.verify import (
    cameras, images, pair_sampson, points, sha256, stats,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/"build-opencv/colmap-sparse/heldout-v2-001"
THREE_VIEW = ROOT/"build-opencv/three-view-audit/run-001/report.json"
DEFAULT_OUTPUT = ROOT/"build-opencv/foreground-roi/run-001"


def point_support(model_dir, image_manifest, masks):
    ims = images(model_dir/"images.txt")
    pts = points(model_dir/"points3D.txt")
    by_name = {item["name"]: item for item in image_manifest}
    summary = dict(point_count=len(pts), all_inside=0, any_inside=0,
                   majority_inside=0, exactly_half_inside=0,
                   no_inside=0, total_track_links=0, inside_track_links=0,
                   outside_track_links=0)
    per_point = []
    for pid, point in pts.items():
        inside = []
        for iid, compact_idx in point["track"]:
            im = ims[iid]
            mapping = by_name[im["name"]]["compact_to_original"]
            if compact_idx < 0 or compact_idx >= len(mapping):
                raise ValueError(f"model point {pid} has out-of-range mapper row")
            inside.append(masks[im["name"]][mapping[compact_idx]])
        count = sum(inside); n = len(inside)
        summary["total_track_links"] += n
        summary["inside_track_links"] += count
        summary["outside_track_links"] += n-count
        summary["all_inside"] += count == n
        summary["any_inside"] += count > 0
        summary["majority_inside"] += count*2 > n
        summary["exactly_half_inside"] += count*2 == n
        summary["no_inside"] += count == 0
        per_point.append(dict(point_id=pid, track_links=n, inside_links=count,
                              all_inside=count == n, majority_inside=count*2 > n))
    return summary, per_point


def pair_support(matches, masks):
    summary = dict(total=len(matches), both_inside=0, exactly_one_inside=0, neither_inside=0)
    for match in matches:
        a = masks[match["image1"]][match["feature1"]]
        b = masks[match["image2"]][match["feature2"]]
        summary["both_inside"] += a and b
        summary["exactly_one_inside"] += a != b
        summary["neither_inside"] += not a and not b
    return summary


def heldout_geometry(matches, masks, image_manifest, model_dir):
    cameras_by_id = cameras(model_dir/"cameras.txt")
    model_images = {im["name"]: im for im in images(model_dir/"images.txt").values()}
    input_images = {im["name"]: im for im in image_manifest}
    subsets = {key: dict(raw_count=0, unregistered_count=0,
                         numerical_invalid_count=0, values=[])
               for key in ("both_inside", "other")}
    for match in matches:
        a, b = match["image1"], match["image2"]
        key = "both_inside" if (masks[a][match["feature1"]] and
                                masks[b][match["feature2"]]) else "other"
        bucket = subsets[key]
        bucket["raw_count"] += 1
        if a not in model_images or b not in model_images:
            bucket["unregistered_count"] += 1
            continue
        ia, ib = model_images[a], model_images[b]
        ca, cb = cameras_by_id[ia["camera_id"]], cameras_by_id[ib["camera_id"]]
        xya = input_images[a]["features"][match["feature1"]]
        xyb = input_images[b]["features"][match["feature2"]]
        try:
            value = pair_sampson(ia, ib, ca, cb, xya, xyb)
        except (ValueError, OverflowError, ZeroDivisionError):
            value = None
        if value is None or not math.isfinite(value):
            bucket["numerical_invalid_count"] += 1
        bucket["values"].append(value)
    result = {}
    for key, bucket in subsets.items():
        score = stats(bucket.pop("values"))
        bucket["scored_geometry"] = score
        bucket["finite_over_4px_count"] = score["finite_count"]-score["within_4px"]
        bucket["missing_inclusive_over_4px_or_unavailable_count"] = (
            bucket["raw_count"]-score["within_4px"])
        result[key] = bucket
    return result


def cycle_support(cycles, masks):
    subsets = {"all_three_inside": [], "other": []}
    for cycle in cycles:
        key = "all_three_inside" if all(
            masks[ref["image"]][ref["feature_id"]] for ref in cycle["refs"]) else "other"
        subsets[key].append(cycle)
    result = {}
    for key, group in subsets.items():
        statuses = collections.Counter(c["lexical"]["status"] for c in group)
        residuals = [c["lexical"]["third_reprojection_px"]
                     for c in group if c["lexical"]["status"] == "scored"]
        result[key] = dict(cycle_count=len(group), lexical_status_counts=dict(statuses),
                           scored_third_reprojection=stats(residuals),
                           cycle_ids=[c["id"] for c in group])
    return result


def overlay_html(polygons, image_manifest, output_dir, counts):
    cards = []
    for image in image_manifest:
        name = image["name"]
        src = html.escape(os.path.relpath(image["path"], output_dir), quote=True)
        shapes = []
        for polygon in polygons[name]:
            vertices = " ".join(f"{x:g},{y:g}" for x, y in polygon)
            shapes.append(f'<polygon points="{vertices}" fill="#11cc55" fill-opacity="0.22" '
                          'stroke="#00ff66" stroke-width="2" vector-effect="non-scaling-stroke"/>')
        label = html.escape(name)
        svg = (f'<svg viewBox="0 0 768 512" role="img" aria-label="Manual ROI on {label}">'
               f'<image href="{src}" x="0" y="0" width="768" height="512"/>'
               +"".join(shapes)+"</svg>")
        if counts is None:
            support_label = "Support metrics deferred until overlay review."
        else:
            details = counts[name]
            support_label = (f'Inside: training {details["training_inside"]}/{details["training_count"]}; '
                             f'held-out {details["heldout_inside"]}/{details["heldout_count"]}')
        cards.append(f'<article><h2>{label}</h2><p>{support_label}</p>'
                     '<details><summary>Open full-size overlay</summary>'
                     +svg+'</details><div class="thumb">'+svg+'</div></article>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Manual foreground ROI overview</title>'
            '<style>body{font:14px system-ui;margin:24px;color:#172027}header{max-width:950px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:18px}'
            'article{border:1px solid #bbb;padding:12px}article h2{margin:0}svg{max-width:100%;height:auto}'
            '.thumb svg{width:384px}details svg{width:768px}</style><header><h1>Manual target-tree envelopes</h1>'
            '<p>Green polygon is a human-supplied approximate envelope. Gaps may contain background. '
            'A feature centre inside does not mean its descriptor excludes background. '
            'These polygons are not ground truth or automatic segmentation.</p></header><main class="grid">'
            +"".join(cards)+'</main></html>')


def audit(polygon_path, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    polygon_path = Path(polygon_path).resolve(strict=True)
    if output_dir.exists():
        raise ValueError("output directory must be fresh")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < (10 << 30)+(50 << 20):
        raise ValueError("need 10 GiB reserve plus 50 MiB output cap")
    manifest_path = FROZEN/"holdout.json"
    verifier_path = FROZEN/"verification-supervisor.json"
    model_dir = FROZEN/"model_text"
    model_paths = [model_dir/name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    inputs = [polygon_path, manifest_path, verifier_path, THREE_VIEW, *model_paths]
    manifest = json.loads(manifest_path.read_text())
    verifier = json.loads(verifier_path.read_text())
    three_view = json.loads(THREE_VIEW.read_text())
    code_paths = [Path(__file__), Path(__file__).with_name("polygon.py"),
                  Path(verify_math.__file__)]
    code_before = {path.name: sha256(path) for path in code_paths}
    if verifier["integrity_status"] != "pass" or not verifier["heldout_valid"]:
        raise ValueError("frozen observation holdout is not integrity-valid")
    by_name = {im["name"]: im for im in manifest["images"]}
    polygons = load_contract(polygon_path, by_name)
    inputs += [Path(im["path"]) for im in manifest["images"]]
    hashes_before = {str(path): sha256(path) for path in inputs}
    if hashes_before[str(manifest_path)] != verifier["hashes"]["manifest"]:
        raise ValueError("manifest differs from frozen verifier")
    if hashes_before[str(manifest_path)] != three_view["source_hashes_before"][str(manifest_path)]:
        raise ValueError("three-view audit used a different holdout manifest")
    for path in model_paths:
        if hashes_before[str(path)] != verifier["hashes"]["model"][path.name]:
            raise ValueError(f"model differs from frozen verifier: {path.name}")
        if hashes_before[str(path)] != three_view["source_hashes_before"][str(path)]:
            raise ValueError(f"three-view audit used a different model: {path.name}")
    for path in (manifest_path, *model_paths):
        if hashes_before[str(path)] != three_view["source_hashes_after"][str(path)]:
            raise ValueError(f"three-view input changed during its audit: {path.name}")
    if code_before[Path(verify_math.__file__).name] != verifier["hashes"]["verifier"]:
        raise ValueError("imported camera math differs from frozen verifier")
    for image in manifest["images"]:
        if hashes_before[image["path"]] != image["sha256"]:
            raise ValueError(f"PNG changed: {image['name']}")
    masks = {im["name"]: tuple(in_roi(polygons[im["name"]], xy[:2]) for xy in im["features"])
             for im in manifest["images"]}
    image_counts = {}
    for im in manifest["images"]:
        mask = masks[im["name"]]
        image_counts[im["name"]] = dict(
            training_count=len(im["compact_to_original"]),
            training_inside=sum(mask[idx] for idx in im["compact_to_original"]),
            heldout_count=len(im["heldout_ids"]),
            heldout_inside=sum(mask[idx] for idx in im["heldout_ids"]))
    tracks, per_point = point_support(model_dir, manifest["images"], masks)
    training_pairs = pair_support(manifest["training_matches"], masks)
    heldout_pairs = pair_support(manifest["heldout_pairs"], masks)
    geometry = heldout_geometry(manifest["heldout_pairs"], masks, manifest["images"], model_dir)
    cycles = cycle_support(three_view["cycles"], masks)
    result = dict(schema="foreground_roi_support_audit_v1", polygon_schema="foreground_roi_v1",
                  interpretation="post hoc development support only; approximate manual envelope",
                  image_feature_counts=image_counts, training_pairs=training_pairs,
                  raw_heldout_pairs=heldout_pairs, model_tracks=tracks,
                  per_point_support=per_point,
                  scored_heldout_pair_geometry_by_roi=geometry,
                  three_view_cycles_by_roi=cycles,
                  frozen_full_set=dict(heldout=verifier["heldout"],
                                       three_view=three_view["lexical_third_reprojection"]),
                  input_hashes_before=hashes_before,
                  input_hashes_after={str(path): sha256(path) for path in inputs},
                  code_hashes_before=code_before,
                  code_hashes_after={path.name: sha256(path) for path in code_paths})
    if result["input_hashes_before"] != result["input_hashes_after"]:
        raise ValueError("source changed during ROI audit")
    if result["code_hashes_before"] != result["code_hashes_after"]:
        raise ValueError("audit code changed during ROI audit")
    overview = overlay_html(polygons, manifest["images"], output_dir, image_counts)
    report_text = json.dumps(result, indent=2)+"\n"
    if len(report_text.encode())+len(overview.encode()) > (50 << 20):
        raise ValueError("ROI output exceeds 50 MiB")
    output_dir.mkdir()
    (output_dir/"report.json").write_text(report_text)
    (output_dir/"overlays.html").write_text(overview)
    return {key: value for key, value in result.items()
            if key not in ("per_point_support", "input_hashes_before", "input_hashes_after",
                           "image_feature_counts", "frozen_full_set")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.polygons, args.output), indent=2))


if __name__ == "__main__":
    main()
