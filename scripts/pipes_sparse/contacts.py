#!/usr/bin/env python3
"""Read-only photo contacts for three frozen sparse outliers."""

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
POINTS = ROOT / "build-opencv/pipes-sparse/run-003/points.json"
REPORT = ROOT / "build-opencv/pipes-sparse/run-003/report.json"
SCORE = ROOT / "build-opencv/pipes-score/run-002/score.json"
OUTPUT = ROOT / "build-opencv/pipes-sparse/contacts-001"
IDS = (212, 104, 214)
PHOTO_PREFIX = ROOT / "build-opencv/pipes-prepare/run-001/pipes/images/dslr_images_undistorted"
CROP_WIDTH, CROP_HEIGHT, DISPLAY_SCALE = 960, 640, .625


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_xy(resized_xy, scales):
    if len(resized_xy) != 2 or len(scales) != 2 or any(
            not math.isfinite(float(v)) for v in (*resized_xy, *scales)) or min(scales) <= 0:
        raise ValueError("invalid coordinate or resize scales")
    return [resized_xy[0] / scales[0], resized_xy[1] / scales[1]]


def photo_url(page, photo):
    relative = os.path.relpath(photo, page.parent).replace(os.sep, "/")
    return quote(relative, safe="/")


def render_card(point, observation, distance, view, page):
    name = observation["image"]
    photo = PHOTO_PREFIX / name
    if photo != Path(view["path"]) or photo.is_symlink() or not photo.is_file():
        raise ValueError("photo path differs from frozen report")
    if digest(photo) != view["sha256"]:
        raise ValueError("source photo hash mismatch")
    x, y = source_xy(observation["xy_edge_frame"], view["resize_scale_xy"])
    width, height = view["source_camera"]["width"], view["source_camera"]["height"]
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError("observation outside source image")
    # Original JPEG pixels are shown at a fixed CSS scale. Clipping yields a
    # 960x640-original-pixel contact centered exactly on the observed point.
    css_w, css_h = width * DISPLAY_SCALE, height * DISPLAY_SCALE
    frame_w, frame_h = CROP_WIDTH * DISPLAY_SCALE, CROP_HEIGHT * DISPLAY_SCALE
    left, top = frame_w / 2 - x * DISPLAY_SCALE, frame_h / 2 - y * DISPLAY_SCALE
    title = (f"Point {point['id']} · {name} · laser proximity {distance:.3f} m · "
             f"parallax {point['max_ray_parallax_deg']:.2f}° · "
             f"reprojection {observation['reprojection_px']:.2f} px")
    return (f'<article><h2>{html.escape(title)}</h2>'
            f'<div class="crop" style="width:{frame_w:.0f}px;height:{frame_h:.0f}px">'
            f'<img src="{html.escape(photo_url(page, photo), quote=True)}" '
            f'alt="{html.escape(name, quote=True)} source photograph" '
            f'style="width:{css_w:.3f}px;height:{css_h:.3f}px;'
            f'left:{left:.3f}px;top:{top:.3f}px">'
            f'<span class="cross" style="left:{frame_w/2:.3f}px;top:{frame_h/2:.3f}px" '
            f'aria-label="observed SIFT feature"></span></div>'
            f'<p>Observed source-photo pixel-edge coordinate ({x:.2f}, {y:.2f}); '
            f'feature {observation["original_feature_id"]}. Cross marks the recorded observation.</p></article>')


def build(output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("contact output must be fresh with existing parent")
    points_data = json.loads(POINTS.read_text())
    report = json.loads(REPORT.read_text())
    score = json.loads(SCORE.read_text())
    if points_data.get("schema") != "fixed_camera_sparse_points_v1" or score.get("status") != "pass":
        raise ValueError("unexpected frozen result schema/status")
    hashes = {str(p): digest(p) for p in (POINTS, REPORT, SCORE, Path(__file__))}
    for path in (POINTS, REPORT):
        if score["input_sha256"].get(str(path)) != hashes[str(path)]:
            raise ValueError("score did not use these frozen sparse inputs")
    by_id = {p["id"]: p for p in points_data["points"]}
    scored = sorted(score["point_distances_m"], key=lambda r: r["distance_m"], reverse=True)
    if [r["id"] for r in scored[:3]] != list(IDS):
        raise ValueError("requested IDs are not exact three worst score rows")
    by_view = {v["name"]: v for v in report["views"]}
    page = output / "index.html"
    cards = []
    rows = []
    for ranked in scored[:3]:
        point = by_id[ranked["id"]]
        if len(point["observations"]) != 2:
            raise ValueError("expected two recorded observations per point")
        for observation in point["observations"]:
            view = by_view[observation["image"]]
            cards.append(render_card(point, observation, ranked["distance_m"], view, page))
            photo = PHOTO_PREFIX / observation["image"]
            hashes[str(photo)] = digest(photo)
            rows.append({"point_id": point["id"], "image": observation["image"],
                         "source_xy_edge": source_xy(observation["xy_edge_frame"],
                                                     view["resize_scale_xy"]),
                         "distance_m": ranked["distance_m"]})
    body = "\n".join(cards)
    page_html = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                 "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                 "<title>Three sparse outlier photo contacts</title><style>"
                 "body{font:15px system-ui,sans-serif;background:#111;color:#eee;margin:24px}"
                 "h1{font-size:24px}p{line-height:1.4;color:#ccc}"
                 ".grid{display:grid;grid-template-columns:repeat(2,max-content);gap:24px 18px}"
                 "article{max-width:600px;background:#242424;padding:12px;border-radius:8px}"
                 "h2{font-size:15px;line-height:1.4;margin:0 0 10px}"
                 ".crop{position:relative;overflow:hidden;background:#333}"
                 ".crop img{position:absolute;max-width:none}"
                 ".cross{position:absolute;display:block;width:34px;height:34px;"
                 "transform:translate(-50%,-50%);border:2px solid #ff3333;border-radius:50%;"
                 "box-shadow:0 0 0 2px #fff,0 0 6px #000}"
                 ".cross:before,.cross:after{content:'';position:absolute;background:#ff3333}"
                 ".cross:before{left:15px;top:-10px;width:2px;height:54px}"
                 ".cross:after{left:-10px;top:15px;width:54px;height:2px}"
                 "</style><h1>Three highest sparse-to-laser distances</h1>"
                 "<p>Each crop uses the unchanged original photograph. A red cross marks the saved "
                 "SIFT observation. Laser distance is a one-way proximity diagnostic; the photos "
                 "alone do not establish a surface correspondence or cause of the error.</p>"
                 f'<main class="grid">{body}</main></html>\n')
    if len(page_html.encode()) > 256 * 1024:
        raise ValueError("contact HTML exceeds bound")
    output.mkdir()
    page.write_text(page_html)
    manifest = {"schema": "pipes_sparse_contacts_v1", "point_ids": list(IDS),
                "crop_source_pixels": [CROP_WIDTH, CROP_HEIGHT],
                "display_scale": DISPLAY_SCALE, "input_sha256": hashes,
                "index_html_sha256": digest(page), "observations": rows,
                "interpretation": "visual review only; no feature moves or truth labels"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output.absolute())
    print(json.dumps({"output": str(args.output.absolute()), "point_ids": manifest["point_ids"],
                      "html_sha256": manifest["index_html_sha256"]}))


if __name__ == "__main__":
    main()
