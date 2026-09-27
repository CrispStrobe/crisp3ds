#!/usr/bin/env python3
"""Training-photo-only contact sheets and manually reviewed coarse pose masks.

No automatic object segmentation is inferred from a single preview. This module
never opens held-out photos, supplied masks/depth/poses, or Google reference meshes.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw


OBJECTS = {"006_mustard_bottle": (350, 220, 900, 800),
           "035_power_drill": (350, 230, 1000, 780)}
PACKAGE_SHA256 = {"006_mustard_bottle": "45b6ed1430627a747eaeb8d4dff7d2c49502083cca0a7914e25aacd45b594425",
                  "035_power_drill": "65cae1f7b7b75bbe2e98b015937627864d1705c8b743211345033533e4fd7f3f"}
NAME = re.compile(r"NP3_(\d{3})\.jpg\Z")
MAX_PHOTO_BYTES = 5_000_000
MAX_TOTAL_PHOTO_BYTES = 150_000_000
MIN_FREE_BYTES = 10 * 1024 ** 3
MAX_OUTPUT_BYTES = 20 * 1024 ** 2
MAX_ARTIFACT_BYTES = MAX_OUTPUT_BYTES - 1024 ** 2
IMAGE_SIZE = (1280, 1024)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def training_records(package):
    if package.get("schema") != "ycb_object_evaluation_package_v1" or package.get("object_id") not in OBJECTS:
        raise ValueError("unsupported frozen YCB evaluation package")
    rows = package.get("training_inputs")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("exactly 48 frozen training photos required")
    expected = [angle for angle in range(0, 360, 6) if (angle // 6) % 5 != 4]
    if [row.get("angle_degrees") for row in rows] != expected:
        raise ValueError("training split or order changed")
    total = 0
    for row, angle in zip(rows, expected):
        name = f"NP3_{angle:03}.jpg"
        if row.get("path") != f"photos/{name}" or not NAME.fullmatch(name):
            raise ValueError("unexpected training image path")
        size, sha = row.get("bytes"), row.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_PHOTO_BYTES:
            raise ValueError("training photo size out of bounds")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("invalid training photo SHA-256")
        total += size
    if total > MAX_TOTAL_PHOTO_BYTES:
        raise ValueError("training photos exceed byte cap")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate training photo SHA-256")
    return rows


def verified_training_photo(root, row):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset root must be a real directory")
    photo_dir = root / "photos"
    if photo_dir.is_symlink() or not photo_dir.is_dir():
        raise ValueError("photos directory must be real")
    photo = photo_dir / Path(row["path"]).name
    if photo.is_symlink() or not photo.is_file():
        raise ValueError("training photo must be a regular file")
    if photo.stat().st_size != row["bytes"] or digest(photo) != row["sha256"]:
        raise ValueError(f"training photo hash/size mismatch: {photo.name}")
    with Image.open(photo) as image:
        if image.size != IMAGE_SIZE:
            raise ValueError("training photo dimensions differ from 1280x1024")
        image.load()
        result = image.convert("RGB")
    if photo.stat().st_size != row["bytes"] or digest(photo) != row["sha256"]:
        raise ValueError(f"training photo changed during decode: {photo.name}")
    return result


def polygon(points, width, height):
    if not isinstance(points, list) or not 3 <= len(points) <= 64:
        raise ValueError("polygon needs 3-64 vertices")
    clean = []
    for point in points:
        if (not isinstance(point, list) or len(point) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool) for value in point)):
            raise ValueError("polygon vertices must be integer [x,y]")
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("polygon vertex outside photo")
        clean.append((x, y))
    twice_area = abs(sum(clean[i][0] * clean[(i + 1) % len(clean)][1] -
                         clean[(i + 1) % len(clean)][0] * clean[i][1]
                         for i in range(len(clean))))
    if twice_area < 100:
        raise ValueError("degenerate or tiny polygon")
    return clean


def render_manual(region, size=IMAGE_SIZE):
    if set(region) != {"include", "exclude"}:
        raise ValueError("manual region must have include and exclude polygon lists")
    includes, excludes = region["include"], region["exclude"]
    if (not isinstance(includes, list) or not 1 <= len(includes) <= 8 or
            not isinstance(excludes, list) or len(excludes) > 8):
        raise ValueError("manual polygon count out of bounds")
    mask = Image.new("L", size, 0)
    drawing = ImageDraw.Draw(mask)
    for vertices in includes:
        drawing.polygon(polygon(vertices, *size), fill=255)
    for vertices in excludes:
        drawing.polygon(polygon(vertices, *size), fill=0)
    pixels = mask.histogram()[255]
    fraction = pixels / (size[0] * size[1])
    if not 0.005 <= fraction <= 0.40:
        raise ValueError(f"manual support area outside 0.5%-40%: {fraction:.4f}")
    return mask, {"support_pixels": pixels, "support_fraction": fraction}


def load_package(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("package must be a small regular JSON file")
    package = json.loads(path.read_text())
    training_records(package)
    if digest(path) != PACKAGE_SHA256[package["object_id"]]:
        raise ValueError("evaluation package differs from pinned VPS-verified report SHA-256")
    return package


def save_bounded_image(image, path, format, used, **options):
    encoded = BytesIO()
    image.save(encoded, format=format, **options)
    payload = encoded.getvalue()
    if used + len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("generated image/mask artifacts exceed 19 MiB reserve")
    path.write_bytes(payload)
    return used + len(payload)


def prepare(package_path, dataset_root, output, *, regions_path=None):
    package_sha_before = digest(package_path)
    package = load_package(package_path)
    if digest(package_path) != package_sha_before:
        raise ValueError("evaluation package changed during validation")
    rows = training_records(package)
    runner_sha_before = digest(__file__)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 10 GiB free at output")
    regions = None
    regions_sha_before = None
    if regions_path is not None:
        path = Path(regions_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("manual ROI configuration must be small regular JSON")
        regions_sha_before = digest(path)
        regions = json.loads(path.read_text())
        if (regions.get("schema") != "ycb_manual_pose_regions_v1" or
                regions.get("object_id") != package["object_id"] or
                set(regions.get("images", {})) != {Path(row["path"]).name for row in rows}):
            raise ValueError("manual ROI configuration must exactly cover 48 training names")
        if digest(path) != regions_sha_before:
            raise ValueError("manual ROI configuration changed during validation")
    output.mkdir()
    result = {"schema": "ycb_object_photo_support_v1", "status": "running",
              "object_id": package["object_id"], "source_package_sha256": package_sha_before,
              "source_roi_sha256": regions_sha_before, "runner_sha256": runner_sha_before,
              "source_scope": "48 training RGB photos only; no held-out/reference/pose/depth files opened",
              "mask_role": "coarse manually reviewed pose support, not exact silhouette or mesh mask",
              "mask_status": "generated_unreviewed" if regions else "not_generated_pending_manual_review",
              "images": []}
    try:
        used = 0
        crop = OBJECTS[package["object_id"]]
        tile = (200, 210)
        sheet = Image.new("RGB", (8 * tile[0], 6 * tile[1]), "white")
        full_sheet = Image.new("RGB", sheet.size, "white")
        overlay_sheet = Image.new("RGB", sheet.size, "white") if regions else None
        draw = ImageDraw.Draw(sheet)
        full_draw = ImageDraw.Draw(full_sheet)
        overlay_draw = ImageDraw.Draw(overlay_sheet) if regions else None
        mask_dir = output / "masks"
        if regions:
            mask_dir.mkdir()
        for index, row in enumerate(rows):
            name = Path(row["path"]).name
            image = verified_training_photo(dataset_root, row)
            preview = image.crop(crop)
            preview.thumbnail((tile[0] - 8, tile[1] - 28), Image.Resampling.LANCZOS)
            x = (index % 8) * tile[0]
            y = (index // 8) * tile[1]
            sheet.paste(preview, (x + (tile[0] - preview.width) // 2, y + 2))
            draw.text((x + 4, y + tile[1] - 22), name, fill="black")
            full_preview = image.copy()
            full_preview.thumbnail((tile[0] - 8, tile[1] - 28), Image.Resampling.LANCZOS)
            full_sheet.paste(full_preview, (x + (tile[0] - full_preview.width) // 2, y + 2))
            full_draw.text((x + 4, y + tile[1] - 22), name, fill="black")
            entry = {"name": name, "image_sha256": row["sha256"]}
            if regions:
                mask, diagnostics = render_manual(regions["images"][name])
                destination = mask_dir / f"{name}.png"
                used = save_bounded_image(mask, destination, "PNG", used)
                entry.update({"mask_sha256": digest(destination), **diagnostics})
                overlay = image.copy()
                overlay.paste(Image.new("RGB", image.size, (255, 0, 0)), (0, 0),
                              mask.point(lambda value: value // 3))
                overlay = overlay.crop(crop)
                overlay.thumbnail((tile[0] - 8, tile[1] - 28), Image.Resampling.LANCZOS)
                overlay_sheet.paste(overlay, (x + (tile[0] - overlay.width) // 2, y + 2))
                overlay_draw.text((x + 4, y + tile[1] - 22), name, fill="black")
            result["images"].append(entry)
        sheet_path = output / "training_contact_sheet.jpg"
        used = save_bounded_image(sheet, sheet_path, "JPEG", used, quality=90)
        result["contact_sheet_sha256"] = digest(sheet_path)
        full_path = output / "training_full_sheet.jpg"
        used = save_bounded_image(full_sheet, full_path, "JPEG", used, quality=90)
        result["full_sheet_sha256"] = digest(full_path)
        if overlay_sheet:
            overlay_path = output / "training_mask_overlay_sheet.jpg"
            used = save_bounded_image(overlay_sheet, overlay_path, "JPEG", used, quality=90)
            result["mask_overlay_sheet_sha256"] = digest(overlay_path)
        if (digest(package_path) != package_sha_before or digest(__file__) != runner_sha_before or
                (regions_path and digest(regions_path) != regions_sha_before)):
            raise ValueError("package, ROI configuration, or runner changed during preparation")
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["failure"] = str(error)
        raise
    finally:
        manifest = (json.dumps(result, indent=2) + "\n").encode()
        if len(manifest) > 1024 ** 2:
            raise ValueError("output manifest exceeds 1 MiB reserve")
        (output / "manifest.json").write_bytes(manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", type=Path, help="manual ROI JSON; omit for contact sheet only")
    args = parser.parse_args()
    report = prepare(args.package, args.dataset_root, args.output, regions_path=args.regions)
    print(json.dumps({"status": report["status"], "mask_status": report["mask_status"],
                      "training_views": len(report["images"])}))


if __name__ == "__main__":
    main()
