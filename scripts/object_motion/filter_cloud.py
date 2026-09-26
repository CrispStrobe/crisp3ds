#!/usr/bin/env python3
"""Filter native dense points by frozen 48/60 image-only pose-mask support.

No depth, reference scan, reconstructed mesh, or supplied pose is read. Retained
OpenMVS PLY vertex records (including RGB, normals, visibility lists) are copied
byte for byte; only the header vertex count changes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.classical_backend import geometry
DEFAULT_CLOUD = ROOT / "build-opencv/classical-ycb-foreground-003-continuation/dense.ply"
DEFAULT_MODEL = ROOT / "build-opencv/object-motion/foreground/models/0"
DEFAULT_MANIFEST = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
DEFAULT_OUTPUT = ROOT / "build-opencv/object-motion/filter-001/filtered.ply"
MIN_SUPPORT = 48
CAMERA_COUNT = 60


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_vertices(path):
    """Read validated point-cloud records and XYZ without changing native bytes."""
    with Path(path).open("rb") as stream:
        vertex, face = geometry._header(stream)
        if face["count"] != 0:
            raise ValueError("filter input must be a point cloud without faces")
        end_header = stream.tell()
        stream.seek(0)
        header = stream.read(end_header)
        records, positions = [], []
        for _ in range(vertex["count"]):
            start = stream.tell()
            xyz = {}
            for name, fmt, item_fmt in vertex["props"]:
                if item_fmt is None:
                    raw = stream.read(struct.calcsize("<" + fmt))
                    if len(raw) != struct.calcsize("<" + fmt):
                        raise ValueError("truncated PLY scalar")
                    if name in ("x", "y", "z"):
                        xyz[name] = struct.unpack("<" + fmt, raw)[0]
                else:
                    count_blob = stream.read(struct.calcsize("<" + fmt))
                    if len(count_blob) != struct.calcsize("<" + fmt):
                        raise ValueError("truncated PLY list count")
                    count = struct.unpack("<" + fmt, count_blob)[0]
                    payload = stream.read(count * struct.calcsize("<" + item_fmt))
                    if len(payload) != count * struct.calcsize("<" + item_fmt):
                        raise ValueError("truncated PLY list")
            stop = stream.tell()
            stream.seek(start)
            records.append(stream.read(stop - start))
            stream.seek(stop)
            positions.append((xyz["x"], xyz["y"], xyz["z"]))
        if stream.read(1):
            raise ValueError("unexpected trailing point-cloud payload")
    return header, records, positions


def filter_header(header, old_count, new_count):
    lines = header.splitlines(keepends=True)
    old = f"element vertex {old_count}\n".encode("ascii")
    matches = [i for i, line in enumerate(lines) if line == old]
    if len(matches) != 1:
        raise ValueError("cannot identify unique vertex-count header line")
    lines[matches[0]] = f"element vertex {new_count}\n".encode("ascii")
    return b"".join(lines)


def select_records(header, records, selected, output):
    """Write only selected original vertex byte slices to a fresh PLY."""
    if len(records) != len(selected):
        raise ValueError("selection length differs from source vertices")
    retained = sum(bool(v) for v in selected)
    if retained == 0:
        raise ValueError("selection retained zero vertices")
    with Path(output).open("xb") as stream:
        stream.write(filter_header(header, len(records), retained))
        for keep, blob in zip(selected, records):
            if keep:
                stream.write(blob)
    return retained


def retain_by_support(counts):
    import numpy as np
    counts = np.asarray(counts)
    if np.any((counts < 0) | (counts > CAMERA_COUNT)):
        raise ValueError("support count outside camera population")
    return counts >= MIN_SUPPORT


def project_simple_radial(points, cam_from_world, params):
    """Return pixel u/v and camera-frame z using COLMAP SIMPLE_RADIAL."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(cam_from_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (3, 4):
        raise ValueError("unexpected point or camera transform shape")
    f, cx, cy, k = map(float, params)
    if not all(math.isfinite(v) for v in (f, cx, cy, k)) or f <= 0:
        raise ValueError("invalid SIMPLE_RADIAL parameters")
    xyz = points @ matrix[:, :3].T + matrix[:, 3]
    z = xyz[:, 2]
    safe_z = np.where(np.isfinite(z) & (np.abs(z) > 1e-9), z, 1)
    x, y = xyz[:, 0] / safe_z, xyz[:, 1] / safe_z
    radial = 1 + k * (x*x + y*y)
    u, v = f * radial * x + cx, f * radial * y + cy
    return u, v, z


def support_for_camera(points, cam_from_world, params, scanlines, width, height):
    """Vectorized SIMPLE_RADIAL projection into one photo-derived scanline mask."""
    import numpy as np
    u, v, z = project_simple_radial(points, cam_from_world, params)
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 1e-9)
    valid &= np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    rows = np.floor(np.where(valid, v, 0)).astype(np.int64)
    lo = np.full(height, np.inf)
    hi = np.full(height, -np.inf)
    for row, left, right in scanlines:
        if not 0 <= row < height or not 0 <= left <= right < width:
            raise ValueError("invalid mask scanline")
        lo[row], hi[row] = left, right
    return valid & (u >= lo[rows]) & (u <= hi[rows])


def validate_inputs(cloud, model_dir, manifest_path):
    import pycolmap
    cloud, model_dir, manifest_path = Path(cloud), Path(model_dir), Path(manifest_path)
    cloud_report = geometry.inspect(cloud)
    if cloud_report["faces"]:
        raise ValueError("source cloud has faces")
    manifest = json.loads(manifest_path.read_text())
    images = manifest.get("images", [])
    if manifest.get("schema") != "object_motion_prepare_v1" or len(images) != CAMERA_COUNT:
        raise ValueError("expected 60 prepared RGB photo masks")
    if [r["name"] for r in images] != [f"NP3_{a:03}.jpg" for a in range(0, 360, 6)]:
        raise ValueError("unexpected mask/photo order")
    summary_path = model_dir.parent.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    model_hashes = {name: digest(model_dir / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}
    if summary.get("model_files_sha256") != model_hashes or summary.get("registered") != CAMERA_COUNT:
        raise ValueError("model differs from sealed 60-camera producer")
    if summary.get("input_manifest_sha256") != digest(manifest_path):
        raise ValueError("mask manifest differs from sealed producer")
    mask_hashes, photo_hashes = {}, {}
    for rec in images:
        name = rec["name"]
        if len(rec["pose_support_scanlines"]) == 0:
            raise ValueError("empty pose-support scanlines")
        mask_hashes[name] = digest(rec["pose_support_mask"])
        photo_hashes[name] = digest(rec["path"])
        if mask_hashes[name] != rec["mask_sha256"] or photo_hashes[name] != rec["sha256"]:
            raise ValueError(f"changed photo or mask: {name}")
    model = pycolmap.Reconstruction(str(model_dir))
    if model.num_reg_images() != CAMERA_COUNT or len(model.cameras) != 1:
        raise ValueError("expected all 60 registered and one shared camera")
    by_name = {image.name: image for image in model.images.values()}
    if set(by_name) != set(r["name"] for r in images):
        raise ValueError("model image names differ from RGB manifest")
    return cloud_report, images, by_name, model, model_hashes, mask_hashes, photo_hashes


def filter_cloud(cloud, model_dir, manifest_path, output, report_path):
    import numpy as np
    cloud, model_dir, manifest_path = map(Path, (cloud, model_dir, manifest_path))
    output, report_path = Path(output), Path(report_path)
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("filtered PLY and report paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB free")
    original_hash = digest(cloud)
    cloud_report, images, by_name, model, model_hashes, mask_hashes, photo_hashes = validate_inputs(
        cloud, model_dir, manifest_path)
    manifest_hash = digest(manifest_path)
    header, records, positions = raw_vertices(cloud)
    if len(records) != cloud_report["vertices"]:
        raise ValueError("point count differs after validation")
    points = np.asarray(positions, dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.uint8)
    per_camera = {}
    for rec in images:
        image = by_name[rec["name"]]
        camera = model.cameras[image.camera_id]
        if camera.model.name != "SIMPLE_RADIAL" or (camera.width, camera.height) != (1280, 1024):
            raise ValueError("expected 1280x1024 SIMPLE_RADIAL camera")
        support = support_for_camera(points, image.cam_from_world.matrix(), camera.params,
                                     rec["pose_support_scanlines"], camera.width, camera.height)
        counts += support.astype(np.uint8)
        per_camera[rec["name"]] = int(support.sum())
    keep = retain_by_support(counts)
    retained = int(keep.sum())
    if retained == 0:
        raise ValueError("frozen threshold retained zero points")
    try:
        select_records(header, records, keep, output)
        filtered_report = geometry.inspect(output)
        if filtered_report["vertices"] != retained or filtered_report["faces"] != 0:
            raise ValueError("filtered PLY failed validation")
        if digest(cloud) != original_hash:
            raise ValueError("source cloud changed during filtering")
        if digest(manifest_path) != manifest_hash or any(
                digest(model_dir / name) != value for name, value in model_hashes.items()):
            raise ValueError("camera model or mask manifest changed during filtering")
        if any(digest(rec["path"]) != photo_hashes[rec["name"]] or
               digest(rec["pose_support_mask"]) != mask_hashes[rec["name"]] for rec in images):
            raise ValueError("photo or pose-support mask changed during filtering")
        result = {"schema": "object_motion_point_support_v1",
                  "source_cloud": str(cloud.resolve()), "source_cloud_sha256": original_hash,
                  "filtered_cloud": str(output.resolve()), "filtered_cloud_sha256": digest(output),
                  "model_dir": str(model_dir.resolve()), "model_files_sha256": model_hashes,
                  "manifest_sha256": manifest_hash, "mask_sha256": mask_hashes,
                  "image_sha256": photo_hashes, "camera_count": CAMERA_COUNT,
                  "threshold_min_support": MIN_SUPPORT, "input_vertices": len(records),
                  "retained_vertices": retained, "removed_vertices": len(records) - retained,
                  "support_histogram": {str(i): int((counts == i).sum()) for i in range(CAMERA_COUNT + 1)},
                  "support_counts_by_source_vertex": counts.astype(int).tolist(),
                  "supported_points_per_camera": per_camera,
                  "method": "project every original dense world point through every registered SIMPLE_RADIAL camera; count if in photo-derived pose-support scanline; retain >=48/60",
                  "limits": "coarse pose masks may remove valid object surface or include checkerboard; this is pre-mesh support filtering, not shape truth"}
        report_path.write_text(json.dumps(result, indent=2) + "\n")
    except BaseException:
        output.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", type=Path, default=DEFAULT_CLOUD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT.parent / "report.json")
    args = parser.parse_args()
    result = filter_cloud(args.cloud, args.model, args.manifest, args.output, args.report)
    print(json.dumps({k: result[k] for k in ("input_vertices", "retained_vertices", "removed_vertices", "filtered_cloud_sha256")}))


if __name__ == "__main__":
    main()
