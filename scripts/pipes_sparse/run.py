#!/usr/bin/env python3
"""Four-view SIFT tracks triangulated with supplied ETH3D camera poses."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import resource
import shutil
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import cv2
import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fixed_camera.run import build_tracks, reconstruct
from scripts.sparse_verify.verify import rotation

NAMES = tuple(f"DSC_{n:04d}.JPG" for n in range(634, 638))
MAX_EDGE = 1024
FEATURES = 4096
RATIO = .8
SAMPSON_PX = 2.0
MAX_RESULT_BYTES = 128 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_image(data, record):
    name, rel = record["name"], record["path"]
    pure = PurePosixPath(rel)
    if (name not in NAMES or pure.is_absolute() or str(pure) != rel or
            "\\" in rel or any(p in (".", "..") for p in pure.parts) or
            rel != f"pipes/images/dslr_images_undistorted/{name}"):
        raise ValueError("image path outside four-view allowlist")
    target = data.joinpath(*pure.parts)
    if data.is_symlink() or target.is_symlink():
        raise ValueError("image missing or symlinked")
    if any((data.joinpath(*pure.parts[:i])).is_symlink() for i in range(1, len(pure.parts))):
        raise ValueError("image missing or symlinked")
    if not target.is_file():
        raise ValueError("image missing or symlinked")
    if target.stat().st_size != record["size_bytes"] or sha(target) != record["sha256"]:
        raise ValueError("selected image size/hash mismatch")
    return target


def resize_camera(camera):
    if camera["model"] != "PINHOLE" or len(camera["params"]) != 4:
        raise ValueError("expected supplied PINHOLE camera")
    w, h = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = map(float, camera["params"])
    if min(w, h) <= 0 or min(fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError("invalid supplied camera")
    scale = min(1., MAX_EDGE / max(w, h))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    sx, sy = rw / w, rh / h
    return {"model": "PINHOLE", "width": rw, "height": rh,
            "params": [fx * sx, fy * sy, cx * sx, cy * sy]}, [sx, sy]


def edge_xy(keypoint):
    """OpenCV KeyPoint.pt uses zero at first centre; COLMAP uses 0.5."""
    return [float(keypoint.pt[0]) + .5, float(keypoint.pt[1]) + .5]


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def fundamental(a, b):
    ka = np.array([[a["camera"]["params"][0], 0, a["camera"]["params"][2]],
                   [0, a["camera"]["params"][1], a["camera"]["params"][3]], [0, 0, 1]])
    kb = np.array([[b["camera"]["params"][0], 0, b["camera"]["params"][2]],
                   [0, b["camera"]["params"][1], b["camera"]["params"][3]], [0, 0, 1]])
    ra, rb = np.asarray(a["R"]), np.asarray(b["R"])
    ta, tb = np.asarray(a["t"]), np.asarray(b["t"])
    relative_r = rb @ ra.T
    relative_t = tb - relative_r @ ta
    f = np.linalg.inv(kb).T @ skew(relative_t) @ relative_r @ np.linalg.inv(ka)
    norm = np.linalg.norm(f)
    if not math.isfinite(norm) or norm <= 1e-15:
        raise ValueError("degenerate supplied camera baseline")
    return f / norm


def sampson(f, xy_a, xy_b):
    a, b = np.array([*xy_a, 1.]), np.array([*xy_b, 1.])
    fa, fb = f @ a, f.T @ b
    den = fa[0]**2 + fa[1]**2 + fb[0]**2 + fb[1]**2
    return abs(float(b @ fa)) / math.sqrt(den) if den > 1e-12 else math.inf


def mutual_ratio(desc_a, desc_b):
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(desc_a, desc_b, k=2)
    backward = matcher.knnMatch(desc_b, desc_a, k=2)
    reverse = {(m[0].trainIdx, m[0].queryIdx) for m in backward
               if len(m) == 2 and m[0].distance < RATIO * m[1].distance}
    return sorted((m[0].queryIdx, m[0].trainIdx) for m in forward
                  if len(m) == 2 and m[0].distance < RATIO * m[1].distance and
                  (m[0].queryIdx, m[0].trainIdx) in reverse)


def selected_views(data):
    manifest_path = data / "prepare-metadata.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 2 * 1024**2:
        raise ValueError("missing or oversized prepared manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "validated":
        raise ValueError("prepared data not validated")
    all_images = manifest.get("images")
    if (not isinstance(all_images, list) or len(all_images) != 14 or
            len({r["name"] for r in all_images}) != 14):
        raise ValueError("expected 14 distinct prepared images")
    selected = sorted(all_images, key=lambda r: r["name"])[:4]
    if [r["name"] for r in selected] != list(NAMES):
        raise ValueError("first four lexical prepared images changed")
    views = []
    for record in sorted(selected, key=lambda x: x["name"]):
        if record.get("pose_convention") != "world_to_camera":
            raise ValueError("unexpected supplied pose convention")
        path = checked_image(data, record)
        if manifest.get("file_sha256", {}).get(record["path"]) != record["sha256"]:
            raise ValueError("prepared file digest mismatch")
        camera, scales = resize_camera(record["camera"])
        r = np.asarray(rotation(record["qvec"]), dtype=float)
        t = np.asarray(record["tvec"], dtype=float)
        if t.shape != (3,) or not np.isfinite(t).all():
            raise ValueError("invalid fixed pose")
        views.append({"id": record["id"], "name": record["name"],
                      "path": str(path), "sha256": record["sha256"],
                      "camera_id": record["camera_id"], "camera": camera,
                      "source_camera": record["camera"], "resize_scale_xy": scales,
                      "qvec": record["qvec"], "R": r.tolist(), "t": t.tolist(),
                      "center": (-r.T @ t).tolist()})
    return views, sha(manifest_path)


def dependencies():
    import numpy._core._multiarray_umath as core
    from PIL import _imaging
    cv_binary = sorted(Path(cv2.__file__).resolve().parent.glob("cv2*.so"))
    if not cv_binary:
        raise ValueError("OpenCV binary not found")
    paths = [Path(sys.executable), cv_binary[0], Path(core.__file__),
             Path(PIL.__file__), Path(_imaging.__file__), Path(__file__), ROOT / "scripts/fixed_camera/run.py",
             ROOT / "scripts/sparse_verify/verify.py",
             ROOT / "scripts/research_job/guard.py",
             ROOT / "docs/PIPES-EVALUATION-PROTOCOL.md"]
    return {"versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                         "numpy": np.__version__, "pillow": PIL.__version__},
            "binary_and_source_sha256": {str(p.resolve()): sha(p) for p in paths}}


def configure_opencv_threads():
    # macOS GCD ignores positive limits here; disabling its pool reports one worker.
    cv2.setNumThreads(0)
    actual = cv2.getNumThreads()
    if actual != 1:
        raise ValueError(f"OpenCV thread cap did not apply: {actual}")
    return actual


def worker(data, output):
    started = time.monotonic()
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} - {"child.log"}:
            raise ValueError("output directory is not fresh")
    elif not output.parent.is_dir():
        raise ValueError("output parent does not exist")
    if shutil.disk_usage(output.parent).free < 10 * 1024**3 + MAX_RESULT_BYTES:
        raise ValueError("less than 10 GiB free plus output allowance")
    actual_threads = configure_opencv_threads()
    views, manifest_sha = selected_views(data)
    before = {v["path"]: sha(v["path"]) for v in views}
    deps = dependencies()
    detector = cv2.SIFT_create(nfeatures=FEATURES)
    features = {}
    for view in views:
        gray = cv2.imread(view["path"], cv2.IMREAD_GRAYSCALE)
        source = view["source_camera"]
        if gray is None or gray.shape != (source["height"], source["width"]):
            raise ValueError("source image does not match camera dimensions")
        size = (view["camera"]["width"], view["camera"]["height"])
        if (gray.shape[1], gray.shape[0]) != size:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints = keypoints[:FEATURES]
        descriptors = descriptors[:FEATURES] if descriptors is not None else None
        features[view["name"]] = ([edge_xy(k) for k in keypoints], descriptors)
    edges, pairs = [], []
    for a, b in itertools.combinations(views, 2):
        xy_a, d_a = features[a["name"]]
        xy_b, d_b = features[b["name"]]
        matches = mutual_ratio(d_a, d_b)
        f = fundamental(a, b)
        kept = [(ia, ib) for ia, ib in matches if sampson(f, xy_a[ia], xy_b[ib]) <= SAMPSON_PX]
        edges.extend(((a["name"], ia), (b["name"], ib)) for ia, ib in kept)
        pairs.append({"images": [a["name"], b["name"]], "mutual_ratio_matches": len(matches),
                      "fixed_pose_epipolar_matches": len(kept), "supplied_fundamental": f.tolist()})
    tracks, conflicts, components = build_tracks(edges)
    images = {v["name"]: v for v in views}
    cameras = {v["camera_id"]: v["camera"] for v in views}
    xy = {name: vals for name, (vals, _) in features.items()}
    reasons = Counter(conflicts)
    accepted = []
    for track in tracks:
        if len(track) < 2:
            reasons["short_track"] += 1
            continue
        point, reason = reconstruct(track, images, cameras, xy)
        if reason:
            reasons[reason] += 1
        else:
            point["id"] = len(accepted) + 1
            for observation in point["observations"]:
                observation["xy_edge_frame"] = xy[observation["image"]][observation["original_feature_id"]]
            accepted.append(point)
    if sum(reasons.values()) + len(accepted) != components:
        raise ValueError("component accounting mismatch")
    if {v["path"]: sha(v["path"]) for v in views} != before or sha(data / "prepare-metadata.json") != manifest_sha:
        raise ValueError("prepared inputs changed during run")
    if dependencies() != deps:
        raise ValueError("binary, source, or frozen protocol changed during run")
    output.mkdir(exist_ok=True)  # Research guard may precreate this directory.
    if (output / "report.json").exists() or (output / "points.json").exists():
        raise ValueError("result already exists")
    report = {"schema": "eth3d_pipes_fixed_camera_sparse_v1", "purpose": "four_training_view_research_baseline",
              "reference_geometry_used": False, "fixed_camera_oracle_lane": True,
              "source_manifest": str(data / "prepare-metadata.json"), "source_manifest_sha256": manifest_sha,
              "dependencies": deps, "opencv_threads_requested": 0,
              "opencv_threads": actual_threads,
              "host": {"system": platform.system(), "machine": platform.machine(),
                       "processor": platform.processor()},
              "settings": {"selection": list(NAMES), "max_long_edge": MAX_EDGE,
                           "sift_features_per_view": FEATURES, "descriptor": "SIFT_L2",
                           "mutual_ratio_both_directions": RATIO,
                           "fixed_pose_sqrt_sampson_max_px": SAMPSON_PX,
                           "min_parallax_deg": 1., "max_reprojection_px_all_observations": 4.,
                           "max_normal_matrix_condition": 1e8, "positive_depth_all_observations": True,
                           "coordinates": "pixel-edge frame; cv2.KeyPoint.pt + 0.5; K scales by actual output/source dimensions"},
              "views": [{k: v[k] for k in ("id", "name", "path", "sha256", "camera_id", "source_camera",
                                          "camera", "resize_scale_xy", "qvec", "R", "t", "center")}
                        for v in views],
              "feature_counts": {name: len(vals[0]) for name, vals in features.items()},
              "pairs": pairs, "epipolar_edges": len(edges), "connected_components": components,
              "rejected_components": dict(sorted(reasons.items())), "accepted_tracks": len(accepted),
              "accepted_tracks_three_or_more_views": sum(len(p["observations"]) >= 3 for p in accepted),
              "accepted_observations": sum(len(p["observations"]) for p in accepted),
              "accepted_observations_by_image": {v["name"]: sum(
                  o["image"] == v["name"] for p in accepted for o in p["observations"])
                  for v in views},
              "accepted_tracks_by_support_views": dict(sorted(Counter(
                  len(p["observations"]) for p in accepted).items())),
              "wall_seconds_at_report": round(time.monotonic() - started, 3),
              "peak_rss_bytes_at_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1024 if platform.system() == "Linux" else 1)}
    payloads = {"report.json": report, "points.json": {"schema": "fixed_camera_sparse_points_v1", "points": accepted}}
    encoded = {name: (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode()
               for name, obj in payloads.items()}
    if sum(map(len, encoded.values())) > MAX_RESULT_BYTES:
        raise ValueError("128 MiB result cap exceeded")
    for name, content in encoded.items():
        (output / name).write_bytes(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="validated prepared data directory")
    parser.add_argument("--output", type=Path, required=True, help="fresh guarded output directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, output = args.data.absolute(), args.output.absolute()
    if args.worker:
        report = worker(data, output)
        print(json.dumps({"accepted_tracks": report["accepted_tracks"],
                          "accepted_tracks_three_or_more_views": report["accepted_tracks_three_or_more_views"]}))
        if report["accepted_tracks"] == 0:
            raise SystemExit(2)
    else:
        from scripts.research_job.guard import run_child
        status = run_child([sys.executable, "-m", "scripts.pipes_sparse.run", "--worker",
                            "--data", str(data), "--output", str(output)], cwd=ROOT,
                           output_dir=output, timeout_seconds=300,
                           max_output_bytes=MAX_RESULT_BYTES, reserve_bytes=10 * 1024**3,
                           max_log_bytes=10 * 1024**2)
        print(json.dumps(status, sort_keys=True))
        if status["status"] != "succeeded":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
