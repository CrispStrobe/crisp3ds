#!/usr/bin/env python3
"""Reconstruct foreground training tracks with frozen baseline cameras."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.foreground_roi.polygon import in_roi, load_contract
from scripts.sparse_verify import verify as checker

BASE = ROOT / "build-opencv/colmap-sparse/heldout-v2-001"
FORE = ROOT / "build-opencv/colmap-sparse/foreground-001"
ROI = ROOT / "tests/datasets/tree-envelope-v1.json"
ROI_SHA = "4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853"
OUTPUT = ROOT / "build-opencv/colmap-sparse/fixed-camera-object-001"
PAIR_BASE = 2147483647
MAX_REPROJECTION_PX = 4.0
MIN_PARALLAX_DEG = 1.0
MAX_CONDITION = 1e8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs(base_manifest):
    import numpy as np
    import numpy._core._multiarray_umath as np_core
    paths = [ROI, BASE / "database.db", BASE / "holdout.json", BASE / "provenance.json",
             BASE / "verification-supervisor.json", FORE / "database.db", FORE / "holdout.json",
             FORE / "provenance.json", FORE / "verification-supervisor.json",
             FORE / "roi-eligibility.json", Path(__file__),
             ROOT / "scripts/foreground_roi/polygon.py", ROOT / "scripts/sparse_verify/verify.py",
             ROOT / "scripts/research_job/guard.py"]
    paths += [Path(np.__file__), Path(np_core.__file__)]
    paths += [lane / "model_text" / name for lane in (BASE, FORE)
              for name in ("cameras.txt", "images.txt", "points3D.txt")]
    paths += [Path(im["path"]) for im in base_manifest["images"]]
    return {str(p.resolve(strict=True)): sha(p) for p in paths}


def build_tracks(rows):
    """Connected components of verified correspondences; reject full conflicts."""
    parent = {}
    rank = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for a, b in rows:
        if a == b:
            raise ValueError("self correspondence")
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank.get(ra, 0) < rank.get(rb, 0):
                ra, rb = rb, ra
            parent[rb] = ra
            if rank.get(ra, 0) == rank.get(rb, 0):
                rank[ra] = rank.get(ra, 0) + 1
    groups = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    good = []
    rejected = Counter()
    for group in groups.values():
        if len({name for name, _ in group}) != len(group):
            rejected["same_image_conflict"] += 1
        else:
            good.append(tuple(sorted(group)))
    return sorted(good), dict(rejected), len(groups)


def read_verified_edges(db_path, manifest):
    by_id = {im["image_id"]: im for im in manifest["images"]}
    rows = []
    pair_count = 0
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro&immutable=1", uri=True) as conn:
        for pair, count, cols, blob in conn.execute("SELECT pair_id,rows,cols,data FROM two_view_geometries"):
            if count == 0:
                continue
            if cols != 2 or blob is None or len(blob) != count * 8:
                raise ValueError(f"bad verified geometry blob: {pair}")
            second = pair % PAIR_BASE
            first = (pair - second) // PAIR_BASE
            if first not in by_id or second not in by_id or first >= second:
                raise ValueError(f"invalid geometry image IDs: {pair}")
            a, b = by_id[first], by_id[second]
            ma, mb = a["compact_to_original"], b["compact_to_original"]
            pair_count += 1
            for ia, ib in struct.iter_unpack("<II", blob):
                if ia >= len(ma) or ib >= len(mb):
                    raise ValueError(f"geometry index out of range: {pair}:{ia}:{ib}")
                rows.append(((a["name"], ma[ia]), (b["name"], mb[ib])))
    return rows, pair_count


def reconstruct(track, images, cameras, features):
    """Least-squares intersection of world rays with fixed world-to-camera poses."""
    import numpy as np
    centers, rays = [], []
    for name, fid in track:
        im = images[name]
        camera = cameras[im["camera_id"]]
        xy = features[name][fid]
        try:
            x, y = checker.undistort(camera, xy)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None, "invalid_undistortion"
        direction = np.asarray(checker.mv(checker.mt(im["R"]), [x, y, 1.0]), dtype=float)
        direction /= np.linalg.norm(direction)
        if not np.isfinite(direction).all():
            return None, "nonfinite_ray"
        centers.append(np.asarray(im["center"], dtype=float))
        rays.append(direction)
    parallax = max(math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1))))
                   for i, a in enumerate(rays) for b in rays[i+1:])
    if parallax < MIN_PARALLAX_DEG:
        return None, "low_parallax"
    identity = np.eye(3)
    matrix = sum((identity - np.outer(d, d) for d in rays), np.zeros((3, 3)))
    rhs = sum(((identity - np.outer(d, d)) @ c for d, c in zip(rays, centers)), np.zeros(3))
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > MAX_CONDITION:
        return None, "ill_conditioned"
    try:
        xyz = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None, "ill_conditioned"
    if not np.isfinite(xyz).all():
        return None, "nonfinite_solution"
    repro = []
    for name, fid in track:
        im = images[name]
        cam_xyz = checker.mv(im["R"], xyz)
        cam_xyz = [v + t for v, t in zip(cam_xyz, im["t"])]
        if not math.isfinite(cam_xyz[2]) or cam_xyz[2] <= 0:
            return None, "nonpositive_depth"
        uv = checker.project(cameras[im["camera_id"]], cam_xyz)
        error = math.hypot(uv[0] - features[name][fid][0], uv[1] - features[name][fid][1])
        if not math.isfinite(error):
            return None, "nonfinite_reprojection"
        repro.append(error)
    if max(repro) > MAX_REPROJECTION_PX:
        return None, "high_reprojection"
    return {"xyz": xyz.tolist(), "observations": [{"image": n, "original_feature_id": f,
            "reprojection_px": e} for (n, f), e in zip(track, repro)],
            "source_track": [[n, f] for n, f in track],
            "max_reprojection_px": max(repro), "max_ray_parallax_deg": parallax,
            "normal_matrix_condition": condition}, None


def baseline_all_inside(points, baseline, polygons):
    by_id = {im["image_id"]: im for im in baseline["images"]}
    by_name = {im["name"]: im for im in baseline["images"]}
    count = 0
    for point in points.values():
        refs = []
        for iid, idx in point["track"]:
            im = by_id[iid]
            refs.append((im["name"], im["compact_to_original"][idx]))
        if refs and all(in_roi(polygons[name], by_name[name]["features"][fid])
                        for name, fid in refs):
            count += 1
    return count


def worker(output):
    if sha(ROI) != ROI_SHA:
        raise ValueError("ROI SHA mismatch")
    baseline = json.loads((BASE / "holdout.json").read_text())
    foreground = json.loads((FORE / "holdout.json").read_text())
    before = inputs(baseline)
    reports = {"baseline": json.loads((BASE / "verification-supervisor.json").read_text()),
               "foreground": json.loads((FORE / "verification-supervisor.json").read_text())}
    for lane, report in reports.items():
        if report.get("integrity_status") != "pass" or not report.get("heldout_valid"):
            raise ValueError(f"{lane} independent integrity report failed")
    # Re-run independent model/database checks against the pinned inputs.
    for lane, manifest, stored in ((BASE, baseline, reports["baseline"]),
                                   (FORE, foreground, reports["foreground"])):
        check = checker.verify(lane / "holdout.json", lane / "model_text", lane / "database.db")
        if check["integrity_status"] != "pass":
            raise ValueError(f"{lane.name} current integrity failed: {check['errors'][:5]}")
        if check["hashes"] != stored["hashes"] or check["database"] != stored["database"]:
            raise ValueError(f"{lane.name} differs from saved independent integrity report")
    if (baseline["heldout_pairs"] != foreground["heldout_pairs"] or
            [(i["name"], i["features"], i["heldout_ids"]) for i in baseline["images"]] !=
            [(i["name"], i["features"], i["heldout_ids"]) for i in foreground["images"]]):
        raise ValueError("heldout population changed")
    if len(baseline["heldout_pairs"]) != 1121 or len(foreground["images"]) != 10:
        raise ValueError("unexpected heldout population")
    expected = {i["name"]: {"width": i["width"], "height": i["height"], "sha256": i["sha256"]}
                for i in baseline["images"]}
    polygons = load_contract(ROI, expected)
    cameras = checker.cameras(BASE / "model_text/cameras.txt")
    registered = checker.images(BASE / "model_text/images.txt")
    by_name = {im["name"]: im for im in registered.values()}
    if len(by_name) != 9:
        raise ValueError("baseline must have nine registered cameras")
    baseline_points = checker.points(BASE / "model_text/points3D.txt")
    if len(baseline_points) != 1773:
        raise ValueError("baseline point count changed")
    all_inside = baseline_all_inside(baseline_points, baseline, polygons)
    if all_inside != 151:
        raise ValueError(f"baseline all-inside count changed: {all_inside}")
    lanes = {}
    stage_counts = {}
    for lane, path, manifest in (("baseline_roi", BASE, baseline), ("foreground", FORE, foreground)):
        raw_edges, pair_count = read_verified_edges(path / "database.db", manifest)
        if lane == "foreground" and len(raw_edges) != 1105:
            raise ValueError(f"foreground geometry row count changed: {len(raw_edges)}")
        features = {im["name"]: im["features"] for im in manifest["images"]}
        heldout = {im["name"]: set(im["heldout_ids"]) for im in manifest["images"]}
        edges = []
        for edge in raw_edges:
            for name, fid in edge:
                if fid in heldout[name]:
                    raise ValueError("verified row references heldout feature")
            if all(in_roi(polygons[name], features[name][fid]) for name, fid in edge):
                edges.append(edge)
        if lane == "foreground" and len(edges) != len(raw_edges):
            raise ValueError("foreground verified row outside ROI")
        eligible = len(edges)
        edges = [edge for edge in edges if all(name in by_name for name, _ in edge)]
        tracks, conflicts, components = build_tracks(edges)
        reasons = Counter(conflicts)
        accepted = []
        for track in tracks:
            if len(track) < 2:
                reasons["short_track"] += 1
                continue
            point, reason = reconstruct(track, by_name, cameras, features)
            if reason:
                reasons[reason] += 1
            else:
                point["id"] = len(accepted) + 1
                accepted.append(point)
        seen = set()
        for point in accepted:
            for obs in point["observations"]:
                key = (obs["image"], obs["original_feature_id"])
                if key in seen or obs["reprojection_px"] > MAX_REPROJECTION_PX:
                    raise ValueError("candidate integrity failed")
                seen.add(key)
        if sum(reasons.values()) + len(accepted) != components:
            raise ValueError("stage accounting failed")
        lanes[lane] = accepted
        stage_counts[lane] = {"verified_geometry_pairs": pair_count,
            "verified_geometry_rows": len(raw_edges), "roi_eligible_rows": eligible,
            "roi_excluded_rows": len(raw_edges) - eligible,
            "unregistered_endpoint_rows": eligible - len(edges),
            "connected_components": components, "rejected_by_reason": dict(sorted(reasons.items())),
            "accepted_tracks": len(accepted), "accepted_tracks_three_or_more_views":
            sum(len(p["observations"]) >= 3 for p in accepted),
            "accepted_observations": len(seen),
            "registered_images_covered": len({o["image"] for p in accepted for o in p["observations"]}),
            "accepted_observations_by_image": dict(sorted(Counter(
                o["image"] for p in accepted for o in p["observations"]).items()))}
    after = inputs(baseline)
    if before != after:
        raise ValueError("pinned inputs changed")
    result = {"schema": "fixed_camera_object_tracks_v1", "method": "linear least-squares world-ray intersection",
              "cameras": "heldout-v2-001 fixed nine-camera text model", "lanes": lanes}
    report = {"schema": "fixed_camera_object_report_v1", "status": "pass",
              "predeclared_screens": {"positive_depth_all_observations": True,
                  "max_reprojection_px": MAX_REPROJECTION_PX, "min_max_ray_parallax_deg": MIN_PARALLAX_DEG,
                  "max_normal_matrix_condition": MAX_CONDITION},
              "stage_counts": stage_counts,
              "camera_source": {"cameras_text": str(BASE / "model_text/cameras.txt"),
                  "images_text": str(BASE / "model_text/images.txt")},
              "roi_source": str(ROI),
              "comparison_only": {"baseline_registered_images": 9, "baseline_points": 1773,
                  "baseline_all_observations_inside_roi_points": all_inside},
              "candidate_integrity": {"unique_observations": True, "all_positive_depth": True,
                  "all_reprojection_within_4px": True, "all_registered": True, "all_training_roi": True},
              "input_integrity": {k: {"stored_status": v["integrity_status"], "current_status": "pass"}
                                  for k, v in reports.items()},
              "heldout": {"pair_count": 1121, "population_unchanged": True,
                  "interpretation": "Pair and three-view errors depend on fixed cameras and heldout observations; new points cannot change them. No improvement claim."},
              "input_sha256_before": before, "input_sha256_after": after}
    import numpy as np
    report["numeric_runtime"] = {"numpy_version": np.__version__, "openblas_threads":
        os.environ.get("OPENBLAS_NUM_THREADS", "unset")}
    provenance = {"schema": "fixed_camera_object_provenance_v1",
        "baseline_source": str(BASE), "foreground_source": str(FORE),
        "roi_source": str(ROI), "camera_source": report["camera_source"],
        "numeric_runtime": report["numeric_runtime"],
        "input_sha256_before": before, "input_sha256_after": after,
        "saved_integrity_reports": report["input_integrity"]}
    (output / "points.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(stage_counts), flush=True)


def main(argv=None):
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if args.worker:
        if not output.is_dir():
            raise ValueError("worker requires guard-created output directory")
        worker(output)
        return
    from scripts.research_job.guard import run_child
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_child([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)],
                       cwd=ROOT, output_dir=output, timeout_seconds=300,
                       max_output_bytes=100 << 20, reserve_bytes=10 << 30, max_log_bytes=10 << 20)
    print(json.dumps(result, indent=2))
    if result["status"] != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
