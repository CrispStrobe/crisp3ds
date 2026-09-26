#!/usr/bin/env python3
"""Independent geometry audit of points made with frozen baseline cameras.

This module deliberately does not import the point producer or its triangulator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from collections import Counter, defaultdict

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_manifest(before, after, required):
    if before != after:
        raise ValueError("producer changed an input")
    if not required.issubset(before):
        raise ValueError("producer input hashes omit a required source")
    for filename, expected in before.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"source hash mismatch: {filename}")


def rotation_from_colmap(q):
    """COLMAP scalar-first quaternion to world-to-camera rotation."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def camera_rows(images_path: Path, cameras_path: Path):
    cams = {}
    for line in cameras_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 7 or f[1] != "SIMPLE_PINHOLE":
            raise ValueError("expected frozen SIMPLE_PINHOLE camera")
        cid = int(f[0])
        cams[cid] = {"width": int(f[2]), "height": int(f[3]),
                     "f": float(f[4]), "cx": float(f[5]), "cy": float(f[6])}
    rows = images_path.read_text().splitlines()
    rows = [r for r in rows if not r.startswith("#")]
    if len(rows) % 2:
        raise ValueError("incomplete COLMAP image record")
    result = {}
    for header in rows[::2]:
        f = header.split()
        if len(f) != 10:
            raise ValueError("invalid COLMAP image header")
        r = rotation_from_colmap([float(v) for v in f[1:5]])
        t = np.asarray([float(v) for v in f[5:8]], dtype=np.float64)
        camera_id = int(f[8])
        if camera_id not in cams or f[9] in result:
            raise ValueError("unknown camera or repeated image name")
        result[f[9]] = {**cams[camera_id], "R": r, "t": t,
                        "image_id": int(f[0]), "camera_id": camera_id}
    return result


def project(cam, xyz):
    xyz_cam = cam["R"] @ np.asarray(xyz, dtype=np.float64) + cam["t"]
    if not np.isfinite(xyz_cam).all() or xyz_cam[2] <= 0:
        raise ValueError("point is not in front of camera")
    return np.array([cam["f"]*xyz_cam[0]/xyz_cam[2]+cam["cx"],
                     cam["f"]*xyz_cam[1]/xyz_cam[2]+cam["cy"]])


def dlt_point(observations, cameras):
    """Independent homogeneous DLT reference for exact synthetic pinhole rays."""
    equations = []
    for name, xy in observations:
        cam = cameras[name]
        k = np.array(((cam["f"], 0, cam["cx"]),
                      (0, cam["f"], cam["cy"]), (0, 0, 1)))
        p = k @ np.column_stack((cam["R"], cam["t"]))
        u, v = xy
        equations.extend((u*p[2]-p[0], v*p[2]-p[1]))
    _, _, vh = np.linalg.svd(np.asarray(equations))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("DLT point at infinity")
    return homogeneous[:3]/homogeneous[3]


def ray_normal_residual(observations, cameras, xyz):
    """Gradient residual of sum of squared distances to back-projected rays."""
    normal = np.zeros((3, 3))
    rhs = np.zeros(3)
    for name, pixel in observations:
        cam = cameras[name]
        u, v = pixel
        direction_camera = np.array(((u-cam["cx"])/cam["f"],
                                     (v-cam["cy"])/cam["f"], 1.0))
        direction = cam["R"].T @ direction_camera
        direction /= np.linalg.norm(direction)
        center = -cam["R"].T @ cam["t"]
        projector = np.eye(3)-np.outer(direction, direction)
        normal += projector
        rhs += projector @ center
    singular_values = np.linalg.svd(normal, compute_uv=False)
    if singular_values[-1] <= 1e-10 * singular_values[0]:
        raise ValueError("degenerate ray geometry")
    xyz = np.asarray(xyz, dtype=np.float64)
    return float(np.linalg.norm(normal @ xyz-rhs)), float(np.linalg.norm(rhs)), singular_values


def load_json(path):
    def reject(token):
        raise ValueError(f"nonfinite JSON number: {token}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def inside_polygon(vertices, point):
    """Closed-boundary winding parity, independently evaluated from ROI code."""
    x, y = point
    parity = False
    for a, b in zip(vertices, vertices[1:]+vertices[:1]):
        ax, ay = a
        bx, by = b
        cross = (bx-ax)*(y-ay)-(by-ay)*(x-ax)
        if abs(cross) <= 1e-9 and min(ax, bx)-1e-9 <= x <= max(ax, bx)+1e-9 and min(ay, by)-1e-9 <= y <= max(ay, by)+1e-9:
            return True
        if (ay > y) != (by > y) and x < ax+(y-ay)*(bx-ax)/(by-ay):
            parity = not parity
    return parity


def ray_direction(cam, pixel):
    u, v = pixel
    direction = cam["R"].T @ np.array(((u-cam["cx"])/cam["f"],
                                       (v-cam["cy"])/cam["f"], 1.0))
    return direction/np.linalg.norm(direction)


def parallax_degrees(observations, cameras):
    directions = [ray_direction(cameras[name], xy) for name, xy in observations]
    return max(min((angle := math.degrees(math.acos(np.clip(float(a @ b), -1, 1)))), 180-angle)
               for i, a in enumerate(directions) for b in directions[i+1:])


def verify_lane(points, cameras, features, polygons, heldout):
    """Check every published accepted point against source centres and geometry."""
    if not isinstance(points, list):
        raise ValueError("points lane must be an array")
    used_ids = set()
    unique_refs = set()
    all_errors = []
    for point in points:
        pid = point["id"]
        if pid in used_ids:
            raise ValueError(f"duplicate point id {pid}")
        used_ids.add(pid)
        xyz = np.asarray(point["xyz"], dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"invalid point coordinates {pid}")
        rows = point["observations"]
        if len(rows) < 2 or len({o["image"] for o in rows}) != len(rows):
            raise ValueError(f"point {pid} needs at least two distinct cameras")
        obs = []
        errors = []
        refs = []
        for item in rows:
            name, fid = item["image"], item["original_feature_id"]
            if name not in cameras or name not in features or not isinstance(fid, int) or fid < 0 or fid >= len(features[name]):
                raise ValueError(f"invalid feature reference in {pid}")
            ref = (name, fid)
            if ref in unique_refs:
                raise ValueError(f"feature reused across points: {ref}")
            unique_refs.add(ref)
            if fid in heldout[name]:
                raise ValueError(f"held-out observation in {pid}: {ref}")
            xy = np.asarray(features[name][fid][:2], dtype=np.float64)
            if not any(inside_polygon(poly, xy) for poly in polygons[name]):
                raise ValueError(f"observation outside ROI in {pid}: {ref}")
            prediction = project(cameras[name], xyz)
            error = float(np.linalg.norm(prediction-xy))
            if error > 4+1e-8 or not math.isclose(error, item["reprojection_px"], abs_tol=1e-7):
                raise ValueError(f"reprojection mismatch/threshold in {pid}: {ref}, {error}")
            obs.append((name, xy))
            errors.append(error)
            refs.append([name, fid])
        if point["source_track"] != sorted(refs):
            raise ValueError(f"source track differs from observations in {pid}")
        maximum = max(errors)
        if not math.isclose(maximum, point["max_reprojection_px"], abs_tol=1e-7):
            raise ValueError(f"maximum reprojection mismatch in {pid}")
        parallax = parallax_degrees(obs, cameras)
        if parallax < 1-1e-8 or not math.isclose(parallax, point["max_ray_parallax_deg"], abs_tol=1e-7):
            raise ValueError(f"parallax mismatch/threshold in {pid}")
        residual, scale, singular = ray_normal_residual(obs, cameras, xyz)
        if residual > 1e-7*(1+scale):
            raise ValueError(f"point {pid} is not ray least-squares optimum: {residual}")
        condition = float(singular[0]/singular[-1])
        if condition > 1e8+1e-5 or not math.isclose(condition, point["normal_matrix_condition"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"condition mismatch/threshold in {pid}")
        all_errors.extend(errors)
    return {"points_checked": len(points), "observations_checked": len(all_errors),
            "max_reprojection_px": max(all_errors, default=None)}


def verified_graph(db_path, manifest, polygons, registered):
    """Decode COLMAP verified rows and find components with an independent union-find."""
    by_id = {row["image_id"]: row for row in manifest["images"]}
    parent = {}
    size = {}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def add(node):
        if node not in parent:
            parent[node] = node
            size[node] = 1

    counters = Counter()
    with sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro&immutable=1", uri=True) as conn:
        for pair_id, n, cols, blob in conn.execute("SELECT pair_id,rows,cols,data FROM two_view_geometries"):
            if n == 0:
                continue
            if cols != 2 or blob is None or len(blob) != n*8:
                raise ValueError(f"invalid verified geometry row {pair_id}")
            b_id = pair_id % 2147483647
            a_id = (pair_id-b_id)//2147483647
            if a_id not in by_id or b_id not in by_id or a_id >= b_id:
                raise ValueError(f"invalid pair id {pair_id}")
            a, b = by_id[a_id], by_id[b_id]
            counters["verified_geometry_pairs"] += 1
            for i, j in struct.iter_unpack("<II", blob):
                counters["verified_geometry_rows"] += 1
                if i >= len(a["compact_to_original"]) or j >= len(b["compact_to_original"]):
                    raise ValueError("compact feature index outside manifest")
                af, bf = a["compact_to_original"][i], b["compact_to_original"][j]
                if af in a["heldout_ids"] or bf in b["heldout_ids"]:
                    raise ValueError("held-out feature in verified geometry")
                ar, br = (a["name"], af), (b["name"], bf)
                if not (any(inside_polygon(poly, a["features"][af][:2]) for poly in polygons[ar[0]]) and
                        any(inside_polygon(poly, b["features"][bf][:2]) for poly in polygons[br[0]])):
                    counters["roi_excluded_rows"] += 1
                    continue
                counters["roi_eligible_rows"] += 1
                if ar[0] not in registered or br[0] not in registered:
                    counters["unregistered_endpoint_rows"] += 1
                    continue
                add(ar)
                add(br)
                ra, rb = root(ar), root(br)
                if ra != rb:
                    if size[ra] < size[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    size[ra] += size[rb]
    groups = defaultdict(set)
    for node in parent:
        groups[root(node)].add(node)
    components = {frozenset(group) for group in groups.values()}
    counters["connected_components"] = len(components)
    counters["same_image_conflict"] = sum(len({name for name, _ in group}) != len(group)
                                          for group in components)
    return counters, components


def verify_counts(points, saved, graph_counts, components):
    for key in ("verified_geometry_pairs", "verified_geometry_rows", "roi_eligible_rows",
                "roi_excluded_rows", "unregistered_endpoint_rows", "connected_components"):
        if saved[key] != graph_counts[key]:
            raise ValueError(f"stage count mismatch: {key}")
    if saved["rejected_by_reason"].get("same_image_conflict", 0) != graph_counts["same_image_conflict"]:
        raise ValueError("conflicted component count mismatch")
    if saved["accepted_tracks"] != len(points) or sum(saved["rejected_by_reason"].values())+len(points) != len(components):
        raise ValueError("accepted/rejected component accounting mismatch")
    tracks = [frozenset((o["image"], o["original_feature_id"]) for o in p["observations"])
              for p in points]
    if len(set(tracks)) != len(tracks) or not set(tracks).issubset(components):
        raise ValueError("accepted point is not a verified connected component")
    per_image = Counter(o["image"] for p in points for o in p["observations"])
    if (saved["accepted_tracks_three_or_more_views"] != sum(len(t) >= 3 for t in tracks) or
            saved["accepted_observations"] != sum(map(len, tracks)) or
            saved["registered_images_covered"] != len(per_image) or
            saved["accepted_observations_by_image"] != dict(sorted(per_image.items()))):
        raise ValueError("accepted point summary mismatch")
    return {"components_checked": len(components), "conflicts_checked": graph_counts["same_image_conflict"],
            "rejected_accounting_checked": True, "rejection_reasons_independently_classified": False}


def audit(points_path, report_path, source_root, roi_path):
    """Verify both lanes and the exact frozen source bytes used by the producer."""
    points_path, report_path = Path(points_path), Path(report_path)
    source_root, roi_path = Path(source_root), Path(roi_path)
    produced, report = load_json(points_path), load_json(report_path)
    if produced.get("schema") != "fixed_camera_object_tracks_v1":
        raise ValueError("unexpected points schema")
    if produced.get("cameras") != "heldout-v2-001 fixed nine-camera text model":
        raise ValueError("points claim a different camera model")
    if report.get("schema") != "fixed_camera_object_report_v1" or report.get("status") != "pass":
        raise ValueError("unexpected report schema/status")
    if report.get("predeclared_screens") != {
            "positive_depth_all_observations": True, "max_reprojection_px": 4.0,
            "min_max_ray_parallax_deg": 1.0,
            "max_normal_matrix_condition": 1e8}:
        raise ValueError("reported screens differ from checked limits")
    before, after = report["input_sha256_before"], report["input_sha256_after"]
    required = {str((source_root/lane/name).resolve())
                for lane in ("heldout-v2-001", "foreground-001")
                for name in ("database.db", "holdout.json", "provenance.json",
                             "verification-supervisor.json", "model_text/cameras.txt",
                             "model_text/images.txt", "model_text/points3D.txt")}
    required.add(str(roi_path.resolve()))
    verify_hash_manifest(before, after, required)
    baseline = source_root/"heldout-v2-001"
    if report.get("camera_source") != {
            "cameras_text": str(baseline/"model_text/cameras.txt"),
            "images_text": str(baseline/"model_text/images.txt")}:
        raise ValueError("camera source does not name frozen baseline model")
    if report.get("roi_source") != str(roi_path):
        raise ValueError("ROI source path differs")
    frozen = load_json(baseline/"verification-supervisor.json")
    if frozen["status"] != "pass" or not frozen["heldout_valid"]:
        raise ValueError("baseline frozen verification failed")
    for name, digest in frozen["hashes"]["model"].items():
        if sha256(baseline/"model_text"/name) != digest:
            raise ValueError(f"baseline camera model changed: {name}")
    source_manifest_path = baseline/"holdout.json"
    if sha256(source_manifest_path) != frozen["hashes"]["manifest"]:
        raise ValueError("baseline holdout manifest changed")
    if sha256(baseline/"database.db") != frozen["database"]["sha256"]:
        raise ValueError("baseline verified database changed")
    source_manifest = load_json(source_manifest_path)
    image_records = {row["name"]: row for row in source_manifest["images"]}
    for name, row in image_records.items():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen image changed: {name}")
    cams = camera_rows(baseline/"model_text"/"images.txt", baseline/"model_text"/"cameras.txt")
    if len(cams) != 9 or not set(cams).issubset(image_records):
        raise ValueError("expected nine baseline camera poses")
    if report.get("comparison_only") != {
            "baseline_registered_images": 9, "baseline_points": 1773,
            "baseline_all_observations_inside_roi_points": 151}:
        raise ValueError("comparison-only baseline metadata differs")
    if report.get("heldout", {}).get("pair_count") != 1121 or not report["heldout"].get("population_unchanged"):
        raise ValueError("heldout report mismatch")
    roi = load_json(roi_path)
    if sha256(roi_path) != "4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853":
        raise ValueError("approved ROI bytes changed")
    if roi.get("schema") != "foreground_roi_v1":
        raise ValueError("unexpected ROI contract")
    polygons = {r["name"]: r["polygons"] for r in roi["images"]}
    if set(polygons) != set(image_records):
        raise ValueError("ROI does not cover all source images")
    for name, row in image_records.items():
        roi_row = next(r for r in roi["images"] if r["name"] == name)
        if roi_row["original_sha256"] != row["sha256"] or (roi_row["width"], roi_row["height"]) != (row["width"], row["height"]):
            raise ValueError(f"ROI image metadata mismatch: {name}")
    features = {n: row["features"] for n, row in image_records.items()}
    heldout = {n: set(row["heldout_ids"]) for n, row in image_records.items()}
    lanes = produced["lanes"]
    if set(lanes) != {"baseline_roi", "foreground"}:
        raise ValueError("expected baseline ROI and foreground lanes")
    result = {}
    for name, rows in lanes.items():
        lane_dir = baseline if name == "baseline_roi" else source_root/"foreground-001"
        lane_manifest = source_manifest if name == "baseline_roi" else load_json(lane_dir/"holdout.json")
        if [(r["name"], r["features"], r["heldout_ids"]) for r in lane_manifest["images"]] != [
                (r["name"], r["features"], r["heldout_ids"]) for r in source_manifest["images"]]:
            raise ValueError(f"heldout population differs in {name}")
        geometry = verify_lane(rows, cams, features, polygons, heldout)
        graph_counts, components = verified_graph(lane_dir/"database.db", lane_manifest, polygons, set(cams))
        counts = verify_counts(rows, report["stage_counts"][name], graph_counts, components)
        result[name] = {**geometry, **counts}
    return {"status": "pass", "lanes": result, "frozen_camera_count": len(cams),
            "source_hash_count": len(before), "counts_are_ground_truth_quality": False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2]/"build-opencv/colmap-sparse")
    parser.add_argument("--roi", type=Path, default=Path(__file__).resolve().parents[2]/"tests/datasets/tree-envelope-v1.json")
    parser.add_argument("--output", type=Path, help="fresh path for a persistent independent verification record")
    args = parser.parse_args()
    source_before = load_json(args.report)["input_sha256_before"]
    artifacts = {"points": args.points.resolve(), "report": args.report.resolve(),
                 "verifier_code": Path(__file__).resolve()}
    artifact_before = {key: sha256(path) for key, path in artifacts.items()}
    result = audit(args.points, args.report, args.source_root, args.roi)
    source_after = {path: sha256(Path(path)) for path in source_before}
    artifact_after = {key: sha256(path) for key, path in artifacts.items()}
    if source_before != source_after or artifact_before != artifact_after:
        raise ValueError("source, artifact, or verifier changed during independent audit")
    result["evidence"] = {
        "points_path": str(artifacts["points"]), "report_path": str(artifacts["report"]),
        "verifier_path": str(artifacts["verifier_code"]),
        "artifact_sha256_before": artifact_before,
        "artifact_sha256_after": artifact_after,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "numeric_runtime": {"numpy_version": np.__version__}}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = args.output.resolve()
        if not target.parent.is_dir():
            raise ValueError("output parent directory does not exist")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
