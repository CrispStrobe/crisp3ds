#!/usr/bin/env python3
"""Independent, stdlib-only audit of an estimated COLMAP text reconstruction.

Coordinates are COLMAP keypoint coordinates (pixel centres start at 0.5).
No pose, intrinsics, feature match, or model parameter is estimated here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def finite(values):
    return all(math.isfinite(float(v)) for v in values)


def cameras(path):
    result = {}
    expected = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
                "RADIAL": 5, "OPENCV": 8}
    for line in lines(path):
        fields = line.split()
        cid, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        params = list(map(float, fields[4:]))
        if cid in result or model not in expected or len(params) != expected[model]:
            raise ValueError(f"bad or unsupported camera {cid}: {model}")
        if width <= 0 or height <= 0 or not finite(params):
            raise ValueError(f"invalid camera {cid}")
        result[cid] = dict(model=model, width=width, height=height, params=params)
    return result


def rotation(q):
    n = math.sqrt(sum(v * v for v in q))
    if abs(n - 1) > 1e-5:
        raise ValueError(f"non-unit quaternion: {n}")
    w, x, y, z = [v / n for v in q]
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def mv(a, v):
    return [dot(row, v) for row in a]


def mt(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    bt = mt(b)
    return [[dot(row, col) for col in bt] for row in a]


def images(path):
    raw = []
    pending = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        if pending is None:
            if line.strip():
                pending = line.strip()
        else:
            raw.extend((pending, line.strip()))
            pending = None
    if pending is not None:
        raise ValueError("images.txt has incomplete two-line record")
    result = {}
    names = set()
    for head, obs in zip(raw[::2], raw[1::2]):
        fields = head.split()
        if len(fields) < 10:
            raise ValueError("short images.txt header")
        iid = int(fields[0]); q = list(map(float, fields[1:5])); t = list(map(float, fields[5:8]))
        cid = int(fields[8]); name = " ".join(fields[9:])
        data = obs.split()
        if iid in result or name in names or len(data) % 3 or not finite(q+t):
            raise ValueError(f"invalid image record {iid}")
        r = rotation(q)
        if max(abs(dot(row, other) - (1 if i == j else 0))
               for i, row in enumerate(r) for j, other in enumerate(r)) > 1e-5:
            raise ValueError(f"image {iid} rotation is not SO(3)")
        determinant = dot(r[0], [r[1][1]*r[2][2]-r[1][2]*r[2][1],
                                  r[1][2]*r[2][0]-r[1][0]*r[2][2],
                                  r[1][0]*r[2][1]-r[1][1]*r[2][0]])
        if abs(determinant - 1) > 1e-5:
            raise ValueError(f"image {iid} improper rotation")
        xy = [(float(data[i]), float(data[i+1]), int(data[i+2])) for i in range(0, len(data), 3)]
        if not finite(v for x, y, _ in xy for v in (x, y)):
            raise ValueError(f"image {iid} nonfinite observation")
        center = mv(mt(r), [-v for v in t])
        result[iid] = dict(name=name, camera_id=cid, R=r, t=t, center=center, xy=xy)
        names.add(name)
    return result


def points(path):
    result = {}
    for line in lines(path):
        f = line.split()
        pid = int(f[0]); xyz = list(map(float, f[1:4])); error = float(f[7])
        track = [(int(f[i]), int(f[i+1])) for i in range(8, len(f), 2)]
        if pid in result or len(f) < 10 or (len(f)-8) % 2 or not finite(xyz+[error]) or error < 0:
            raise ValueError(f"invalid point {pid}")
        result[pid] = dict(xyz=xyz, error=error, track=track)
    return result


def intrinsic(c):
    p = c["params"]
    if c["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    return p[:4]


def distort(c, x, y):
    p = c["params"]; model = c["model"]
    r2 = x*x+y*y
    if model == "SIMPLE_RADIAL":
        scale = 1 + p[3]*r2
    elif model == "RADIAL":
        scale = 1 + p[3]*r2 + p[4]*r2*r2
    elif model == "OPENCV":
        k1, k2, p1, p2 = p[4:8]
        scale = 1+k1*r2+k2*r2*r2
        return (x*scale+2*p1*x*y+p2*(r2+2*x*x),
                y*scale+p1*(r2+2*y*y)+2*p2*x*y)
    else:
        scale = 1
    return x*scale, y*scale


def project(c, xyz):
    if xyz[2] <= 0:
        return None
    fx, fy, cx, cy = intrinsic(c)
    x, y = distort(c, xyz[0]/xyz[2], xyz[1]/xyz[2])
    return fx*x+cx, fy*y+cy


def undistort(c, xy):
    fx, fy, cx, cy = intrinsic(c)
    if fx <= 0 or fy <= 0:
        raise ValueError("nonpositive focal length")
    xd, yd = (xy[0]-cx)/fx, (xy[1]-cy)/fy
    x, y = xd, yd
    for _ in range(30):
        dx, dy = distort(c, x, y)
        ex, ey = xd-dx, yd-dy
        if max(abs(ex), abs(ey)) < 1e-12:
            break
        step = 1e-6
        j11 = (distort(c, x+step, y)[0]-dx)/step
        j21 = (distort(c, x+step, y)[1]-dy)/step
        j12 = (distort(c, x, y+step)[0]-dx)/step
        j22 = (distort(c, x, y+step)[1]-dy)/step
        det = j11*j22-j12*j21
        if abs(det) < 1e-12:
            raise ValueError("singular distortion inverse")
        x += (j22*ex-j12*ey)/det
        y += (-j21*ex+j11*ey)/det
    if not finite((x, y)) or max(abs(distort(c, x, y)[0]-xd), abs(distort(c, x, y)[1]-yd)) > 1e-8:
        raise ValueError("distortion inverse did not converge")
    return x, y


def pair_sampson(a, b, ca, cb, xya, xyb):
    # E = [t_b - R_b R_a^T t_a]x R_b R_a^T, using world-to-camera poses.
    r = mm(b["R"], mt(a["R"]))
    t = [v-u for v, u in zip(b["t"], mv(r, a["t"]))]
    tx = [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]
    e = mm(tx, r)
    xa = list(undistort(ca, xya))+[1]
    xb = list(undistort(cb, xyb))+[1]
    # Convert normalized-coordinate E to undistorted pixel-coordinate F.
    fxa, fya, _, _ = intrinsic(ca); fxb, fyb, _, _ = intrinsic(cb)
    ea = mv(e, xa); etb = mv(mt(e), xb)
    den = (ea[0]/fxb)**2+(ea[1]/fyb)**2+(etb[0]/fxa)**2+(etb[1]/fya)**2
    if den <= 1e-20 or not math.isfinite(den):
        return None
    return abs(dot(xb, ea))/math.sqrt(den)


def stats(values):
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    n = len(clean)
    return dict(count=len(values), finite_count=n, invalid_count=len(values)-n,
                median_px=(clean[(n-1)//2]+clean[n//2])/2 if n else None,
                p90_px=clean[math.ceil(.9*n)-1] if n else None,
                within_1px=sum(v <= 1 for v in clean),
                within_2px=sum(v <= 2 for v in clean),
                within_4px=sum(v <= 4 for v in clean))


def spatial_split_overlap(features, heldout_ids, radius=0.25):
    """Conservative clone audit: any heldout/train centres within radius leak."""
    buckets = {}
    for i, p in enumerate(features):
        if i not in heldout_ids:
            key = (math.floor(p[0]/radius), math.floor(p[1]/radius))
            buckets.setdefault(key, []).append(p)
    overlap = 0
    for i in heldout_ids:
        p = features[i]; ix, iy = math.floor(p[0]/radius), math.floor(p[1]/radius)
        if any(math.hypot(p[0]-q[0], p[1]-q[1]) <= radius
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for q in buckets.get((ix+dx, iy+dy), ())):
            overlap += 1
    return overlap


def closed_cycles(pairs):
    edges = set()
    neighbors = {}
    for (a, b) in pairs:
        if a[0] == b[0]:
            continue
        edge = frozenset((a, b)); edges.add(edge)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    cycles = set()
    for a, adjacent in neighbors.items():
        ordered = sorted(adjacent)
        for i, b in enumerate(ordered):
            for c in ordered[i+1:]:
                if len({a[0], b[0], c[0]}) == 3 and frozenset((b, c)) in edges:
                    cycles.add(tuple(sorted((a, b, c))))
    return sorted(cycles)


def three_view(cycles, names, cameras_by_id, features):
    outcome = dict(closed_cycles=len(cycles), registered_cycles=0,
                   low_parallax=0, invalid_triangulation=0, invalid_distortion=0,
                   nonpositive_depth=0, third_view_errors=[])
    for cycle in cycles:
        if any(name not in names for name, _ in cycle):
            continue
        outcome["registered_cycles"] += 1
        (na, ia), (nb, ib), (nc, ic) = cycle  # lexical image order is frozen
        a, b, c = names[na], names[nb], names[nc]
        ca, cb, cc = [cameras_by_id[v["camera_id"]] for v in (a, b, c)]
        try:
            xa = list(undistort(ca, features[na][ia]))+[1]
            xb = list(undistort(cb, features[nb][ib]))+[1]
        except (ValueError, OverflowError, ZeroDivisionError):
            outcome["invalid_distortion"] += 1
            outcome["third_view_errors"].append(None)
            continue
        da, db = mv(mt(a["R"]), xa), mv(mt(b["R"]), xb)
        da = [v/math.sqrt(dot(da, da)) for v in da]
        db = [v/math.sqrt(dot(db, db)) for v in db]
        cosine = max(-1, min(1, dot(da, db)))
        angle = math.degrees(math.acos(cosine))
        if angle < 1.0:
            outcome["low_parallax"] += 1
            continue
        denominator = 1-cosine*cosine
        if denominator < 1e-12:
            outcome["invalid_triangulation"] += 1
            continue
        delta = [v-u for u, v in zip(a["center"], b["center"])]
        right_a, right_b = dot(da, delta), -dot(db, delta)
        sa = (right_a+cosine*right_b)/denominator
        sb = (cosine*right_a+right_b)/denominator
        pa = [v+sa*d for v, d in zip(a["center"], da)]
        pb = [v+sb*d for v, d in zip(b["center"], db)]
        point = [(u+v)/2 for u, v in zip(pa, pb)]
        depths = [mv(v["R"], [p-q for p, q in zip(point, v["center"])])[2]
                  for v in (a, b, c)]
        if min(depths) <= 0:
            outcome["nonpositive_depth"] += 1
            outcome["third_view_errors"].append(None)
            continue
        camera_point = [u+v for u, v in zip(mv(c["R"], point), c["t"])]
        predicted = project(cc, camera_point)
        observed = features[nc][ic]
        outcome["third_view_errors"].append(math.hypot(predicted[0]-observed[0],
                                                         predicted[1]-observed[1]))
    outcome["third_view_reprojection"] = stats(outcome.pop("third_view_errors"))
    return outcome


def audit_database(path, manifest, errors):
    """Check the actual mapper feature table and all pair matches, not just JSON claims."""
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    by_name = {im["name"]: im for im in manifest["images"]}
    image_rows = db.execute("SELECT image_id,name FROM images").fetchall()
    if set(name for _, name in image_rows) != set(by_name):
        errors.append("database image names differ from manifest")
    by_id = dict(image_rows)
    for iid, name in image_rows:
        if name not in by_name:
            continue
        im = by_name[name]; mapping = im.get("compact_to_original")
        if mapping is None:
            errors.append(f"missing compact mapper mapping: {name}")
            continue
        row = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (iid,)).fetchone()
        if row is None:
            errors.append(f"missing database keypoints: {name}")
            continue
        count, cols, blob = row
        if count != len(mapping) or len(blob) != count*cols*4 or cols < 2:
            errors.append(f"database keypoint table shape mismatch: {name}")
            continue
        for idx, original in enumerate(mapping):
            if original < 0 or original >= len(im["features"]):
                errors.append(f"invalid mapping: {name}:{idx}")
                continue
            x, y = struct.unpack_from("<ff", blob, idx*cols*4)
            expected = im["features"][original]
            if math.hypot(x-expected[0], y-expected[1]) > 1e-4:
                errors.append(f"database keypoint coordinate mismatch: {name}:{idx}")
            if original in set(im["heldout_ids"]):
                errors.append(f"heldout row in database: {name}:{idx}")
    pair_base = 2147483647
    actual_matches = set()
    for pair_id, rows, cols, blob in db.execute("SELECT pair_id,rows,cols,data FROM matches"):
        if rows == 0 and blob is None:
            continue
        if cols != 2 or len(blob) != rows*cols*4:
            errors.append(f"database match table shape mismatch: {pair_id}")
            continue
        id2 = pair_id % pair_base; id1 = (pair_id-id2)//pair_base
        if id1 not in by_id or id2 not in by_id:
            errors.append(f"database match has missing image: {pair_id}")
            continue
        na, nb = by_id[id1], by_id[id2]
        if na not in by_name or nb not in by_name:
            continue
        ma, mb = by_name[na].get("compact_to_original", []), by_name[nb].get("compact_to_original", [])
        for ia, ib in struct.iter_unpack("<II", blob):
            if ia >= len(ma) or ib >= len(mb):
                errors.append(f"database match index out of range: {pair_id}:{ia}:{ib}")
                continue
            actual_matches.add((na, ma[ia], nb, mb[ib]))
    exported_matches = {(r["image1"], r["feature1"], r["image2"], r["feature2"])
                        for r in manifest["training_matches"]}
    if actual_matches != exported_matches or len(exported_matches) != len(manifest["training_matches"]):
        errors.append(f"database/manifest training matches differ: database {len(actual_matches)}, manifest {len(exported_matches)}")
    db.close()
    return dict(image_count=len(image_rows), training_match_count=len(actual_matches), sha256=sha256(path))


def verify(manifest_path, model_dir, database_path=None):
    manifest_path = Path(manifest_path).resolve(); model_dir = Path(model_dir).resolve()
    m = json.loads(manifest_path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    cs = cameras(model_dir/"cameras.txt"); ims = images(model_dir/"images.txt"); pts = points(model_dir/"points3D.txt")
    errors = []
    names = {v["name"]: v for v in ims.values()}
    for iid, im in ims.items():
        if im["camera_id"] not in cs:
            errors.append(f"image {iid}: missing camera")
    features = {}
    held = {}
    split_overlap_count = 0
    input_paths = []
    for im in m["images"]:
        name = im["name"]
        if name in features:
            errors.append(f"duplicate manifest image: {name}")
        features[name] = im["features"]
        held[name] = set(im["heldout_ids"])
        if len(held[name]) != len(im["heldout_ids"]) or any(not isinstance(i, int) or i < 0 or i >= len(im["features"]) for i in held[name]):
            errors.append(f"invalid heldout feature IDs: {name}")
        overlap = spatial_split_overlap(im["features"], held[name])
        split_overlap_count += overlap
        if overlap:
            errors.append(f"spatially colocated train/heldout features: {name}: {overlap} heldout rows within 0.25 px")
        p = Path(im["path"])
        input_paths.append(p)
        if sha256(p) != im["sha256"]:
            errors.append(f"input hash mismatch: {p}")
        if name in names:
            c = cs[names[name]["camera_id"]]
            if (c["width"], c["height"]) != (im["width"], im["height"]):
                errors.append(f"image dimensions mismatch: {name}")
            mapping = im.get("compact_to_original")
            if mapping is not None:
                model_xy = names[name]["xy"]
                if len(mapping) != len(model_xy):
                    errors.append(f"mapper row count mismatch: {name}")
                else:
                    for idx, original in enumerate(mapping):
                        if not isinstance(original, int) or original < 0 or original >= len(features[name]):
                            errors.append(f"invalid mapper row {name}:{idx}")
                            continue
                        if original in held[name]:
                            errors.append(f"heldout feature in mapper row {name}:{idx}")
                        observed_xy = model_xy[idx][:2]
                        expected_xy = features[name][original]
                        if math.hypot(observed_xy[0]-expected_xy[0], observed_xy[1]-expected_xy[1]) > 1e-3:
                            errors.append(f"mapper coordinate mismatch {name}:{idx}")
    if len(features) != 10:
        errors.append(f"expected ten input images, got {len(features)}")
    if set(names) - set(features):
        errors.append("registered image absent from manifest")
    db_audit = audit_database(database_path, m, errors) if database_path else None
    for k, match in enumerate(m["training_matches"]):
        for side in (1, 2):
            name, fid = match[f"image{side}"], match[f"feature{side}"]
            if name not in features or not isinstance(fid, int) or fid < 0 or fid >= len(features[name]):
                errors.append(f"training match {k}: invalid feature reference")
            elif fid in held[name]:
                errors.append(f"training match {k}: heldout feature leaked: {name}:{fid}")
    heldout = []
    for k, match in enumerate(m["heldout_pairs"]):
        refs = []
        for side in (1, 2):
            name, fid = match[f"image{side}"], match[f"feature{side}"]
            if name not in features or not isinstance(fid, int) or fid < 0 or fid >= len(features[name]):
                errors.append(f"heldout pair {k}: invalid feature reference")
                break
            if fid not in held[name]:
                errors.append(f"heldout pair {k}: non-heldout feature: {name}:{fid}")
            refs.append((name, fid))
        if len(refs) == 2:
            heldout.append(refs)
    # Every model track is checked both ways; this is independent of the export's summary counts.
    observed = set(); repro = []; depth_positive = 0; depth_nonpositive = 0
    multi_image_observation_tracks = []
    for pid, point in pts.items():
        if len(point["track"]) < 2 or len(set(point["track"])) != len(point["track"]):
            errors.append(f"point {pid}: short or duplicate track")
        if len({iid for iid, _ in point["track"]}) != len(point["track"]):
            multi_image_observation_tracks.append(pid)
        for iid, idx in point["track"]:
            if iid not in ims or not 0 <= idx < len(ims[iid]["xy"]):
                errors.append(f"point {pid}: invalid track reference {iid}:{idx}")
                continue
            if (iid, idx) in observed:
                errors.append(f"observation {iid}:{idx} belongs to multiple points")
            observed.add((iid, idx))
            x, y, linked = ims[iid]["xy"][idx]
            if linked != pid:
                errors.append(f"point {pid}: asymmetric track link {iid}:{idx}")
            camxyz = [u+v for u, v in zip(mv(ims[iid]["R"], point["xyz"]), ims[iid]["t"])]
            if camxyz[2] <= 0:
                depth_nonpositive += 1
            else:
                depth_positive += 1
                if ims[iid]["camera_id"] in cs:
                    uv = project(cs[ims[iid]["camera_id"]], camxyz)
                    repro.append(math.hypot(uv[0]-x, uv[1]-y))
    for iid, im in ims.items():
        for idx, (_, _, pid) in enumerate(im["xy"]):
            if pid != -1 and (iid, idx) not in observed:
                errors.append(f"image {iid}: dangling point link {idx}:{pid}")
    if depth_nonpositive:
        errors.append(f"{depth_nonpositive} sparse track observations have nonpositive depth")
    grouped = {}
    scored = 0
    for (na, fa), (nb, fb) in heldout:
        key = tuple(sorted((na, nb)))
        if na not in names or nb not in names or names[na]["camera_id"] not in cs or names[nb]["camera_id"] not in cs:
            continue
        ca, cb = cs[names[na]["camera_id"]], cs[names[nb]["camera_id"]]
        try:
            val = pair_sampson(names[na], names[nb], ca, cb, features[na][fa], features[nb][fb])
        except (ValueError, OverflowError, ZeroDivisionError):
            val = None
        grouped.setdefault(key, []).append(val)
        scored += 1
    cycle_report = three_view(closed_cycles(heldout), names, cs, features)
    input_hashes_after = {str(p): sha256(p) for p in input_paths}
    for im in m["images"]:
        if input_hashes_after[im["path"]] != im["sha256"]:
            errors.append(f"input changed during verification: {im['path']}")
    integrity_status = "pass" if not errors and scored else "fail"
    return dict(status=integrity_status, integrity_status=integrity_status,
                quality_accepted=False,
                quality_reason="diagnostic run; no predeclared reconstruction acceptance gate",
                heldout_valid=(split_overlap_count == 0),
                split_overlap_count=split_overlap_count,
                errors=errors,
                model=dict(registered_images=len(ims), cameras=len(cs),
                           camera_models={str(cid): cam for cid, cam in cs.items()}, points=len(pts),
                           multi_observation_same_image_tracks=multi_image_observation_tracks,
                           strict_one_observation_per_image_points=len(pts)-len(multi_image_observation_tracks),
                           track_observations=len(observed), positive_depth_observations=depth_positive,
                           nonpositive_depth_observations=depth_nonpositive,
                           reprojection=stats(repro), camera_centers={im["name"]: im["center"] for im in ims.values()}),
                heldout=dict(total_pairs=len(heldout), scored_pairs=scored,
                             unscored_pairs=len(heldout)-scored,
                             all=stats([v for vals in grouped.values() for v in vals]),
                             by_pair={" | ".join(k): stats(v) for k, v in sorted(grouped.items())},
                             three_view=cycle_report),
                database=db_audit,
                hashes=dict(manifest=sha256(manifest_path), verifier=sha256(__file__), inputs=input_hashes_after,
                            model={p.name: sha256(p) for p in
                            (model_dir/"cameras.txt", model_dir/"images.txt", model_dir/"points3D.txt")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="COLMAP text model directory")
    ap.add_argument("--database", help="actual COLMAP mapper database; recommended for independent exclusion audit")
    ap.add_argument("--output", help="optional JSON report path")
    args = ap.parse_args()
    result = verify(args.manifest, args.model, args.database)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized+"\n")
    print(serialized)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
