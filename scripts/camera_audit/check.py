#!/usr/bin/env python3
"""Independent, read-only audit of the pinned COLMAP-to-scene camera mapping."""

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

RESERVE = 10 * 1024**3


def bounded(path, limit):
    if path.stat().st_size > limit:
        raise ValueError(f"input exceeds {limit} bytes: {path}")
    return path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def mul(a, s):
    return tuple(x*s for x in a)


def unit(a):
    return mul(a, 1/math.sqrt(dot(a, a)))


def rotate(q, v):
    # Quaternion sandwich q (0,v) q^-1, expressed as cross products.
    n = math.sqrt(dot(q, q))
    w, x, y, z = (a/n for a in q)
    u = (x, y, z)
    return add(v, add(mul(cross(u, v), 2*w), mul(cross(u, cross(u, v)), 2)))


def rows(q):
    # Matrix columns are rotated basis vectors, independent of importer formula.
    cols = [rotate(q, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return [tuple(c[i] for c in cols) for i in range(3)]


def mv(R, v):
    return tuple(dot(r, v) for r in R)


def trans(R):
    return tuple(zip(*R))


def mm(A, B):
    return tuple(tuple(dot(r, c) for c in trans(B)) for r in A)


def center(R, t):
    return mul(mv(trans(R), t), -1)


def project(K, R, t, X):
    x, y, z = add(mv(R, X), t)
    assert z > 0
    return (K[0]*x/z+K[2], K[1]*y/z+K[3])


def resize_K(K, source_size, output_size):
    sx, sy = output_size[0]/source_size[0], output_size[1]/source_size[1]
    return (K[0]*sx, K[1]*sy, K[2]*sx-.5, K[3]*sy-.5)


def image_size(path):
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG: {path}")
        while True:
            marker = f.read(1)
            if not marker:
                raise ValueError(f"no JPEG dimensions: {path}")
            if marker != b"\xff":
                continue
            code = f.read(1)
            while code == b"\xff":
                code = f.read(1)
            if code in (b"\xd8", b"\xd9", b"\x01"):
                continue
            length = struct.unpack(">H", f.read(2))[0]
            if code[0] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                data = f.read(5)
                return (struct.unpack(">H", data[3:5])[0], struct.unpack(">H", data[1:3])[0])
            f.seek(length-2, 1)


def png_size(path):
    with path.open("rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR"
    return struct.unpack(">II", head[16:24])


def parse_cameras(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        assert p[1] == "PINHOLE" and len(p) == 8
        result[int(p[0])] = ((int(p[2]), int(p[3])), tuple(map(float, p[4:8])))
    return result


def parse_images(path):
    lines = path.read_text().splitlines()
    result = {}
    for i, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        p = line.split(maxsplit=9)
        if len(p) == 10 and p[0].isdigit():
            assert i+1 < len(lines)
            result[int(p[0])] = (tuple(map(float, p[1:5])), tuple(map(float, p[5:8])), int(p[8]), p[9])
    return result


def match_record(entry, pose, camera, dimensions):
    q, t, camera_id, name = pose
    assert entry["name"] == name, "image ID/name pairing mismatch"
    assert entry["camera_id"] == camera_id, "image/camera ID mismatch"
    assert entry["camera"]["model"] == "PINHOLE"
    assert (entry["camera"]["width"], entry["camera"]["height"]) == camera[0] == dimensions, "image/camera dimensions mismatch"
    assert max(abs(a-b) for a, b in zip(entry["camera"]["params"], camera[1])) < 1e-8, "camera parameter mismatch"
    assert max(abs(a-b) for a, b in zip(entry["qvec"], q)) < 1e-8
    assert max(abs(a-b) for a, b in zip(entry["tvec"], t)) < 1e-8


def flat_error(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def geometry_checks(views):
    maximum = 0.0
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            A, B = views[i], views[j]
            Ra, ta, Ka = A["R"], A["t"], A["K"]
            Rb, tb, Kb = B["R"], B["t"], B["K"]
            relR = mm(Rb, trans(Ra))
            relT = sub(tb, mv(relR, ta))
            assert flat_error(relT, sub(mv(Rb, center(Ra, ta)), mv(Rb, center(Rb, tb)))) < 1e-9
            ca, cb = center(Ra, ta), center(Rb, tb)
            baseline = sub(cb, ca)
            assert math.sqrt(dot(baseline, baseline)) > 1e-8
            # A valid common rectified frame: horizontal baseline, averaged forward axis.
            ex = unit(baseline)
            forward = add(mv(trans(Ra), (0, 0, 1)), mv(trans(Rb), (0, 0, 1)))
            ez = unit(sub(forward, mul(ex, dot(forward, ex))))
            ey = cross(ez, ex)
            for depth in (3.0, 7.0):
                X = add(add(mul(add(ca, cb), .5), mul(ez, depth)), mul(ey, .13*depth))
                pa, pb = project(Ka, Ra, ta, X), project(Kb, Rb, tb, X)
                na = ((pa[0]-Ka[2])/Ka[0], (pa[1]-Ka[3])/Ka[1], 1)
                nb = ((pb[0]-Kb[2])/Kb[0], (pb[1]-Kb[3])/Kb[1], 1)
                epipolar = dot(nb, cross(relT, mv(relR, na)))
                maximum = max(maximum, abs(epipolar))
                assert abs(epipolar) < 1e-9
                ya = dot(ey, sub(X, ca))/dot(ez, sub(X, ca))
                yb = dot(ey, sub(X, cb))/dot(ez, sub(X, cb))
                assert abs(ya-yb) < 1e-10
    return maximum


def self_test():
    q = unit((.4, -.2, .5, .7))
    R = rows(q)
    t = (.7, -1.2, 2.3)
    C = center(R, t)
    assert flat_error(C, t) > .1 and math.dist(add(mv(R, C), t), (0, 0, 0)) < 1e-12
    source = (5031, 3191)
    output = (741, 469)
    K = (3883.2, 3911.7, 2411.4, 1733.8)
    scaled = resize_K(K, source, output)
    for src, dst, s in ((K[2], scaled[2], output[0]/source[0]), (K[3], scaled[3], output[1]/source[1])):
        assert abs(dst-((src-.5+.5)*s-.5)) < 1e-12
    X = add(C, mul(mv(trans(R), (0, 0, 1)), 8))
    assert math.dist(project(scaled, R, t, X), (scaled[2], scaled[3])) < 1e-10
    entry = {"name":"a.jpg", "camera_id":4, "camera":{"model":"PINHOLE", "width":5031, "height":3191, "params":list(K)}, "qvec":list(q), "tvec":list(t)}
    pose = (q, t, 4, "a.jpg")
    match_record(entry, pose, (source, K), source)
    for bad in ((q, t, 5, "a.jpg"), (q, t, 4, "b.jpg")):
        try:
            match_record(entry, bad, (source, K), source)
        except AssertionError:
            pass
        else:
            raise AssertionError("mismatched image/camera record escaped")
    other_q = unit((.6, .1, -.3, .7))
    views = [{"R":R, "t":t, "K":scaled}, {"R":rows(other_q), "t":(-.4, .6, 1.2), "K":(720, 711, 371.2, 235.9)}]
    geometry_checks(views)


def audit(dataset, scene):
    manifest_path = dataset/"manifest.json"
    bounded(manifest_path, 1024**2)
    manifest = json.loads(manifest_path.read_text())
    cameras_path = dataset/manifest["colmap"]["cameras"]
    images_path = dataset/manifest["colmap"]["images"]
    bounded(cameras_path, 1024**2)
    bounded(images_path, 1024**2)
    cameras, images = parse_cameras(cameras_path), parse_images(images_path)
    hashes = {str(manifest_path):sha256(manifest_path)}
    for item in manifest["source_files"]:
        path = dataset/item["path"]
        bounded(path, 1024**2)
        assert path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"]
        hashes[str(path)] = item["sha256"]
    tsv_path = scene/"views.tsv"
    bounded(tsv_path, 1024**2)
    hashes[str(tsv_path)] = sha256(tsv_path)
    conversion_path = scene/"conversion.json"
    bounded(conversion_path, 1024**2)
    hashes[str(conversion_path)] = sha256(conversion_path)
    lines = tsv_path.read_text().splitlines()
    assert len(lines) == len(manifest["selected_images"]) == 10
    conversion = json.loads(conversion_path.read_text())
    assert len(conversion["views"]) == 10
    views = []
    max_intrinsic_error = max_rotation_error = max_translation_error = max_meta_error = 0.0
    for index, (entry, line, converted) in enumerate(zip(manifest["selected_images"], lines, conversion["views"])):
        source = dataset/entry["path"]
        bounded(source, 32*1024**2)
        assert source.stat().st_size == entry["size_bytes"] and sha256(source) == entry["sha256"]
        hashes[str(source)] = entry["sha256"]
        pose = images[entry["id"]]
        original = cameras[pose[2]]
        match_record(entry, pose, original, image_size(source))
        p = line.split()
        assert len(p) == 20 and int(p[0]) == index
        out_path = Path(p[3])
        assert out_path.resolve() == (scene/f"views/view_{index:04d}.mve/undistorted.png").resolve()
        bounded(out_path, 20*1024**2)
        output_size = (int(p[1]), int(p[2]))
        assert png_size(out_path) == output_size
        hashes[str(out_path)] = sha256(out_path)
        assert converted["id"] == entry["id"] and converted["name"] == entry["name"]
        assert Path(converted["source"]).resolve() == source.resolve()
        assert Path(converted["output_image"]).resolve() == out_path.resolve()
        K = resize_K(original[1], original[0], output_size)
        R = rows(pose[0])
        actual_R = tuple(float(x) for x in p[8:17])
        actual_t = tuple(float(x) for x in p[17:20])
        max_intrinsic_error = max(max_intrinsic_error, flat_error(K, tuple(map(float, p[4:8]))))
        max_rotation_error = max(max_rotation_error, flat_error(tuple(x for row in R for x in row), actual_R))
        max_translation_error = max(max_translation_error, flat_error(pose[1], actual_t))
        assert flat_error(converted["rotation"], actual_R) < 1e-12
        assert flat_error(converted["translation"], actual_t) < 1e-12
        meta_path = scene/f"views/view_{index:04d}.mve/meta.ini"
        bounded(meta_path, 1024**2)
        hashes[str(meta_path)] = sha256(meta_path)
        meta = {}
        for line in meta_path.read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                meta[key] = value
        assert int(meta["id"]) == index and int(meta["name"]) == entry["id"]
        max_meta_error = max(max_meta_error, flat_error(tuple(map(float, meta["rotation"].split())), actual_R), flat_error(tuple(map(float, meta["translation"].split())), actual_t))
        assert flat_error(tuple(map(float, meta["principal_point"].split())), ((K[2]+.5)/output_size[0], (K[3]+.5)/output_size[1])) < 1e-10
        aspect = K[1]/K[0]
        focal = K[0]/output_size[0] if output_size[0]/output_size[1]*aspect >= 1 else K[1]/output_size[1]
        assert abs(float(meta["focal_length"])-focal) < 1e-10
        assert abs(float(meta["pixel_aspect"])-aspect) < 1e-10
        views.append({"R":R, "t":pose[1], "K":K, "center":center(R, pose[1]), "id":entry["id"], "camera_id":pose[2]})
    assert max_intrinsic_error < 1e-9 and max_rotation_error < 1e-9 and max_translation_error < 1e-9 and max_meta_error < 1e-8
    epipolar = geometry_checks(views)
    return {"result":"pass", "source_revision":manifest["source"]["revision"], "view_count":len(views), "pair_count":len(views)*(len(views)-1)//2, "max_intrinsic_error":max_intrinsic_error, "max_rotation_error":max_rotation_error, "max_translation_error":max_translation_error, "max_meta_error":max_meta_error, "max_analytic_epipolar_error":epipolar, "view_records":[{"index":i, "image_id":v["id"], "camera_id":v["camera_id"], "center":v["center"]} for i,v in enumerate(views)], "input_sha256":hashes}


def main():
    if sys.flags.optimize:
        raise RuntimeError("camera audit requires assertions; do not run Python with -O")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--scene", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    self_test()
    if a.self_test:
        print("synthetic camera, resize, pairing, and geometry checks passed")
        return
    if not (a.dataset and a.scene and a.report):
        p.error("--dataset, --scene and --report are required")
    if a.report.exists():
        p.error("report already exists")
    if shutil.disk_usage(a.report.parent).free < RESERVE:
        p.error("10 GiB free-space reserve required")
    result = audit(a.dataset, a.scene)
    payload = json.dumps(result, indent=2)+"\n"
    if len(payload.encode()) > 100*1024:
        raise ValueError("report exceeds 100 KiB")
    a.report.write_text(payload)
    print(f"{result['result']}: {result['view_count']} views, {result['pair_count']} analytic pairs; {a.report}")


if __name__ == "__main__":
    main()
