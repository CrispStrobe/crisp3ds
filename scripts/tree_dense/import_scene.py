#!/usr/bin/env python3
"""Bounded test-only COLMAP pose/photo import with measured ORB seed tracks."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from PIL import Image

RESERVE = 10 * 1024**3
MAX_BATCH = 1024**3


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("source path must be relative")
    resolved = (root / name).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("source path escapes dataset")
    return resolved


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras(path):
    cameras = {}
    for line in records(path):
        words = line.split()
        if len(words) < 7:
            raise ValueError("malformed COLMAP camera")
        camera_id, model, width, height = int(words[0]), words[1], int(words[2]), int(words[3])
        if model == "PINHOLE" and len(words) == 8:
            fx, fy, cx, cy = map(float, words[4:])
        elif model == "SIMPLE_PINHOLE" and len(words) == 7:
            f, cx, cy = map(float, words[4:])
            fx = fy = f
        else:
            raise ValueError(f"unsupported/distorted COLMAP camera model: {model}")
        if camera_id in cameras or width <= 0 or height <= 0 or width*height > 25_000_000 or not all(
            math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            raise ValueError("invalid COLMAP camera")
        cameras[camera_id] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def quaternion_rotation(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError("invalid quaternion")
    norm = math.sqrt(sum(x*x for x in q))
    if abs(norm-1) > 1e-3:
        raise ValueError("COLMAP quaternion is not unit length")
    w, x, y, z = (value/norm for value in q)
    return [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]


def read_images(path):
    images = {}
    with path.open(encoding="utf-8") as stream:
        while True:
            header = stream.readline()
            if not header:
                break
            if not header.strip() or header.startswith("#"):
                continue
            words = header.strip().split(maxsplit=9)
            if len(words) != 10:
                raise ValueError("malformed COLMAP image header")
            image_id = int(words[0])
            if image_id in images:
                raise ValueError("duplicate COLMAP image ID")
            q = list(map(float, words[1:5]))
            t = list(map(float, words[5:8]))
            if not all(math.isfinite(x) for x in t):
                raise ValueError("invalid COLMAP translation")
            observation_line = stream.readline()
            if observation_line == "":
                raise ValueError("missing COLMAP image observation line")
            images[image_id] = dict(id=image_id, camera_id=int(words[8]), name=words[9],
                                    rotation=quaternion_rotation(q), translation=t,
                                    source_observations=len(observation_line.split()) // 3)
    return images


def scaled_camera(camera, width, height):
    sx, sy = width/camera["width"], height/camera["height"]
    if abs(sx-sy) > 0.002:
        raise ValueError("resize altered image aspect ratio")
    # COLMAP uses the upper-left pixel center at (0.5, 0.5), whereas OpenCV
    # feature coordinates use (0, 0). Convert once before center-aware resize.
    return dict(width=width, height=height, fx=camera["fx"]*sx, fy=camera["fy"]*sy,
                cx=camera["cx"]*sx-.5, cy=camera["cy"]*sy-.5)


def mve_intrinsics(camera):
    width, height = camera["width"], camera["height"]
    fx, fy = camera["fx"], camera["fy"]
    aspect = fy/fx
    if (width/height)*aspect >= 1:
        focal = fx/width
    else:
        focal = fy/height
    return focal, aspect, (camera["cx"]+.5)/width, (camera["cy"]+.5)/height


def project(camera, rotation, translation, point):
    x, y, z = [sum(rotation[3*r+c]*point[c] for c in range(3)) + translation[r]
               for r in range(3)]
    if z <= 0:
        raise ValueError("seed behind camera")
    return camera["fx"]*x/z+camera["cx"], camera["fy"]*y/z+camera["cy"]


def parse_seeds(path, views):
    seeds, errors = [], []
    for line in records(path):
        words = line.split()
        if len(words) != 11:
            raise ValueError("malformed measured seed row")
        x, y, z = map(float, words[:3]); i=int(words[3]); u=float(words[4]); v=float(words[5])
        j=int(words[6]); s=float(words[7]); t=float(words[8])
        if not (0 <= i < j < len(views)) or not all(math.isfinite(a) for a in (x,y,z,u,v,s,t)):
            raise ValueError("invalid measured seed row")
        point = (x,y,z)
        for index, pixel in ((i,(u,v)), (j,(s,t))):
            view = views[index]
            reproj = project(view["camera"], view["rotation"], view["translation"], point)
            error = math.dist(reproj, pixel)
            if error > 2.05:
                raise ValueError(f"measured seed reprojection exceeds 2.05 px: {error}")
            errors.append(error)
        seeds.append((point, i, j))
    if not seeds or len(seeds) > 10000:
        raise ValueError("no measured seed tracks or too many")
    return seeds, errors


def convert(dataset, scene, seed_tool, max_width=768):
    dataset = dataset.resolve(strict=True)
    scene = scene.absolute()
    if scene.exists() or scene.is_symlink():
        raise FileExistsError(scene)
    if not 512 <= max_width <= 800:
        raise ValueError("max width must be 512..800")
    manifest_path = dataset/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["source_files"]:
        path = relative_file(dataset, entry["path"])
        if path.stat().st_size != entry["size_bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"source metadata hash/size mismatch: {entry['path']}")
    colmap = manifest["colmap"]
    cameras_path = relative_file(dataset, colmap["cameras"])
    images_path = relative_file(dataset, colmap["images"])
    cameras, images = read_cameras(cameras_path), read_images(images_path)
    selected = manifest["selected_images"]
    if not 2 <= len(selected) <= 12 or len({x["id"] for x in selected}) != len(selected):
        raise ValueError("selected images must contain 2..12 unique IDs")
    if sum(x["size_bytes"] for x in selected) > 250_000_000:
        raise ValueError("selected source images exceed 250 MB")
    source_paths = []
    for entry in selected:
        view = images[entry["id"]]
        if view["name"] != entry["name"] or view["camera_id"] not in cameras:
            raise ValueError("selected image differs from COLMAP pose record")
        source = relative_file(dataset, entry["path"])
        if source.stat().st_size != entry["size_bytes"] or digest(source) != entry["sha256"]:
            raise ValueError("selected image hash/size mismatch")
        source_paths.append(source)
    scene.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scene.parent).free < RESERVE + MAX_BATCH:
        raise RuntimeError("10 GiB disk reserve plus 1 GiB batch capacity required")
    scene.mkdir(parents=True)
    views_dir = scene/"views"
    views_dir.mkdir()
    views = []
    seed_rows = []
    for index, (entry, source) in enumerate(zip(selected, source_paths)):
        pose = images[entry["id"]]
        original = cameras[pose["camera_id"]]
        with Image.open(source) as image:
            if image.size != (original["width"], original["height"]):
                raise ValueError("photo dimensions differ from COLMAP camera")
            if image.width*image.height>25_000_000:
                raise ValueError("photo exceeds 25 megapixels")
            width = min(max_width, image.width)
            height = round(image.height*width/image.width)
            reduced = image.convert("RGB").resize((width,height), Image.Resampling.LANCZOS)
        camera = scaled_camera(original,width,height)
        view_dir = views_dir/f"view_{index:04d}.mve"
        view_dir.mkdir()
        path = view_dir/"undistorted.png"
        reduced.save(path, optimize=True)
        if path.stat().st_size > 20*1024**2:
            raise ValueError("resized image exceeds 20 MiB")
        focal, aspect, ppx, ppy = mve_intrinsics(camera)
        rotation, translation = pose["rotation"], pose["translation"]
        fields = ["[view]",f"id = {index}",f"name = {entry['id']}","[camera]",
                  f"focal_length = {focal:.12g}","radial_distortion = 0 0",
                  f"pixel_aspect = {aspect:.12g}",f"principal_point = {ppx:.12g} {ppy:.12g}",
                  "rotation = "+" ".join(f"{x:.12g}" for x in rotation),
                  "translation = "+" ".join(f"{x:.12g}" for x in translation)]
        (view_dir/"meta.ini").write_text("\n".join(fields)+"\n")
        views.append(dict(id=entry["id"], name=entry["name"], source=str(source),
                          camera=camera, rotation=rotation, translation=translation,
                          output_image=str(path), source_observations=pose["source_observations"]))
        seed_rows.append(" ".join(map(str, [index,width,height,path.resolve(),camera["fx"],
                                             camera["fy"],camera["cx"],camera["cy"],
                                             *rotation,*translation])))
    (scene/"views.tsv").write_text("\n".join(seed_rows)+"\n")
    with (scene/"seed-tool.log").open("w") as log:
        subprocess.run([str(seed_tool.resolve(strict=True)), str(scene/"views.tsv"),
                        str(scene/"measured-seeds.txt")], stdout=log, stderr=log,
                       check=True, timeout=120)
    seeds, errors = parse_seeds(scene/"measured-seeds.txt", views)
    bundle = ["drews 1.0",f"{len(views)} {len(seeds)}"]
    for view in views:
        focal, _, _, _ = mve_intrinsics(view["camera"])
        bundle.append(f"{focal:.12g} 0 0")
        for row in range(3):
            bundle.append(" ".join(f"{view['rotation'][3*row+col]:.12g}" for col in range(3)))
        bundle.append(" ".join(f"{x:.12g}" for x in view["translation"]))
    for number, (point, i, j) in enumerate(seeds):
        bundle.extend((" ".join(f"{x:.12g}" for x in point), "255 255 255",
                       f"2 {i} {number} 0 {j} {number} 0"))
    (scene/"synth_0.out").write_text("\n".join(bundle)+"\n")
    summary = dict(views=views, measured_seed_tracks=len(seeds), measured_observations=len(errors),
                   reprojection_rms_px=math.sqrt(sum(e*e for e in errors)/len(errors)),
                   reprojection_max_px=max(errors), image_width=max_width,
                   seed_method="mutual-ratio ORB, fixed-pose triangulation; <=2 px reprojection; >=1 degree parallax; 500/pair and 10000 total",
                   source_hashes_sha256={"manifest.json":digest(manifest_path),
                                         colmap["cameras"]:digest(cameras_path),
                                         colmap["images"]:digest(images_path)},
                   source_scale="arbitrary; supplied COLMAP poses have no verified physical scale")
    (scene/"conversion.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    if sum(p.stat().st_size for p in scene.rglob("*") if p.is_file()) > MAX_BATCH:
        raise RuntimeError("scene exceeded 1 GiB batch cap")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--seed-tool",type=Path,required=True)
    parser.add_argument("--max-width",type=int,default=768)
    args=parser.parse_args()
    summary=convert(args.dataset,args.scene,args.seed_tool,args.max_width)
    print(json.dumps({k:summary[k] for k in ("measured_seed_tracks","reprojection_rms_px","reprojection_max_px")},indent=2))


if __name__=="__main__":
    main()
