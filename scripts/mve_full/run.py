"""Bounded, camera-unknown photographs to MVE point cloud and mesh.

This runner uses only the selected, locally built MVE tools. It records actual
outputs and refuses to call a successful process a valid reconstruction when
the required geometric artifacts are empty.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import shutil
import signal
import subprocess
import time

from scripts.mve_full.verify_source import verify as verify_selected_source

try:
    import resource
except ImportError:  # Windows
    resource = None


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e"
NAMES = ("makescene", "sfmrecon", "dmrecon", "scene2pset", "fssrecon", "meshclean")
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ppm"}
RESERVE = 10 << 30
POLL = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            entry = Path(base) / name
            if entry.is_symlink():
                raise ValueError(f"symlink in output: {entry}")
        for name in files:
            total += (Path(base) / name).stat().st_size
    return total


def selected_images(directory: Path, list_file: Path | None, max_views: int) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("images must be a real directory")
    if list_file is None:
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES)
    else:
        names = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate image names")
        if any(Path(name).name != name for name in names):
            raise ValueError("image list must contain basenames only")
        paths = [directory / name for name in names]
    paths = paths[:max_views]
    if len(paths) < 3:
        raise ValueError("at least three images are required")
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"invalid image: {path}")
    return paths


def ply_counts(path: Path, *, allow_degenerate: bool = False) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty PLY: {path}")
    types = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
             "int": "i", "uint": "I", "float": "f", "double": "d"}
    counts: dict[str, int] = {}
    props: dict[str, list[tuple[str, ...]]] = {"vertex": [], "face": []}
    current = None
    with path.open("rb") as stream:
        if stream.readline() != b"ply\n":
            raise ValueError(f"invalid PLY header: {path}")
        fmt = None
        for _ in range(200):
            line = stream.readline().decode("ascii", errors="strict").strip()
            tokens = line.split()
            if tokens[:1] == ["format"]:
                fmt = tokens[1]
            elif tokens[:1] == ["element"]:
                current = tokens[1]
                if current not in props:
                    raise ValueError(f"unsupported PLY element: {current}")
                counts[current] = int(tokens[2])
            elif tokens[:1] == ["property"]:
                if current is None:
                    raise ValueError("PLY property before element")
                props[current].append(tuple(tokens[1:]))
            if line == "end_header":
                break
        else:
            raise ValueError(f"PLY header not terminated: {path}")
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY encoding: {fmt}")
        vertices, faces = counts.get("vertex", 0), counts.get("face", 0)
        if vertices < 0 or vertices > 5_000_000 or faces < 0 or faces > 10_000_000:
            raise ValueError("PLY element count out of bounds")
        vertex_props = props["vertex"]
        if any(len(prop) != 2 or prop[0] not in types for prop in vertex_props):
            raise ValueError("unsupported vertex property")
        names = [prop[1] for prop in vertex_props]
        if not all(name in names for name in ("x", "y", "z")):
            raise ValueError("PLY missing XYZ")
        vertex_format = struct.Struct("<" + "".join(types[prop[0]] for prop in vertex_props))
        if vertices * vertex_format.size > path.stat().st_size - stream.tell():
            raise ValueError(f"truncated PLY payload: {path}")
        positions = []
        for _ in range(vertices):
            values = vertex_format.unpack(stream.read(vertex_format.size))
            if any(not math.isfinite(value) for value in values if isinstance(value, float)):
                raise ValueError("nonfinite PLY vertex attribute")
            positions.append(tuple(values[names.index(axis)] for axis in ("x", "y", "z")))
        if faces:
            if props["face"] != [("list", "uchar", "int", "vertex_indices")]:
                raise ValueError("unsupported face property")
            for _ in range(faces):
                count = stream.read(1)
                if count != b"\x03":
                    raise ValueError("PLY face is not a triangle")
                data = stream.read(12)
                if len(data) != 12:
                    raise ValueError(f"truncated PLY payload: {path}")
                a, b, c = struct.unpack("<iii", data)
                if min(a, b, c) < 0 or max(a, b, c) >= vertices or len({a, b, c}) < 3:
                    raise ValueError("invalid PLY triangle indices")
                p, q, r = positions[a], positions[b], positions[c]
                u = tuple(q[i] - p[i] for i in range(3))
                v = tuple(r[i] - p[i] for i in range(3))
                cross = (u[1] * v[2] - u[2] * v[1],
                         u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0])
                if sum(x * x for x in cross) <= 0 and not allow_degenerate:
                    raise ValueError("degenerate PLY triangle")
        if stream.read(1):
            raise ValueError("unexpected trailing PLY payload")
        return vertices, faces


def registered_views(scene: Path) -> int:
    count = 0
    for view in (scene / "views").glob("view_*.mve"):
        meta = view / "meta.ini"
        if not meta.is_file():
            continue
        camera = meta.read_text().split("[camera]")
        if len(camera) != 2:
            continue
        for line in camera[1].splitlines():
            if line.startswith("focal_length ="):
                count += float(line.partition("=")[2]) > 0
                break
    return count


def stage(command: list[str], name: str, run: Path, max_bytes: int,
          timeout: float, started: float, max_log_bytes: int) -> dict:
    log = run / f"{name}.log"
    temporary = run / "tmp"
    temporary.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(temporary)
    env["TMP"] = str(temporary)
    env["TEMP"] = str(temporary)
    reason = "succeeded"
    stage_started = time.monotonic()
    error_type = None
    proc = None
    with log.open("xb") as sink:
        try:
            group_kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                            if platform.system() == "Windows" else {"start_new_session": True})
            proc = subprocess.Popen(command, cwd=run, env=env, stdin=subprocess.DEVNULL,
                                    stdout=sink, stderr=subprocess.STDOUT, **group_kwargs)
            while True:
                if time.monotonic() - started > timeout:
                    reason = "total_timeout"
                elif log.stat().st_size > max_log_bytes:
                    reason = "log_cap"
                elif tree_bytes(run) > max_bytes:
                    reason = "output_cap"
                elif shutil.disk_usage(run).free < RESERVE:
                    reason = "free_space_reserve"
                if reason != "succeeded" or proc.poll() is not None:
                    break
                time.sleep(POLL)
        except Exception as exc:
            reason = "guard_error"
            error_type = type(exc).__name__
        finally:
            if proc is not None:
                # The leader may have left workers behind in its process group.
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    if proc.poll() is None:
                        proc.kill()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        if proc.poll() is None:
                            proc.kill()
                proc.wait()
        code = None if proc is None else proc.returncode
    if reason == "succeeded" and code:
        reason = "nonzero_exit"
    result = {"stage": name, "command": command, "status": reason,
              "returncode": code, "duration_seconds": round(time.monotonic() - stage_started, 2),
              "elapsed_total_seconds": round(time.monotonic() - started, 2),
              "log": log.name}
    if error_type:
        result["error_type"] = error_type
    return result


def run(args: argparse.Namespace) -> dict:
    if platform.system() not in {"Darwin", "Linux", "Windows"}:
        raise ValueError("unsupported operating system")
    if args.max_views < 3 or args.scale < 0 or args.scale > 5 or args.max_pixels <= 0:
        raise ValueError("invalid max_views, scale or max_pixels")
    if not math.isfinite(args.max_gib) or not math.isfinite(args.timeout_minutes) or args.max_gib <= 0 or args.timeout_minutes <= 0 or args.max_log_mib <= 0:
        raise ValueError("resource caps must be positive")
    if not math.isfinite(args.min_registered_fraction) or not 0 <= args.min_registered_fraction <= 1:
        raise ValueError("min_registered_fraction must be between zero and one")
    archive = ROOT / ".local-tools/mve-spike/mve-bf2279f.tar.gz"
    verified_sources = verify_selected_source(archive, SOURCE)
    images = selected_images(args.images.resolve(), args.image_list, args.max_views)
    extension = ".exe" if platform.system() == "Windows" else ""
    binaries = {name: (((args.binary_dir / (name + extension)).absolute()) if args.binary_dir
                       else SOURCE / "apps" / name / (name + extension)) for name in NAMES}
    for name, binary in binaries.items():
        if not binary.is_file() or (platform.system() != "Windows" and not os.access(binary, os.X_OK)):
            raise ValueError(f"selected MVE binary unavailable: {name}: {binary}")
    # Ensure the source change that excludes SURF is in effect for this local build.
    sfm_source = (SOURCE / "apps/sfmrecon/sfmrecon.cc").read_text()
    sfm_makefile = (SOURCE / "libs/sfm/Makefile").read_text()
    if "feature_types = sfm::FeatureSet::FEATURE_SIFT;" not in sfm_source or "filter-out surf.cc" not in sfm_makefile:
        raise ValueError("SIFT-only selected source patch is absent")
    if args.binary_dir is None:
        archive_members = subprocess.check_output(["ar", "-t", str(SOURCE / "libs/sfm/libmve_sfm.a")], text=True)
        if "surf.o" in archive_members.splitlines():
            raise ValueError("selected SfM archive contains surf.o")
    out = args.output.absolute()
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("output must be a fresh directory with an existing parent")
    cap = int(args.max_gib * (1 << 30))
    if shutil.disk_usage(out.parent).free < RESERVE + cap:
        raise ValueError("free disk lacks output allowance plus 10 GiB reserve")
    out.mkdir()
    copied = out / "input"
    copied.mkdir()
    report = {"schema": "mve_full_unknown_pose_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha256(Path(__file__)),
              "source_commit": "bf2279f161ba962072ecac85224c15e82bc5f52e",
              "selected_source_file_count": verified_sources,
              "binary_provenance": ("external_directory_unverified" if args.binary_dir else
                                    "selected_local_build_unverified_for_distribution"),
              "shipping_approved": False, "camera_poses_input": False, "scale_verified": False,
              "surface_quality_accepted": False,
              "host": platform.platform(), "input": [], "binaries": {}, "stages": [],
              "selected_source_sha256": {rel: sha256(SOURCE / rel) for rel in
                                         ("apps/sfmrecon/sfmrecon.cc", "libs/sfm/feature_set.cc", "libs/sfm/Makefile")},
              "settings": {"scale": args.scale, "max_pixels": args.max_pixels,
                           "max_views": args.max_views, "max_gib": args.max_gib,
                           "timeout_minutes": args.timeout_minutes,
                           "min_registered_fraction": args.min_registered_fraction}}
    status = out / "result.json"
    started = time.monotonic()
    try:
        for image in images:
            if tree_bytes(out) + image.stat().st_size > cap or shutil.disk_usage(out).free < RESERVE + image.stat().st_size:
                raise ValueError("input copy would exceed output allowance or free-space reserve")
            original_hash = sha256(image)
            shutil.copy2(image, copied / image.name)
            copied_hash = sha256(copied / image.name)
            if copied_hash != original_hash or sha256(image) != original_hash:
                raise ValueError(f"source image changed during copy: {image}")
            report["input"].append({"name": image.name, "bytes": (copied / image.name).stat().st_size,
                                    "sha256": copied_hash})
        for name, binary in binaries.items():
            report["binaries"][name] = {"path": str(binary), "sha256": sha256(binary)}
        scene = out / "scene"
        pointset = out / "oriented-points.ply"
        raw_mesh = out / "mesh-raw.ply"
        native_mesh = out / "mesh-native.ply"
        mesh = out / "mesh.ply"
        cmds = [
            ("makescene", [str(binaries["makescene"]), "--images-only", f"--max-pixels={args.max_pixels}", str(copied), str(scene)]),
            ("sfmrecon", [str(binaries["sfmrecon"]), f"--max-pixels={args.max_pixels}", str(scene)]),
        ]
        if args.stop_after != "sfm":
            cmds.extend([
                ("dmrecon", [str(binaries["dmrecon"]), f"--scale={args.scale}", "--neighbors=4",
                              "--local-neighbors=2", "--progress=simple", str(scene)]),
                ("scene2pset", [str(binaries["scene2pset"]), f"--fssr={args.scale}", str(scene), str(pointset)]),
                ("fssrecon", [str(binaries["fssrecon"]), str(pointset), str(raw_mesh)]),
                ("meshclean", [str(binaries["meshclean"]), "--threshold=0", "--component-size=0",
                               str(raw_mesh), str(native_mesh)]),
            ])
        for name, cmd in cmds:
            result = stage(cmd, name, out, cap, args.timeout_minutes * 60,
                           started, args.max_log_mib << 20)
            report["stages"].append(result)
            if result["status"] != "succeeded":
                raise RuntimeError(f"{name}: {result['status']}; inspect {out / result['log']}")
            if name == "makescene":
                report["imported_views"] = len(list((scene / "views").glob("view_*.mve")))
                if report["imported_views"] != len(images):
                    raise ValueError("imported view count differs from selected images")
            elif name == "sfmrecon":
                bundle = scene / "synth_0.out"
                if not bundle.is_file() or bundle.stat().st_size == 0:
                    raise ValueError("SfM did not create a nonempty bundle")
                report["registered_views"] = registered_views(scene)
                report["registration_fraction"] = round(report["registered_views"] / len(images), 4)
                if report["registered_views"] < 3 or report["registration_fraction"] < args.min_registered_fraction:
                    raise ValueError("insufficient registered views for declared fraction and three-view minimum")
            elif name == "scene2pset":
                report["point_count"], _ = ply_counts(pointset)
                if report["point_count"] < 100:
                    raise ValueError("oriented point cloud has fewer than 100 points")
            elif name == "fssrecon":
                report["raw_mesh_vertices"], report["raw_mesh_faces"] = ply_counts(raw_mesh, allow_degenerate=True)
                if report["raw_mesh_faces"] < 1:
                    raise ValueError("surface reconstruction has zero faces")
            elif name == "meshclean":
                from scripts.mve_full.sanitize_mesh import remove_zero_area_faces
                report.update(remove_zero_area_faces(native_mesh, mesh))
                if report["mesh_faces"] < 1:
                    raise ValueError("cleaned mesh has zero faces")
        report["status"] = "succeeded"
        report["output_bytes"] = tree_bytes(out)
        if time.monotonic() - started > args.timeout_minutes * 60:
            raise ValueError("final validation exceeded total timeout")
        if report["output_bytes"] > cap or shutil.disk_usage(out).free < RESERVE:
            raise ValueError("final output breached byte cap or free-space reserve")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        # RUSAGE_CHILDREN is the largest reaped child RSS, not aggregate or tree peak.
        if resource is not None:
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            report["peak_single_child_rss_mib"] = round(rss / (1 << 20) if platform.system() == "Darwin" else rss / 1024, 2)
        else:
            report["peak_single_child_rss_mib"] = None
        status.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary-dir", type=Path, help="CMake directory containing the six selected executables")
    parser.add_argument("--image-list", type=Path, help="optional basenames, one per line")
    parser.add_argument("--max-views", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=1_500_000)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--max-gib", type=float, default=4)
    parser.add_argument("--timeout-minutes", type=float, default=120)
    parser.add_argument("--max-log-mib", type=int, default=64)
    parser.add_argument("--min-registered-fraction", type=float, default=0.7,
                        help="declared structural registration gate, not shape acceptance")
    parser.add_argument("--stop-after", choices=("sfm", "mesh"), default="mesh")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output.absolute()),
                      "registered_views": result["registered_views"],
                      "mesh_faces": result.get("mesh_faces")}, sort_keys=True))


if __name__ == "__main__":
    main()
