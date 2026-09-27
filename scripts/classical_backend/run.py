"""Bounded, image-only PyCOLMAP -> OpenMVS CPU experiment.

Outputs are research artifacts. No metric scale or product integration is implied.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from scripts.classical_backend.geometry import counts as ply_counts

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / ".local-tools/classical-backend/bin"
PYTHON = ROOT / ".local-tools/colmap-sparse/venv/bin/python"
RESERVE = 10 << 30
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")


def tool_path(directory: Path, name: str, *, windows: bool | None = None) -> Path:
    if windows is None:
        windows = os.name == "nt"
    return directory / (name + (".exe" if windows else ""))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def folder_bytes(path: Path) -> int:
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            entry = Path(base) / name
            try:
                if entry.is_symlink():
                    raise ValueError("output contains a symlink")
                if name in files:
                    total += entry.stat().st_size
            except FileNotFoundError:
                # Native tools atomically replace depth maps; a vanished entry
                # can be counted on the next poll or at final validation.
                continue
    return total


def photos(directory: Path, list_file: Path | None, max_views: int) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("--images must be a real directory")
    if list_file:
        names = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError("image list requires unique basenames")
        chosen = [directory / n for n in names]
    else:
        chosen = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES)
    chosen = chosen[:max_views]
    if len(chosen) < 3:
        raise ValueError("at least three photographs are required")
    if any(p.is_symlink() or not p.is_file() or p.suffix.lower() not in SUFFIXES for p in chosen):
        raise ValueError("invalid photograph in selection")
    return chosen


def checked_ply(path: Path, element: str) -> int:
    """Parse every vertex and face, including finite positions and triangle indices."""
    vertices, faces = ply_counts(path)
    count = {"vertex": vertices, "face": faces}[element]
    if count < 1:
        raise ValueError(f"PLY has no {element}: {path}")
    return count


def nonempty_bytes(path: Path) -> int:
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"missing or empty artifact: {path}")
    return path.stat().st_size


def obj_counts(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size < 100:
        raise ValueError(f"missing or empty OBJ: {path}")
    vertices = faces = texcoords = 0
    material = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            xyz = [float(v) for v in line.split()[1:4]]
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError("OBJ has nonfinite or incomplete vertex")
            vertices += 1
        elif line.startswith("f "):
            refs = line.split()[1:]
            if len(refs) < 3:
                raise ValueError("OBJ face has fewer than three vertices")
            ids = [int(ref.split("/")[0]) for ref in refs]
            if any(i <= 0 or i > vertices for i in ids) or len(set(ids)) < 3:
                raise ValueError("OBJ face has invalid vertex indices")
            uv_ids = [int(parts[1]) for ref in refs if len(parts := ref.split("/")) > 1 and parts[1]]
            if len(uv_ids) != len(refs) or any(i <= 0 or i > texcoords for i in uv_ids):
                raise ValueError("OBJ face has invalid texture coordinates")
            faces += 1
        elif line.startswith("vt "):
            uv = [float(v) for v in line.split()[1:3]]
            if len(uv) != 2 or not all(math.isfinite(v) for v in uv):
                raise ValueError("OBJ has nonfinite or incomplete texture coordinate")
            texcoords += 1
        elif line.startswith("mtllib "):
            name = line.split(maxsplit=1)[1]
            if Path(name).name != name:
                raise ValueError("OBJ material path is not a basename")
            material = path.parent / name
    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices or faces: {path}")
    if not material or not material.is_file():
        raise ValueError("textured OBJ has no material file")
    textures = []
    for line in material.read_text(errors="replace").splitlines():
        if line.startswith("map_Kd "):
            name = line.split(maxsplit=1)[1]
            if Path(name).name != name:
                raise ValueError("texture path is not a basename")
            texture = material.parent / name
            if not texture.is_file() or texture.stat().st_size == 0:
                raise ValueError(f"missing texture: {texture}")
            textures.append(texture)
    if not textures:
        raise ValueError("material has no diffuse texture")
    return vertices, faces


def init_pair_ids(database: Path, selected_names: list[str], pair: list[str] | None) -> tuple[int, int] | None:
    """Resolve a user-selected seed pair against the actual COLMAP image IDs."""
    if pair is None:
        return None
    if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(selected_names):
        raise ValueError("initial pair needs two distinct selected image names")
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT image_id, name FROM images WHERE name IN (?, ?)", pair).fetchall()
    ids = {name: image_id for image_id, name in rows}
    if len(ids) != 2:
        raise ValueError("initial pair names missing from COLMAP database")
    return ids[pair[0]], ids[pair[1]]


def sfm_options(pycolmap_module, max_models: int, min_model_size: int,
                pair_ids: tuple[int, int] | None):
    if not 1 <= max_models <= 10 or min_model_size < 2:
        raise ValueError("invalid SfM model retry bounds")
    options = pycolmap_module.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = max_models > 1
    options.max_num_models = max_models
    options.min_model_size = min_model_size
    if pair_ids:
        options.init_image_id1, options.init_image_id2 = pair_ids
    return options


def worker(kind: str, run: Path, max_pixels: int, external_model: Path | None = None,
           camera_model: str = "SIMPLE_RADIAL", matching: str = "exhaustive",
           sequential_overlap: int = 8, sift_max_features: int = 8192,
           sift_max_image_size: int = 3200, seed: int = 0,
           init_image_pair: list[str] | None = None, sfm_max_models: int = 5,
           sfm_min_model_size: int = 10) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pycolmap

    image_dir = run / "images"
    names = [item["name"] for item in json.loads((run / "inputs.json").read_text())]
    options_path = run / "pycolmap-options.json"
    recorded = json.loads(options_path.read_text()) if options_path.exists() else {
        "version": pycolmap.__version__, "binary_sha256": digest(Path(pycolmap._core.__file__)),
        "seed": seed, "device": "cpu", "camera_mode": "SINGLE", "camera_model": camera_model,
        "matching_strategy": matching}
    pycolmap.set_random_seed(seed)
    if kind == "features":
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = camera_model
        reader.default_focal_length_factor = 1.2
        if (run / "masks").is_dir():
            reader.mask_path = str(run / "masks")
        sift = pycolmap.SiftExtractionOptions()
        sift.num_threads = 2
        sift.max_num_features = sift_max_features
        sift.max_image_size = sift_max_image_size
        recorded.update(image_reader=reader.todict(), sift_extraction=sift.todict())
        options_path.write_text(json.dumps(recorded, indent=2, default=str) + "\n")
        pycolmap.extract_features(str(run / "database.db"), str(image_dir), names,
                                  camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                  camera_model=reader.camera_model, sift_options=sift,
                                  device=pycolmap.Device.cpu)
    elif kind == "matching":
        sift = pycolmap.SiftMatchingOptions()
        sift.num_threads = 2
        verification = pycolmap.TwoViewGeometryOptions()
        if matching == "sequential":
            pairer = pycolmap.SequentialMatchingOptions()
            pairer.overlap = sequential_overlap
            pairer.quadratic_overlap = False
            pairer.loop_detection = False
            recorded.update(sift_matching=sift.todict(), sequential_matching=pairer.todict(),
                            two_view_geometry=verification.todict())
            options_path.write_text(json.dumps(recorded, indent=2, default=str) + "\n")
            pycolmap.match_sequential(str(run / "database.db"), sift_options=sift,
                                      matching_options=pairer, verification_options=verification,
                                      device=pycolmap.Device.cpu)
        else:
            pairer = pycolmap.ExhaustiveMatchingOptions()
            recorded.update(sift_matching=sift.todict(), exhaustive_matching=pairer.todict(),
                            two_view_geometry=verification.todict())
            options_path.write_text(json.dumps(recorded, indent=2, default=str) + "\n")
            pycolmap.match_exhaustive(str(run / "database.db"), sift_options=sift,
                                      matching_options=pairer, verification_options=verification,
                                      device=pycolmap.Device.cpu)
    elif kind in ("sfm", "reuse_sfm"):
        if kind == "sfm":
            pair_ids = init_pair_ids(run / "database.db", names, init_image_pair)
            options = sfm_options(pycolmap, sfm_max_models, sfm_min_model_size, pair_ids)
            if pair_ids:
                recorded["init_image_pair_names"] = init_image_pair
            recorded.update(incremental_pipeline=options.todict())
            options_path.write_text(json.dumps(recorded, indent=2, default=str) + "\n")
            models = pycolmap.incremental_mapping(str(run / "database.db"), str(image_dir),
                                                    str(run / "models"), options=options)
            if not models:
                raise ValueError("SfM produced no model")
            selected_index, model = max(models.items(), key=lambda indexed:
                                        (indexed[1].num_reg_images(), indexed[1].num_points3D(),
                                         -indexed[0]))
        else:
            if external_model is None:
                raise ValueError("reuse_sfm requires --sparse-model")
            model = pycolmap.Reconstruction(str(external_model))
        registered_names = sorted(model.images[image_id].name for image_id in model.reg_image_ids())
        if set(registered_names) - set(names):
            raise ValueError("sparse model contains images outside selected input")
        info = {"registered_images": model.num_reg_images(), "sparse_points": model.num_points3D(),
                "registered_names": registered_names,
                "missing_names": sorted(set(names) - set(registered_names)),
                "camera_models": sorted({camera.model.name for camera in model.cameras.values()})}
        if kind == "sfm":
            info["selected_model_index"] = selected_index
            info["candidate_models"] = [{"index": index, "registered_images": candidate.num_reg_images(),
                                          "sparse_points": candidate.num_points3D()}
                                         for index, candidate in sorted(models.items())]
        (run / "sfm.json").write_text(json.dumps(info, indent=2) + "\n")
        if info["registered_images"] < 3 or info["sparse_points"] < 1:
            raise ValueError("SfM model has fewer than 3 cameras or no points")
        sparse = run / "sparse" / "0"
        sparse.mkdir(parents=True)
        model.write(str(sparse))
    elif kind == "undistort":
        options = pycolmap.UndistortCameraOptions()
        options.max_image_size = max_pixels
        pycolmap.undistort_images(str(run / "dense"), str(run / "sparse" / "0"),
                                  str(image_dir), output_type="COLMAP", undistort_options=options)
    else:
        raise ValueError(f"unknown worker stage: {kind}")


def stage(run: Path, name: str, command: list[str], deadline: float,
          max_bytes: int, max_log_bytes: int, max_rss_bytes: int,
          extra_reserve_paths: tuple[Path, ...] = ()) -> dict:
    log_path = run / f"{name}.log"
    started = time.monotonic()
    result = {"name": name, "command": command, "log": str(log_path), "status": "running"}
    env = os.environ.copy()
    temp = run / "tmp"
    temp.mkdir(exist_ok=True)
    env.update(TMPDIR=str(temp), TMP=str(temp), OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
    process = None
    reason = None
    peak_rss_bytes = 0 if os.name == "posix" else None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=run, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "deadline exceeded"
                elif shutil.disk_usage(run).free < RESERVE:
                    reason = "10 GiB disk reserve reached"
                elif any(shutil.disk_usage(path).free < RESERVE for path in extra_reserve_paths):
                    reason = "extra 10 GiB disk reserve reached"
                elif folder_bytes(run) > max_bytes:
                    reason = "output byte limit reached"
                elif log_path.stat().st_size > max_log_bytes:
                    reason = "stage log byte limit reached"
                if os.name == "posix" and process.poll() is None:
                    sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                            capture_output=True, text=True, check=False)
                    if sample.returncode == 0 and sample.stdout.strip().isdigit():
                        peak_rss_bytes = max(peak_rss_bytes, int(sample.stdout.strip()) * 1024)
                        if peak_rss_bytes > max_rss_bytes:
                            reason = "resident memory limit reached"
                if reason:
                    break
                time.sleep(0.5)
    except BaseException as error:
        reason = str(error)
        if isinstance(error, KeyboardInterrupt):
            reason = "interrupted"
    finally:
        if process is not None and process.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
    measured_bytes = folder_bytes(run)
    if not reason and measured_bytes > max_bytes:
        reason = "output byte limit reached after stage"
    if not reason and log_path.stat().st_size > max_log_bytes:
        reason = "stage log byte limit reached after stage"
    if not reason and shutil.disk_usage(run).free < RESERVE:
        reason = "10 GiB disk reserve reached after stage"
    if not reason and any(shutil.disk_usage(path).free < RESERVE for path in extra_reserve_paths):
        reason = "extra 10 GiB disk reserve reached after stage"
    result.update(status="failed" if reason or process is None or process.returncode else "complete",
                  exit_code=process.returncode if process else None, seconds=round(time.monotonic() - started, 3),
                  output_bytes=measured_bytes, peak_sampled_child_rss_bytes=peak_rss_bytes,
                  failure=reason)
    if result["status"] == "failed":
        raise StageError(result)
    return result


class StageError(Exception):
    def __init__(self, result: dict):
        super().__init__(f"{result['name']} failed: {result.get('failure') or 'exit ' + str(result['exit_code'])}")
        self.result = result


def check_toolchain(binary_dir: Path, python: Path) -> dict:
    if not python.is_file():
        raise ValueError(f"PyCOLMAP venv missing: {python}")
    hashes = {}
    for name in TOOLS:
        path = tool_path(binary_dir, name)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"OpenMVS executable missing: {path}")
        hashes[name] = digest(path)
    return {"openmvs_binaries": hashes, "python": str(python),
            "runner_sha256": digest(Path(__file__))}


def reserve_paths_for_devices(output_device: int, workspace_device: int) -> tuple[Path, ...]:
    """Monitor internal free space too when a run writes to another volume."""
    return (ROOT,) if output_device != workspace_device else ()


def run(args: argparse.Namespace) -> dict:
    if args.output.is_symlink():
        raise ValueError("output must not be a symlink")
    output = args.output.resolve()
    binary_dir = args.binary_dir.resolve()
    if output.exists():
        raise ValueError(f"output must be fresh: {output}")
    extra_reserve_paths = reserve_paths_for_devices(output.parent.stat().st_dev, ROOT.stat().st_dev)
    if shutil.disk_usage(output.parent).free < RESERVE + (256 << 20):
        raise ValueError("insufficient free disk above 10 GiB reserve")
    if any(shutil.disk_usage(path).free < RESERVE for path in extra_reserve_paths):
        raise ValueError("internal 10 GiB reserve not met for external output")
    selection = photos(args.images.resolve(), args.image_list, args.max_views)
    args.sfm_min_model_size = effective_sfm_min_model_size(args.sfm_min_model_size, len(selection))
    software = None
    output.mkdir(parents=True)
    (output / "images").mkdir()
    report = {"schema": "classical_backend_v1", "status": "running", "platform": platform.platform(),
              "machine": platform.machine(), "software": software,
              "limits": {"max_views": args.max_views, "max_image_size": args.max_image_size,
                         "max_output_bytes": args.max_gib * (1 << 30),
                         "max_child_rss_bytes": args.max_rss_gib * (1 << 30),
                         "timeout_minutes": args.timeout_minutes, "min_free_bytes": RESERVE,
                         "extra_reserve_paths": [str(path) for path in extra_reserve_paths]},
              "inputs": [], "stages": [], "shipping_approved": False,
              "metric_scale_verified": False, "quality_accepted": False,
              "sfm_camera_mode": "from_imported_model" if args.sparse_model else "SINGLE",
              "sfm_camera_model": "from_imported_model" if args.sparse_model else args.camera_model,
              "pose_masks": []}
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        software = check_toolchain(binary_dir, args.python.resolve())
        report["software"] = software
        if args.sparse_model:
            model_path = args.sparse_model.resolve()
            model_files = {name: model_path / name for name in ("cameras.bin", "images.bin", "points3D.bin")}
            if any(not path.is_file() or path.is_symlink() for path in model_files.values()):
                raise ValueError("--sparse-model must contain real COLMAP binary model files")
            report["sfm_source"] = {"kind": "external_image_only_model", "path": str(model_path),
                                    "files_sha256": {name: digest(path) for name, path in model_files.items()},
                                    "options": "See producer report; external SfM settings are not inferred here"}
            if args.sparse_provenance:
                provenance_path = args.sparse_provenance.resolve()
                if not args.sparse_result:
                    raise ValueError("--sparse-result required with producer provenance")
                producer_result_path = args.sparse_result.resolve()
                producer_result = json.loads(producer_result_path.read_text())
                model_hashes = report["sfm_source"]["files_sha256"]
                if (producer_result.get("model_count", 0) < 1 or
                        Path(producer_result.get("model_dir", "")).resolve() != model_path or
                        producer_result.get("model_files_sha256") != model_hashes):
                    raise ValueError("producer summary does not bind this binary sparse model")
                producer_provenance = json.loads(provenance_path.read_text())
                if (producer_provenance.get("schema") != "object_motion_run_v1" or
                        producer_provenance.get("arm") not in ("raw", "foreground") or
                        producer_result.get("arm") != producer_provenance.get("arm") or
                        producer_result.get("producer_provenance_sha256") != digest(provenance_path)):
                    raise ValueError("sparse producer summary and provenance disagree")
                if not args.sparse_manifest:
                    raise ValueError("--sparse-manifest required with --sparse-provenance")
                manifest_path = args.sparse_manifest.resolve()
                if digest(manifest_path) != producer_provenance.get("input_manifest_sha256"):
                    raise ValueError("sparse source manifest hash differs from producer provenance")
                manifest = json.loads(manifest_path.read_text())
                manifest_hashes = {item["name"]: item["sha256"] for item in manifest["images"]}
                if producer_provenance.get("image_hashes") != manifest_hashes:
                    raise ValueError("producer image hashes differ from preparation manifest")
                if producer_provenance.get("mask_hashes") != {item["name"]: item["mask_sha256"] for item in manifest["images"]}:
                    raise ValueError("producer mask hashes differ from preparation manifest")
                source_kind = ("external_raw_image_only_verified_model" if producer_provenance["arm"] == "raw" else
                               "external_photo_mask_image_only_verified_model")
                report["sfm_source"].update(kind=source_kind, producer_arm=producer_provenance["arm"],
                                             provenance_path=str(provenance_path),
                                             provenance_sha256=digest(provenance_path),
                                             producer_result_path=str(producer_result_path),
                                             producer_result_sha256=digest(producer_result_path),
                                             manifest_path=str(manifest_path),
                                             manifest_sha256=digest(manifest_path),
                                             producer_options=producer_provenance.get("options"))
            else:
                report["sfm_source"]["kind"] = "external_unverified_model"
        else:
            report["sfm_source"] = {"kind": "internal_image_only_pycolmap", "random_seed": args.seed,
                                    "matching": args.matching, "camera_mode": "SINGLE",
                                    "camera_model": args.camera_model,
                                    "init_image_pair_names": args.init_image_pair,
                                    "sfm_max_models": args.sfm_max_models,
                                    "sfm_min_model_size": args.sfm_min_model_size,
                                    "sift_max_features": args.sift_max_features,
                                    "sift_max_image_size": args.sift_max_image_size,
                                    "sequential_overlap": args.sequential_overlap}
        save()
        for source in selection:
            if any(shutil.disk_usage(path).free < RESERVE for path in extra_reserve_paths) or \
                    shutil.disk_usage(output).free < RESERVE + source.stat().st_size or \
                    folder_bytes(output) + source.stat().st_size > args.max_gib * (1 << 30):
                raise ValueError("photo copy would exceed disk reserve or output limit")
            target = output / "images" / source.name
            shutil.copyfile(source, target)
            original_hash = digest(source)
            if original_hash != digest(target):
                raise ValueError(f"copied photograph hash differs: {source}")
            report["inputs"].append({"name": source.name, "source": str(source.resolve()),
                                     "sha256": original_hash, "bytes": source.stat().st_size})
        if args.pose_mask_dir:
            masks = args.pose_mask_dir.resolve()
            if not masks.is_dir() or masks.is_symlink():
                raise ValueError("--pose-mask-dir must be a real directory")
            (output / "masks").mkdir()
            for item in report["inputs"]:
                mask = masks / (item["name"] + ".png")
                if mask.is_symlink() or not mask.is_file() or mask.stat().st_size < 1:
                    raise ValueError(f"missing pose mask: {mask}")
                if (any(shutil.disk_usage(path).free < RESERVE for path in extra_reserve_paths) or
                        shutil.disk_usage(output).free < RESERVE + mask.stat().st_size or
                        folder_bytes(output) + mask.stat().st_size > args.max_gib * (1 << 30)):
                    raise ValueError("pose mask copy would exceed disk reserve or output limit")
                target = output / "masks" / mask.name
                shutil.copyfile(mask, target)
                mask_hash = digest(mask)
                if digest(target) != mask_hash:
                    raise ValueError(f"copied pose mask differs: {mask}")
                report["pose_masks"].append({"name": mask.name, "source": str(mask), "sha256": mask_hash})
        (output / "inputs.json").write_text(json.dumps(report["inputs"], indent=2) + "\n")
        if args.init_image_pair and (len(set(args.init_image_pair)) != 2 or
                                     not set(args.init_image_pair).issubset({item["name"] for item in report["inputs"]})):
            raise ValueError("--init-image-pair needs two distinct selected photo names")
        if args.sparse_model and args.sparse_provenance:
            chosen_hashes = {item["name"]: item["sha256"] for item in report["inputs"]}
            if chosen_hashes != producer_provenance["image_hashes"]:
                raise ValueError("selected photo hashes differ from sparse producer")
        save()
        deadline = time.monotonic() + args.timeout_minutes * 60
        def execute(name, command, validate=None):
            result = stage(output, name, [str(x) for x in command], deadline,
                           int(args.max_gib * (1 << 30)), args.max_log_mib << 20,
                           int(args.max_rss_gib * (1 << 30)),
                           extra_reserve_paths=extra_reserve_paths)
            if validate:
                try:
                    result["artifact"] = validate()
                except Exception as error:
                    result.update(status="failed", failure=str(error))
                    raise StageError(result) from error
            report["stages"].append(result)
            if name == "sfm":
                report["sfm_source"]["effective_options"] = json.loads((output / "pycolmap-options.json").read_text())
            save()
        kinds = (("reuse_sfm",) if args.stop_after_sfm else ("reuse_sfm", "undistort")) if args.sparse_model else \
                (("features", "matching", "sfm") if args.stop_after_sfm else
                 ("features", "matching", "sfm", "undistort"))
        for kind in kinds:
            check = None
            if kind in ("sfm", "reuse_sfm"):
                check = lambda: json.loads((output / "sfm.json").read_text())
            elif kind == "undistort":
                def check_undistort():
                    exported = list((output / "dense" / "images").glob("*"))
                    if len(exported) < 3:
                        raise ValueError("undistortion exported fewer than 3 images")
                    return {"images": len(exported)}
                check = check_undistort
            execute(kind, [args.python, "-m", "scripts.classical_backend.run", "--worker", kind,
                           "--output", output, "--max-image-size", args.max_image_size,
                           "--camera-model", args.camera_model,
                           "--matching", args.matching, "--sequential-overlap", args.sequential_overlap,
                           "--sift-max-features", args.sift_max_features,
                           "--sift-max-image-size", args.sift_max_image_size,
                           "--seed", args.seed,
                           "--sfm-max-models", args.sfm_max_models,
                           "--sfm-min-model-size", args.sfm_min_model_size,
                           *(["--init-image-pair", *args.init_image_pair] if args.init_image_pair else []),
                           *(["--sparse-model", args.sparse_model.resolve()] if args.sparse_model else [])], check)
            if kind in ("sfm", "reuse_sfm") and report["stages"][-1]["artifact"]["registered_images"] / len(selection) < args.min_registered_fraction:
                raise ValueError("registered image fraction below threshold")
        if args.stop_after_sfm:
            report["status"] = "sparse_complete"
            report["completion_scope"] = "SfM only; no undistortion or OpenMVS stages"
        else:
            dense = output / "dense"
            common = ["--max-threads", str(args.max_threads), "--working-folder", str(output)]
            execute("import", [tool_path(binary_dir, "InterfaceCOLMAP"), "-i", dense,
                               "-o", output / "scene.mvs", "--image-folder", dense / "images", *common],
                    lambda: {"bytes": nonempty_bytes(output / "scene.mvs")})
            execute("densify", [tool_path(binary_dir, "DensifyPointCloud"), "-i", output / "scene.mvs",
                                "-o", output / "dense.mvs", "--resolution-level", "2", *common],
                    lambda: {"points": checked_ply(output / "dense.ply", "vertex")})
            execute("mesh", [tool_path(binary_dir, "ReconstructMesh"), "-i", output / "dense.mvs",
                             "-p", output / "dense.ply", "-o", output / "mesh.mvs", *common],
                    lambda: {"faces": checked_ply(output / "mesh.ply", "face")})
            execute("refine", [tool_path(binary_dir, "RefineMesh"), "-i", output / "dense.mvs",
                               "-m", output / "mesh.ply", "-o", output / "refined.mvs",
                               "--resolution-level", "1", "--scales", "1", *common],
                    lambda: {"faces": checked_ply(output / "refined.ply", "face")})
            execute("texture", [tool_path(binary_dir, "TextureMesh"), "-i", output / "dense.mvs",
                                "-m", output / "refined.ply", "-o", output / "textured.mvs",
                                "--export-type", "obj", *common],
                    lambda: dict(zip(("vertices", "faces"), obj_counts(output / "textured.obj"))))
            report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = shutil.disk_usage(output).free
    report["changed_source_images"] = [item["name"] for item in report["inputs"]
                                       if digest(Path(item["source"])) != item["sha256"]]
    report["changed_pose_masks"] = [item["name"] for item in report["pose_masks"]
                                    if digest(Path(item["source"])) != item["sha256"]]
    report["changed_binaries"] = [name for name, before in (software or {}).get("openmvs_binaries", {}).items()
                                  if digest(tool_path(binary_dir, name)) != before]
    if args.sparse_model and report.get("sfm_source"):
        report["changed_sparse_model_files"] = [name for name, before in report["sfm_source"]["files_sha256"].items()
                                                if digest(args.sparse_model.resolve() / name) != before]
        if args.sparse_provenance and "provenance_path" in report["sfm_source"]:
            for label, key in (("provenance", "provenance_sha256"),
                               ("producer_result", "producer_result_sha256"),
                               ("manifest", "manifest_sha256")):
                if digest(Path(report["sfm_source"][label + "_path"])) != report["sfm_source"][key]:
                    report.setdefault("changed_sparse_source_files", []).append(label)
    if (report["changed_source_images"] or report["changed_pose_masks"] or report["changed_binaries"] or
            report.get("changed_sparse_model_files") or report.get("changed_sparse_source_files")):
        report.update(status="failed", failure="input or OpenMVS binary changed during run")
    save()
    return report


def effective_sfm_min_model_size(configured: int | None, selected_count: int) -> int:
    if selected_count < 3:
        raise ValueError("at least three selected photos required")
    if configured is None:
        return min(10, selected_count)
    if not 2 <= configured <= selected_count:
        raise ValueError("explicit SfM minimum model size exceeds selected photo count")
    return configured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--pose-mask-dir", type=Path,
                        help="optional photo-derived pose masks named IMAGE_NAME.png")
    parser.add_argument("--sparse-model", type=Path, help="reuse an independently produced image-only COLMAP model")
    parser.add_argument("--sparse-provenance", type=Path, help="producer's object-motion raw provenance.json")
    parser.add_argument("--sparse-result", type=Path, help="producer's sealed summary.json with model hashes")
    parser.add_argument("--sparse-manifest", type=Path, help="producer's image preparation manifest")
    parser.add_argument("--binary-dir", type=Path, default=BIN)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--max-views", type=int, default=60)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--camera-model", choices=("SIMPLE_RADIAL", "SIMPLE_PINHOLE"),
                        default="SIMPLE_RADIAL")
    parser.add_argument("--matching", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--sequential-overlap", type=int, default=8)
    parser.add_argument("--sift-max-features", type=int, default=8192)
    parser.add_argument("--sift-max-image-size", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-image-pair", nargs=2, metavar=("IMAGE1", "IMAGE2"),
                        help="optional manual COLMAP initial pair by selected filenames; not an automatic robustness fix")
    parser.add_argument("--sfm-max-models", type=int, default=5,
                        help="COLMAP mapper model attempts; 5 enables generic retries after weak starts")
    parser.add_argument("--sfm-min-model-size", type=int,
                        help="minimum COLMAP candidate model size; default min(10, selected view count)")
    parser.add_argument("--stop-after-sfm", action="store_true", help="record a sparse-only diagnostic run")
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--min-registered-fraction", type=float, default=0.7)
    parser.add_argument("--max-gib", type=float, default=2.0)
    parser.add_argument("--max-log-mib", type=int, default=32)
    parser.add_argument("--max-rss-gib", type=float, default=10.0)
    parser.add_argument("--timeout-minutes", type=float, default=30)
    parser.add_argument("--worker", choices=("features", "matching", "sfm", "reuse_sfm", "undistort"))
    args = parser.parse_args()
    if args.worker:
        if args.sfm_min_model_size is None:
            args.sfm_min_model_size = min(10, args.max_views)
        worker(args.worker, args.output, args.max_image_size, args.sparse_model, args.camera_model,
               args.matching, args.sequential_overlap, args.sift_max_features,
               args.sift_max_image_size, args.seed, args.init_image_pair,
               args.sfm_max_models, args.sfm_min_model_size)
        return 0
    if (not args.images or not (3 <= args.max_views <= 200) or args.max_threads < 1
            or not math.isfinite(args.max_gib) or args.max_gib <= 0
            or not math.isfinite(args.max_rss_gib) or args.max_rss_gib <= 0
            or not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0
            or not math.isfinite(args.min_registered_fraction)
            or not 0 < args.min_registered_fraction <= 1 or args.max_image_size < 1
            or args.sequential_overlap < 1 or args.sift_max_features < 1 or args.sift_max_image_size < 1
            or args.seed < 0 or (args.pose_mask_dir and args.sparse_model)
            or not 1 <= args.sfm_max_models <= 10 or
            (args.sfm_min_model_size is not None and
             not 2 <= args.sfm_min_model_size <= args.max_views)
            or (args.init_image_pair and args.sparse_model)
            or bool(args.sparse_provenance) != bool(args.sparse_manifest)
            or bool(args.sparse_provenance) != bool(args.sparse_result)
            or (args.sparse_provenance and not args.sparse_model)):
        parser.error("valid --images, --max-views, --max-threads and --max-gib required")
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output.resolve()),
                      "failure": result.get("failure")}))
    return 0 if result["status"] in ("complete", "sparse_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
