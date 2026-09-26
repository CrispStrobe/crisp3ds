#!/usr/bin/env python3
"""Bounded 60-photo raw/shared-intrinsics versus foreground-support SfM trial."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
DEFAULT_OUTPUT = ROOT / "build-opencv/object-motion"
VENV_PYTHON = ROOT / ".local-tools/colmap-sparse/venv/bin/python"
LOG_CAP = 4 * 1024 * 1024
OUTPUT_CAP = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(manifest):
    images = manifest["images"]
    if manifest.get("schema") != "object_motion_prepare_v1" or len(images) != 60:
        raise ValueError("expected prepared 60-photo manifest")
    names = [r["name"] for r in images]
    if names != [f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)]:
        raise ValueError("unexpected photo order")
    source_manifest = Path(manifest["source_manifest"])
    if source_manifest.is_symlink() or digest(source_manifest) != manifest["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    expected_dir = ROOT / ".local-tools/test-data/ycb-cracker-box/photos"
    for image in images:
        if Path(image["path"]).is_symlink() or Path(image["pose_support_mask"]).is_symlink():
            raise ValueError(f"symlink input: {image['name']}")
        if Path(image["path"]).resolve().parent != expected_dir.resolve() or Path(image["path"]).name != image["name"]:
            raise ValueError(f"unexpected source photo: {image['name']}")
        if digest(image["path"]) != image["sha256"] or digest(image["pose_support_mask"]) != image["mask_sha256"]:
            raise ValueError(f"changed photo or mask: {image['name']}")
    if len({str(Path(r["path"]).parent) for r in images}) != 1:
        raise ValueError("photos require one image directory")
    return images


def worker(manifest_path, output, arm):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pycolmap
    from scripts.object_motion.diagnose import diagnose
    manifest = json.loads(manifest_path.read_text())
    images = validate(manifest)
    image_dir = Path(images[0]["path"]).parent
    output.mkdir(parents=True, exist_ok=True)
    pycolmap.set_random_seed(20260927)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.default_focal_length_factor = 1.2
    if arm == "foreground":
        reader.mask_path = str(Path(images[0]["pose_support_mask"]).parent)
    sift = pycolmap.SiftExtractionOptions()
    sift.max_image_size = 1200
    sift.max_num_features = 1800
    sift.num_threads = 2
    matching = pycolmap.SiftMatchingOptions()
    matching.num_threads = 2
    sequential = pycolmap.SequentialMatchingOptions()
    sequential.overlap = 8
    sequential.quadratic_overlap = False
    sequential.loop_detection = False
    verification = pycolmap.TwoViewGeometryOptions()
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 2
    mapping.mapper.num_threads = 2
    mapping.multiple_models = False
    mapping.max_num_models = 1
    mapping.min_model_size = 2
    provenance = {"schema": "object_motion_run_v1", "arm": arm,
                  "input_manifest_sha256": digest(manifest_path),
                  "image_count": len(images), "image_hashes": {i["name"]: i["sha256"] for i in images},
                  "mask_hashes": {i["name"]: i["mask_sha256"] for i in images},
                  "pycolmap_version": pycolmap.__version__, "pycolmap_binary_sha256": digest(pycolmap._core.__file__),
                  "options": {"camera_mode": "SINGLE", "image_reader": reader.todict(),
                              "sift": sift.todict(), "matching": matching.todict(),
                              "sequential": sequential.todict(), "verification": verification.todict(),
                              "mapping": mapping.todict()},
                  "scope": "same 60 raw JPEGs; photo-derived mask changes feature extraction in foreground arm; arbitrary Sim(3) scale"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    db = output / "database.db"
    start = time.monotonic()
    pycolmap.extract_features(str(db), str(image_dir), [r["name"] for r in images],
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model=reader.camera_model, reader_options=reader,
                              sift_options=sift, device=pycolmap.Device.cpu)
    pycolmap.match_sequential(str(db), sift_options=matching,
                              matching_options=sequential,
                              verification_options=verification, device=pycolmap.Device.cpu)
    models = pycolmap.incremental_mapping(str(db), str(image_dir), str(output / "models"), options=mapping)
    report = {"arm": arm, "model_count": len(models), "elapsed_seconds": time.monotonic()-start,
              "registered": 0, "total_images": 60, "points3D": 0}
    if models:
        model = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        model_dir = output / "model_text"
        model_dir.mkdir()
        model.write_text(str(model_dir))
        report.update(diagnose(model, manifest))
    validate(manifest)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["model_count"]:
        seal(output)
    print(json.dumps({k: report[k] for k in ("arm", "registered", "points3D", "elapsed_seconds")}))


def seal(output):
    """Bind the successful binary model to photo hashes and mapping options."""
    summary_path = output / "summary.json"
    provenance_path = output / "provenance.json"
    report = json.loads(summary_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if report.get("model_count", 0) < 1 or report.get("registered", 0) < 2:
        raise ValueError("no usable model to seal")
    if report.get("model_files_sha256"):
        raise ValueError("model already sealed")
    models = sorted((output / "models").iterdir())
    if len(models) != report["model_count"]:
        raise ValueError("model directory count differs from summary")
    # This experiment fixes max_num_models=1; require an unambiguous binary model.
    if len(models) != 1 or not models[0].is_dir():
        raise ValueError("expected one model directory")
    model_dir = models[0]
    names = ("cameras.bin", "images.bin", "points3D.bin")
    hashes = {}
    for name in names:
        path = model_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing model file {path}")
        hashes[name] = digest(path)
    report["model_dir"] = str(model_dir.resolve())
    report["model_files_sha256"] = hashes
    report["producer_provenance_sha256"] = digest(provenance_path)
    report["input_manifest_sha256"] = provenance["input_manifest_sha256"]
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def recover_diagnostics(manifest_path, output):
    """Continue only diagnostics after a completed mapper and failed reporter."""
    import pycolmap
    from scripts.object_motion.diagnose import diagnose
    status = json.loads((output / "trial_status.json").read_text())
    if status["status"] != "failed" or (output / "summary.json").exists():
        raise ValueError("recovery requires a failed reporter with no summary")
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance["input_manifest_sha256"] != digest(manifest_path):
        raise ValueError("manifest differs from original run")
    manifest = json.loads(manifest_path.read_text())
    validate(manifest)
    model_dir = output / "models" / "0"
    if not all((model_dir / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        raise ValueError("mapper did not finish binary model")
    model = pycolmap.Reconstruction(str(model_dir))
    report = {"arm": provenance["arm"], "model_count": 1,
              "elapsed_seconds": status["wall_seconds"], "registered": 0,
              "total_images": 60, "points3D": 0,
              "continuation": "diagnostics only after mapper completed; original failed trial status preserved"}
    report.update(diagnose(model, manifest))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    sealed = seal(output)
    (output / "recovery.json").write_text(json.dumps({"schema": "object_motion_diagnostic_recovery_v1",
        "original_status": status, "model_files_sha256": sealed["model_files_sha256"]}, indent=2) + "\n")
    return sealed


def output_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def drain_log(pipe, path, state):
    consumed = 0
    with path.open("wb") as log:
        for chunk in iter(lambda: pipe.read(65536), b""):
            available = max(0, LOG_CAP - consumed)
            if available:
                log.write(chunk[:available])
                consumed += min(len(chunk), available)
            if len(chunk) > available:
                state["truncated"] = True
        if state["truncated"]:
            log.write(b"\n[log truncated at 4 MiB; child output continued to be drained]\n")


def run_trial(cmd, dest, timeout_seconds, disk_check_path):
    """Run one child with bounded log, output size, free space, and wall time."""
    dest.mkdir(parents=True)
    start = time.monotonic()
    state = {"truncated": False}
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    drain = threading.Thread(target=drain_log, args=(process.stdout, dest / "trial.log", state), daemon=True)
    drain.start()
    status = None
    try:
        while process.poll() is None:
            if time.monotonic() - start >= timeout_seconds:
                status = "timed_out"
                process.kill()
                break
            if shutil.disk_usage(disk_check_path).free < 10 * 1024 ** 3:
                status = "disk_floor"
                process.kill()
                break
            if output_bytes(dest) > OUTPUT_CAP:
                status = "output_limit"
                process.kill()
                break
            time.sleep(2)
        code = process.wait()
    except BaseException as exc:
        process.kill()
        code = process.wait()
        (dest / "trial_status.json").write_text(json.dumps({
            "status": "interrupted", "returncode": code,
            "wall_seconds": time.monotonic() - start, "timeout_seconds": timeout_seconds,
            "exception": type(exc).__name__}, indent=2) + "\n")
        raise
    finally:
        drain.join(timeout=10)
        process.stdout.close()
    if status is None:
        status = "failed" if code else "completed"
    if status == "completed":
        summary = dest / "summary.json"
        if not summary.is_file() or json.loads(summary.read_text()).get("model_count", 0) == 0:
            status = "no_model"
        elif json.loads(summary.read_text()).get("registered", 0) < 3:
            status = "insufficient_sparse"
    result = {"status": status, "returncode": code, "wall_seconds": time.monotonic() - start,
              "timeout_seconds": timeout_seconds, "log_truncated": state["truncated"],
              "output_bytes": output_bytes(dest)}
    (dest / "trial_status.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--arm", choices=("raw", "foreground", "both"), default="both")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--seal", action="store_true", help="seal an existing successful output directory")
    p.add_argument("--recover-diagnostics", action="store_true", help="continue diagnostics after mapper succeeded")
    p.add_argument("--timeout-seconds", type=int, default=590)
    a = p.parse_args()
    if not 1 <= a.timeout_seconds <= 600:
        p.error("--timeout-seconds must be 1..600")
    if a.seal:
        print(json.dumps({k: seal(a.output)[k] for k in ("model_dir", "model_files_sha256")}))
        return
    if a.recover_diagnostics:
        report = recover_diagnostics(a.manifest, a.output)
        print(json.dumps({k: report[k] for k in ("registered", "points3D", "model_files_sha256")}))
        return
    if a.worker:
        if a.arm == "both":
            raise ValueError("worker needs one arm")
        worker(a.manifest, a.output, a.arm)
        return
    manifest = json.loads(a.manifest.read_text())
    validate(manifest)
    if shutil.disk_usage(a.output.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB free")
    any_failure = False
    for arm in ("raw", "foreground") if a.arm == "both" else (a.arm,):
        dest = a.output / arm
        if dest.exists():
            raise FileExistsError(dest)
        cmd = [str(VENV_PYTHON), str(Path(__file__)), "--worker", "--manifest", str(a.manifest),
               "--output", str(dest), "--arm", arm]
        result = run_trial(cmd, dest, a.timeout_seconds, a.output.parent)
        print(json.dumps({"arm": arm, **result}), flush=True)
        any_failure |= result["status"] != "completed"
    if any_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
