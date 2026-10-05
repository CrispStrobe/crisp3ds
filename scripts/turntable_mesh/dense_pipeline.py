"""One command from a calibrated photo set to a closed STL.

Stages, each a separate bounded process so two interpreters can be used:

  inputs   camera table + undistorted masks for every registered view
  stereo   mask repair, silhouette hull, multiscale stereo, TSDF   (needs Torch)
  mesh     hull-bounded surface extraction to a binary STL       (needs SciPy, scikit-image)
  check    silhouette agreement with the photos and a preview     (needs OpenCV)

Starting point is an AliceVision SfM scene with one shared radialk3 lens, its
native undistorted images and one object mask per source photo; or an existing
``--inputs`` directory, which skips the first stage. Camera recovery and
segmentation are separate earlier steps.

Interpreters: ``--python`` runs inputs/mesh/check, ``--torch-python`` runs
stereo. Both default to the environment variables CRISP3DS_PYTHON and
CRISP3DS_TORCH_PYTHON, then to the interpreter running this script.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .dense_config import add_arguments, build, describe

REPOSITORY = Path(__file__).resolve().parents[2]


def stop_group(process, force):
    """End a stage and everything it started."""
    if os.name == "nt":  # no process groups to signal: end the process tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)


def bounded(command, log_path, timeout, environment):
    """Run in its own process group; kill the whole group at the deadline."""
    started = time.monotonic()
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPOSITORY,
                                   env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            stop_group(process, force=False)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_group(process, force=True)
                process.wait()
            code, timed_out = None, True
    return {"command": [str(c) for c in command], "exit_code": code, "timed_out": timed_out,
            "seconds": time.monotonic() - started, "log": str(log_path)}


def run(args):
    config = build(args.config, args.set)
    output = Path(args.output).absolute()
    if output.exists():
        raise FileExistsError(output)
    if args.inputs is None and not (args.scene and args.prepared and args.raw_masks):
        raise ValueError("give --inputs, or all of --scene, --prepared and --raw-masks")
    free = shutil.disk_usage(output.parent if output.parent.exists() else REPOSITORY).free
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f"only {free / 2**30:.1f} GiB free; need {args.minimum_free_gib} (see --minimum-free-gib)")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config.to_json(), indent=2) + "\n")
    python = args.python or os.environ.get("CRISP3DS_PYTHON") or sys.executable
    torch_python = args.torch_python or os.environ.get("CRISP3DS_TORCH_PYTHON") or sys.executable
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY), "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = str(args.threads)
    report = {"output": str(output), "device": args.device, "stages": {}, "status": "running"}
    started = time.monotonic()

    def stage(name, command, timeout):
        print(f"[{name}] ...", flush=True)
        result = bounded(command, output / f"{name}.log", timeout, environment)
        report["stages"][name] = result
        (output / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
        if result["timed_out"] or result["exit_code"]:
            report["status"] = f"failed in {name}"
            (output / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
            tail = Path(result["log"]).read_text()[-2000:]
            raise RuntimeError(f"stage {name} failed ({'deadline' if result['timed_out'] else result['exit_code']}):\n{tail}")
        print(f"[{name}] {result['seconds']:.1f}s", flush=True)

    inputs = Path(args.inputs).absolute() if args.inputs else output / "inputs"
    if args.inputs is None:
        stage("inputs", [python, "-m", "scripts.turntable_mesh.dense_all_views_inputs", "--scene", args.scene,
                         "--prepared", args.prepared, "--raw-masks", args.raw_masks, "--output", inputs], 600)
    stage("stereo", [torch_python, "-m", "scripts.turntable_mesh.multiscale_stereo", "--inputs", inputs,
                     "--output", output / "stereo", "--device", args.device, "--config", output / "config.json",
                     *(["--reuse-depths", Path(args.reuse_depths).absolute()] if args.reuse_depths else [])],
          args.stereo_timeout)
    stage("mesh", [python, "-m", "scripts.turntable_mesh.tsdf_hull_mesh", "--volume", output / "stereo/volume.npz",
                   "--output", output / "mesh", "--config", output / "config.json"], 900)
    if not args.skip_check:
        stage("check", [python, "-m", "scripts.turntable_mesh.mesh_photo_check", "--inputs", inputs,
                        "--mesh", output / "mesh/mesh.stl", "--output", output / "check",
                        "--repaired-masks", output / "stereo/masks-repaired",
                        "--preview-views", "0" if args.no_preview else "3"], 900)
        report["photo_check"] = json.loads((output / "check/result.json").read_text())
    if not args.keep_volume:
        (output / "stereo/volume.npz").unlink()
    mesh = json.loads((output / "mesh/result.json").read_text())
    report.update(status="complete", mesh=str(output / "mesh/mesh.stl"), closed=mesh["closed"],
                  triangles=mesh["triangles"], genus=mesh["genus"], seconds=time.monotonic() - started,
                  reference_used=False, physical_scale_established=False)
    (output / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="fresh output directory")
    source = parser.add_argument_group("photo set (either --inputs or the other three)")
    source.add_argument("--inputs", type=Path, help="existing inputs directory (cameras.json, sparse_points.npy)")
    source.add_argument("--scene", type=Path, help="AliceVision .sfm with poses and one radialk3 intrinsic")
    source.add_argument("--prepared", type=Path, help="native undistorted images named <viewId>.png")
    source.add_argument("--raw-masks", type=Path, help="0/255 masks named like the source photos")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--python", help="interpreter with NumPy, SciPy, scikit-image, OpenCV, Pillow")
    parser.add_argument("--torch-python", help="interpreter with Torch, NumPy, Pillow")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stereo-timeout", type=int, default=3600, help="seconds")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--reuse-depths", type=Path, help="depths.npz of an earlier run on the same inputs: skip matching, re-fuse only")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--keep-volume", action="store_true", help="keep stereo/volume.npz for re-meshing")
    parser.add_argument("--list-settings", action="store_true", help="print every --set key with its default")
    add_arguments(parser)
    args = parser.parse_args()
    if args.list_settings:
        print(describe())
        return
    if args.output is None:
        parser.error("--output is required")
    report = run(args)
    print(json.dumps({k: report[k] for k in ("status", "mesh", "closed", "triangles", "genus", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
