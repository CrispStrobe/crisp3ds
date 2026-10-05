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
from .dense_events import SCHEMA, EventLog

REPOSITORY = Path(__file__).resolve().parents[2]


def stop_group(process, force):
    """End a stage and everything it started."""
    if os.name == "nt":  # no process groups to signal: end the process tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)


def bounded(command, log_path, timeout, environment, tick=None, cancel=None):
    """Run in its own process group; stop the whole group at the deadline or on cancel."""
    started = time.monotonic()
    posix = os.name == "posix"
    with open(log_path, "w") as log:
        process = subprocess.Popen([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, cwd=REPOSITORY,
                                   env=environment, start_new_session=posix)
        timed_out = cancelled = False
        while process.poll() is None:
            if tick:
                tick()
            cancelled = bool(cancel and cancel())
            timed_out = time.monotonic() - started > timeout
            if timed_out or cancelled:
                stop_group(process, force=False)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_group(process, force=True)
                    process.wait()
                break
            time.sleep(0.5)
        if tick:
            tick()
    return {"command": [str(c) for c in command], "exit_code": None if timed_out or cancelled else process.returncode,
            "timed_out": timed_out, "cancelled": cancelled, "seconds": time.monotonic() - started, "log": str(log_path)}


def input_sheet(inputs, path, count=8):
    """Contact sheet of evenly spaced photos with their mask outlines. Needs OpenCV; skipped without it."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    from .multiscale_stereo import load_views

    rows = load_views(inputs)
    tiles = []
    for index in np.linspace(0, len(rows), min(count, len(rows)), endpoint=False, dtype=int):
        photo = cv2.imread(rows[index]["image"])
        mask = cv2.imread(rows[index]["mask"], cv2.IMREAD_GRAYSCALE)
        if photo is None or mask is None:
            return False
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(photo, contours, -1, (60, 220, 60), 2)
        tiles.append(cv2.resize(photo, (420, round(420 * photo.shape[0] / photo.shape[1]))))
    columns = 4
    height = tiles[0].shape[0]
    sheet = np.full((height * ((len(tiles) + columns - 1) // columns), 420 * columns, 3), 30, np.uint8)
    for n, tile in enumerate(tiles):
        sheet[(n // columns) * height : (n // columns) * height + tile.shape[0], (n % columns) * 420 : (n % columns + 1) * 420] = tile[:height]
    cv2.imwrite(str(path), sheet)
    return True


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
    events_path = output / "events.jsonl"
    events = EventLog(events_path)
    report = {"output": str(output), "device": args.device, "stages": {}, "status": "running"}
    started = time.monotonic()
    events.emit("run_started", schema=SCHEMA, configuration=config.to_json(), device=args.device,
                inputs=str(args.inputs or args.scene))

    def cancelled():
        return (output / "cancel").exists()

    def finish(status):
        report["status"] = status
        report["seconds"] = time.monotonic() - started
        (output / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
        events.emit("run_finished", status="complete" if status == "complete" else ("cancelled" if cancelled() else "failed"),
                    seconds=report["seconds"])

    def stage(name, command, timeout, tick=None):
        print(f"[{name}] ...", flush=True)
        events.emit("stage_started", stage=name)
        result = bounded(command, output / f"{name}.log", timeout, environment, tick, cancelled)
        report["stages"][name] = result
        (output / "pipeline.json").write_text(json.dumps(report, indent=2) + "\n")
        if result["timed_out"] or result["cancelled"] or result["exit_code"]:
            reason = "cancelled" if result["cancelled"] else "deadline" if result["timed_out"] else f"exit code {result['exit_code']}"
            tail = Path(result["log"]).read_text()[-2000:]
            events.emit("error", stage=name, message=f"{reason}: {tail[-400:]}")
            finish(f"failed in {name}")
            raise RuntimeError(f"stage {name} failed ({reason}):\n{tail}")
        events.emit("stage_finished", stage=name, seconds=result["seconds"])
        print(f"[{name}] {result['seconds']:.1f}s", flush=True)

    inputs = Path(args.inputs).absolute() if args.inputs else output / "inputs"
    if args.inputs is None:
        stage("inputs", [python, "-m", "scripts.turntable_mesh.dense_all_views_inputs", "--scene", args.scene,
                         "--prepared", args.prepared, "--raw-masks", args.raw_masks, "--output", inputs], 600)
    if input_sheet(inputs, output / "input-sheet.png"):
        events.artifact("input_sheet", output / "input-sheet.png", "Photos with mask outlines", stage="inputs")

    # Preview volumes appear while matching runs; mesh them coarsely one at a time.
    meshing = {"process": None, "done": set()}

    def mesh_previews():
        if args.no_live_previews:
            return
        running = meshing["process"]
        if running is not None and running.poll() is None:
            return
        for volume in sorted((output / "stereo/preview").glob("*.npz")):
            if volume.name in meshing["done"]:
                continue
            meshing["done"].add(volume.name)
            names = {"00-hull": "Silhouette hull", "01-hull-repaired": "Silhouette hull after mask repair"}
            label = names[volume.stem] if volume.stem in names else "Surface after level " + str(int(volume.stem.split("-")[-1]) + 1)
            log = open(output / "stereo" / "preview" / f"{volume.stem}.log", "w")
            meshing["process"] = subprocess.Popen(
                [str(python), "-m", "scripts.turntable_mesh.tsdf_hull_mesh", "--volume", str(volume),
                 "--output", str(volume.with_suffix("")), "--config", str(output / "config.json"),
                 "--step", str(args.preview_step), "--events", str(events_path), "--label", label],
                stdout=log, stderr=subprocess.STDOUT, cwd=REPOSITORY, env=environment)
            return

    stage("stereo", [torch_python, "-m", "scripts.turntable_mesh.multiscale_stereo", "--inputs", inputs,
                     "--output", output / "stereo", "--device", args.device, "--config", output / "config.json",
                     "--events", events_path, *([] if args.no_live_previews else ["--previews"]),
                     *(["--reuse-depths", Path(args.reuse_depths).absolute()] if args.reuse_depths else [])],
          args.stereo_timeout, mesh_previews)
    stage("mesh", [python, "-m", "scripts.turntable_mesh.tsdf_hull_mesh", "--volume", output / "stereo/volume.npz",
                   "--output", output / "mesh", "--config", output / "config.json", "--events", events_path], 900,
          mesh_previews)
    while not args.no_live_previews:  # let outstanding previews finish; they are small
        mesh_previews()
        if meshing["process"] is None or (meshing["process"].poll() is not None
                                          and all(v.name in meshing["done"] for v in (output / "stereo/preview").glob("*.npz"))):
            break
        time.sleep(0.5)
    for volume in (output / "stereo/preview").glob("*.npz") if (output / "stereo/preview").is_dir() else ():
        volume.unlink()
    if not args.skip_check:
        stage("check", [python, "-m", "scripts.turntable_mesh.mesh_photo_check", "--inputs", inputs,
                        "--mesh", output / "mesh/mesh.stl", "--output", output / "check",
                        "--repaired-masks", output / "stereo/masks-repaired", "--events", events_path,
                        "--preview-views", "0" if args.no_preview else "3"], 900)
        report["photo_check"] = json.loads((output / "check/result.json").read_text())
    if args.reference:
        stage("evaluate", [python, "-m", "scripts.turntable_mesh.scan_evaluate", "--mesh", output / "mesh/mesh.stl",
                           "--reference", args.reference, "--output", output / "scan"], 900)
        events.artifact("scan_overlay", output / "scan/overlay.png", "Distance to the reference scan", stage="evaluate")
        events.artifact("report", output / "scan/result.json", "Scanner evaluation", stage="evaluate")
    if not args.keep_volume:
        (output / "stereo/volume.npz").unlink()
    mesh = json.loads((output / "mesh/result.json").read_text())
    report.update(mesh=str(output / "mesh/mesh.stl"), closed=mesh["closed"],
                  triangles=mesh["triangles"], genus=mesh["genus"],
                  reference_used=False, physical_scale_established=False)
    events.artifact("report", output / "mesh/result.json", "Mesh report", stage="mesh")
    finish("complete")
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
    parser.add_argument("--reference", type=Path, help="independent scan (PLY) to score the result against afterwards; never used as input")
    parser.add_argument("--no-live-previews", action="store_true", help="do not mesh intermediate surfaces while matching")
    parser.add_argument("--preview-step", type=int, default=2, help="marching-cubes step of live preview meshes")
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
