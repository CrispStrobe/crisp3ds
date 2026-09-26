#!/usr/bin/env python3
"""Replay the saved original-JPEG COLMAP database through the pinned mapper."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_RUN = ROOT / "build-opencv/colmap-sparse/originals-supervisor-001"
DEFAULT_OUTPUT = ROOT / "build-opencv/colmap-sparse/originals-mapper-replay-001"
CONVERSION = ROOT / "build-opencv/tree-refine/baseline/scene/conversion.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_inputs():
    source = json.loads(CONVERSION.read_text())
    if len(source["views"]) != 10:
        raise ValueError("expected ten original images")
    images = {}
    for view in source["views"]:
        path = Path(view["source"]).resolve(strict=True)
        if path.name != view["name"] or path.suffix.lower() != ".jpg":
            raise ValueError(f"unexpected original image {path}")
        images[path.name] = {"path": str(path), "sha256": sha256(path)}
    if len(images) != 10 or len({Path(x["path"]).parent for x in images.values()}) != 1:
        raise ValueError("original image set is incomplete or spans directories")
    return images


def source_hashes(source_run, images, pycolmap):
    return {"database": sha256(source_run / "database.db"),
            "baseline_provenance": sha256(source_run / "provenance.json"),
            "conversion": sha256(CONVERSION),
            "diagnose_script": sha256(Path(__file__)),
            "pycolmap_binary": sha256(Path(pycolmap._core.__file__)),
            "images": {name: sha256(Path(item["path"])) for name, item in images.items()}}


def assert_source_consistent(baseline, images, source_hash):
    if baseline.get("lane") != "baseline_originals" or baseline.get("random_seed") != {"pycolmap": 0}:
        raise ValueError("source is not the pinned original-image baseline")
    if baseline.get("input_hashes_before") != baseline.get("input_hashes_after"):
        raise ValueError("original baseline inputs changed during its run")
    expected = baseline["input_hashes_after"]["images"]
    if set(expected) != set(images):
        raise ValueError("original image set differs from baseline")
    for name, item in images.items():
        if expected[name]["source"] != item["sha256"] or source_hash["images"][name] != item["sha256"]:
            raise ValueError(f"original image digest differs from baseline: {name}")
    if baseline["input_hashes_after"]["conversion_json"] != source_hash["conversion"]:
        raise ValueError("conversion manifest differs from baseline")
    if baseline["software_hashes_before"] != baseline["software_hashes_after"]:
        raise ValueError("baseline software changed during its run")
    if baseline["software_hashes_before"]["pycolmap_binary"] != source_hash["pycolmap_binary"]:
        raise ValueError("installed PyCOLMAP binary differs from baseline")


def sqlite_header_changes(source_path, copied_path):
    """Allow only SQLite's two coordinated header counters to advance."""
    original = Path(source_path).read_bytes()
    replayed = Path(copied_path).read_bytes()
    if (len(original) != len(replayed) or
            original[:16] != b"SQLite format 3\x00" or
            replayed[:16] != b"SQLite format 3\x00"):
        raise ValueError("copied database changed size or format")
    changed = [i for i, (a, b) in enumerate(zip(original, replayed)) if a != b]
    allowed = set(range(24, 28)) | set(range(92, 96))
    if not set(changed).issubset(allowed):
        raise ValueError(f"copied database changed outside SQLite header counters: {changed[:12]}")
    if replayed[24:28] != replayed[92:96]:
        raise ValueError("SQLite header counter and mirror differ")
    return changed


def sqlite_content_sha256(path):
    contents = bytearray(Path(path).read_bytes())
    if contents[:16] != b"SQLite format 3\x00":
        raise ValueError("not a SQLite database")
    contents[24:28] = b"\x00" * 4
    contents[92:96] = b"\x00" * 4
    return hashlib.sha256(contents).hexdigest()


def mapper_options(pycolmap, baseline, snapshot_path):
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    base_options = baseline["options"]["incremental_pipeline"]
    actual = json.loads(json.dumps(options.todict(), default=str))
    if actual != base_options:
        raise ValueError("installed mapper defaults differ from original baseline")
    options.snapshot_path = str(snapshot_path)
    options.snapshot_images_freq = 1
    return options


def model_state(model):
    cameras = []
    for camera_id, camera in sorted(model.cameras.items()):
        cameras.append({"camera_id": int(camera_id), "model": str(camera.model),
                        "width": int(camera.width), "height": int(camera.height),
                        "params": [float(x) for x in camera.params],
                        "prior_focal_length": bool(camera.has_prior_focal_length)})
    return {"registered_images": model.num_reg_images(),
            "registered_image_names": sorted(model.image(i).name for i in model.reg_image_ids()),
            "points3D": model.num_points3D(), "cameras": cameras}


def manager_state(manager, stage):
    return {"stage": stage, "models": [model_state(manager.get(i)) for i in range(manager.size())]}


def replay(source_run, out):
    import pycolmap
    import numpy as np
    baseline = json.loads((source_run / "provenance.json").read_text())
    images = raw_inputs()
    before = source_hashes(source_run, images, pycolmap)
    assert_source_consistent(baseline, images, before)
    if baseline["pycolmap_version"] != pycolmap.__version__:
        raise ValueError("installed PyCOLMAP version differs from baseline")
    out.mkdir(parents=True, exist_ok=True)
    database = out / "database.db"
    shutil.copyfile(source_run / "database.db", database)
    if sha256(database) != before["database"]:
        raise ValueError("database copy differs from source")
    snapshots = out / "snapshots"
    snapshots.mkdir()
    options = mapper_options(pycolmap, baseline, snapshots)
    provenance = {"schema": "colmap_mapper_replay_v1",
                  "source_run": str(source_run.resolve()),
                  "pycolmap_version": pycolmap.__version__, "numpy_version": np.__version__,
                  "random_seed": 0,
                  "logging_verbose_level": 1,
                  "mapper_options": json.loads(json.dumps(options.todict(), default=str)),
                  "source_hashes_before": before, "source_hashes_after": None,
                  "copied_database_sha256_before": sha256(database),
                  "copied_database_sha256_after": None,
                  "copied_database_content_sha256_before": sqlite_content_sha256(database),
                  "copied_database_content_sha256_after": None,
                  "copied_database_changed_header_offsets": None,
                  "instrumentation": "snapshot every new image after local/optional global refinement; callbacks after init, each next image, and final model"}
    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    pycolmap.logging.verbose_level = 1
    pycolmap.set_random_seed(0)
    manager = pycolmap.ReconstructionManager()
    image_dir = str(Path(next(iter(images.values()))["path"]).parent)
    pipeline = pycolmap.IncrementalPipeline(options, image_dir, str(database), manager)
    events = []
    events_path = out / "events.jsonl"

    def record(stage):
        event = manager_state(manager, stage)
        event["snapshot_directories"] = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        events.append(event)
        with events_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    pipeline.add_callback(pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK,
                          lambda: record("after_initial_global_BA_and_filter"))
    pipeline.add_callback(pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK,
                          lambda: record("after_next_image_local_or_global_refinement"))
    pipeline.add_callback(pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK,
                          lambda: record("after_final_reconstruction"))
    pipeline.run()
    record("after_pipeline_run")
    binary_dir = out / "model_binary"
    binary_dir.mkdir()
    manager.write(str(binary_dir))
    snapshot_states = []
    snapshot_text = out / "snapshot_text"
    snapshot_text.mkdir()
    for snap in sorted(snapshots.iterdir()):
        if not snap.is_dir():
            continue
        model = pycolmap.Reconstruction(str(snap))
        text_dir = snapshot_text / snap.name
        text_dir.mkdir()
        model.write_text(str(text_dir))
        snapshot_states.append({"directory": str(snap.relative_to(out)), **model_state(model)})
    (out / "snapshot_summary.json").write_text(json.dumps(snapshot_states, indent=2) + "\n")
    if manager.size():
        model = max((manager.get(i) for i in range(manager.size())),
                    key=lambda m: (m.num_reg_images(), m.num_points3D()))
        text_dir = out / "model_text"
        text_dir.mkdir()
        model.write_text(str(text_dir))
        model.export_PLY(str(out / "points.ply"))
    after = source_hashes(source_run, images, pycolmap)
    provenance["source_hashes_after"] = after
    provenance["copied_database_sha256_after"] = sha256(database)
    provenance["copied_database_content_sha256_after"] = sqlite_content_sha256(database)
    provenance["copied_database_changed_header_offsets"] = sqlite_header_changes(
        source_run / "database.db", database)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if before != after:
        raise ValueError("source images, database, or software changed during mapper replay")
    if (provenance["copied_database_content_sha256_before"] !=
            provenance["copied_database_content_sha256_after"]):
        raise ValueError("copied database content changed during mapper replay")
    report = {"model_count": manager.size(),
              "final_models": manager_state(manager, "final")["models"],
              "callback_count": len(events), "snapshot_count": len(snapshot_states),
              "source_database_sha256": before["database"],
              "validation": "mapper-only diagnostic; no independent accuracy measure"}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    source_run = args.source_run.resolve(strict=True)
    output = args.output.absolute()
    if args.worker:
        replay(source_run, output)
        return
    from scripts.research_job.guard import run_child
    output.parent.mkdir(parents=True, exist_ok=True)
    status = run_child([sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--source-run", str(source_run), "--output", str(output)],
                       cwd=ROOT, output_dir=output, timeout_seconds=300,
                       max_output_bytes=1 << 30, reserve_bytes=10 << 30,
                       max_log_bytes=10 << 20)
    print(json.dumps(status, indent=2))
    if status["status"] != "succeeded":
        sys.exit(1)


if __name__ == "__main__":
    main()
