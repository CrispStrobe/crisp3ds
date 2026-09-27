#!/usr/bin/env python3
"""Fresh, hash-checked refine/texture continuation after a scene handoff error."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.classical_backend import geometry
from scripts.classical_backend.run import obj_counts
from scripts.upstream_control.fetch import OUTPUT as SAMPLE, sha256
from scripts.upstream_control.run import (BIN, MAX_SECONDS, MAX_TOTAL_BYTES, RESERVE,
                                          folder_bytes, run_stage, verify_sample)

SOURCE = ROOT / "build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001"
OUTPUT = ROOT / "build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001-continuation"


def continue_run(source=SOURCE, output=OUTPUT, sample=SAMPLE):
    source, output, sample = map(Path, (source, output, sample))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    sample_manifest, sample_hash = verify_sample(sample)
    prior_report_path = source / "report.json"
    prior = json.loads(prior_report_path.read_text())
    if prior.get("status") != "failed" or len(prior.get("stages", [])) != 3 or (
            [s["name"] for s in prior["stages"]] != ["densify", "mesh", "refine"] or
            [s["status"] for s in prior["stages"]] != ["completed", "completed", "failed"] or
            prior.get("sample_manifest_sha256") != sample_hash):
        raise ValueError("unexpected original failed run")
    artifact_names = ("scene_dense.ply", "scene_dense_mesh.ply")
    for stage, name in zip(prior["stages"][:2], artifact_names):
        if sha256(source / name) != stage["artifact"]["sha256"]:
            raise ValueError(f"upstream control stage artifact changed: {name}")
    if geometry.counts(source / "scene_dense_mesh.ply")[1] < 1:
        raise ValueError("no rough mesh faces to refine")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < RESERVE + MAX_TOTAL_BYTES:
        raise RuntimeError("insufficient room for bounded continuation")
    output.mkdir()
    (output / "images").mkdir()
    copied = ["scene_dense.mvs", "scene_dense_mesh.ply"] + [f"images/{i:05}.jpg" for i in range(11)]
    for name in copied:
        target = output / name
        shutil.copyfile(source / name, target)
        if sha256(target) != sha256(source / name):
            raise ValueError(f"copied continuation input differs: {name}")
    report = {"schema": "upstream_sceaux_v23_stage_oracle_v24_cpu_continuation_v1",
              "lane": "stage-oracle continuation, upstream supplied cameras; no image-only SfM claim",
              "source_run_report_sha256": sha256(prior_report_path),
              "source_run_status": prior["status"], "sample_manifest_sha256": sample_hash,
              "copied_inputs_sha256": {name: sha256(output / name) for name in copied},
              "stages": [], "status": "running", "deadline_seconds": MAX_SECONDS}
    report_path = output / "report.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    deadline = time.monotonic() + MAX_SECONDS
    common = ["--working-folder", output, "--max-threads", "2"]
    specs = [
        ("refine", [BIN / "RefineMesh", "-i", output / "scene_dense.mvs", "-m", output / "scene_dense_mesh.ply",
                    "-o", output / "scene_dense_mesh_refine.mvs", "--resolution-level", "2", "--scales", "1", *common],
         output / "scene_dense_mesh_refine.ply"),
        ("texture", [BIN / "TextureMesh", "-i", output / "scene_dense.mvs", "-m", output / "scene_dense_mesh_refine.ply",
                     "-o", output / "scene_dense_mesh_refine_texture.mvs", "--resolution-level", "2",
                     "--export-type", "obj", *common], output / "scene_dense_mesh_refine_texture.obj"),
    ]
    try:
        for name, command, artifact in specs:
            stage = run_stage(name, command, output, sample_manifest["total_bytes"], deadline)
            report["stages"].append(stage)
            save()
            if stage["status"] != "completed":
                raise RuntimeError(f"{name} {stage['status']}")
            if name == "refine":
                counts = geometry.inspect(artifact)
                if counts["faces"] < 1:
                    raise ValueError("refined mesh has no faces")
            else:
                vertices, faces = obj_counts(artifact)
                counts = {"vertices": vertices, "faces": faces}
                if faces < 1:
                    raise ValueError("textured OBJ has no faces")
            stage["artifact"] = {"path": str(artifact), "sha256": sha256(artifact), **counts}
            save()
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        raise
    finally:
        report["output_bytes"] = folder_bytes(output)
        save()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    args = parser.parse_args()
    report = continue_run(args.source, args.output, args.sample)
    print(json.dumps({"status": report["status"], "stages": [s["name"] for s in report["stages"]]}))


if __name__ == "__main__":
    main()
