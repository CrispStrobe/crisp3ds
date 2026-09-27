"""Evaluation-only frozen YCB scoring for a completed fresh masked OpenMVS run.

The Google mesh and Berkeley camera diagnostic are post hoc oracles. Neither is
used to choose native options, filter triangles, or repair a failed producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.classical_backend import fresh_masked_complete as producer
from scripts.classical_backend import geometry
from scripts.classical_backend.dense_masks import MODEL_FILES, model_hashes
from scripts.classical_backend.run import RESERVE, digest
from scripts.object_dataset import align, evaluate, surface_metrics


REFERENCE_SHA256 = "6e0187aa961aef4fa21dfc753023a4be7e82924e6398dc0875a2b0d609363ed4"
STAGES = ("copy_inputs", "undistort", "warp_masks", "import", "densify",
          "dmap_mask_check", "mesh", "refine", "texture")
THRESHOLD_FRACTIONS = (0.005, 0.01, 0.02)


def checked_run(run: Path) -> dict:
    """Require a complete, hash-sealed producer and its original camera gate."""
    run = Path(run)
    result_path = run / "result.json"
    if run.is_symlink() or not run.is_dir() or result_path.is_symlink():
        raise ValueError("missing or linked fresh run")
    report = json.loads(result_path.read_text())
    if (report.get("schema") != producer.SCHEMA or report.get("status") != "complete" or
            report.get("sources_unchanged") is not True or
            report.get("quality_accepted") is not False or
            report.get("metric_scale_verified") is not False or
            [row.get("name") for row in report.get("stages", [])] != list(STAGES) or
            any(row.get("status") != "complete" for row in report["stages"])):
        raise ValueError("fresh producer is incomplete or unsealed")
    source = report.get("source", {})
    required = ("model", "images", "pose_masks", "manifest", "producer", "camera_gate",
                "producer_sha256", "camera_gate_sha256", "model_sha256")
    if any(key not in source for key in required):
        raise ValueError("fresh producer lacks source binding")
    expected_model = source["model_sha256"]
    if set(expected_model) != set(MODEL_FILES):
        raise ValueError("fresh producer model hash inventory differs")
    bound = producer.input_binding(*(Path(source[key]) for key in
        ("model", "images", "pose_masks", "manifest", "producer", "camera_gate")),
        source["producer_sha256"], source["camera_gate_sha256"], expected_model)
    for key, value in bound.items():
        if source.get(key) != value:
            raise ValueError(f"fresh producer source binding changed: {key}")
    if model_hashes(run / "sparse" / "0") != expected_model:
        raise ValueError("copied sparse model differs from reviewed camera model")
    artifacts = report.get("artifact_sha256", {})
    if not all(name in artifacts for name in ("refined.ply", "mesh.ply", "textured.obj")):
        raise ValueError("fresh producer lacks final geometry hashes")
    for name, expected in artifacts.items():
        path = run / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"fresh native artifact hash differs: {name}")
    if (run / "refined.ply").stat().st_size == 0:
        raise ValueError("fresh refined mesh is empty")
    return {"result_sha256": digest(result_path), "source": source,
            "refined_sha256": artifacts["refined.ply"], "sparse_sha256": expected_model}


def score(run: Path, reference: Path, output: Path, *, parent_run: Path | None = None) -> dict:
    run, reference, output = map(Path, (run, reference, output))
    if output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("evaluation output must be a fresh directory under a real parent")
    if reference.is_symlink() or not reference.is_file() or digest(reference) != REFERENCE_SHA256:
        raise ValueError("Google reference differs from frozen geometry hash")
    evidence = checked_run(run)
    if shutil.disk_usage(output.parent).free < RESERVE + (128 << 20):
        raise ValueError("evaluation volume lacks 10 GiB floor plus headroom")
    reference_mesh = evaluate.inspect_ply(reference, geometry=True)
    if reference_mesh[0]["nontriangle_faces"]:
        raise ValueError("reference is not a triangle mesh")
    output.mkdir()
    normalized = output / "refined-geometry.ply"
    export = geometry.export_geometry(run / "refined.ply", normalized)
    mesh = evaluate.inspect_ply(normalized, geometry=True)
    if mesh[0]["nontriangle_faces"]:
        raise ValueError("fresh mesh is not a triangle mesh")
    deadline = time.monotonic() + align.MAX_SECONDS
    matrix, fit_diagnostics = align.align_points(
        evaluate.sample_surface(*reference_mesh[1:], 1024, align.SEED),
        evaluate.sample_surface(*mesh[1:], 1024, align.SEED), deadline=deadline)
    reference_diagonal = float(np.linalg.norm(np.ptp(np.asarray(reference_mesh[1]), axis=0)))
    thresholds = [reference_diagonal * fraction for fraction in THRESHOLD_FRACTIONS]
    primary = surface_metrics.compare(reference_mesh[1:], mesh[1:], matrix,
                                      thresholds=thresholds, count=2048, seed=2027)
    control = surface_metrics.compare(reference_mesh[1:], reference_mesh[1:], np.eye(4),
                                      thresholds=thresholds, count=2048, seed=2027)
    shared = {"status": "unavailable", "reason": "no exact shared sparse model and frozen parent transform supplied"}
    if parent_run is not None:
        parent_run = Path(parent_run)
        parent_model = model_hashes(parent_run / "dense" / "sparse")
        if parent_model == evidence["sparse_sha256"]:
            parent_fit = parent_run / "reference-fit-v1.json"
            parent_mesh = parent_run / "refined-geometry.ply"
            provenance, parent_matrix, _ = evaluate.load_sim3(parent_fit, reference, parent_mesh)
            if provenance["registration_basis"] != "reference-fit":
                raise ValueError("shared parent transform is not reference-fitted")
            shared = {"status": "available", "parent_run": str(parent_run.resolve()),
                      "parent_transform_sha256": digest(parent_fit),
                      "parent_sparse_sha256": parent_model,
                      "comparison": surface_metrics.compare(reference_mesh[1:], mesh[1:], parent_matrix,
                                                            thresholds=thresholds, count=2048, seed=2027)}
        else:
            shared = {"status": "unavailable", "reason": "candidate and parent sparse model hashes differ",
                      "parent_sparse_sha256": parent_model}
    report = {"schema": "classical_fresh_complete_evaluation_v1",
              "producer": evidence, "geometry_export": export,
              "reference_sha256": REFERENCE_SHA256,
              "normalization": "OpenMVS XYZ promoted to float64; native coordinates and triangles preserved",
              "reference_fit": {"matrix": matrix.tolist(), "registration_basis": "reference-fit",
                                "fitted_scale": float(np.linalg.svd(matrix[:3, :3], compute_uv=False).mean()),
                                "samples_per_mesh": 1024, "seed": align.SEED,
                                "diagnostics": fit_diagnostics},
              "threshold_fractions_of_reference_diagonal": THRESHOLD_FRACTIONS,
              "primary": primary, "reference_self_control": control, "shared_gauge": shared,
              "metric_accuracy_claim_allowed": False, "quality_accepted": False,
              "interpretation": "development-object shape comparison after reference-fitted Sim(3); neither physical scale nor held-out quality is established"}
    if digest(reference) != REFERENCE_SHA256 or checked_run(run) != evidence:
        raise ValueError("reference or fresh producer changed during evaluation")
    with (output / "score.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path,
                        help="optional 006 parent; shared gauge only if sparse hashes match exactly")
    args = parser.parse_args()
    result = score(args.run, args.reference, args.output, parent_run=args.parent_run)
    print(json.dumps({"score": str(args.output / "score.json"),
                      "f1": [row["f_score"] for row in result["primary"]["threshold_scores"]],
                      "shared_gauge": result["shared_gauge"]["status"]}))


if __name__ == "__main__":
    main()
