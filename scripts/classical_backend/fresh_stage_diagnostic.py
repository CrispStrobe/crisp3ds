"""Read-only stage attribution for sealed fresh YCB OpenMVS geometry.

Google geometry is an evaluation oracle only. All stages use the already frozen
006 reference transform composed with the camera-only fresh-to-006 transport.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from scripts.classical_backend import dense_masks, geometry, score_fresh_complete
from scripts.classical_backend.fusion_point_diagnostic import load_openmvs_cloud
from scripts.classical_backend.run import RESERVE, digest
from scripts.object_dataset import evaluate, surface_metrics

COUNT = 2048
SEED = 2027
FRACTIONS = (0.005, 0.01, 0.02)
MAX_REPORT_BYTES = 2 << 20


def bounds(points):
    points = np.asarray(points, dtype=np.float64)
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist(),
            "extent": np.ptp(points, axis=0).tolist()}


def proximity(points, matrix, reference_tree, reference_bounds, thresholds):
    """Fixed point sample to exact Google triangles; no surface F1 claim."""
    if len(points) < COUNT:
        raise ValueError("dense cloud has fewer points than fixed sample")
    selected = np.random.default_rng(SEED + 1).choice(len(points), COUNT, replace=False)
    mapped = points[selected] @ matrix[:3, :3].T + matrix[:3, 3]
    distances = reference_tree.distances(mapped)
    lower, upper = reference_bounds
    return {"method": "uniform dense-vertex sample to exact reference triangles; not area-weighted mesh F1",
            "sample_count": COUNT, "seed": SEED + 1,
            "distance": {**evaluate.summarize_distances(distances),
                         "p95": float(np.quantile(distances, .95))},
            "within_threshold_fraction": [float(np.mean(distances <= threshold)) for threshold in thresholds],
            "within_reference_bbox_fraction": float(np.mean(np.all((mapped >= lower) & (mapped <= upper), axis=1)))}


def projection_support(points, run, mask_report):
    """Count projection into the original image-derived foreground masks.

    This is a silhouette support check, without occlusion or depth agreement.
    """
    import pycolmap

    selected = np.random.default_rng(SEED + 2).choice(len(points), min(COUNT, len(points)), replace=False)
    xyz = np.asarray(points[selected], dtype=np.float64)
    model_dir = run / "dense" / "sparse"
    model = pycolmap.Reconstruction(str(model_dir))
    if dense_masks.model_hashes(model_dir) != mask_report["undistorted_model_sha256"]:
        raise ValueError("projection camera model differs from sealed mask model")
    listed = {row["name"]: row for row in mask_report["images"]}
    if set(listed) != {image.name for image in model.images.values()}:
        raise ValueError("mask and camera image sets differ")
    visible = np.zeros(len(xyz), dtype=np.int32)
    supported = np.zeros(len(xyz), dtype=np.int32)
    mask_hashes = {}
    for image in sorted(model.images.values(), key=lambda item: item.name):
        camera = model.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise ValueError("projection support requires undistorted PINHOLE cameras")
        path = run / "masks" / listed[image.name]["native_mask_name"]
        if path.is_symlink() or digest(path) != listed[image.name]["mask_sha256"]:
            raise ValueError("foreground mask differs from sealed mask report")
        mask_hashes[image.name] = listed[image.name]["mask_sha256"]
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"))
        if mask.shape != (camera.height, camera.width) or not np.isin(mask, [0, 255]).all():
            raise ValueError("foreground mask dimensions or labels differ")
        rotation = image.cam_from_world.rotation.matrix()
        camera_xyz = xyz @ rotation.T + image.cam_from_world.translation
        z = camera_xyz[:, 2]
        good = z > 0
        indices = np.flatnonzero(good)
        fx, fy, cx, cy = camera.params
        px = np.floor(fx * camera_xyz[good, 0] / z[good] + cx).astype(np.int64)
        py = np.floor(fy * camera_xyz[good, 1] / z[good] + cy).astype(np.int64)
        inside = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
        indices, px, py = indices[inside], px[inside], py[inside]
        visible[indices] += 1
        supported[indices] += (mask[py, px] == 255)
    return {"sample_count": len(xyz), "seed": SEED + 2,
            "camera_count": len(listed), "mask_sha256": mask_hashes,
            "fraction_visible_in_any_camera": float(np.mean(visible > 0)),
            "fraction_foreground_in_any_camera": float(np.mean(supported > 0)),
            "fraction_foreground_in_at_least_two_cameras": float(np.mean(supported >= 2)),
            "median_visible_views": float(np.median(visible)),
            "median_foreground_views": float(np.median(supported)),
            "limitation": "projection into image-derived masks only; no occlusion or depth test"}


def run_diagnostic(run: Path, reference: Path, score_path: Path, parent_run: Path,
                   output: Path) -> dict:
    run, reference, score_path, parent_run, output = map(Path, (run, reference, score_path, parent_run, output))
    if output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a fresh directory under a real parent")
    if shutil.disk_usage(output.parent).free < RESERVE + (64 << 20):
        raise ValueError("evaluation volume lacks 10 GiB floor plus headroom")
    evidence = score_fresh_complete.checked_run(run)
    if reference.is_symlink() or digest(reference) != score_fresh_complete.REFERENCE_SHA256:
        raise ValueError("Google reference hash differs")
    if score_path.is_symlink() or not score_path.is_file():
        raise ValueError("missing frozen fresh score")
    score_hash = digest(score_path)
    score = json.loads(score_path.read_text())
    if (score.get("schema") != "classical_fresh_complete_evaluation_v1" or
            score.get("producer") != evidence or score.get("reference_sha256") != score_fresh_complete.REFERENCE_SHA256 or
            score.get("shared_gauge", {}).get("status") != "available" or
            score["shared_gauge"].get("parent_transform_sha256") != score_fresh_complete.PARENT_FIT_SHA256 or
            Path(score["shared_gauge"]["parent_run"]).resolve() != parent_run.resolve()):
        raise ValueError("fresh score is not bound to these frozen inputs")
    parent_fit = parent_run / "reference-fit-v1.json"
    if digest(parent_fit) != score_fresh_complete.PARENT_FIT_SHA256 or \
            dense_masks.model_hashes(parent_run / "dense" / "sparse") != score_fresh_complete.PARENT_MODEL_SHA256:
        raise ValueError("frozen 006 parent differs")
    evaluate.load_sim3(parent_fit, reference, parent_run / "refined-geometry.ply")
    transport = score_fresh_complete.transported_gauge(run / "sparse" / "0", parent_run / "dense" / "sparse", evidence["sparse_sha256"])
    if transport["status"] != "available":
        raise ValueError("camera-only transport unavailable")
    parent_matrix = np.asarray(evaluate.load_sim3(parent_fit, reference, parent_run / "refined-geometry.ply")[1])
    matrix = parent_matrix @ np.asarray(transport["candidate_to_parent_matrix"])
    if not np.allclose(matrix, score["shared_gauge"]["matrix_used_output_to_reference"], rtol=0, atol=1e-10):
        raise ValueError("fresh score's shared gauge differs from recomputed camera transport")
    mask_path = run / "masks" / "report.json"
    if digest(mask_path) != next(row["artifact"]["report_sha256"] for row in json.loads((run / "result.json").read_text())["stages"] if row["name"] == "warp_masks"):
        raise ValueError("mask report differs from sealed producer")
    mask_hash = digest(mask_path)
    mask_report = json.loads(mask_path.read_text())
    ref_info, ref_vertices, ref_faces = evaluate.inspect_ply(reference, geometry=True)
    if ref_info["nontriangle_faces"]:
        raise ValueError("reference has nontriangle faces")
    ref_vertices = np.asarray(ref_vertices, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(ref_vertices, axis=0)))
    thresholds = [diagonal * fraction for fraction in FRACTIONS]
    output.mkdir()
    stages = {}
    cloud_path = run / "dense.ply"
    cloud_hash = digest(cloud_path)
    if cloud_hash != json.loads((run / "result.json").read_text())["artifact_sha256"]["dense.ply"]:
        raise ValueError("dense cloud hash differs from producer")
    cloud = load_openmvs_cloud(cloud_path)
    if len(cloud) != next(row["artifact"]["points"] for row in json.loads((run / "result.json").read_text())["stages"] if row["name"] == "densify"):
        raise ValueError("dense cloud count differs from producer")
    triangles, dropped = surface_metrics.positive_triangles(ref_vertices, ref_faces)
    if dropped:
        raise ValueError("reference has zero-area faces")
    ref_tree = surface_metrics.TriangleBVH(triangles)
    stages["dense"] = {"source_sha256": cloud_hash, "points": len(cloud),
                       "reference_gauge_bounds": bounds(cloud @ matrix[:3, :3].T + matrix[:3, 3]),
                       "proximity": proximity(cloud, matrix, ref_tree, (ref_vertices.min(axis=0), ref_vertices.max(axis=0)), thresholds),
                       "foreground_projection": projection_support(cloud, run, mask_report)}
    for label, name in (("rough", "mesh.ply"), ("refined", "refined.ply")):
        native = run / name
        expected = json.loads((run / "result.json").read_text())["artifact_sha256"][name]
        if digest(native) != expected:
            raise ValueError(f"{name} differs from producer")
        normalized = output / f"{label}-geometry.ply"
        export = geometry.export_geometry(native, normalized)
        info, vertices, faces = evaluate.inspect_ply(normalized, geometry=True)
        if info["nontriangle_faces"]:
            raise ValueError(f"{name} has nontriangle faces")
        vertices = np.asarray(vertices, dtype=np.float64)
        comparison = surface_metrics.compare((ref_vertices, ref_faces), (vertices, faces), matrix,
                                             thresholds=thresholds, count=COUNT, seed=SEED)
        stages[label] = {"source_sha256": expected, "geometry_export": export,
                         "reference_gauge_bounds": bounds(vertices @ matrix[:3, :3].T + matrix[:3, 3]),
                         "surface_comparison": comparison,
                         "foreground_projection": projection_support(vertices, run, mask_report)}
        normalized.unlink()
        stages[label]["geometry_export"]["temporary_output_removed"] = True
    if (score_fresh_complete.checked_run(run) != evidence or digest(reference) != score_fresh_complete.REFERENCE_SHA256 or
            digest(score_path) != score_hash or digest(parent_fit) != score_fresh_complete.PARENT_FIT_SHA256 or
            digest(mask_path) != mask_hash):
        raise ValueError("sealed inputs changed during evaluation")
    report = {"schema": "classical_fresh_stage_diagnostic_v1", "status": "complete",
              "producer": evidence, "source_sha256": {"score": score_hash, "reference": digest(reference),
               "parent_fit": digest(parent_fit), "mask_report": digest(mask_path)},
              "output_to_reference_matrix": matrix.tolist(), "camera_transport": transport,
              "reference_bounds": bounds(ref_vertices), "reference_bbox_diagonal": diagonal,
              "threshold_fractions": FRACTIONS, "thresholds": thresholds, "stages": stages,
              "interpretation": "Conditional shape comparison in frozen 006 reference-fitted gauge; Google mesh and YCB photos have cross-scanner frame uncertainty. Dense point proximity and mask support are not surface F1. Reference used only after reconstruction."}
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds 2 MiB cap")
    (output / "report.json").write_bytes(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "score", "parent-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = run_diagnostic(args.run, args.reference, args.score, args.parent_run, args.output)
    print(json.dumps({"status": result["status"], "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
