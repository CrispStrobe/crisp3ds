"""Evaluation-only frozen scanner score for the sealed cracker visual hull.

Never feeds reference geometry or a fitted transform back to the producer.
Baseline replay is mandatory before scoring the candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

from scripts.classical_backend import geometry
from scripts.classical_backend.run import stage
from scripts.object_dataset import align, evaluate, surface_metrics
from scripts.classical_backend.cracker_visual_hull import ROOT, DATA, digest


HULL = DATA / "cracker-visual-hull-001"
PRODUCER = DATA / "turntable-fresh-openmvs-005"
OLD_SCORE = DATA / "turntable-fresh-openmvs-score-005" / "score.json"
OLD_STAGE = DATA / "turntable-fresh-stage-diagnostic-005" / "report.json"
REFERENCE = ROOT / ".local-tools/test-data/ycb-cracker-box/reference/google_64k_geometry_f64.ply"
OUTPUT = DATA / "cracker-visual-hull-score-001"
HASHES = {
    "hull_result": "6bbef03514e8ac85139307afff59eaa767f58ba84a04f8c209e344337c0e06a8",
    "hull_report": "b0e88ce667f34e6e569aace1b21987f0fc3291813666887816e3c1556b05f2af",
    "hull_mesh": "050f230c49f17d5e54fb5b15d8a2ddf4878e250d9662a9ad94fdd9eabdbc2de4",
    "old_score": "165b8d5169e8bdb917f008864b01c434f4e03455b6ffe688d8ae8a824b2847c9",
    "old_stage": "5f6b1a6bf769516d2b0ff211cc4eb6204828b364fe5d98b7279032b41b6e48e2",
    "rough_native": "7eab999003f08892f0541269588635e3311b4f91b08ebcd3d639df57e20efa9e",
    "refined_native": "56bade52185e3ef7ac662e46b7aa931064b0503442bcea185199986d6c658c0a",
    "rough_normalized": "f21fe1a64fbedfc3a53c09f1363bf2a72c4b9bffc6f524fd7edb8cfc65c8efa7",
    "refined_normalized": "86195546639f79be4c3158e9d9a259805320a69f49948bbb2533d276dea10bd5",
    "reference": "6e0187aa961aef4fa21dfc753023a4be7e82924e6398dc0875a2b0d609363ed4",
}
FRACTIONS = (0.005, 0.01, 0.02)
COUNT = 2048
SEED = 2027
MAX_OUTPUT = 20 * 1024**2
MAX_LOG = 2 * 1024**2
MAX_RSS = 2 * 1024**3
MIN_FREE = 10 * 1024**3
WORKER_SECONDS = 270
SUPERVISOR_SECONDS = 300


def sealed_json(path: Path, expected: str) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2 or digest(path) != expected:
        raise ValueError(f"sealed input differs: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("sealed JSON is not an object")
    return value


def verify_inputs() -> tuple[dict, dict, dict]:
    hull_result = sealed_json(HULL / "result.json", HASHES["hull_result"])
    hull_report = sealed_json(HULL / "hull-report.json", HASHES["hull_report"])
    old_score = sealed_json(OLD_SCORE, HASHES["old_score"])
    old_stage = sealed_json(OLD_STAGE, HASHES["old_stage"])
    for path, key in ((HULL / "hull.ply", "hull_mesh"), (REFERENCE, "reference"),
                      (PRODUCER / "mesh.ply", "rough_native"),
                      (PRODUCER / "refined.ply", "refined_native")):
        if path.is_symlink() or not path.is_file() or digest(path) != HASHES[key]:
            raise ValueError(f"sealed geometry differs: {key}")
    if (hull_result.get("status") != "complete" or hull_report.get("status") != "complete" or
            hull_result.get("hull_report_sha256") != HASHES["hull_report"] or
            hull_report.get("neutral_mesh", {}).get("sha256") != HASHES["hull_mesh"] or
            hull_report.get("boundary_touched") is not False or
            hull_report.get("occupied_voxels") != 55815 or
            old_score.get("shared_gauge", {}).get("status") != "available" or
            old_score.get("threshold_fractions_of_reference_diagonal") != list(FRACTIONS) or
            old_score.get("reference_sha256") != HASHES["reference"] or
            old_stage.get("source_sha256", {}).get("score") != HASHES["old_score"]):
        raise ValueError("sealed hull or baseline protocol differs")
    matrix = np.asarray(old_score["shared_gauge"]["matrix_used_output_to_reference"], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=0):
        raise ValueError("frozen shared-gauge matrix invalid")
    return hull_report, old_score, old_stage


def disk_floor(*, allowance: int = 0) -> dict[str, int]:
    free = {"internal": shutil.disk_usage(ROOT).free, "external": shutil.disk_usage(DATA).free}
    if free["internal"] < MIN_FREE or free["external"] < MIN_FREE + allowance:
        raise ValueError("10 GiB disk floor or output allowance breached")
    return free


def thresholds_from_reference(vertices: np.ndarray, old_score: dict, old_stage: dict) -> list[float]:
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    thresholds = [diagonal * fraction for fraction in FRACTIONS]
    recorded = old_stage.get("thresholds")
    if (not np.isfinite(diagonal) or diagonal <= 0 or not isinstance(recorded, list) or
            len(recorded) != 3 or not np.allclose(thresholds, recorded, rtol=0, atol=1e-15) or
            not np.isclose(diagonal, old_score["shared_gauge"]["comparison"]["reference_bbox_diagonal"],
                           rtol=0, atol=1e-12)):
        raise ValueError("reference diagonal or frozen thresholds differ")
    return thresholds


def assert_baseline_replay(actual: dict, expected: dict, name: str) -> None:
    """Abstain before candidate scoring if the old fixed-gauge F is not replayed."""
    rows = actual.get("threshold_scores", [])
    frozen = expected.get("threshold_scores", [])
    if len(rows) != 3 or len(frozen) != 3:
        raise ValueError(f"{name} baseline lacks three thresholds")
    for row, prior in zip(rows, frozen):
        for key in ("threshold", "precision", "recall", "f_score"):
            if not np.isclose(row[key], prior[key], rtol=0, atol=1e-12):
                raise ValueError(f"{name} fixed-gauge baseline failed deterministic replay: {key}")


def reconstruct_occupancy_from_exposed_faces(vertices: np.ndarray, faces: np.ndarray,
                                             center: np.ndarray, side: float, grid: int,
                                             expected_count: int) -> np.ndarray:
    """Recover exact voxels by parity along X through the producer's lattice faces."""
    pitch = side / grid
    lattice_float = (np.asarray(vertices, float) - (center - side / 2)) / pitch
    lattice = np.rint(lattice_float).astype(np.int32)
    if (not np.isfinite(lattice_float).all() or not np.allclose(lattice_float, lattice, rtol=0, atol=1e-6) or
            np.any(lattice < 0) or np.any(lattice > grid)):
        raise ValueError("hull vertices are not on sealed voxel lattice")
    xfaces: dict[tuple[int, int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int32):
        triangle = lattice[face]
        if np.all(triangle[:, 0] == triangle[0, 0]):
            lo = triangle.min(axis=0)
            hi = triangle.max(axis=0)
            if hi[1] - lo[1] != 1 or hi[2] - lo[2] != 1 or not 0 <= lo[1] < grid or not 0 <= lo[2] < grid:
                raise ValueError("hull X-face is not a unit voxel face")
            key = (int(lo[0]), int(lo[1]), int(lo[2]))
            xfaces[key] = xfaces.get(key, 0) + 1
    if not xfaces or any(count != 2 for count in xfaces.values()):
        raise ValueError("hull X-faces are missing or duplicated")
    rows: dict[tuple[int, int], list[int]] = {}
    for i, j, k in xfaces:
        rows.setdefault((j, k), []).append(i)
    occupied = np.zeros((grid, grid, grid), dtype=bool)
    for (j, k), boundaries in rows.items():
        boundaries.sort()
        if len(boundaries) % 2 or boundaries[0] < 0 or boundaries[-1] > grid:
            raise ValueError("hull X-ray parity is invalid")
        for left, right in zip(boundaries[::2], boundaries[1::2]):
            if left >= right:
                raise ValueError("hull X-ray interval is empty")
            occupied[left:right, j, k] = True
    if int(occupied.sum()) != expected_count:
        raise ValueError("reconstructed occupancy disagrees with sealed count")
    return occupied


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    triangles = vertices[faces] - vertices.mean(axis=0)
    return float(abs(np.einsum("ij,ij->i", triangles[:, 0],
                               np.cross(triangles[:, 1], triangles[:, 2])).sum()) / 6)


def summarize(comparison: dict) -> dict:
    return {"comparison": comparison,
            "threshold_interpretation": [{"threshold": row["threshold"],
                                          "extra_surface_fraction": 1 - row["precision"],
                                          "unsupported_reference_surface_fraction": 1 - row["recall"]}
                                         for row in comparison["threshold_scores"]]}


def _timeout(_signum, _frame):
    raise TimeoutError("hull score worker exceeded 270 seconds")


def worker(output: Path) -> None:
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(WORKER_SECONDS)
    try:
        disk_floor()
        hull_report, old_score, old_stage = verify_inputs()
        ref_info, ref_vertices, ref_faces = evaluate.inspect_ply(REFERENCE, geometry=True)
        ref_vertices, ref_faces = np.asarray(ref_vertices), np.asarray(ref_faces)
        thresholds = thresholds_from_reference(ref_vertices, old_score, old_stage)
        matrix = np.asarray(old_score["shared_gauge"]["matrix_used_output_to_reference"])
        normalized = {}
        baselines = {}
        for label, name in (("rough", "mesh.ply"), ("refined", "refined.ply")):
            destination = output / f"{label}-geometry.ply"
            normalized[label] = geometry.export_geometry(PRODUCER / name, destination)
            if normalized[label]["output_sha256"] != HASHES[f"{label}_normalized"]:
                raise ValueError(f"{label} normalized baseline bytes differ")
            _, vertices, faces = evaluate.inspect_ply(destination, geometry=True)
            baselines[label] = surface_metrics.compare((ref_vertices, ref_faces), (vertices, faces), matrix,
                                                       thresholds=thresholds, count=COUNT, seed=SEED)
            expected = (old_stage["stages"]["rough"]["surface_comparison"] if label == "rough"
                        else old_score["shared_gauge"]["comparison"])
            assert_baseline_replay(baselines[label], expected, label)
        hull_info, hull_vertices, hull_faces = evaluate.inspect_ply(HULL / "hull.ply", geometry=True)
        hull_vertices, hull_faces = np.asarray(hull_vertices), np.asarray(hull_faces)
        fixed = surface_metrics.compare((ref_vertices, ref_faces), (hull_vertices, hull_faces), matrix,
                                        thresholds=thresholds, count=COUNT, seed=SEED)
        self_control = surface_metrics.compare((ref_vertices, ref_faces), (ref_vertices, ref_faces),
                                               np.eye(4), thresholds=thresholds, count=COUNT, seed=SEED)
        assert_baseline_replay(self_control, old_score["reference_self_control"], "reference self-control")
        fit_matrix, fit_diagnostics = align.align_points(
            evaluate.sample_surface(ref_vertices, ref_faces, 1024, align.SEED),
            evaluate.sample_surface(hull_vertices, hull_faces, 1024, align.SEED),
            deadline=time.monotonic() + align.MAX_SECONDS)
        fitted = surface_metrics.compare((ref_vertices, ref_faces), (hull_vertices, hull_faces), fit_matrix,
                                         thresholds=thresholds, count=COUNT, seed=SEED)
        grid = hull_report["grid"]
        occupied = reconstruct_occupancy_from_exposed_faces(hull_vertices, hull_faces,
                        np.asarray(grid["center"]), grid["side"], grid["side_cells"],
                        hull_report["occupied_voxels"])
        indices = np.argwhere(occupied)
        centers = np.asarray(grid["center"]) - grid["side"] / 2 + \
                  (indices + 0.5) * (grid["side"] / grid["side_cells"])
        centers_ref = centers @ matrix[:3, :3].T + matrix[:3, 3]
        ref_min, ref_max = ref_vertices.min(axis=0), ref_vertices.max(axis=0)
        outside = int(np.count_nonzero(np.any((centers_ref < ref_min) | (centers_ref > ref_max), axis=1)))
        fixed_vertices = hull_vertices @ matrix[:3, :3].T + matrix[:3, 3]
        ref_topology = fixed["topology"]["reference"]
        hull_topology = fixed["topology"]["output"]
        watertight = all(topology["boundary_edges"] == topology["nonmanifold_edges"] == 0
                         for topology in (ref_topology, hull_topology))
        ref_volume = signed_mesh_volume(ref_vertices, ref_faces) if watertight else None
        hull_volume = signed_mesh_volume(fixed_vertices, hull_faces) if watertight else None
        report = {"schema": "cracker_visual_hull_scanner_score_v1", "status": "complete",
                  "candidate_sha256": HASHES["hull_mesh"], "candidate_report_sha256": HASHES["hull_report"],
                  "reference_sha256": HASHES["reference"], "old_score_sha256": HASHES["old_score"],
                  "old_stage_diagnostic_sha256": HASHES["old_stage"],
                  "runner_sha256": digest(Path(__file__)),
                  "protocol": {"threshold_fractions_of_reference_diagonal": FRACTIONS,
                               "absolute_thresholds": thresholds, "samples_per_direction": COUNT,
                               "reference_seed": SEED, "output_seed": SEED + 1,
                               "alignment_samples_per_mesh": 1024, "alignment_seed": align.SEED,
                               "shared_gauge_matrix_unchanged": matrix.tolist()},
                  "baseline_replay": {label: summarize(value) for label, value in baselines.items()},
                  "reference_self_control": summarize(self_control),
                  "hull_fixed_shared_gauge": summarize(fixed),
                  "hull_reference_fitted_secondary": {**summarize(fitted), "matrix": fit_matrix.tolist(),
                                                      "diagnostics": fit_diagnostics},
                  "reference": ref_info, "hull": hull_info, "normalized_baselines": normalized,
                  "fixed_gauge_volume": {"watertight_required": watertight,
                                         "reference": ref_volume, "hull": hull_volume,
                                         "hull_over_reference": hull_volume / ref_volume if watertight and ref_volume > 0 else None},
                  "outside_reference_aabb_voxel_proxy": {"outside_centers": outside,
                                                           "occupied_centers": len(indices),
                                                           "fraction": outside / len(indices),
                                                           "meaning": "fraction of fixed-gauge occupied voxel centers outside scanner AABB; lower-bound extra-volume proxy, not IoU"},
                  "interpretation": "evaluation-only conditional shape comparison; shared gauge includes 006 reference fit, so no physical pose/scale claim"}
        verify_inputs()
        disk_floor()
        payload = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
        if len(payload) > 2 * 1024**2:
            raise ValueError("score receipt exceeds 2 MiB")
        with (output / "score.json").open("xb") as stream:
            stream.write(payload)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output
    if output != OUTPUT:
        raise ValueError("only frozen scanner-score output path is allowed")
    if args.worker:
        worker(output)
        return
    if output.exists() or output.is_symlink() or output.parent != DATA or DATA.is_symlink() or not DATA.is_dir():
        raise ValueError("scanner-score output must be fresh on external volume")
    disk_floor(allowance=MAX_OUTPUT)
    verify_inputs()
    disk_floor(allowance=MAX_OUTPUT)
    output.mkdir()
    started = time.monotonic()
    status = {"schema": "cracker_visual_hull_score_supervisor_v1", "status": "running",
              "runner_sha256": digest(Path(__file__)), "candidate_sha256": HASHES["hull_mesh"]}
    try:
        executed = stage(output, "score", [sys.executable, "-m", "scripts.classical_backend.score_cracker_visual_hull",
                                            "--worker", "--output", str(output)],
                         started + SUPERVISOR_SECONDS, MAX_OUTPUT, MAX_LOG, MAX_RSS,
                         extra_reserve_paths=(ROOT,))
        path = output / "score.json"
        scored = json.loads(path.read_text())
        if scored.get("status") != "complete" or scored.get("runner_sha256") != status["runner_sha256"]:
            raise ValueError("worker did not seal exact score receipt")
        status.update(status="complete", stage=executed, score_sha256=digest(path))
    except Exception as error:
        status.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["seconds"] = round(time.monotonic() - started, 3)
        with (output / "result.json").open("x") as stream:
            json.dump(status, stream, sort_keys=True, indent=2)
            stream.write("\n")
        disk_floor()


if __name__ == "__main__":
    main()
