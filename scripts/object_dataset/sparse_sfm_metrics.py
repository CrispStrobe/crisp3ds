"""Read-only, hash-bound sparse-SfM diagnostics for the frozen mustard train split.

No reference, held-out image, depth, mask generation, or reconstruction is used.
PyCOLMAP is imported only by the CLI; pure metric contracts remain CI-testable.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SHA = "45b6ed1430627a747eaeb8d4dff7d2c49502083cca0a7914e25aacd45b594425"
PAIR_BASE = 2147483647
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
MAX_DB_BYTES = 2 * 1024**3
MAX_MODEL_FILE_BYTES = 256 * 1024**2
MAX_REPORT_BYTES = 1024**2
MAX_LOG_BYTES = 32 * 1024**2
MIN_FREE = 10 * 1024**3
NAME = re.compile(r"NP3_\d{3}\.jpg\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def graph_components(names: list[str], edges: list[tuple[str, str]]) -> dict:
    if not names or len(names) != len(set(names)):
        raise ValueError("graph needs distinct selected image names")
    parent = {name: name for name in names}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in edges:
        if left not in parent or right not in parent or left == right:
            raise ValueError("verified pair is outside selected images or self-linked")
        parent[root(left)] = root(right)
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(root(name), []).append(name)
    components = sorted((sorted(group) for group in groups.values()),
                        key=lambda group: (-len(group), group))
    return {"verified_pair_edges": len(set(tuple(sorted(edge)) for edge in edges)),
            "component_count": len(components), "component_sizes": [len(g) for g in components],
            "components": components}


def verified_graph(database: Path, names: list[str]) -> dict:
    """Nonempty COLMAP two_view_geometries edges, including isolated images."""
    if database.is_symlink() or not database.is_file() or database.stat().st_size > MAX_DB_BYTES:
        raise ValueError("missing, linked, or oversized COLMAP database")
    if any(database.with_name(database.name + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("database has mutable WAL/SHM sidecar; seal it first")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        image_rows = connection.execute("SELECT image_id,name FROM images").fetchall()
        ids = {int(image_id): name for image_id, name in image_rows}
        if len(ids) != len(names) or set(ids.values()) != set(names):
            raise ValueError("database image inventory differs from frozen training names")
        edges = []
        for pair_id, count in connection.execute("SELECT pair_id,rows FROM two_view_geometries"):
            if count is None or count < 0:
                raise ValueError("invalid verified geometry row count")
            if count:
                first, second = divmod(int(pair_id), PAIR_BASE)
                if first not in ids or second not in ids:
                    raise ValueError("verified pair references missing image")
                edges.append((ids[first], ids[second]))
    return graph_components(names, edges)


def summarize_model(model, names: list[str]) -> dict:
    """Duck-typed PyCOLMAP model, permitting analytic CI fixtures."""
    registered = {int(image_id): model.images[image_id] for image_id in model.reg_image_ids()}
    by_name = {image.name: image for image in registered.values()}
    if len(by_name) != len(registered) or not set(by_name).issubset(names):
        raise ValueError("model has duplicate or non-training registered image names")
    lengths: list[int] = []
    distinct_image_lengths: list[int] = []
    residuals: list[float] = []
    invalid = 0
    repeated_image_tracks = 0
    repeated_image_observations = 0
    observations: dict[str, int] = {name: 0 for name in by_name}
    for point_id, point in model.points3D.items():
        xyz = tuple(float(value) for value in point.xyz)
        if len(xyz) != 3 or not all(map(math.isfinite, xyz)):
            raise ValueError("nonfinite or invalid sparse point")
        elements = point.track.elements
        if len(elements) < 2:
            raise ValueError("sparse point has a track shorter than two")
        seen_images = set()
        seen_observations = set()
        for element in elements:
            image_id = int(element.image_id)
            image = registered.get(image_id)
            if image is None:
                raise ValueError("track refers to unregistered image")
            index = int(element.point2D_idx)
            if not 0 <= index < len(image.points2D):
                raise ValueError("track 2D observation index outside image")
            if (image_id, index) in seen_observations:
                raise ValueError("duplicate track observation")
            seen_observations.add((image_id, index))
            if image_id in seen_images:
                repeated_image_observations += 1
            seen_images.add(image_id)
            point2d = image.points2D[index]
            if int(point2d.point3D_id) != int(point_id):
                raise ValueError("track and image 2D-to-3D backlink disagree")
            observed = tuple(float(value) for value in point2d.xy)
            if len(observed) != 2 or not all(map(math.isfinite, observed)):
                raise ValueError("nonfinite measured 2D observation")
            observations[image.name] += 1
            projected = image.project_point(point.xyz)
            if projected is None:
                invalid += 1
                continue
            predicted = tuple(float(value) for value in projected)
            if len(predicted) != 2 or not all(map(math.isfinite, predicted)):
                invalid += 1
                continue
            residuals.append(math.hypot(predicted[0] - observed[0],
                                        predicted[1] - observed[1]))
        lengths.append(len(elements))
        distinct_image_lengths.append(len(seen_images))
        repeated_image_tracks += len(seen_images) != len(elements)
    total_observations = sum(lengths)
    if len(residuals) + invalid != total_observations:
        raise ValueError("reprojection denominator mismatch")
    return {"registered_names": sorted(by_name), "registered_count": len(by_name),
            "selected_count": len(names), "sparse_points": len(lengths),
            "triangulated_observations_by_image": dict(sorted(observations.items())),
            "track_length": {"point_denominator": len(lengths),
                             "median": statistics.median(lengths) if lengths else None,
                             "p95": percentile(lengths, 0.95),
                             "at_least_three_count": sum(length >= 3 for length in lengths),
                             "at_least_three_fraction": (sum(length >= 3 for length in lengths) / len(lengths)
                                                         if lengths else None)},
            "distinct_image_track_length": {
                "point_denominator": len(distinct_image_lengths),
                "median": statistics.median(distinct_image_lengths) if distinct_image_lengths else None,
                "p95": percentile(distinct_image_lengths, 0.95),
                "at_least_three_count": sum(length >= 3 for length in distinct_image_lengths),
                "at_least_three_fraction": (sum(length >= 3 for length in distinct_image_lengths) /
                                            len(distinct_image_lengths) if distinct_image_lengths else None)},
            "repeated_image_tracks": repeated_image_tracks,
            "repeated_image_observations": repeated_image_observations,
            "reprojection_l2_pixels": {"track_observation_denominator": total_observations,
                                       "finite_count": len(residuals), "invalid_count": invalid,
                                       "mean": statistics.fmean(residuals) if residuals else None,
                                       "median": statistics.median(residuals) if residuals else None,
                                       "p95": percentile(residuals, 0.95)}}


def _regular(path: Path, maximum: int) -> None:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"missing, linked, empty, or oversized evidence: {path}")


def _sealed_snapshot(source: Path, snapshot: Path) -> None:
    wal = source.with_name(source.name + "-wal")
    shm = source.with_name(source.name + "-shm")
    if (snapshot.resolve() == source.resolve() or
            any(snapshot.with_name(snapshot.name + suffix).exists()
                for suffix in ("-wal", "-shm")) or
            (wal.exists() and (wal.is_symlink() or not wal.is_file() or wal.stat().st_size != 0)) or
            (shm.exists() and (shm.is_symlink() or not shm.is_file())) or
            sha256(snapshot) != sha256(source)):
        raise ValueError("database snapshot is not a sidecar-free byte copy of sealed source")


def evaluate(run: Path, package_path: Path, output: Path) -> dict:
    """Inspect an existing sparse-only run; never modify its inputs."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if run.is_symlink() or not run.is_dir() or output.parent.is_symlink():
        raise ValueError("run and output parent must be real directories")
    _regular(package_path, MAX_REPORT_BYTES)
    if sha256(package_path) != PACKAGE_SHA:
        raise ValueError("mustard training package SHA differs from frozen validation")
    package = json.loads(package_path.read_text())
    selected = package.get("training_inputs")
    if (package.get("schema") != "ycb_object_evaluation_package_v1" or
            package.get("object_id") != "006_mustard_bottle" or not isinstance(selected, list) or
            len(selected) != 48):
        raise ValueError("not the frozen mustard 48-training package")
    names = [Path(item["path"]).name for item in selected]
    if (len(names) != len(set(names)) or any(not NAME.fullmatch(name) or
            item["path"] != f"photos/{name}" for name, item in zip(names, selected))):
        raise ValueError("training name inventory invalid")
    result_path, sfm_path, db = run / "result.json", run / "sfm.json", run / "database.db"
    model_dir = run / "sparse/0"
    inputs = [package_path, result_path, sfm_path, db] + [model_dir / name for name in MODEL_FILES]
    runner_path = Path(__file__)
    for path in inputs[1:]:
        _regular(path, MAX_DB_BYTES if path == db else
                 MAX_MODEL_FILE_BYTES if path.parent == model_dir else MAX_REPORT_BYTES)
    if model_dir.is_symlink() or (run / "sparse").is_symlink():
        raise ValueError("sparse model directory is linked")
    before = {str(path.resolve()): sha256(path) for path in inputs + [runner_path]}
    result = json.loads(result_path.read_text())
    actual = result.get("inputs", [])
    if (result.get("schema") != "classical_backend_v1" or
            result.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap" or
            [row.get("name") for row in actual] != names or
            any(row.get("sha256") != expected["sha256"] for row, expected in zip(actual, selected)) or
            len(actual) != 48):
        raise ValueError("producer input lineage differs from frozen 48 training photos")
    stages = {stage.get("name"): stage.get("status") for stage in result.get("stages", [])}
    if any(stages.get(name) != "complete" for name in ("features", "matching", "sfm")):
        raise ValueError("producer sparse stages are not complete")
    if result.get("status") not in ("sparse_complete", "failed") or (
            result.get("status") == "failed" and
            result.get("failure") != "registered image fraction below threshold"):
        raise ValueError("producer did not seal a supported sparse outcome")
    sfm = json.loads(sfm_path.read_text())
    if sorted(sfm.get("registered_names", [])) != sorted(set(sfm.get("registered_names", []))):
        raise ValueError("duplicate registered image in SfM summary")
    import pycolmap  # local research runtime only; pure unit tests need no dependency
    model = pycolmap.Reconstruction(str(model_dir))
    model_metrics = summarize_model(model, names)
    if (sfm.get("registered_images") != model_metrics["registered_count"] or
            sfm.get("sparse_points") != model_metrics["sparse_points"] or
            sorted(sfm.get("registered_names", [])) != model_metrics["registered_names"]):
        raise ValueError("SfM summary and saved binary reconstruction disagree")
    graph = verified_graph(db, names)
    after = {str(path.resolve()): sha256(path) for path in inputs + [runner_path]}
    if before != after:
        raise ValueError("evidence changed during sparse diagnostic")
    report = {"schema": "mustard_train_sparse_sfm_metrics_v1", "status": "complete",
              "scope": "48 training RGB only; image-estimated SfM, no held-out/GT/depth/reference",
              "producer_status": result["status"], "producer_failure": result.get("failure"),
              "source_sha256": before, "runner_sha256": before[str(runner_path.resolve())],
              "matching_graph": graph, "model": model_metrics,
              "candidate_models": sfm.get("candidate_models", []),
              "reprojection_definition": "L2 pixels over finite projections of each saved 3D track observation",
              "quality_claim": False}
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_REPORT_BYTES or shutil.disk_usage(output.parent).free < MIN_FREE:
        raise ValueError("diagnostic report cap or 10 GiB output reserve")
    with output.open("xb") as stream:
        stream.write(encoded)
    return report


def evaluate_failed_sfm(run: Path, package_path: Path, output: Path,
                        database_snapshot: Path, producer_source: Path) -> dict:
    """Diagnose two *failed* mapper outcomes without inventing a sparse success.

    A no-model failure has graph metrics only. A saved two-camera candidate is
    diagnostic geometry, never the producer's rejected sparse/0 export.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if run.is_symlink() or not run.is_dir() or output.parent.is_symlink():
        raise ValueError("run and output parent must be real directories")
    result_path = run / "result.json"
    database = run / "database.db"
    options_path = run / "pycolmap-options.json"
    logs = [run / f"{name}.log" for name in ("features", "matching", "sfm")]
    runner_path = Path(__file__)
    producer_path = producer_source
    for path, cap in ((package_path, MAX_REPORT_BYTES), (result_path, MAX_REPORT_BYTES),
                      (database, MAX_DB_BYTES), (database_snapshot, MAX_DB_BYTES),
                      (options_path, MAX_REPORT_BYTES),
                      *((path, MAX_LOG_BYTES) for path in logs),
                      (runner_path, MAX_REPORT_BYTES), (producer_path, MAX_REPORT_BYTES)):
        _regular(path, cap)
    if sha256(package_path) != PACKAGE_SHA:
        raise ValueError("mustard training package SHA differs from frozen validation")
    _sealed_snapshot(database, database_snapshot)
    package = json.loads(package_path.read_text())
    selected = package.get("training_inputs")
    if (package.get("schema") != "ycb_object_evaluation_package_v1" or
            package.get("object_id") != "006_mustard_bottle" or
            not isinstance(selected, list) or len(selected) != 48):
        raise ValueError("not the frozen mustard 48-training package")
    names = [Path(item["path"]).name for item in selected]
    if (len(names) != len(set(names)) or any(not NAME.fullmatch(name) or
            item["path"] != f"photos/{name}" for name, item in zip(names, selected))):
        raise ValueError("training name inventory invalid")
    result = json.loads(result_path.read_text())
    actual = result.get("inputs", [])
    stages = result.get("stages", [])
    if (result.get("schema") != "classical_backend_v1" or result.get("status") != "failed" or
            result.get("failure") != "sfm failed: exit 1" or
            result.get("sfm_source", {}).get("kind") != "internal_image_only_pycolmap" or
            [row.get("name") for row in actual] != names or len(actual) != 48 or
            any(row.get("sha256") != expected["sha256"] for row, expected in zip(actual, selected)) or
            [(stage.get("name"), stage.get("status"), stage.get("exit_code")) for stage in stages] !=
            [("features", "complete", 0), ("matching", "complete", 0), ("sfm", "failed", 1)] or
            any(Path(stage.get("log", "")) != log for stage, log in zip(stages, logs)) or
            any(result.get(key) for key in ("changed_source_images", "changed_pose_masks", "changed_binaries")) or
            result.get("software", {}).get("runner_sha256") != sha256(producer_path)):
        raise ValueError("producer is not the sealed failed mustard SfM outcome")
    options = json.loads(options_path.read_text())
    source = result["sfm_source"]
    if (options.get("version") != "3.11.1" or options.get("seed") != source.get("random_seed") or
            options.get("camera_model") != source.get("camera_model") or
            options.get("camera_mode") != source.get("camera_mode") or
            options.get("matching_strategy") != source.get("matching")):
        raise ValueError("effective mapper options disagree with producer profile")
    if ((run / "sparse").is_symlink() or (run / "sparse/0").exists() or
            (run / "sparse/0").is_symlink()):
        raise ValueError("failed mapper unexpectedly has accepted sparse/0 export")
    log_tail = logs[-1].read_bytes()[-4096:].decode(errors="replace").rstrip()
    model_root = run / "models"
    if model_root.is_symlink() or not model_root.is_dir():
        raise ValueError("failed mapper model directory missing or linked")
    sfm_path = run / "sfm.json"
    if not sfm_path.exists() and not sfm_path.is_symlink():
        if not log_tail.endswith("ValueError: SfM produced no model") or list(model_root.iterdir()):
            raise ValueError("no-model failure marker or empty candidate directory differs")
        failure_kind = "failed_no_model"
        model_dir = None
        sfm = None
    else:
        _regular(sfm_path, MAX_REPORT_BYTES)
        if not log_tail.endswith("ValueError: SfM model has fewer than 3 cameras or no points"):
            raise ValueError("degenerate-model failure marker differs")
        sfm = json.loads(sfm_path.read_text())
        index = sfm.get("selected_model_index")
        if (type(index) is not int or index < 0 or
                [item.get("index") for item in sfm.get("candidate_models", [])] != [index] or
                sfm.get("registered_images") != 2 or sfm.get("sparse_points", 0) < 1 or
                sorted(sfm.get("registered_names", [])) != sorted(set(sfm.get("registered_names", []))) or
                {item.name for item in model_root.iterdir()} != {str(index)}):
            raise ValueError("saved candidate does not match the supported two-camera failure")
        model_dir = model_root / str(index)
        if (model_dir.is_symlink() or not model_dir.is_dir() or
                {item.name for item in model_dir.iterdir()} != set(MODEL_FILES)):
            raise ValueError("candidate model files missing, extra, or linked")
        for name in MODEL_FILES:
            _regular(model_dir / name, MAX_MODEL_FILE_BYTES)
        failure_kind = "failed_degenerate_model"
    evidence = [package_path, result_path, database, database_snapshot, options_path,
                *logs, runner_path, producer_path]
    if sfm is not None:
        evidence += [sfm_path] + [model_dir / name for name in MODEL_FILES]
    before = {str(path.resolve()): sha256(path) for path in evidence}
    graph = verified_graph(database_snapshot, names)
    model_metrics = None
    if model_dir is not None:
        import pycolmap  # only for a saved, explicitly rejected candidate
        model_metrics = summarize_model(pycolmap.Reconstruction(str(model_dir)), names)
        if (model_metrics["registered_count"] != sfm["registered_images"] or
                model_metrics["sparse_points"] != sfm["sparse_points"] or
                model_metrics["registered_names"] != sorted(sfm["registered_names"])):
            raise ValueError("rejected candidate binary and SfM summary disagree")
    after = {str(path.resolve()): sha256(path) for path in evidence}
    _sealed_snapshot(database, database_snapshot)
    if before != after:
        raise ValueError("failed SfM evidence changed during diagnostic")
    report = {"schema": "mustard_train_sparse_failure_metrics_v1", "status": failure_kind,
              "scope": "48 training RGB only; image-estimated SfM failure, no held-out/GT/depth/reference",
              "producer_status": "failed", "producer_failure": result["failure"],
              "database_source": str(database.resolve()),
              "database_snapshot": str(database_snapshot.resolve()),
              "producer_source": str(producer_path.resolve()),
              "source_sha256": before, "runner_sha256": before[str(runner_path.resolve())],
              "matching_graph": graph, "model": model_metrics,
              "model_role": "unavailable_no_model" if model_metrics is None else "rejected_candidate_model",
              "accepted_sparse_export": False, "quality_claim": False,
              "geometry_metrics_interpretation": "unavailable" if model_metrics is None else
              "diagnostic only: producer rejected fewer than three cameras"}
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_REPORT_BYTES or shutil.disk_usage(output.parent).free < MIN_FREE:
        raise ValueError("failed diagnostic report cap or 10 GiB output reserve")
    with output.open("xb") as stream:
        stream.write(encoded)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-sfm", action="store_true",
                        help="diagnose sealed no-model or rejected two-camera failure")
    parser.add_argument("--database-snapshot", type=Path,
                        help="sidecar-free byte-identical snapshot of the producer database")
    parser.add_argument("--producer-source", type=Path,
                        help="archived producer source matching result.software.runner_sha256")
    args = parser.parse_args()
    if args.failed_sfm and (args.database_snapshot is None or args.producer_source is None):
        parser.error("--failed-sfm requires --database-snapshot and --producer-source")
    report = (evaluate_failed_sfm(args.run, args.package, args.output,
                                  args.database_snapshot, args.producer_source) if args.failed_sfm else
              evaluate(args.run, args.package, args.output))
    print(json.dumps({"status": report["status"], "registered":
                      report["model"]["registered_count"] if report["model"] is not None else None,
                      "graph_components": report["matching_graph"]["component_count"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
