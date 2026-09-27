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
    residuals: list[float] = []
    invalid = 0
    observations: dict[str, int] = {name: 0 for name in by_name}
    for point_id, point in model.points3D.items():
        xyz = tuple(float(value) for value in point.xyz)
        if len(xyz) != 3 or not all(map(math.isfinite, xyz)):
            raise ValueError("nonfinite or invalid sparse point")
        elements = point.track.elements
        if len(elements) < 2:
            raise ValueError("sparse point has a track shorter than two")
        seen = set()
        for element in elements:
            image_id = int(element.image_id)
            image = registered.get(image_id)
            if image is None or image_id in seen:
                raise ValueError("track refers to unregistered or duplicate image")
            seen.add(image_id)
            index = int(element.point2D_idx)
            if not 0 <= index < len(image.points2D):
                raise ValueError("track 2D observation index outside image")
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
            "reprojection_l2_pixels": {"track_observation_denominator": total_observations,
                                       "finite_count": len(residuals), "invalid_count": invalid,
                                       "mean": statistics.fmean(residuals) if residuals else None,
                                       "median": statistics.median(residuals) if residuals else None,
                                       "p95": percentile(residuals, 0.95)}}


def _regular(path: Path, maximum: int) -> None:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"missing, linked, empty, or oversized evidence: {path}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.package, args.output)
    print(json.dumps({"status": report["status"], "registered": report["model"]["registered_count"],
                      "graph_components": report["matching_graph"]["component_count"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
