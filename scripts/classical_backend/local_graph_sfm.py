"""One bounded image-only mapper replay with cyclic local verified matches.

Only the copied COLMAP database's matches and two_view_geometries tables are
restricted. Features, cameras, keypoints, descriptors, RGB, mapper options,
seed and automatic initialization are identical to the sealed exhaustive arm.
No angle, pose, calibration, held-out or reference-geometry prior is injected.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

from scripts.classical_backend import sparse_masked_dense as seal
from scripts.classical_backend.run import RESERVE, StageError, digest, folder_bytes, stage


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RESULT_SHA = "5a83c3e05e2b66887411799ecc8c5f3f41e3b8af3bbd775d308055ef7ee0041e"
SOURCE_DB_SHA = "5858be16dfa3f0fc50303d0eb4a001e21ee14036f67755a9997d5dac374b3075"
STAGE_REPORT_SHA = "bbcba624f6a51f82e0ebdebf6e7e4cda10804bc8df34cfd0d5b3242983a225d0"
SEED = 20260927
WINDOW = 4
MAX_OUTPUT_BYTES = 512 << 20
MAX_RSS_BYTES = 4 << 30
MAX_LOG_BYTES = 32 << 20
TIMEOUT_SECONDS = 600
PAIR_SCALE = 2147483647
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
PROTECTED_TABLES = ("cameras", "images", "keypoints", "descriptors")
NAME_RE = re.compile(r"NP3_(\d{3})\.jpg\Z")


def immutable_connection(path: Path) -> sqlite3.Connection:
    if (path.is_symlink() or not path.is_file() or
            any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))):
        raise ValueError("sealed source database is missing, linked or has live sidecars")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def sorted_images(connection: sqlite3.Connection, expected_names: list[str]) -> list[tuple[int, str]]:
    images = list(connection.execute("SELECT image_id,name FROM images"))
    if (len(images) != 48 or len(expected_names) != 48 or
            {name for _, name in images} != set(expected_names) or
            len(set(expected_names)) != 48):
        raise ValueError("database must contain exactly sealed 48 TRAIN names")
    for _, name in images:
        if NAME_RE.fullmatch(name) is None:
            raise ValueError("database contains a non-TRAIN acquisition name")
    return sorted(images, key=lambda row: int(NAME_RE.fullmatch(row[1]).group(1)))


def pair_id(first: int, second: int) -> int:
    if not 0 < first < PAIR_SCALE or not 0 < second < PAIR_SCALE or first == second:
        raise ValueError("invalid COLMAP image pair")
    return min(first, second) * PAIR_SCALE + max(first, second)


def allowed_pairs(images: list[tuple[int, str]], window: int = WINDOW) -> set[int]:
    n = len(images)
    if not 0 < window < n // 2 or len({row[0] for row in images}) != n:
        raise ValueError("invalid cyclic acquisition graph")
    return {pair_id(images[i][0], images[j][0])
            for i in range(n) for j in range(i+1, n)
            if min(j-i, n-(j-i)) <= window}


def graph_audit(connection: sqlite3.Connection, images: list[tuple[int, str]],
                allowed: set[int]) -> dict:
    matches = dict(connection.execute("SELECT pair_id,rows FROM matches"))
    verified = dict(connection.execute("SELECT pair_id,rows FROM two_view_geometries"))
    if not allowed <= matches.keys() or not allowed <= verified.keys():
        raise ValueError("local graph lacks a required match or verified geometry row")
    id_to_name = dict(images)
    graph = {image_id: set() for image_id, _ in images}
    for i, (first, _) in enumerate(images):
        for j in range(i+1, len(images)):
            second = images[j][0]
            edge = pair_id(first, second)
            if edge in allowed and verified[edge] > 0:
                graph[first].add(second)
                graph[second].add(first)
    seen = set()
    components = []
    for first in graph:
        if first in seen:
            continue
        stack = [first]
        seen.add(first)
        count = 0
        while stack:
            vertex = stack.pop()
            count += 1
            for neighbor in graph[vertex] - seen:
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(count)
    degrees = [len(graph[image_id]) for image_id, _ in images]
    return {"images": len(images), "window_selected_positions": WINDOW,
            "allowed_pairs": len(allowed), "match_rows": len(matches),
            "verified_rows": len(verified),
            "positive_allowed_match_rows": sum(matches[edge] > 0 for edge in allowed),
            "positive_allowed_verified_rows": sum(verified[edge] > 0 for edge in allowed),
            "positive_verified_components": components,
            "positive_verified_min_degree": min(degrees),
            "positive_verified_median_degree": sorted(degrees)[len(degrees)//2],
            "positive_verified_max_degree": max(degrees),
            "initial_pair_3_6_retained": pair_id(3, 6) in allowed,
            "initial_pair_names": [id_to_name[3], id_to_name[6]]}


def table_rows(connection: sqlite3.Connection, table: str) -> list[tuple]:
    if table not in (*PROTECTED_TABLES, "matches", "two_view_geometries"):
        raise ValueError("unrecognized protected COLMAP table")
    return list(connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))


def copy_filter_database(source_path: Path, destination: Path,
                         expected_names: list[str]) -> dict:
    """SQLite backup from immutable source, then delete only nonlocal pair rows."""
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("graph database destination must be fresh")
    source_sha = digest(source_path)
    if source_sha != SOURCE_DB_SHA:
        raise ValueError("source database SHA differs from frozen exhaustive run")
    with closing(immutable_connection(source_path)) as source:
        images = sorted_images(source, expected_names)
        allowed = allowed_pairs(images)
        before = graph_audit(source, images, allowed)
        if (len(allowed) != 192 or before["match_rows"] != 1128 or
                before["verified_rows"] != 1128 or
                before["positive_allowed_match_rows"] != 174 or
                before["positive_allowed_verified_rows"] != 173 or
                before["positive_verified_components"] != [48] or
                before["positive_verified_min_degree"] < 4 or
                before["initial_pair_names"] != ["NP3_012.jpg", "NP3_036.jpg"]):
            raise ValueError("sealed exhaustive graph differs from reviewed 48-view case")
        with closing(sqlite3.connect(destination)) as copy:
            source.backup(copy)
            copy.commit()
            for table in ("matches", "two_view_geometries"):
                placeholders = ",".join("?" for _ in allowed)
                copy.execute(f"DELETE FROM {table} WHERE pair_id NOT IN ({placeholders})",
                             tuple(sorted(allowed)))
            copy.commit()
            if any(table_rows(source, table) != table_rows(copy, table)
                   for table in PROTECTED_TABLES):
                raise ValueError("graph copy modified camera, image or feature rows")
            for table in ("matches", "two_view_geometries"):
                original = {row[0]: row for row in table_rows(source, table) if row[0] in allowed}
                copied = {row[0]: row for row in table_rows(copy, table)}
                if copied != original:
                    raise ValueError(f"graph copy modified surviving {table} rows")
            if copy.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("filtered SQLite database failed integrity_check")
            after = graph_audit(copy, images, allowed)
    if digest(source_path) != source_sha:
        raise ValueError("immutable source database bytes changed during backup/filter")
    if (after["match_rows"] != 192 or after["verified_rows"] != 192 or
            after["positive_allowed_verified_rows"] != 173 or
            after["positive_verified_components"] != [48]):
        raise ValueError("filtered local graph differs from predeclared connectivity")
    if any(Path(str(destination) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("filtered SQLite backup has an unsealed sidecar")
    return {"source_database_sha256": source_sha,
            "filtered_database_sha256": digest(destination),
            "selection_basis": "sorted accepted TRAIN acquisition positions; symmetric cyclic distance <=4",
            "before": before, "after": after,
            "removed_match_rows": before["match_rows"] - after["match_rows"],
            "removed_verified_rows": before["verified_rows"] - after["verified_rows"],
            "protected_table_rows_identical": True,
            "surviving_match_and_geometry_rows_identical": True}


def mapper_options():
    import pycolmap
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = True
    options.max_num_models = 5
    options.min_model_size = 10
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    return options


def json_options(options) -> dict:
    return json.loads(json.dumps(options.todict(), default=str))


def validate_source(baseline_run: Path, stage_root: Path) -> dict:
    baseline_result = baseline_run / "result.json"
    database = baseline_run / "database.db"
    stage_report = stage_root / "stage-report.json"
    if (any(path.is_symlink() or not path.is_file() for path in
            (baseline_result, database, stage_report)) or
            digest(baseline_result) != SOURCE_RESULT_SHA or
            digest(database) != SOURCE_DB_SHA or
            digest(stage_report) != STAGE_REPORT_SHA):
        raise ValueError("baseline result, database or accepted TRAIN stage seal differs")
    baseline = json.loads(baseline_result.read_text())
    stage_data = json.loads(stage_report.read_text())
    source = baseline.get("sfm_source", {})
    if (baseline.get("status") != "sparse_complete" or
            source.get("kind") != "internal_image_only_pycolmap" or
            source.get("matching") != "exhaustive" or
            source.get("intrinsics_policy") != "fixed-initial" or
            source.get("random_seed") != SEED or
            stage_data.get("schema") != "mustard_train_only_stage_v1" or
            stage_data.get("status") != "complete"):
        raise ValueError("baseline differs from frozen image-only exhaustive/fixed arm")
    import pycolmap
    if (pycolmap.__version__ != "3.11.1" or
            source.get("effective_options", {}).get("binary_sha256") !=
            digest(Path(pycolmap._core.__file__)) or
            source["effective_options"].get("incremental_pipeline") !=
            json_options(mapper_options())):
        raise ValueError("mapper version/binary or effective options differ from baseline")
    names = stage_data.get("train_names")
    if (not isinstance(names, list) or len(names) != 48 or
            {row.get("name") for row in baseline.get("inputs", [])} != set(names) or
            {row["name"]: row["sha256"] for row in baseline["inputs"]} !=
            stage_data.get("train_photo_sha256")):
        raise ValueError("baseline and accepted stage differ in TRAIN pixels")
    photos = stage_root / "images"
    if photos.is_symlink() or not photos.is_dir() or {p.name for p in photos.iterdir()} != set(names):
        raise ValueError("stage photo folder differs from sealed TRAIN inventory")
    for name in names:
        photo = photos / name
        if photo.is_symlink() or digest(photo) != stage_data["train_photo_sha256"][name]:
            raise ValueError(f"sealed TRAIN photo differs: {name}")
    with closing(immutable_connection(database)) as connection:
        images = sorted_images(connection, names)
        audit = graph_audit(connection, images, allowed_pairs(images))
    if (audit["positive_verified_components"] != [48] or
            audit["initial_pair_names"] != ["NP3_012.jpg", "NP3_036.jpg"]):
        raise ValueError("source verified graph is disconnected or initial pair differs")
    return {"baseline": baseline, "stage": stage_data, "names": names,
            "database": database, "photos": photos,
            "source_model_hashes": {name: digest(baseline_run / "sparse/0" / name)
                                    for name in MODEL_FILES},
            "initial_graph": audit}


def worker(output: Path) -> None:
    import pycolmap
    saved_options = json.loads((output / "mapper-options.json").read_text())
    options = mapper_options()
    if json_options(options) != saved_options:
        raise ValueError("worker mapper options differ from frozen parent preflight")
    pycolmap.set_random_seed(SEED)
    pycolmap.incremental_mapping(str(output / "database.db"), str(output / "images"),
                                  str(output / "models"), options=options)


def reconstruction_summary(disk_models: list[dict]) -> dict:
    selected = max(disk_models,
                   key=lambda item: (item["registered"], item["points3D"], -item["index"]),
                   default=None)
    registered = selected["registered"] if selected else 0
    gate_met = registered >= 34  # ceil(0.70 * 48), registration only.
    return {"largest_registered": registered, "selected_model": selected,
            "registration_gate": {"required_of_48": 34, "met": gate_met},
            "reconstruction_status": (
                "no_model" if selected is None else
                "insufficient_registration" if not gate_met else
                "registration_gate_only_not_camera_or_shape_acceptance")}


def run(args: argparse.Namespace) -> dict:
    from scripts.object_motion import sam_m1_parity as mac
    import pycolmap
    started = time.monotonic()
    mac.host_preflight(args.output, check_memory=False)
    if (args.output.exists() or args.output.is_symlink() or args.output.parent.is_symlink() or
            not args.output.parent.is_dir()):
        raise ValueError("local-graph mapper output must be fresh under real parent")
    bound = validate_source(args.baseline_run, args.stage_root)
    output = args.output.resolve()
    photo_bytes = sum((bound["photos"] / name).stat().st_size for name in bound["names"])
    if (photo_bytes + bound["database"].stat().st_size >= MAX_OUTPUT_BYTES or
            shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT_BYTES + (256 << 20) or
            shutil.disk_usage(ROOT).free < RESERVE + (256 << 20)):
        raise ValueError("512 MiB cap or dual-disk reserve unavailable")
    output.mkdir()
    report = {"schema": "mustard_sfm_local_verified_graph_v1", "status": "running",
              "scope": "image-only automatic-mapper local graph intervention; no dense or quality acceptance",
              "matching_graph_strategy": "verified_exhaustive_graph_filtered_cyclic_selected_local4",
              "baseline_result_sha256": SOURCE_RESULT_SHA,
              "source_database_sha256": SOURCE_DB_SHA,
              "stage_report_sha256": STAGE_REPORT_SHA,
              "source_model_sha256": bound["source_model_hashes"],
              "photo_sha256": bound["stage"]["train_photo_sha256"],
              "software": {"runner_sha256": digest(Path(__file__)),
                           "pycolmap_version": pycolmap.__version__,
                           "pycolmap_core_sha256": digest(Path(pycolmap._core.__file__)),
                           "python_path": str(Path(sys.executable).absolute())},
              "mapper_options": json_options(mapper_options()),
              "seed": SEED, "automatic_initialization": True,
              "resource_limits": {"seconds": TIMEOUT_SECONDS,
                                  "child_rss_bytes": MAX_RSS_BYTES,
                                  "output_bytes": MAX_OUTPUT_BYTES,
                                  "log_bytes": MAX_LOG_BYTES,
                                  "free_disk_floor_each_bytes": RESERVE},
              "stages": [], "quality_accepted": False,
              "metric_scale_verified": False, "shipping_approved": False}

    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    deadline = started + TIMEOUT_SECONDS
    save()
    try:
        copy = {"name": "copy_photos_and_restrict_database", "status": "running"}
        report["stages"].append(copy)
        save()
        (output / "images").mkdir()
        for name in bound["names"]:
            source, target = bound["photos"] / name, output / "images" / name
            if (time.monotonic() > deadline or folder_bytes(output) + source.stat().st_size > MAX_OUTPUT_BYTES or
                    shutil.disk_usage(output).free < RESERVE + source.stat().st_size or
                    shutil.disk_usage(ROOT).free < RESERVE):
                raise ValueError("photo copy exceeds bound")
            shutil.copyfile(source, target)
            if digest(target) != bound["stage"]["train_photo_sha256"][name]:
                raise ValueError(f"copied TRAIN photo differs: {name}")
        copy["graph"] = copy_filter_database(bound["database"], output / "database.db", bound["names"])
        (output / "mapper-options.json").write_text(json.dumps(report["mapper_options"], indent=2) + "\n")
        copy["status"] = "complete"
        save()
        if (time.monotonic() > deadline or folder_bytes(output) > MAX_OUTPUT_BYTES or
                min(shutil.disk_usage(output).free, shutil.disk_usage(ROOT).free) < RESERVE):
            raise ValueError("graph copy postflight bound exceeded")
        report["stages"].append(stage(output, "mapper",
                                      [str(Path(sys.executable).absolute()), "-m",
                                       "scripts.classical_backend.local_graph_sfm",
                                       "--worker", "--output", str(output)],
                                      deadline, MAX_OUTPUT_BYTES, MAX_LOG_BYTES, MAX_RSS_BYTES,
                                      extra_reserve_paths=(ROOT,)))
        save()
        disk_models = []
        model_root = output / "models"
        if model_root.is_dir():
            for directory in sorted(model_root.iterdir()):
                if not directory.is_dir() or directory.is_symlink():
                    raise ValueError("mapper output contains an invalid model folder")
                if not directory.name.isdecimal():
                    raise ValueError("mapper output model folder index is not numeric")
                hashes = {name: digest(directory / name) for name in MODEL_FILES}
                model = pycolmap.Reconstruction(str(directory))
                disk_models.append({"directory": str(directory), "index": int(directory.name),
                                    "model_files_sha256": hashes,
                                    "registered": model.num_reg_images(),
                                    "points3D": model.num_points3D(),
                                    "registered_names": sorted(model.images[i].name
                                                               for i in model.reg_image_ids())})
        report["disk_models"] = disk_models
        report.update(reconstruction_summary(disk_models))
        report["status"] = "complete"
    except StageError as error:
        report["stages"].append(error.result)
        report.update(status="failed", failure=str(error))
    except Exception as error:
        if report["stages"] and report["stages"][-1].get("status") == "running":
            report["stages"][-1].update(status="failed", failure=str(error))
        report.update(status="failed", failure=str(error))
    report["output_bytes"] = folder_bytes(output)
    report["free_bytes_after"] = {"external": shutil.disk_usage(output).free,
                                  "internal": shutil.disk_usage(ROOT).free}
    if (time.monotonic() > deadline or report["output_bytes"] > MAX_OUTPUT_BYTES or
            min(report["free_bytes_after"].values()) < RESERVE):
        report.update(status="failed", failure="mapper postflight time/output/disk bound exceeded")
    try:
        validate_source(args.baseline_run, args.stage_root)
        report["sources_unchanged"] = (
            digest(Path(__file__)) == report["software"]["runner_sha256"] and
            digest(Path(pycolmap._core.__file__)) == report["software"]["pycolmap_core_sha256"])
    except Exception:
        report["sources_unchanged"] = False
    if not report["sources_unchanged"]:
        report.update(status="failed", failure="sealed source, PyCOLMAP binary or runner changed")
    first_stage = report["stages"][0] if report["stages"] else {}
    if first_stage.get("status") == "complete":
        try:
            report["prepared_unchanged"] = (
                digest(output / "database.db") == first_stage["graph"]["filtered_database_sha256"] and
                all(digest(output / "images" / name) == bound["stage"]["train_photo_sha256"][name]
                    for name in bound["names"]))
        except Exception:
            report["prepared_unchanged"] = False
        if not report["prepared_unchanged"]:
            report.update(status="failed", failure="filtered graph or copied RGB changed after mapper")
    else:
        report["prepared_unchanged"] = None
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return 0
    if args.baseline_run is None or args.stage_root is None:
        parser.error("--baseline-run and --stage-root are required")
    result = run(args)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json"),
                      "largest_registered": result.get("largest_registered"),
                      "failure": result.get("failure")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
