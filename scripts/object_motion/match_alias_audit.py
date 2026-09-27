"""Read-only, TRAIN-only COLMAP match graph and pose-alias diagnostic."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

import numpy as np


MAX_IMAGE_ID = 2147483647
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
INITIALIZER = re.compile(r"Initializing with image pair #(\d+) and #(\d+)")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_ids(pair_id):
    if not isinstance(pair_id, int) or pair_id < 0:
        raise ValueError("invalid COLMAP pair ID")
    a, b = pair_id // MAX_IMAGE_ID, pair_id % MAX_IMAGE_ID
    if not 0 < a < b < MAX_IMAGE_ID:
        raise ValueError("invalid decoded COLMAP pair")
    return a, b


def cyclic_separation(index_a, index_b, count):
    if (not 0 <= index_a < count or not 0 <= index_b < count or index_a == index_b):
        raise ValueError("invalid cyclic image indices")
    delta = abs(index_a - index_b)
    return min(delta, count - delta)


def bucket(distance):
    return "neighbor_1_to_4" if distance <= 4 else "middle_5_to_15" if distance < 16 else "long_16_to_24"


def rotation_degrees(left, right):
    relative = right @ left.T
    return math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def sidecars(path):
    result = {}
    for suffix in ("-wal", "-shm"):
        item = Path(str(path) + suffix)
        if item.is_symlink() or (item.exists() and (not item.is_file() or item.stat().st_size != 0)):
            raise ValueError("nonempty or linked SQLite sidecar prevents immutable main-DB audit")
        result[suffix] = {"exists": item.exists(), "bytes": item.stat().st_size if item.exists() else 0}
    return result


def source_snapshot(run, repair_report, expected_repair_sha):
    import pycolmap

    report_path = Path(repair_report)
    if (report_path.is_symlink() or not report_path.is_file() or
            not 0 < report_path.stat().st_size <= 1 << 20 or sha(report_path) != expected_repair_sha):
        raise ValueError("repair report seal differs")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "mustard_sparse_track_repair_v1" or report.get("status") != "candidate_unreviewed":
        raise ValueError("not accepted candidate producer report")
    if (pycolmap.__version__ != "3.11.1" or
            sha(pycolmap._core.__file__) != report["software"]["pycolmap_core_sha256"]):
        raise ValueError("active PyCOLMAP enum/binary differs from sealed producer")
    run = Path(run)
    if run.is_symlink() or not run.is_dir():
        raise ValueError("source run directory missing or linked")
    db, log, producer = run / "database.db", run / "sfm.log", run / "result.json"
    for path, limit in ((db, 256 << 20), (log, 32 << 20), (producer, 1 << 20)):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise ValueError("source DB/log/result missing, linked, empty, or oversized")
    producer_sha = sha(producer)
    if producer_sha != report["baseline"]["result_sha256"]:
        raise ValueError("database run producer differs from sealed repair baseline")
    model_dir = report_path.parent / "model"
    if (model_dir.is_symlink() or not model_dir.is_dir() or
            str(model_dir.resolve()) != report["output"]["model_dir"] or
            {item.name for item in model_dir.iterdir()} != set(MODEL_FILES)):
        raise ValueError("repaired model inventory differs")
    model_sha = {}
    for name in MODEL_FILES:
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 256 << 20:
            raise ValueError("repaired model file invalid")
        model_sha[name] = sha(path)
    if model_sha != report["output"]["model_files_sha256"]:
        raise ValueError("repaired model seal differs")
    return {"db": db, "db_sha256": sha(db), "db_sidecars": sidecars(db),
            "sfm_log": log, "sfm_log_sha256": sha(log), "model_dir": model_dir,
            "producer_result": producer, "producer_result_sha256": producer_sha,
            "model_sha256": model_sha, "report": report,
            "model": pycolmap.Reconstruction(str(model_dir))}


def analyze(snapshot):
    import pycolmap

    db, model = snapshot["db"], snapshot["model"]
    configurations = {int(value): name for name, value in
                      pycolmap.TwoViewGeometryConfiguration.__members__.items()}
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        images = dict(connection.execute("SELECT image_id,name FROM images"))
        if (len(images) != 48 or len(set(images.values())) != 48 or
                set(images.values()) != set(snapshot["report"]["output"]["registered_names"])):
            raise ValueError("database images differ from repaired TRAIN inventory")
        names = sorted(images.values())
        ordered = {name: index for index, name in enumerate(names)}
        by_id = {image_id: ordered[name] for image_id, name in images.items()}
        model_images = {image.name: image for image in model.images.values() if image.has_pose}
        if set(model_images) != set(names):
            raise ValueError("model registered names differ")
        if (set(model.images) != set(images) or
                any(model.images[image_id].name != name for image_id, name in images.items())):
            raise ValueError("COLMAP database image ID-to-name map differs from repaired model")
        rotations = {name: np.asarray(image.cam_from_world.matrix(), dtype=float)[:, :3]
                     for name, image in model_images.items()}
        for rotation in rotations.values():
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)):
                raise ValueError("invalid recovered rotation")
        counts = {key: {"possible_pairs": 0, "verified_pairs": 0, "inlier_correspondences": 0,
                        "configuration_counts": {},
                        "recovered_rotation_degrees": []}
                  for key in ("neighbor_1_to_4", "middle_5_to_15", "long_16_to_24")}
        all_pairs = []
        native_pose_nondefault = 0
        for pair_id, rows, cols, config, qblob, tblob in connection.execute(
                "SELECT pair_id,rows,cols,config,qvec,tvec FROM two_view_geometries"):
            a, b = pair_ids(pair_id)
            if a not in images or b not in images or cols != 2 or rows < 0:
                raise ValueError("invalid verified pair row or image IDs")
            separation = cyclic_separation(by_id[a], by_id[b], len(images))
            group = bucket(separation)
            counts[group]["possible_pairs"] += 1
            if rows == 0:
                continue
            counts[group]["verified_pairs"] += 1
            counts[group]["inlier_correspondences"] += rows
            configuration_name = configurations.get(config)
            if configuration_name is None:
                raise ValueError("unknown pinned PyCOLMAP two-view geometry configuration")
            histogram = counts[group]["configuration_counts"]
            histogram[configuration_name] = histogram.get(configuration_name, 0) + 1
            q = np.frombuffer(qblob, dtype=np.float64) if qblob else np.asarray([])
            t = np.frombuffer(tblob, dtype=np.float64) if tblob else np.asarray([])
            if q.shape == (4,) and t.shape == (3,) and np.isfinite(q).all() and np.isfinite(t).all():
                native_pose_nondefault += int(not np.array_equal(q, [1, 0, 0, 0]) or
                                              not np.array_equal(t, [0, 0, 0]))
            orientation = rotation_degrees(rotations[images[a]], rotations[images[b]])
            counts[group]["recovered_rotation_degrees"].append(orientation)
            all_pairs.append({"a": images[a], "b": images[b], "cyclic_index_separation": separation,
                              "verified_inliers": rows, "configuration": config,
                              "recovered_full_rotation_degrees": orientation})
        if sum(item["possible_pairs"] for item in counts.values()) != 48 * 47 // 2:
            raise ValueError("expected exhaustive 48-view pair inventory")
        for item in counts.values():
            values = item.pop("recovered_rotation_degrees")
            item["verified_edge_fraction"] = item["verified_pairs"] / item["possible_pairs"]
            item["median_recovered_rotation_degrees"] = float(np.median(values)) if values else None
            item["p95_recovered_rotation_degrees"] = float(np.percentile(values, 95)) if values else None
        long_tracks, opposite_tracks = 0, 0
        for point in model.points3D.values():
            indices = [by_id[int(element.image_id)] for element in point.track.elements]
            largest = max((cyclic_separation(a, b, len(images)) for a in indices for b in indices if a != b),
                          default=0)
            long_tracks += largest >= 16
            opposite_tracks += largest >= 20
        log = snapshot["sfm_log"].read_text()
        initializer = INITIALIZER.findall(log)
        if len(initializer) != 1:
            raise ValueError("SfM log does not identify exactly one selected initial pair")
        init_a, init_b = (int(item) for item in initializer[0])
        if init_a not in images or init_b not in images:
            raise ValueError("initializer IDs absent from image inventory")
        init_names = [images[init_a], images[init_b]]
        init_edge = next((item for item in all_pairs if {item["a"], item["b"]} == set(init_names)), None)
        if init_edge is None:
            raise ValueError("initializer absent from verified graph")
        return {"images": 48, "verified_pair_support_by_cyclic_index_separation": counts,
                "two_view_configuration_enum_source": "active pinned PyCOLMAP 3.11.1 TwoViewGeometryConfiguration",
                "native_two_view_relative_pose_nondefault_pairs": native_pose_nondefault,
                "native_two_view_rotation_comparison": "unavailable: all qvec identity and tvec zero"
                if native_pose_nondefault == 0 else "nondefault native relative pose present; inspect convention separately",
                "model_tracks": len(model.points3D), "tracks_spanning_cyclic_16plus": long_tracks,
                "tracks_spanning_cyclic_20plus": opposite_tracks,
                "initializer": {"image_ids": [init_a, init_b], "names": init_names,
                                "cyclic_index_separation": cyclic_separation(by_id[init_a], by_id[init_b], 48),
                                "verified_edge": init_edge},
                "strongest_long_edges": sorted((item for item in all_pairs if item["cyclic_index_separation"] >= 16),
                                                 key=lambda item: item["verified_inliers"], reverse=True)[:12],
                "strongest_neighbor_edges": sorted((item for item in all_pairs if item["cyclic_index_separation"] <= 4),
                                                     key=lambda item: item["verified_inliers"], reverse=True)[:12],
                "long_edges_with_small_recovered_rotation": sorted(
                    (item for item in all_pairs if item["cyclic_index_separation"] >= 16),
                    key=lambda item: (item["recovered_full_rotation_degrees"], -item["verified_inliers"]))[:12]}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--repair-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink() or args.output.parent.is_symlink() or
            not args.output.parent.is_dir()):
        raise FileExistsError("fresh output under a real parent required")
    source = source_snapshot(args.run, args.repair_report, args.repair_report_sha256)
    result = {"schema": "mustard_match_alias_audit_v1", "status": "diagnostic_only",
              "scope": "TRAIN-only immutable original COLMAP DB and repaired camera/track model; no GT/heldout",
              "cyclic_index_caveat": "sorted NP3 filenames give acquisition order, not supplied camera angles or poses",
              "source": {"database_sha256": source["db_sha256"], "database_sidecars": source["db_sidecars"],
                         "sfm_log_sha256": source["sfm_log_sha256"],
                         "producer_result_sha256": source["producer_result_sha256"],
                         "repair_report_sha256": args.repair_report_sha256,
                         "model_files_sha256": source["model_sha256"],
                         "runner_sha256": sha(__file__)},
              "audit": analyze(source)}
    if (sha(source["db"]) != source["db_sha256"] or sidecars(source["db"]) != source["db_sidecars"] or
            sha(source["sfm_log"]) != source["sfm_log_sha256"] or
            sha(source["producer_result"]) != source["producer_result_sha256"] or
            sha(args.repair_report) != args.repair_report_sha256 or
            any(sha(source["model_dir"] / name) != digest for name, digest in source["model_sha256"].items()) or
            sha(__file__) != result["source"]["runner_sha256"]):
        raise ValueError("source changed during immutable audit")
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > 128 << 10:
        raise ValueError("audit report exceeds 128 KiB")
    with args.output.open("xb") as stream:
        stream.write(payload)


if __name__ == "__main__":
    main()
