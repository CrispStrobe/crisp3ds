#!/usr/bin/env python3
"""Read-only YCB initial-pair evidence and image-only candidate ranking."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build-opencv/object-motion"
DATABASES = {
    "foreground": BASE / "foreground/database.db",
    "wrapper_001": BASE / "calibration-ablation-001/image_only_reverified/database.db",
    "serial_002": BASE / "calibration-ablation-002/image_only_reverified/database.db",
}
MODELS = {
    "foreground": BASE / "foreground/models/0",
    "wrapper_001": BASE / "calibration-ablation-001/image_only_reverified/models/0",
    "serial_002": BASE / "calibration-ablation-002/image_only_reverified/models/0",
}
LOGS = {
    "wrapper_001": BASE / "calibration-ablation-001/image_only_reverified/trial.log",
    "serial_002": BASE / "calibration-ablation-002/image_only_reverified/trial.log",
}
OUTPUT = BASE / "initialization-diagnostic-002"
MAX_IMAGE_ID = 2147483647
SEED_NAMES = ("NP3_162.jpg", "NP3_192.jpg")
CONFIG_UNCALIBRATED = 3  # Verified against pinned PyCOLMAP 3.11.1 enum.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_pair(pair_id):
    first, second = divmod(int(pair_id), MAX_IMAGE_ID)
    if first < 1 or second <= first or second > MAX_IMAGE_ID:
        raise ValueError("invalid COLMAP pair ID")
    return first, second


def decode_matches(blob, rows, cols):
    if rows == 0 and blob is None and cols == 2:
        return np.empty((0, 2), dtype=np.uint32)
    if rows < 0 or cols != 2 or blob is None or len(blob) != rows * 2 * 4:
        raise ValueError("invalid two-view inlier array")
    return np.frombuffer(blob, dtype="<u4").reshape(rows, 2).copy()


def load_geometry(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 20_000_000:
        raise ValueError("unexpected geometry database size")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        names = dict(connection.execute("SELECT image_id,name FROM images"))
        if len(names) != 60 or len(set(names.values())) != 60:
            raise ValueError("unexpected image inventory")
        pairs = {}
        for pair_id, rows, cols, blob, config in connection.execute(
                "SELECT pair_id,rows,cols,data,config FROM two_view_geometries ORDER BY pair_id"):
            first, second = decode_pair(pair_id)
            if first not in names or second not in names:
                raise ValueError("geometry references unknown image")
            pairs[(first, second)] = {"config": int(config),
                                      "matches": decode_matches(blob, rows, cols)}
        return names, pairs
    finally:
        connection.close()


def triple_support(seed_pair, pairs):
    """Count seed correspondences independently agreeing through a third image."""
    first, second = seed_pair
    outgoing = {}
    for (left, right), item in pairs.items():
        if not len(item["matches"]):
            continue
        outgoing[(left, right)] = {}
        outgoing[(right, left)] = {}
        for a, b in item["matches"]:
            outgoing[(left, right)].setdefault(int(a), set()).add(int(b))
            outgoing[(right, left)].setdefault(int(b), set()).add(int(a))
    scores = {}
    for third in {b for a, b in outgoing if a == first} & {b for a, b in outgoing if a == second}:
        if third in seed_pair:
            continue
        a_to_k = outgoing[(first, third)]
        b_to_k = outgoing[(second, third)]
        scores[third] = sum(bool(a_to_k.get(int(a), set()) & b_to_k.get(int(b), set()))
                            for a, b in pairs[seed_pair]["matches"])
    return scores


def angle_gap(name1, name2):
    match1 = re.fullmatch(r"NP3_(\d{3})\.jpg", name1)
    match2 = re.fullmatch(r"NP3_(\d{3})\.jpg", name2)
    if not match1 or not match2:
        raise ValueError("unexpected YCB photo name")
    difference = abs(int(match1.group(1)) - int(match2.group(1)))
    return min(difference, 360 - difference)


def candidate_rows(names, pairs, exclude_seed=True):
    rows = []
    for seed, item in pairs.items():
        first, second = seed
        label = (names[first], names[second])
        gap = angle_gap(*label)
        if item["config"] != CONFIG_UNCALIBRATED or len(item["matches"]) < 80:
            continue
        if exclude_seed and set(label) == set(SEED_NAMES):
            continue
        support = triple_support(seed, pairs)
        ranked = sorted(support.items(), key=lambda kv: (-kv[1], names[kv[0]]))
        if not ranked:
            continue
        rows.append({"pair": list(label), "acquisition_angle_gap_degrees": gap,
                     "verified_inliers": int(len(item["matches"])),
                     "best_third_name": names[ranked[0][0]],
                     "best_exact_triplet_support": ranked[0][1],
                     "third_views_with_at_least_40_exact_triples": sum(score >= 40 for score in support.values())})
    return sorted(rows, key=lambda row: (-row["best_exact_triplet_support"],
                                         -row["third_views_with_at_least_40_exact_triples"],
                                         -row["verified_inliers"], row["pair"]))


def model_evidence(model_path):
    import pycolmap
    model = pycolmap.Reconstruction(str(model_path))
    if len(model.cameras) != 1:
        raise ValueError("expected one shared camera")
    camera = next(iter(model.cameras.values()))
    return {"registered": model.num_reg_images(), "points3D": model.num_points3D(),
            "camera_model": camera.model.name, "camera_params": list(map(float, camera.params)),
            "model_files_sha256": {name: sha256(model_path / name)
                                   for name in ("cameras.bin", "images.bin", "points3D.bin")}}


def log_evidence(path):
    content = Path(path).read_text()
    initialized = re.findall(r"Initializing with image pair #(\d+) and #(\d+)", content)
    attempted = [int(value) for value in re.findall(r"Registering image #(\d+) \(3\)", content)]
    seen = [int(value) for value in re.findall(r"Image sees (\d+) / \d+ points", content)]
    rejected = content.count("Could not register, trying another image")
    return {"sha256": sha256(path), "initial_pairs": initialized, "third_image_attempts": attempted,
            "visible_seed_points_by_attempt": seen, "rejections": rejected}


def analyze(output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free before diagnostic report")
    before_hashes = {name: sha256(path) for name, path in DATABASES.items()}
    databases = {name: load_geometry(path) for name, path in DATABASES.items()}
    names = databases["foreground"][0]
    if any(other_names != names for other_names, _ in databases.values()):
        raise ValueError("image inventories differ")
    seed_ids = tuple(sorted(image_id for image_id, name in names.items() if name in SEED_NAMES))
    if len(seed_ids) != 2:
        raise ValueError("seed images missing")
    seed_sets = {}
    seed_rows = {}
    for arm, (_, pairs) in databases.items():
        item = pairs[seed_ids]
        seed_sets[arm] = set(map(tuple, item["matches"]))
        seed_rows[arm] = {"verified_inliers": len(seed_sets[arm]), "two_view_config": item["config"],
                          "inlier_match_sha256": hashlib.sha256(item["matches"].tobytes()).hexdigest(),
                          "best_exact_triplet_support": max(triple_support(seed_ids, pairs).values())}
    candidates = candidate_rows(names, databases["foreground"][1])
    report = {"schema": "ycb_initialization_read_only_v1", "scope": "existing image-only correspondence databases and models only; no new mapping, Berkeley pose, depth, or mesh scoring",
              "source_databases_sha256": before_hashes,
              "seed_pair": list(SEED_NAMES), "seed_image_ids": list(seed_ids),
              "seed_geometry": seed_rows,
              "seed_inlier_overlap": {"foreground_vs_wrapper_001": len(seed_sets["foreground"] & seed_sets["wrapper_001"]),
                                      "foreground_vs_serial_002": len(seed_sets["foreground"] & seed_sets["serial_002"])},
              "models": {name: model_evidence(path) for name, path in MODELS.items()},
              "logs": {name: log_evidence(path) for name, path in LOGS.items()},
              "candidate_policy": "source foreground DB only; pinned 3.11.1 UNCALIBRATED two-view config 3, ≥80 inliers; rank exact seed-inlier triplets corroborated by a third photo, then number of third views ≥40, then inliers; exclude failed fixed seed; filename angle shown only as dataset-specific annotation, not a rank/gate",
              "candidate_count": len(candidates), "top_candidates": candidates[:12],
              "candidates_not_mapper_validated": True, "reference_metadata_used_for_ranking": False}
    if {name: sha256(path) for name, path in DATABASES.items()} != before_hashes:
        raise RuntimeError("source geometry database changed during read-only diagnostic")
    for name, model in MODELS.items():
        for filename, value in report["models"][name]["model_files_sha256"].items():
            if sha256(model / filename) != value:
                raise RuntimeError("source model changed during read-only diagnostic")
    for name, path in LOGS.items():
        if sha256(path) != report["logs"][name]["sha256"]:
            raise RuntimeError("source mapper log changed during read-only diagnostic")
    output.mkdir(parents=True)
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    if target.stat().st_size > 64_000:
        raise RuntimeError("initialization report exceeds 64 KB")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = analyze(args.output)
    print(json.dumps({"candidate_count": report["candidate_count"],
                      "top_pair": report["top_candidates"][0]["pair"]}))


if __name__ == "__main__":
    main()
