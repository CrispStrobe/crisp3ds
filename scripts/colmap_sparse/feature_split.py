"""Deterministic, spatially conservative COLMAP keypoint observation split.

COLMAP keypoints have 2 (x,y), 4 (x,y,scale,orientation), or 6
(x,y,affine 2x2) columns. Only x,y define clone proximity here. This module
does not perform feature matching or reconstruct a scene.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import time


RADIUS_PX = 0.25
HOLDOUT_FRACTION = 0.20
LAYOUTS = {2: "xy", 4: "scale_orientation", 6: "affine"}


@dataclass(frozen=True)
class FeatureSplit:
    layout: str
    train_ids: tuple[int, ...]
    heldout_ids: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    group_for_row: tuple[int, ...]
    heldout_for_row: tuple[bool, ...]
    radius_px: float = RADIUS_PX


def _validated_locations(keypoints):
    if isinstance(keypoints, (str, bytes)):
        raise ValueError("keypoints must be a row collection")
    try:
        count = len(keypoints)
    except TypeError as exc:
        raise ValueError("keypoints must be a row collection") from exc
    if not count:
        raise ValueError("keypoints are empty; column layout is unknown")
    width = None
    locations = []
    for idx, row in enumerate(keypoints):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"keypoint {idx} is not a numeric row")
        try:
            size = len(row)
        except TypeError as exc:
            raise ValueError(f"keypoint {idx} is not a row") from exc
        if width is None:
            width = size
            if width not in LAYOUTS:
                raise ValueError(f"unsupported COLMAP keypoint width {width}")
        if size != width:
            raise ValueError(f"keypoint {idx} has inconsistent width")
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"keypoint {idx} has a nonnumeric field") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"keypoint {idx} has a nonfinite field")
        locations.append((values[0], values[1]))
    return LAYOUTS[width], locations


def _components(locations):
    """Union every pair at distance <=RADIUS_PX, including grid boundaries."""
    parent = list(range(len(locations)))
    rank = [0] * len(locations)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    grid = {}
    radius2 = RADIUS_PX * RADIUS_PX
    for i, (x, y) in enumerate(locations):
        cx, cy = math.floor(x / RADIUS_PX), math.floor(y / RADIUS_PX)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    qx, qy = locations[j]
                    if (x-qx)**2 + (y-qy)**2 <= radius2:
                        union(i, j)
        grid.setdefault((cx, cy), []).append(i)
    members = {}
    for i in range(len(locations)):
        members.setdefault(find(i), []).append(i)
    # Component signatures ignore input row order and affine/orientation fields.
    # Repeated positions remain in the signature, preserving deterministic ties.
    decorated = []
    for indices in members.values():
        signature = tuple(sorted((locations[i][0].hex(), locations[i][1].hex())
                                 for i in indices))
        decorated.append((signature, tuple(sorted(indices))))
    decorated.sort(key=lambda item: item[0])
    return decorated


def split_features(keypoints, image_name, seed):
    """Return immutable original-row IDs and component assignments.

    A seeded SHA-256 rank selects round(20% of components), independent of
    original row ordering. Every keypoint in a connected spatial component
    receives one assignment. `image_name` and `seed` are mandatory stable
    identifiers; neither image content nor geometry influences the split.
    """
    if not isinstance(image_name, str) or not image_name:
        raise ValueError("image_name must be a nonempty string")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty string")
    layout, locations = _validated_locations(keypoints)
    decorated = _components(locations)
    groups = tuple(indices for _, indices in decorated)
    ranks = []
    for gid, (signature, _) in enumerate(decorated):
        key = repr((seed, image_name, signature)).encode("utf-8")
        ranks.append((hashlib.sha256(key).digest(), signature, gid))
    selected = {gid for _, _, gid in sorted(ranks)[:round(HOLDOUT_FRACTION*len(groups))]}
    group_for_row = [0] * len(locations)
    heldout_for_row = [False] * len(locations)
    for gid, group in enumerate(groups):
        for row in group:
            group_for_row[row] = gid
            heldout_for_row[row] = gid in selected
    heldout = tuple(i for i, assigned in enumerate(heldout_for_row) if assigned)
    train = tuple(i for i, assigned in enumerate(heldout_for_row) if not assigned)
    return FeatureSplit(layout, train, heldout, groups,
                        tuple(group_for_row), tuple(heldout_for_row))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_database(database, seed):
    """Read-only dry run; independently check every proposed split's separation."""
    from scripts.sparse_verify.verify import spatial_split_overlap

    started = time.monotonic()
    path = Path(database).resolve(strict=True)
    before = sha256(path)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    records = db.execute("SELECT images.name,keypoints.rows,keypoints.cols,keypoints.data "
                         "FROM images JOIN keypoints USING(image_id) ORDER BY images.name").fetchall()
    image_reports = []
    for name, rows, cols, blob in records:
        if cols not in LAYOUTS or blob is None or len(blob) != rows*cols*4:
            raise ValueError(f"invalid keypoint table: {name}")
        keypoints = list(struct.iter_unpack("<"+"f"*cols, blob))
        split = split_features(keypoints, name, seed)
        # Separate verifier routine uses a spatial search over heldout vs train.
        overlap = spatial_split_overlap(keypoints, set(split.heldout_ids), RADIUS_PX)
        image_reports.append(dict(name=name, layout=split.layout, row_count=rows,
                                  group_count=len(split.groups), train_ids=split.train_ids,
                                  heldout_ids=split.heldout_ids,
                                  spatial_overlap_count=overlap))
    db.close()
    after = sha256(path)
    result = dict(protocol="colmap_spatial_components_v1", seed=seed,
                  radius_px=RADIUS_PX, holdout_fraction_of_groups=HOLDOUT_FRACTION,
                  database=str(path), database_sha256_before=before,
                  database_sha256_after=after, splitter_sha256=sha256(__file__),
                  verifier_sha256=sha256(spatial_split_overlap.__code__.co_filename),
                  elapsed_seconds=round(time.monotonic()-started, 3),
                  image_count=len(image_reports),
                  total_features=sum(r["row_count"] for r in image_reports),
                  total_groups=sum(r["group_count"] for r in image_reports),
                  total_heldout_features=sum(len(r["heldout_ids"]) for r in image_reports),
                  total_spatial_overlap=sum(r["spatial_overlap_count"] for r in image_reports),
                  images=image_reports, quality_claim="split integrity only; no reconstruction")
    result["status"] = "pass" if (len(image_reports) == 10 and before == after and
                                   result["total_spatial_overlap"] == 0) else "fail"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="existing COLMAP database, opened read-only")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", required=True, help="fresh JSON audit path")
    args = parser.parse_args()
    output = Path(args.output)
    if shutil.disk_usage(output.parent).free < (10 << 30):
        raise ValueError("less than 10 GiB free at audit output")
    result = audit_database(args.database, args.seed)
    content = json.dumps(result, separators=(",", ":"))+"\n"
    if len(content.encode()) >= 5_000_000:
        raise ValueError("split audit exceeds 5 MB")
    with output.open("x") as stream:
        stream.write(content)
    print(json.dumps({key: value for key, value in result.items() if key != "images"}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
