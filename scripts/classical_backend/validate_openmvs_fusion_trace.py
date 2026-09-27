"""Validate the bounded evaluation-only OpenMVS fusion trace format.

This checks internal provenance accounting. It does not score shape or verify
that a pre-ROI point remains in the final PLY.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


HEADER = ("point", "pretrim_index", "seed_view", "seed_x", "seed_y", "x", "y", "z",
          "pixels", "views", "truncated")
REASONS = {"out_of_bounds", "no_depth", "already_fused", "low_confidence",
           "depth_disagreement", "reprojection_error", "normal_disagreement",
           "accepted", "neighbor_not_cached"}


def validate(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError("trace missing, linked, or over 1 MiB")
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows or tuple(rows[0]) != HEADER:
        raise ValueError("wrong fusion trace header")
    points: list[dict] = []
    events: dict[int, list[tuple[int, str]]] = {}
    for row in rows[1:]:
        if len(row) != 11:
            raise ValueError("wrong fusion trace row width")
        if row[0] == "event":
            index, view, x, y = map(int, row[1:5])
            values = [float(value) for value in row[5:10]]
            reason = row[10]
            if (index < 0 or view < 0 or not all(math.isfinite(v) for v in values) or
                    reason not in REASONS or
                    (reason == "accepted" and (x < 0 or y < 0 or values[0] <= 0 or values[1] < 0))):
                raise ValueError("invalid fusion trace event")
            events.setdefault(index, []).append((view, reason))
        else:
            index, pretrim, view, x, y = map(int, row[:5])
            xyz = [float(value) for value in row[5:8]]
            pixels, views, truncated = map(int, row[8:11])
            if (index != len(points) or pretrim < 0 or view < 0 or x < 0 or y < 0 or
                    not all(math.isfinite(value) for value in xyz) or pixels < 1 or
                    views < 1 or truncated not in (0, 1)):
                raise ValueError("invalid fused point record")
            if points and pretrim <= points[-1]["pretrim_index"]:
                raise ValueError("pretrim point indices are not increasing")
            points.append({"trace_index": index, "pretrim_index": pretrim,
                           "seed_view": view, "seed_pixel": [x, y], "xyz": xyz,
                           "pixels": pixels, "views": views, "truncated": bool(truncated)})
    if not 1 <= len(points) <= 8 or set(events) != set(range(len(points))):
        raise ValueError("trace must have one to eight points with event records")
    for point in points:
        accepted = [view for view, reason in events[point["trace_index"]] if reason == "accepted"]
        if not point["truncated"] and (len(accepted) != point["pixels"] or
                                       len(set(accepted)) != point["views"] or
                                       point["seed_view"] not in accepted):
            raise ValueError("fused point and accepted-pixel accounting differ")
    return {"schema": "openmvs_eval_fusion_trace_validation_v1", "status": "valid",
            "point_count": len(points), "event_count": sum(map(len, events.values())),
            "complete_event_records": all(not point["truncated"] for point in points),
            "points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.trace), indent=2))


if __name__ == "__main__":
    main()
