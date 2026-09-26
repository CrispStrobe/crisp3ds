"""Versioned manual polygon ROI contract in COLMAP pixel-centre coordinates."""

import json
import math
import re


SCHEMA = "foreground_roi_v1"
PROVENANCE = "approximate target-tree envelope; gaps may contain background"


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on_segment(a, b, p, epsilon=1e-9):
    return (abs(_cross(a, b, p)) <= epsilon and
            min(a[0], b[0])-epsilon <= p[0] <= max(a[0], b[0])+epsilon and
            min(a[1], b[1])-epsilon <= p[1] <= max(a[1], b[1])+epsilon)


def _intersect(a, b, c, d):
    c1, c2, c3, c4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if (c1 > 0 > c2 or c1 < 0 < c2) and (c3 > 0 > c4 or c3 < 0 < c4):
        return True
    return (_on_segment(a, b, c) or _on_segment(a, b, d) or
            _on_segment(c, d, a) or _on_segment(c, d, b))


def validate_polygon(vertices, width, height):
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError("polygon needs at least three vertices")
    points = []
    for i, vertex in enumerate(vertices):
        if not isinstance(vertex, list) or len(vertex) != 2 or any(isinstance(v, bool) for v in vertex):
            raise ValueError(f"invalid polygon vertex {i}")
        try:
            x, y = float(vertex[0]), float(vertex[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"nonnumeric polygon vertex {i}") from exc
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"nonfinite or out-of-bounds vertex {i}")
        points.append((x, y))
    if len(set(points)) != len(points):
        raise ValueError("repeated polygon vertex")
    n = len(points)
    for i in range(n):
        a, b, c = points[i-1], points[i], points[(i+1) % n]
        if (abs(_cross(a, b, c)) <= 1e-9 and
                (b[0]-a[0])*(c[0]-b[0])+(b[1]-a[1])*(c[1]-b[1]) < 0):
            raise ValueError("overlapping adjacent polygon edges")
    area2 = sum(points[i][0]*points[(i+1)%n][1]-points[(i+1)%n][0]*points[i][1]
                for i in range(n))
    if abs(area2) <= 1e-9:
        raise ValueError("zero-area polygon")
    for i in range(n):
        a, b = points[i], points[(i+1)%n]
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            c, d = points[j], points[(j+1)%n]
            if _intersect(a, b, c, d):
                raise ValueError("self-intersecting polygon")
    return tuple(points)


def contains_point(polygon, point):
    """Closed polygon: centres exactly on any edge or vertex count as inside."""
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i+1) % len(polygon)]
        if _on_segment(a, b, (x, y)):
            return True
        if (a[1] > y) != (b[1] > y):
            crossing = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < crossing:
                inside = not inside
    return inside


def in_roi(polygons, point):
    return any(contains_point(polygon, point) for polygon in polygons)


def validate_contract(payload, expected_images):
    """Validate JSON object against frozen PNG names, dimensions, and hashes."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported ROI schema")
    provenance = payload.get("provenance")
    if (not isinstance(provenance, dict) or provenance.get("method") != "manual" or
            provenance.get("description") != PROVENANCE):
        raise ValueError("ROI manual provenance is missing or changed")
    items = payload.get("images")
    if not isinstance(items, list) or len(items) != len(expected_images):
        raise ValueError("ROI must list every frozen image exactly once")
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid ROI image entry")
        name = item.get("name")
        if name in result or name not in expected_images:
            raise ValueError(f"duplicate or unexpected ROI image: {name}")
        source = expected_images[name]
        if (item.get("width"), item.get("height")) != (source["width"], source["height"]):
            raise ValueError(f"ROI dimensions mismatch: {name}")
        digest = item.get("original_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != source["sha256"]:
            raise ValueError(f"ROI PNG hash mismatch: {name}")
        polygons = item.get("polygons")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"ROI polygons missing: {name}")
        result[name] = tuple(validate_polygon(poly, item["width"], item["height"])
                             for poly in polygons)
    return result


def load_contract(path, expected_images):
    def reject(token):
        raise ValueError(f"nonfinite JSON token: {token}")
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject)
    return validate_contract(payload, expected_images)
