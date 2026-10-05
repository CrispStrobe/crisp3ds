"""Read-only checks for native AliceVision pinhole scenes; no camera fitting.

Current JSON focal/offset conventions follow AliceVision 1.2.14. Radial
validity certifies that the increasing branch reaches every sensor corner.
It is a lens/photometric check, not a reconstruction accuracy certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _array(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("invalid native numeric array")
    if any(isinstance(v, bool) for v in value):
        raise ValueError("boolean camera parameter")
    out = np.asarray(value, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite native camera parameter")
    return out


def pixel_intrinsic(row):
    """Convert current native serialization to zero-origin fx, fy, cx, cy, k."""
    if row.get("type") != "pinhole" or row.get("distortionType") not in (
        "none",
        "radialk3",
    ):
        raise ValueError("unsupported camera model")
    for name in ("width", "height", "sensorWidth", "focalLength", "pixelRatio"):
        if isinstance(row.get(name), bool):
            raise ValueError("boolean camera parameter")
    w, h = int(row["width"]), int(row["height"])
    if (
        float(row["width"]) != w
        or float(row["height"]) != h
        or not 1 <= w <= 16384
        or not 1 <= h <= 16384
        or w * h > 64_000_000
    ):
        raise ValueError("invalid image dimensions")
    sensor, focal, ratio = map(
        float, (row["sensorWidth"], row["focalLength"], row["pixelRatio"])
    )
    if not all(math.isfinite(v) and v > 0 for v in (sensor, focal, ratio)):
        raise ValueError("invalid focal length or aspect ratio")
    fy = focal * w / sensor
    fx = fy / ratio
    pp = _array(row["principalPoint"], 2) + [w / 2, h / 2]
    k = (
        _array(row["distortionParams"], 3)
        if row["distortionType"] == "radialk3"
        else np.zeros(3)
    )
    if not np.isfinite([fx, fy, *pp]).all():
        raise ValueError("nonfinite pixel intrinsic")
    return w, h, np.array([fx, fy, *pp]), k


def radial_field_check(w, h, intrinsic, coefficients):
    """Analytic derivative roots bound the increasing radial inverse branch.

    g(r)=r*(1+k1*r²+k2*r⁴+k3*r⁶). The first positive derivative
    root ends its strictly increasing branch. A full image is representable
    on that branch iff its largest distorted corner radius is below g(root).
    No sampled-grid claim is used, and no value is corrected.
    """
    p, k = _array(intrinsic, 4), _array(coefficients, 3)
    if p[0] <= 0 or p[1] <= 0 or not 0 <= p[2] < w or not 0 <= p[3] < h:
        return {"passed": False, "reason": "invalid focal length or principal point"}
    radius = max(
        float(np.linalg.norm([(x - p[2]) / p[0], (y - p[3]) / p[1]]))
        for x in (0, w - 1)
        for y in (0, h - 1)
    )
    polynomial = np.trim_zeros(np.array([7 * k[2], 5 * k[1], 3 * k[0], 1.0]), "f")
    roots = np.roots(polynomial)
    positive = sorted(
        float(v.real)
        for v in roots
        if abs(v.imag) <= 1e-9 * max(1.0, abs(v.real)) and v.real > 0
    )
    first = math.sqrt(positive[0]) if positive else None
    maximum = (
        first * (1 + k[0] * first**2 + k[1] * first**4 + k[2] * first**6)
        if first is not None
        else None
    )
    passed = maximum is None or (
        math.isfinite(maximum) and radius < maximum * (1 - 1e-8)
    )
    return {
        "passed": bool(passed),
        "maximum_distorted_corner_radius": radius,
        "first_radial_derivative_zero": first,
        "increasing_branch_distorted_radius_limit": maximum,
        "method": "analytic first positive root of 1+3k1*r²+5k2*r⁴+7k3*r⁶",
    }


def _stats(values):
    return (
        {
            "count": len(values),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
        if values
        else None
    )


def audit_scene(
    path: Path,
    *,
    expected_names=None,
    expected_calibration=None,
    minimum_coverage=0.8,
    minimum_observations_per_view=20,
    maximum_reprojection_p95=4.0,
):
    """Check sealed native poses, lens and sparse observations; never mutate input.

    Passing supports use as photo reconstruction cameras, not shape accuracy.
    expected_calibration is a declared fixed lens {pixels:[fx,fy,cx,cy],k:[...]}.
    It must match native serialization; calibration source authenticity remains
    the caller's provenance responsibility.
    """
    for v in (minimum_coverage, maximum_reprojection_p95):
        if isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("invalid audit bound")
    if not 0 < minimum_coverage <= 1 or not 0 < maximum_reprojection_p95 <= 100:
        raise ValueError("invalid audit bound")
    if (
        isinstance(minimum_observations_per_view, bool)
        or not isinstance(minimum_observations_per_view, int)
        or not 1 <= minimum_observations_per_view <= 100000
    ):
        raise ValueError("invalid observation bound")
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 128 << 20:
        raise ValueError("native scene missing or exceeds 128 MiB bound")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    scene = json.loads(raw)
    version = tuple(map(int, scene["version"]))
    if version < (1, 2, 11):
        raise ValueError("unsupported legacy focal serialization")
    views, poses, landmarks, intrinsics = (
        scene.get(k, []) for k in ("views", "poses", "structure", "intrinsics")
    )
    if (
        not 3 <= len(views) <= 96
        or len(poses) > len(views)
        or len(landmarks) > 1_000_000
        or not 1 <= len(intrinsics) <= len(views)
    ):
        raise ValueError("invalid bounded native scene inventory")
    names = [Path(v["path"]).name for v in views]
    if (
        len(set(names)) != len(names)
        or expected_names is not None
        and set(names) != set(expected_names)
    ):
        raise ValueError("camera image inventory mismatch")
    vmap = {str(v["viewId"]): v for v in views}
    imap = {str(v["intrinsicId"]): v for v in intrinsics}
    pmap = {str(v["poseId"]): v["pose"]["transform"] for v in poses}
    if (
        len(vmap) != len(views)
        or len(imap) != len(intrinsics)
        or len(pmap) != len(poses)
    ):
        raise ValueError("duplicate native identifier")
    reasons, lens = [], []
    if not landmarks:
        reasons.append("no triangulated landmarks")
    converted = {}
    for key, intrinsic in imap.items():
        w, h, pixels, k = pixel_intrinsic(intrinsic)
        check = radial_field_check(w, h, pixels.tolist(), k.tolist())
        if not check["passed"]:
            reasons.append("lens principal branch does not cover the image field")
        if expected_calibration is not None:
            expected_p = _array(expected_calibration["pixels"], 4)
            expected_k = _array(expected_calibration["k"], 3)
            if (
                not np.allclose(pixels, expected_p, atol=1e-7, rtol=0)
                or not np.allclose(k, expected_k, atol=1e-12, rtol=0)
                or intrinsic.get("locked") not in (True, "true")
            ):
                reasons.append("declared fixed calibration was changed or unlocked")
        converted[key] = (w, h, pixels, k)
        lens.append(
            {
                "intrinsic_id": key,
                "pixel_intrinsic": pixels.tolist(),
                "radial_coefficients": k.tolist(),
                "field_check": check,
            }
        )
    registered = {}
    errors = []
    positive = []
    observation_count = 0
    for key, view in vmap.items():
        if str(view["intrinsicId"]) not in converted:
            raise ValueError("unknown camera intrinsic")
        if str(view["poseId"]) not in pmap:
            continue
        pose = pmap[str(view["poseId"])]
        R = _array(pose["rotation"], 9).reshape(3, 3, order="F")
        C = _array(pose["center"], 3)
        if (
            not np.allclose(R @ R.T, np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(R) - 1) > 1e-8
        ):
            raise ValueError("camera rotation is not proper")
        registered[key] = {
            "name": Path(view["path"]).name,
            "R": R,
            "C": C,
            "errors": [],
            "positive": [],
        }
    if len(registered) / len(views) < minimum_coverage:
        reasons.append("insufficient registered camera coverage")
    landmark_ids = set()
    for point in landmarks:
        landmark_id = str(point["landmarkId"])
        if landmark_id in landmark_ids or len(point.get("observations", [])) < 2:
            raise ValueError("duplicate or untriangulated landmark")
        landmark_ids.add(landmark_id)
        X = _array(point["X"], 3)
        observed = set()
        for obs in point.get("observations", []):
            key = str(obs["observationId"])
            observation_count += 1
            if (
                observation_count > 16_000_000
                or key in observed
                or key not in registered
            ):
                raise ValueError("invalid bounded landmark observations")
            observed.add(key)
            camera = registered[key]
            _, _, p, k = converted[str(vmap[key]["intrinsicId"])]
            q = camera["R"] @ (X - camera["C"])
            valid = bool(q[2] > 1e-12)
            positive.append(valid)
            camera["positive"].append(valid)
            if not valid:
                reasons.append("landmark behind camera")
                continue
            n = q[:2] / q[2]
            rr = float(n @ n)
            gain = 1 + k[0] * rr + k[1] * rr**2 + k[2] * rr**3
            projected = n * gain * p[:2] + p[2:]
            error = float(np.linalg.norm(projected - _array(obs["x"], 2)))
            if not math.isfinite(error):
                raise ValueError("nonfinite reprojection")
            errors.append(error)
            camera["errors"].append(error)
    per_view = []
    for camera in sorted(registered.values(), key=lambda v: v["name"]):
        summary = _stats(camera["errors"])
        if summary is None or summary["count"] < minimum_observations_per_view:
            reasons.append("insufficient distinct landmark support per registered view")
        elif summary["p95"] > maximum_reprojection_p95:
            reasons.append("per-view reprojection exceeds declared bound")
        per_view.append(
            {
                "name": camera["name"],
                "observations": len(camera["positive"]),
                "reprojection_pixels": summary,
                "positive_depth_fraction": float(np.mean(camera["positive"]))
                if camera["positive"]
                else None,
            }
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("native scene changed during audit")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "scene": str(path.absolute()),
        "scene_sha256": sha,
        "input_images": len(views),
        "registered_cameras": len(registered),
        "coverage": len(registered) / len(views),
        "missing_names": sorted(set(names) - {v["name"] for v in registered.values()}),
        "landmarks": len(landmarks),
        "observations": observation_count,
        "reprojection_pixels": _stats(errors),
        "per_view": per_view,
        "lenses": lens,
        "positive_depth_fraction": float(np.mean(positive)) if positive else None,
        "policy": {
            "minimum_coverage": minimum_coverage,
            "minimum_observations_per_view": minimum_observations_per_view,
            "maximum_per_view_reprojection_p95_pixels": maximum_reprojection_p95,
            "fixed_calibration_required": expected_calibration is not None,
        },
        "reference_geometry_used": False,
        "shape_accuracy_claim": False,
        "source_unchanged": True,
    }
