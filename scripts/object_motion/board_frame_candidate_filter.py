"""Synthetic-first fixed-board-pose candidate filter; no I/O or model export.

Inputs are original distorted OpenCV integer-center observations. Accepted
observations are virtual undistorted integer-center pixels for a future,
separately staged PINHOLE image/mask contract. Coarse-mask support is not
proof of object-only geometry.
"""

from dataclasses import dataclass
import math
import re
from typing import Sequence

import cv2
import numpy as np

from scripts.object_motion.board_pose_sparse_contract import ObjectTrack, Observation


@dataclass(frozen=True)
class CalibratedFrame:
    name: str
    width: int
    height: int
    k: tuple[float, float, float, float]
    distortion: tuple[float, float, float, float, float]
    # Camera from checkerboard: x_cam = R @ x_board + t.
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]


@dataclass(frozen=True)
class FilterGates:
    min_posed_views: int = 3
    min_parallax_degrees: float = 1.0
    max_reprojection_pixels: float = 2.0
    min_signed_height_squares: float = 1.0
    board_half_width_squares: float = 5.0
    board_half_height_squares: float = 4.5
    min_camera_plane_distance_squares: float = 5.0


GATES = FilterGates()


@dataclass(frozen=True)
class AcceptedTrack:
    track_id: int
    virtual_observations: tuple[Observation, ...]
    xyz_board_squares: tuple[float, float, float]
    signed_height_squares: float
    parallax_degrees: float
    max_distorted_reprojection_px: float
    max_virtual_reprojection_px: float


@dataclass(frozen=True)
class FilterResult:
    input_tracks: int
    accepted: tuple[AcceptedTrack, ...]
    rejected_by_reason: dict[str, int]
    dropped_unposed_observations: int
    camera_facing_sign: int


class GeometryReject(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _camera(frame: CalibratedFrame):
    if frame.width < 2 or frame.height < 2 or not frame.name:
        raise ValueError('invalid exact frame name or size')
    k = np.asarray(frame.k, dtype=np.float64)
    d = np.asarray(frame.distortion, dtype=np.float64)
    r = np.asarray(frame.rotation, dtype=np.float64)
    t = np.asarray(frame.translation, dtype=np.float64)
    if (k.shape != (4,) or d.shape != (5,) or r.shape != (3, 3) or t.shape != (3,) or
            not np.isfinite(k).all() or not np.isfinite(d).all() or
            not np.isfinite(r).all() or not np.isfinite(t).all() or
            min(k[:2]) <= 0 or not (0 <= k[2] < frame.width and 0 <= k[3] < frame.height) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(r), 1, atol=1e-4)):
        raise ValueError(f'invalid calibrated camera: {frame.name}')
    matrix = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=np.float64)
    return matrix, d, r, t, -r.T @ t


def prepare_frames(frames: Sequence[CalibratedFrame], expected_count: int = 39):
    if len(frames) != expected_count or len({frame.name for frame in frames}) != expected_count:
        raise ValueError('wrong number of exact posed names')
    if expected_count == 39 and any(
            not re.fullmatch(r'NP3_(?:0[0-9]{2}|[12][0-9]{2}|3[0-5][0-9])\.jpg', frame.name) or
            int(frame.name[4:7]) % 6 for frame in frames):
        raise ValueError('non-NP3 posed name in production set')
    prepared = {frame.name: (frame, *_camera(frame)) for frame in frames}
    first_k, first_d = prepared[frames[0].name][1:3]
    if any(not np.array_equal(camera[1], first_k) or not np.array_equal(camera[2], first_d)
           for camera in prepared.values()):
        raise ValueError('all posed NP3 views must share frozen K/distortion')
    signed = np.array([camera[5][2] for camera in prepared.values()], dtype=np.float64)
    middle = float(np.median(signed))
    if (not np.isfinite(middle) or abs(middle) < GATES.min_camera_plane_distance_squares or
            np.any(signed * middle <= 0)):
        raise ValueError('camera-facing board side unavailable')
    return prepared, 1 if middle > 0 else -1


def _undistort(observation: Observation, camera):
    frame, k, d, _, _, _ = camera
    raw = np.array([observation.x, observation.y], dtype=np.float64)
    if (not np.isfinite(raw).all() or not (0 <= raw[0] < frame.width and 0 <= raw[1] < frame.height)):
        raise GeometryReject('raw_coordinate_invalid')
    normalized = cv2.undistortPoints(raw.reshape(1, 1, 2), k, d).reshape(2)
    virtual = cv2.undistortPoints(raw.reshape(1, 1, 2), k, d, P=k).reshape(2)
    if (not np.isfinite(normalized).all() or not np.isfinite(virtual).all() or
            not (0 <= virtual[0] < frame.width and 0 <= virtual[1] < frame.height)):
        raise GeometryReject('virtual_coordinate_invalid')
    return normalized, virtual


def _triangulate(track: ObjectTrack, observations: list[Observation], prepared,
                 side: int, gates: FilterGates):
    rows, normalized, virtuals, centers = [], [], [], []
    for obs in observations:
        camera = prepared[obs.image_name]
        _, _, _, r, t, center = camera
        normal, virtual = _undistort(obs, camera)
        p = np.column_stack((r, t))
        rows.extend((normal[0]*p[2]-p[0], normal[1]*p[2]-p[1]))
        normalized.append(normal)
        virtuals.append(virtual)
        centers.append(center)
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    point = vh[-1]
    if abs(point[3]) < 1e-12:
        raise GeometryReject('point_at_infinity')
    xyz = point[:3] / point[3]
    if not np.isfinite(xyz).all():
        raise GeometryReject('nonfinite_point')
    distorted_errors, virtual_errors, rays = [], [], []
    for obs, virtual, center in zip(observations, virtuals, centers):
        frame, k, d, r, t, _ = prepared[obs.image_name]
        cam = r @ xyz + t
        if cam[2] <= 0:
            raise GeometryReject('cheirality')
        projected_distorted = cv2.projectPoints(xyz.reshape(1, 3), cv2.Rodrigues(r)[0],
                                                 t, k, d)[0].reshape(2)
        projected_virtual = np.array([k[0, 0]*cam[0]/cam[2]+k[0, 2],
                                      k[1, 1]*cam[1]/cam[2]+k[1, 2]])
        distorted_errors.append(float(np.linalg.norm(projected_distorted - [obs.x, obs.y])))
        virtual_errors.append(float(np.linalg.norm(projected_virtual - virtual)))
        ray = xyz - center
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm <= 1e-12:
            raise GeometryReject('cheirality')
        rays.append(ray / ray_norm)
    if (not np.isfinite(distorted_errors).all() or not np.isfinite(virtual_errors).all() or
            max(distorted_errors) > gates.max_reprojection_pixels or
            max(virtual_errors) > gates.max_reprojection_pixels):
        raise GeometryReject('reprojection')
    # Minimum dot is the maximum pair angle: require at least one usable baseline.
    min_dot = min(float(np.dot(a, b)) for i, a in enumerate(rays) for b in rays[i+1:])
    parallax = math.degrees(math.acos(float(np.clip(min_dot, -1, 1))))
    if parallax < gates.min_parallax_degrees:
        raise GeometryReject('parallax')
    height = side * float(xyz[2])
    if height <= 0:
        raise GeometryReject('on_or_behind_board')
    if height < gates.min_signed_height_squares:
        if (abs(xyz[0]) <= gates.board_half_width_squares and
                abs(xyz[1]) <= gates.board_half_height_squares):
            raise GeometryReject('board_plane_footprint')
        raise GeometryReject('near_plane_outside_footprint')
    virtual_obs = tuple(Observation(obs.image_name, float(xy[0]), float(xy[1]))
                        for obs, xy in zip(observations, virtuals))
    return AcceptedTrack(track.track_id, virtual_obs, tuple(map(float, xyz)),
                         height, parallax, max(distorted_errors), max(virtual_errors))


def filter_candidate_tracks(tracks: Sequence[ObjectTrack], frames: Sequence[CalibratedFrame],
                            all_source_names: set[str], gates: FilterGates = GATES,
                            expected_posed_count: int = 39,
                            expected_source_count: int = 48) -> FilterResult:
    """Classify whole tracks under fixed poses; never modify the source tracks."""
    if (gates.min_posed_views != 3 or gates.min_parallax_degrees != 1.0 or
            gates.max_reprojection_pixels != 2.0 or gates.min_signed_height_squares != 1.0 or
            gates.board_half_width_squares != 5.0 or gates.board_half_height_squares != 4.5 or
            gates.min_camera_plane_distance_squares != 5.0):
        raise ValueError('frozen board/geometry gates changed')
    prepared, side = prepare_frames(frames, expected_posed_count)
    if not set(prepared).issubset(all_source_names) or len(all_source_names) != expected_source_count:
        raise ValueError('posed names not a subset of sealed TRAIN names')
    seen_ids, accepted, reasons, dropped = set(), [], {}, 0
    for track in tracks:
        if track.track_id in seen_ids or track.track_id < 0:
            raise ValueError('duplicate or invalid candidate track ID')
        seen_ids.add(track.track_id)
        names = [obs.image_name for obs in track.observations]
        if len(names) != len(set(names)) or any(name not in all_source_names for name in names):
            raise ValueError('candidate contains duplicate/foreign image observation')
        posed = [obs for obs in track.observations if obs.image_name in prepared]
        dropped += len(track.observations) - len(posed)
        if len(posed) < gates.min_posed_views:
            reason = 'fewer_than_three_posed_views'
        else:
            try:
                accepted.append(_triangulate(track, posed, prepared, side, gates))
                continue
            except GeometryReject as failure:
                reason = failure.reason
        reasons[reason] = reasons.get(reason, 0) + 1
    return FilterResult(len(tracks), tuple(accepted), reasons, dropped, side)
