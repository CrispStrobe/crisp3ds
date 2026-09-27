"""Read-only, fixed-pose object-track ingestion for a COLMAP sparse model.

Coordinates in the input are OpenCV integer-pixel-center coordinates. World is
the rigid checkerboard frame, not a supplied object-dataset reference frame.
No feature extraction, pose estimation, dense reconstruction, or disk writing
occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
from PIL import Image as PILImage


class ContractError(ValueError):
    """An input or triangulated point violated the fixed-pose contract."""


@dataclass(frozen=True)
class Frame:
    name: str
    image_sha256: str
    mask_sha256: str
    width: int
    height: int
    # fx, fy, cx, cy, with (0, 0) at the center of the upper-left pixel.
    k: tuple[float, float, float, float]
    # x_camera = R @ x_board + t; R is a proper, right-handed rotation.
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    image_name: str
    x: float
    y: float


@dataclass(frozen=True)
class ObjectTrack:
    track_id: int
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Gates:
    min_views: int = 3
    min_tracks: int = 8
    min_parallax_degrees: float = 1.0
    max_reprojection_pixels: float = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"not a regular non-symlink file: {path}")


def _validate_frame(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    if (not frame.name or Path(frame.name).name != frame.name or "\\" in frame.name
            or not frame.name.lower().endswith((".jpg", ".jpeg"))):
        raise ContractError(f"invalid exact JPEG name: {frame.name!r}")
    if frame.width < 2 or frame.height < 2:
        raise ContractError(f"invalid dimensions: {frame.name}")
    if len(frame.k) != 4 or not np.isfinite(frame.k).all():
        raise ContractError(f"invalid K: {frame.name}")
    fx, fy, cx, cy = frame.k
    if fx <= 0 or fy <= 0 or not (0 <= cx < frame.width) or not (0 <= cy < frame.height):
        raise ContractError(f"invalid PINHOLE K: {frame.name}")
    r = np.asarray(frame.rotation, dtype=np.float64)
    t = np.asarray(frame.translation, dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ContractError(f"invalid camera_from_board: {frame.name}")
    if np.max(np.abs(r.T @ r - np.eye(3))) > 1e-4 or abs(np.linalg.det(r) - 1) > 1e-4:
        raise ContractError(f"not a right-handed rotation: {frame.name}")
    for digest in (frame.image_sha256, frame.mask_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ContractError(f"invalid SHA-256: {frame.name}")
    return r, t


def _triangulate(track: ObjectTrack, frames: dict[str, Frame], poses: dict[str, tuple[np.ndarray, np.ndarray]], gates: Gates) -> np.ndarray:
    rows = []
    centers = []
    for obs in track.observations:
        frame = frames[obs.image_name]
        r, t = poses[obs.image_name]
        fx, fy, cx, cy = frame.k
        u, v = (obs.x - cx) / fx, (obs.y - cy) / fy
        p = np.column_stack((r, t))
        rows.extend((u * p[2] - p[0], v * p[2] - p[1]))
        centers.append(-r.T @ t)
    _, _, vh = np.linalg.svd(np.asarray(rows))
    xh = vh[-1]
    if abs(xh[3]) < 1e-12:
        raise ContractError(f"point at infinity: track {track.track_id}")
    xyz = xh[:3] / xh[3]
    if not np.isfinite(xyz).all():
        raise ContractError(f"nonfinite point: track {track.track_id}")
    rays = []
    for obs, center in zip(track.observations, centers):
        frame = frames[obs.image_name]
        r, t = poses[obs.image_name]
        cam = r @ xyz + t
        if cam[2] <= 0:
            raise ContractError(f"cheirality: track {track.track_id}")
        fx, fy, cx, cy = frame.k
        projected = np.array((fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy))
        if np.linalg.norm(projected - (obs.x, obs.y)) > gates.max_reprojection_pixels:
            raise ContractError(f"reprojection: track {track.track_id}")
        ray = xyz - center
        rays.append(ray / np.linalg.norm(ray))
    # The minimum cosine is the *maximum* pairwise viewing-ray angle.
    cos_min = min(float(np.dot(a, b)) for i, a in enumerate(rays) for b in rays[i + 1:])
    if math.degrees(math.acos(np.clip(cos_min, -1, 1))) < gates.min_parallax_degrees:
        raise ContractError(f"parallax: track {track.track_id}")
    return xyz


def build_sparse_model(
    image_dir: Path,
    mask_dir: Path,
    frame_sequence: Sequence[Frame],
    tracks: Sequence[ObjectTrack],
    gates: Gates = Gates(),
):
    """Validate original RGB/masks and tracks, then return an in-memory model.

    Mask files are named ``<exact JPEG name>.png`` and nonzero means object.
    All listed source hashes, inventory, pixels, poses, and tracks are checked.
    The model is never written by this function.
    """
    import pycolmap

    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if image_dir.is_symlink() or mask_dir.is_symlink() or not image_dir.is_dir() or not mask_dir.is_dir():
        raise ContractError("image/mask directories must exist and not be symlinks")
    frames = {f.name: f for f in frame_sequence}
    if len(frames) != len(frame_sequence) or len(frames) < 3:
        raise ContractError("need at least three uniquely named frames")
    if gates.min_views < 2 or gates.min_tracks < 1 or gates.min_parallax_degrees <= 0 or gates.max_reprojection_pixels <= 0:
        raise ContractError("invalid fixed gates")
    poses = {name: _validate_frame(frame) for name, frame in frames.items()}
    if {p.name for p in image_dir.iterdir()} != set(frames):
        raise ContractError("RGB inventory differs from manifest")
    if {p.name for p in mask_dir.iterdir()} != {name + ".png" for name in frames}:
        raise ContractError("mask inventory differs from manifest")
    rgb, masks = {}, {}
    for name, frame in frames.items():
        source, mask_path = image_dir / name, mask_dir / (name + ".png")
        _regular_file(source)
        _regular_file(mask_path)
        if _sha256(source) != frame.image_sha256 or _sha256(mask_path) != frame.mask_sha256:
            raise ContractError(f"source hash mismatch: {name}")
        with PILImage.open(source) as photo, PILImage.open(mask_path) as mask:
            if photo.format != "JPEG" or photo.mode != "RGB" or photo.size != (frame.width, frame.height):
                raise ContractError(f"original RGB mismatch: {name}")
            if mask.format != "PNG" or mask.size != photo.size or mask.mode not in ("L", "1"):
                raise ContractError(f"mask mismatch: {name}")
            rgb[name], masks[name] = np.asarray(photo).copy(), np.asarray(mask).copy()
    if len(tracks) < gates.min_tracks or len({t.track_id for t in tracks}) != len(tracks):
        raise ContractError("too few or duplicate track IDs")
    point_xyz = {}
    per_image: dict[str, list[tuple[int, Observation]]] = {name: [] for name in frames}
    claimed = set()
    for track in tracks:
        if track.track_id < 0 or len(track.observations) < gates.min_views:
            raise ContractError(f"insufficient track: {track.track_id}")
        if len({o.image_name for o in track.observations}) != len(track.observations):
            raise ContractError(f"repeated image in track: {track.track_id}")
        for obs in track.observations:
            if obs.image_name not in frames or not math.isfinite(obs.x) or not math.isfinite(obs.y):
                raise ContractError(f"invalid observation: {track.track_id}")
            frame = frames[obs.image_name]
            if not (0 <= obs.x < frame.width and 0 <= obs.y < frame.height):
                raise ContractError(f"out-of-bounds observation: {track.track_id}")
            if not masks[obs.image_name][math.floor(obs.y), math.floor(obs.x)]:
                raise ContractError(f"non-object observation: {track.track_id}")
            key = (obs.image_name, obs.x, obs.y)
            if key in claimed:
                raise ContractError(f"reused observation: {track.track_id}")
            claimed.add(key)
            per_image[obs.image_name].append((track.track_id, obs))
        point_xyz[track.track_id] = _triangulate(track, frames, poses, gates)

    model = pycolmap.Reconstruction()
    image_ids, point_indices = {}, {}
    for image_id, name in enumerate(sorted(frames), start=1):
        frame = frames[name]
        fx, fy, cx, cy = frame.k
        camera = pycolmap.Camera(model="PINHOLE", width=frame.width, height=frame.height,
                                  params=[fx, fy, cx + 0.5, cy + 0.5], camera_id=image_id)
        model.add_camera(camera)
        r, t = poses[name]
        ordered = sorted(per_image[name], key=lambda pair: pair[0])
        keypoints = np.asarray([[o.x + 0.5, o.y + 0.5] for _, o in ordered], dtype=np.float64).reshape((-1, 2))
        image = pycolmap.Image(name=name, keypoints=keypoints,
                                cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(r), t),
                                camera_id=image_id, id=image_id)
        model.add_image(image)
        model.register_image(image_id)
        image_ids[name] = image_id
        point_indices.update({(name, track_id): i for i, (track_id, _) in enumerate(ordered)})
    for track in tracks:
        elements = [pycolmap.TrackElement(image_ids[o.image_name], point_indices[o.image_name, track.track_id])
                    for o in track.observations]
        first = track.observations[0]
        color = rgb[first.image_name][math.floor(first.y), math.floor(first.x)]
        model.add_point3D(point_xyz[track.track_id], pycolmap.Track(elements), np.asarray(color, dtype=np.uint8))
    return model


def write_sparse_model(model, output_dir: Path, source_dir: Path, min_free_bytes: int = 10 * 1024**3) -> None:
    """Explicit, fresh-directory export; never invoked by ``build_sparse_model``.

    Source and destination volumes must each retain more than 10 GiB free.
    This does not export RGB/masks or start undistortion/dense reconstruction.
    """
    output_dir, source_dir = Path(output_dir), Path(source_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ContractError("sparse output must be a fresh path")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir() or not source_dir.is_dir():
        raise ContractError("source/output parent must be existing directories")
    if min_free_bytes < 10 * 1024**3:
        raise ContractError("disk reserve cannot be lowered below 10 GiB")
    for path in (source_dir, output_dir.parent):
        if shutil.disk_usage(path).free <= min_free_bytes:
            raise ContractError(f"insufficient disk reserve: {path}")
    output_dir.mkdir()
    try:
        model.write_binary(str(output_dir))
    except Exception:
        # Leave any partial output intact for inspection; never silently retry.
        raise
    if shutil.disk_usage(output_dir).free <= min_free_bytes:
        raise ContractError("disk reserve breached after sparse export")
