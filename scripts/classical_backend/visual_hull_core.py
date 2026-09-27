"""Pure geometry guards for a proposed image-only visual-hull control.

No dataset, scanner, pose-reference, or file I/O is performed here. Input
camera convention is COLMAP ``x_camera = R @ x_world + t`` and COLMAP pixel
coordinates have integer pixel edges (pixel-center coordinates end in .5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeView:
    rotation: np.ndarray
    translation: np.ndarray
    k: tuple[float, float, float, float]
    mask: np.ndarray

    def __post_init__(self):
        r = np.asarray(self.rotation, dtype=np.float64).copy()
        t = np.asarray(self.translation, dtype=np.float64).copy()
        k = np.asarray(self.k, dtype=np.float64)
        mask = np.asarray(self.mask).copy()
        if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or
                not np.isfinite(t).all() or np.max(np.abs(r.T @ r - np.eye(3))) > 1e-5 or
                abs(np.linalg.det(r) - 1) > 1e-5 or k.shape != (4,) or not np.isfinite(k).all() or
                min(k[:2]) <= 0 or mask.ndim != 2 or mask.dtype != np.uint8 or
                not 0 <= k[2] < mask.shape[1] or not 0 <= k[3] < mask.shape[0] or
                not np.isin(mask, (0, 255)).all()):
            raise ValueError("invalid fixed PINHOLE camera or binary mask")
        r.setflags(write=False)
        t.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation", t)
        object.__setattr__(self, "k", tuple(float(value) for value in k))
        object.__setattr__(self, "mask", mask)


def sparse_cube(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Freeze a scanner-free cube from 1/99% sparse-point quantiles.

    The side is 1.5 times the largest quantile span, centered on those bounds.
    This rule is fixed before seeing any visual-hull or scanner result.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("sparse points must be at least eight finite XYZ rows")
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    side = 1.5 * float(np.max(upper - lower))
    if not np.isfinite(side) or side <= 0:
        raise ValueError("sparse cube has zero or nonfinite span")
    return (lower + upper) / 2, side


def _projection(points: np.ndarray, view: PinholeView) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("invalid finite world points")
    r, t, mask = view.rotation, view.translation, view.mask
    camera = points @ r.T + t
    positive = camera[:, 2] > 0
    u = np.full(len(points), np.nan)
    v = np.full(len(points), np.nan)
    fx, fy, cx, cy = view.k
    u[positive] = fx * camera[positive, 0] / camera[positive, 2] + cx
    v[positive] = fy * camera[positive, 1] / camera[positive, 2] + cy
    visible = positive & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    supported = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(visible)
    supported[indices] = mask[np.floor(v[indices]).astype(int), np.floor(u[indices]).astype(int)] == 255
    return visible, supported


def carve_chunk(points: np.ndarray, views: tuple[PinholeView, ...], *, min_visible: int) -> tuple[np.ndarray, np.ndarray]:
    """A voxel survives only if every visible view supports it and >=N see it."""
    if not views or len(views) > 255 or not 1 <= min_visible <= len(views) or len(points) > 65_536:
        raise ValueError("invalid view count or voxel chunk cap")
    visible_count = np.zeros(len(points), dtype=np.uint8)
    accepted = np.ones(len(points), dtype=bool)
    for view in views:
        visible, supported = _projection(points, view)
        visible_count += visible.astype(np.uint8)
        accepted &= ~visible | supported
    return accepted & (visible_count >= min_visible), visible_count


def voxel_centers(start: int, stop: int, size: int, center: np.ndarray, side: float) -> np.ndarray:
    """Lexicographic XYZ cell centers for a cubic grid."""
    if not 0 <= start <= stop <= size**3 or size < 2 or not np.isfinite(side) or side <= 0:
        raise ValueError("invalid fixed voxel grid")
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("invalid voxel-grid center")
    index = np.arange(start, stop, dtype=np.int64)
    ijk = np.column_stack((index // (size * size), (index // size) % size, index % size))
    return center - side / 2 + (ijk + 0.5) * (side / size)


def boundary_touched(occupied: np.ndarray) -> bool:
    occupied = np.asarray(occupied)
    if occupied.ndim != 3 or len(set(occupied.shape)) != 1 or occupied.dtype != bool:
        raise ValueError("occupancy must be a cubic boolean grid")
    return bool(np.any(occupied[0]) or np.any(occupied[-1]) or
                np.any(occupied[:, 0]) or np.any(occupied[:, -1]) or
                np.any(occupied[:, :, 0]) or np.any(occupied[:, :, -1]))


def exposed_cube_mesh(occupied: np.ndarray, center: np.ndarray, side: float,
                      *, max_triangles: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Emit only exposed cube faces, outward wound, without surface smoothing."""
    occupied = np.asarray(occupied)
    if occupied.ndim != 3 or len(set(occupied.shape)) != 1 or occupied.dtype != bool:
        raise ValueError("occupancy must be a cubic boolean grid")
    size = occupied.shape[0]
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(side) or side <= 0:
        raise ValueError("invalid mesh lattice")
    if not occupied.any() or boundary_touched(occupied):
        raise ValueError("empty or grid-truncated hull abstains")
    padded = np.pad(occupied, 1)
    # Corner offsets for +X,-X,+Y,-Y,+Z,-Z, each with outward winding.
    corners = (
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    )
    neighbors = (
        padded[2:, 1:-1, 1:-1], padded[:-2, 1:-1, 1:-1],
        padded[1:-1, 2:, 1:-1], padded[1:-1, :-2, 1:-1],
        padded[1:-1, 1:-1, 2:], padded[1:-1, 1:-1, :-2],
    )
    vertex_ids = {}
    triangles = []
    for neighbor, offsets in zip(neighbors, corners):
        exposed = np.argwhere(occupied & ~neighbor)
        if 2 * (len(triangles) // 2 + len(exposed)) > max_triangles:
            raise ValueError("exposed-face triangle cap exceeded")
        for cell in exposed:
            quad = []
            for offset in offsets:
                lattice = tuple(int(cell[axis]) + offset[axis] for axis in range(3))
                if lattice not in vertex_ids:
                    vertex_ids[lattice] = len(vertex_ids)
                quad.append(vertex_ids[lattice])
            triangles.append((quad[0], quad[1], quad[2]))
            triangles.append((quad[0], quad[2], quad[3]))
    lattice = np.asarray(list(vertex_ids), dtype=np.float64)
    vertices = center - side / 2 + lattice * (side / size)
    return vertices, np.asarray(triangles, dtype=np.int32)
