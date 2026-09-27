"""Read-only verified-index DB edges to candidate mask-supported object tracks.

This never reads saved SfM points or cameras and never estimates camera poses.
Input database keypoints are COLMAP pixel coordinates; output observations use
OpenCV integer-center coordinates (subtract 0.5 on both axes).
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Sequence

import numpy as np
from PIL import Image

from scripts.object_motion.board_pose_sparse_contract import ObjectTrack, Observation


MAX_IMAGE_ID = 2_147_483_647
MAX_DATABASE_BYTES = 20 * 1024**2
MAX_KEYPOINTS_PER_IMAGE = 10_000
MAX_GEOMETRY_ROWS = 1_128
MAX_VERIFIED_EDGES = 1_000_000
MAX_GRAPH_NODES = 200_000
MAX_COMPONENT_NODES = 24
MAX_SECONDS = 60


class TrackAdapterError(ValueError):
    """A sealed source or bounded verified graph violated the contract."""


@dataclass(frozen=True)
class ImageSource:
    name: str
    image_sha256: str
    mask_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class TrackCandidates:
    tracks: tuple[ObjectTrack, ...]
    database_sha256: str
    image_count: int
    geometry_rows: int
    empty_geometry_rows: int
    verified_edges: int
    mask_supported_edges: int
    components: int
    rejected_conflicting_components: int
    rejected_oversize_components: int
    rejected_short_components: int


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sources(image_dir: Path, mask_dir: Path, sources: Sequence[ImageSource], required_images: int):
    if not 3 <= required_images <= 48 or len(sources) != required_images:
        raise TrackAdapterError("wrong exact image count")
    if image_dir.is_symlink() or mask_dir.is_symlink() or not image_dir.is_dir() or not mask_dir.is_dir():
        raise TrackAdapterError("missing or linked source directory")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise TrackAdapterError("duplicate manifest image name")
    if {p.name for p in image_dir.iterdir()} != set(names) or \
            {p.name for p in mask_dir.iterdir()} != {name + ".png" for name in names}:
        raise TrackAdapterError("source inventory differs from exact manifest")
    masks = {}
    for source in sources:
        name = source.name
        if (not name or Path(name).name != name or "\\" in name or
                not name.lower().endswith((".jpg", ".jpeg")) or
                source.width < 2 or source.height < 2 or
                not _sha_text(source.image_sha256) or not _sha_text(source.mask_sha256)):
            raise TrackAdapterError(f"invalid manifest entry: {name!r}")
        photo, mask = image_dir / name, mask_dir / (name + ".png")
        if (photo.is_symlink() or mask.is_symlink() or not photo.is_file() or not mask.is_file() or
                _digest(photo) != source.image_sha256 or _digest(mask) != source.mask_sha256):
            raise TrackAdapterError(f"source hash or file mismatch: {name}")
        with Image.open(photo) as rgb, Image.open(mask) as support:
            if (rgb.format != "JPEG" or rgb.mode != "RGB" or
                    rgb.size != (source.width, source.height) or
                    support.format != "PNG" or support.mode != "L" or support.size != rgb.size):
                raise TrackAdapterError(f"image/mask format or size mismatch: {name}")
            pixels = np.asarray(support).copy()
        if not np.isin(pixels, (0, 255)).all():
            raise TrackAdapterError(f"mask not binary: {name}")
        masks[name] = pixels
    return masks


class _Components:
    def __init__(self):
        self.parent = {}
        self.nodes = {}
        self.sizes = {}
        self.images = {}
        self.conflict = {}
        self.oversize = {}

    def _add(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.nodes[node] = {node}
            self.sizes[node] = 1
            self.images[node] = {node[0]}
            self.conflict[node] = False
            self.oversize[node] = False

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while node != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left, right):
        self._add(left)
        self._add(right)
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # Stable root independent of hash randomization or edge encounter order.
        if b < a:
            a, b = b, a
        conflict = self.conflict[a] or self.conflict[b] or bool(self.images[a] & self.images[b])
        merged_size = self.sizes[a] + self.sizes[b]
        # Keep only a bounded witness once a component is irreversibly invalid.
        merged = self.nodes[a] | self.nodes[b]
        self.parent[b] = a
        self.nodes[a] = set(sorted(merged)[:MAX_COMPONENT_NODES + 1])
        self.sizes[a] = merged_size
        self.images[a] |= self.images[b]
        self.conflict[a] = conflict
        self.oversize[a] = self.oversize[a] or self.oversize[b] or merged_size > MAX_COMPONENT_NODES
        for mapping in (self.nodes, self.sizes, self.images, self.conflict, self.oversize):
            del mapping[b]


def extract_candidate_tracks(
    database: Path,
    expected_database_sha256: str,
    image_dir: Path,
    mask_dir: Path,
    sources: Sequence[ImageSource],
    *,
    required_images: int = 48,
) -> TrackCandidates:
    """Extract deterministic candidates from verified edges only, without writes.

    ``required_images`` differs from 48 only for synthetic tests. A returned
    track is mask-supported, not independently proven to lie on the object.
    """
    database, image_dir, mask_dir = Path(database), Path(image_dir), Path(mask_dir)
    started = time.monotonic()
    if (database.is_symlink() or not database.is_file() or
            not 0 < database.stat().st_size <= MAX_DATABASE_BYTES or
            not _sha_text(expected_database_sha256)):
        raise TrackAdapterError("invalid sealed database path/hash/size")
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise TrackAdapterError("SQLite sidecar present")
    if _digest(database) != expected_database_sha256:
        raise TrackAdapterError("sealed database SHA-256 mismatch")
    masks = _sources(image_dir, mask_dir, sources, required_images)
    manifest = {source.name: source for source in sources}
    components = _Components()
    geometry_rows = empty_geometry_rows = verified_edges = supported_edges = edge_ordinal = 0
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise TrackAdapterError("SQLite integrity check failed")
        image_rows = db.execute("SELECT image_id,name FROM images ORDER BY image_id").fetchall()
        if (len(image_rows) != required_images or len({row[0] for row in image_rows}) != required_images or
                {row[1] for row in image_rows} != set(manifest) or
                any(not isinstance(row[0], int) or not 0 < row[0] < MAX_IMAGE_ID for row in image_rows)):
            raise TrackAdapterError("database image IDs/names differ from exact manifest")
        id_to_name = dict(image_rows)
        if db.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0] != required_images:
            raise TrackAdapterError("keypoint inventory differs from images")
        keypoints = {}
        for image_id, name in image_rows:
            row = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            if (row is None or not isinstance(row[0], int) or not 0 <= row[0] <= MAX_KEYPOINTS_PER_IMAGE or
                    row[1] not in (2, 4, 6) or not isinstance(row[2], bytes) or
                    len(row[2]) != row[0] * row[1] * 4):
                raise TrackAdapterError(f"invalid bounded keypoint blob: {name}")
            xy = np.frombuffer(row[2], dtype="<f4").reshape(row[0], row[1])[:, :2].astype(np.float64) - 0.5
            if not np.isfinite(xy).all():
                raise TrackAdapterError(f"nonfinite keypoints: {name}")
            keypoints[image_id] = xy
        rows = db.execute("SELECT pair_id,rows,cols,data,config FROM two_view_geometries ORDER BY pair_id").fetchall()
        if len(rows) > MAX_GEOMETRY_ROWS:
            raise TrackAdapterError("too many geometry rows")
        if len({row[0] for row in rows}) != len(rows):
            raise TrackAdapterError("duplicate geometry pair ID")
        for pair_id, count, columns, blob, config in rows:
            geometry_rows += 1
            if (not isinstance(pair_id, int) or pair_id <= 0 or
                    not isinstance(count, int) or not 0 <= count <= MAX_KEYPOINTS_PER_IMAGE or
                    columns != 2 or not isinstance(config, int)):
                raise TrackAdapterError("invalid verified geometry row")
            if count == 0:
                if blob is not None or config != 0:
                    raise TrackAdapterError("invalid verified geometry row")
            elif not isinstance(blob, bytes) or len(blob) != count * 8 or config <= 0:
                raise TrackAdapterError("invalid verified geometry row")
            first, second = divmod(pair_id, MAX_IMAGE_ID)
            if first not in id_to_name or second not in id_to_name or first >= second:
                raise TrackAdapterError("invalid COLMAP pair ID")
            if count == 0:
                empty_geometry_rows += 1
                continue
            matches = np.frombuffer(blob, dtype="<u4").reshape(count, 2)
            if (np.any(matches[:, 0] >= len(keypoints[first])) or
                    np.any(matches[:, 1] >= len(keypoints[second])) or
                    len(np.unique(matches, axis=0)) != count):
                raise TrackAdapterError("invalid verified feature-index edge")
            verified_edges += count
            if verified_edges > MAX_VERIFIED_EDGES:
                raise TrackAdapterError("verified edge resource cap exceeded")
            for left_index, right_index in matches:
                edge_ordinal += 1
                left, right = (first, int(left_index)), (second, int(right_index))
                acceptable = True
                for image_id, index in (left, right):
                    name = id_to_name[image_id]
                    x, y = keypoints[image_id][index]
                    source = manifest[name]
                    if (not 0 <= x < source.width or not 0 <= y < source.height or
                            masks[name][int(y), int(x)] == 0):
                        acceptable = False
                        break
                if acceptable:
                    components.union(left, right)
                    supported_edges += 1
                    if len(components.parent) > MAX_GRAPH_NODES:
                        raise TrackAdapterError("graph node resource cap exceeded")
                if edge_ordinal % 8192 == 0 and time.monotonic() - started > MAX_SECONDS:
                    raise TrackAdapterError("extraction wall-time cap exceeded")
    if time.monotonic() - started > MAX_SECONDS:
        raise TrackAdapterError("extraction wall-time cap exceeded")
    if _digest(database) != expected_database_sha256:
        raise TrackAdapterError("sealed database changed during extraction")
    for source in sources:
        if (_digest(image_dir / source.name) != source.image_sha256 or
                _digest(mask_dir / (source.name + ".png")) != source.mask_sha256):
            raise TrackAdapterError("source image/mask changed during extraction")
    if time.monotonic() - started > MAX_SECONDS:
        raise TrackAdapterError("extraction wall-time cap exceeded")
    tracks = []
    conflicting = oversized = short = 0
    for root in sorted(components.nodes):
        if components.conflict[root]:
            conflicting += 1
            continue
        if components.oversize[root]:
            oversized += 1
            continue
        nodes = sorted(components.nodes[root], key=lambda node: (id_to_name[node[0]], node[1]))
        if len(nodes) < 3:
            short += 1
            continue
        observations = tuple(Observation(id_to_name[image_id], *map(float, keypoints[image_id][index]))
                             for image_id, index in nodes)
        tracks.append((nodes, observations))
    tracks.sort(key=lambda item: item[0])
    return TrackCandidates(tuple(ObjectTrack(i, obs) for i, (_, obs) in enumerate(tracks)),
                           expected_database_sha256, required_images, geometry_rows, empty_geometry_rows,
                           verified_edges, supported_edges, len(components.nodes),
                           conflicting, oversized, short)
