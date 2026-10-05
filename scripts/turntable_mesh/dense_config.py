"""Every tunable of the all-view dense pipeline in one validated record.

Lengths are relative (voxels, pixels at a pyramid level, fractions of depth)
so the same values apply to scenes of any SfM scale. Per-level tuples have one
entry per pyramid level in ``sizes``; a shorter tuple repeats its last entry.
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class DenseConfig:
    # --- inputs and pyramid
    sizes: tuple = (256, 512, 1024)  # longest canvas side per level; capped at native
    crop_padding: int = 24  # native pixels around each mask box
    stretch_percentiles: tuple = (1.0, 99.0)  # grey range inside the mask mapped to 0..1
    # --- neighbours
    neighbours: int = 6
    best_of: int = 3  # mean of the best N neighbour scores
    minimum_angle: float = 3.0
    maximum_angle: float = 40.0
    # --- matching
    planes: int = 128  # full inverse-depth sweep at the coarsest level
    windows: tuple = (7, 9, 11)
    aggregates: tuple = (1.0, 1.5, 2.0)  # cost blur sigma in level pixels; 0 disables
    passes: tuple = (1, 2, 2)
    band_steps: tuple = (8, 5)  # search half-width in steps, first and later passes
    min_score: float = 0.55
    min_variance: float = 1e-4
    window_fill: float = 0.6  # minimum jointly valid fraction of a window
    # --- cross-view agreement
    tolerances: tuple = (0.006, 0.003, 0.002)  # relative depth
    min_votes: tuple = (2, 3, 3)
    vote_neighbours: int = 10
    # --- silhouette hull
    grid: int = 400  # voxels along the longest hull side
    hull_dilate: int = 2  # native pixels of mask tolerance
    hull_allowed: int = 2  # views that may disagree
    repair_masks: bool = True
    repair_loose: int = 8
    repair_base_margin: float = 8.0  # voxels above the support kept out of repair
    # --- thin parts
    hull_front: bool = True
    hull_front_level: int = 1
    hull_front_min_score: float = 0.6
    hull_front_margin: float = 0.004  # relative depth by which it must be nearer
    # --- fusion
    fallback_level: bool = True
    rim_fraction: float = 0.55  # silhouette band left to the hull, in windows
    truncation_voxels: float = 3.0
    behind_voxels: float = 12.0
    behind_weight: float = 0.25
    # --- surface extraction
    mesh_smooth: float = 1.0  # voxels
    mesh_fill_sigmas: tuple = (2.0, 4.0)
    mesh_final_smooth: float = 0.6
    mesh_minimum_weight: float = 0.5
    mesh_confidence_cap: float = 4.0
    mesh_taubin_cycles: int = 5

    def level(self, name, index):
        values = getattr(self, name)
        return values[min(index, len(values) - 1)]

    def validate(self):
        def need(condition, message):
            if not condition:
                raise ValueError("dense configuration: " + message)

        need(1 <= len(self.sizes) <= 5 and all(32 <= s <= 4096 for s in self.sizes), "sizes must be 1..5 values in 32..4096")
        need(list(self.sizes) == sorted(self.sizes), "sizes must increase")
        need(2 <= self.best_of <= self.neighbours <= 16, "need 2 <= best_of <= neighbours <= 16")
        need(0 < self.minimum_angle < self.maximum_angle < 90, "invalid neighbour angles")
        need(16 <= self.planes <= 512, "planes must be 16..512")
        need(all(w % 2 == 1 and 3 <= w <= 31 for w in self.windows), "windows must be odd, 3..31")
        need(all(1 <= p <= 4 for p in self.passes), "passes must be 1..4")
        need(len(self.band_steps) == 2 and all(2 <= b <= 32 for b in self.band_steps), "band_steps must be two values in 2..32")
        need(0 < self.min_score < 1 and 0 < self.hull_front_min_score < 1, "scores must be in (0,1)")
        need(self.min_variance > 0 and 0 < self.window_fill <= 1, "invalid texture gates")
        need(all(0 < t < 0.1 for t in self.tolerances), "tolerances must be in (0,0.1)")
        need(all(1 <= v <= self.vote_neighbours for v in self.min_votes) and self.vote_neighbours <= 32, "invalid votes")
        need(64 <= self.grid <= 1024, "grid must be 64..1024")
        need(0 <= self.hull_dilate <= 16 and 0 <= self.hull_allowed <= self.repair_loose <= 64, "invalid hull tolerances")
        need(0 <= self.hull_front_level < len(self.sizes) or not self.hull_front, "hull_front_level outside pyramid")
        need(0 <= self.rim_fraction <= 3 and self.truncation_voxels >= 1, "invalid fusion values")
        need(self.behind_voxels >= self.truncation_voxels and 0 <= self.behind_weight <= 1, "invalid inside vote")
        need(self.mesh_smooth > 0 and self.mesh_confidence_cap > 0 and 0 <= self.mesh_taubin_cycles <= 100, "invalid mesh values")
        return self

    def to_json(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _coerce(field, value):
    default = field.default
    if isinstance(default, tuple):
        if isinstance(value, str):
            value = [v for v in value.replace(",", " ").split() if v]
        kind = type(default[0])
        return tuple(kind(float(v)) if kind is int else kind(v) for v in value)
    if isinstance(default, bool):
        if isinstance(value, str):
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{field.name} needs true or false")
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return type(default)(value)


def build(config_path=None, overrides=()):
    """Defaults, then an optional JSON file, then ``key=value`` overrides."""
    known = {f.name: f for f in fields(DenseConfig)}
    values = {}
    if config_path is not None:
        loaded = json.loads(Path(config_path).read_text())
        loaded = loaded.get("configuration", loaded)
        for key, value in loaded.items():
            if key in known:
                values[key] = _coerce(known[key], value)
    for item in overrides:
        key, separator, value = item.partition("=")
        key = key.strip().replace("-", "_")
        if not separator or key not in known:
            raise ValueError(f"unknown setting {item!r}; known: {', '.join(sorted(known))}")
        values[key] = _coerce(known[key], value)
    return DenseConfig(**values).validate()


def add_arguments(parser):
    parser.add_argument("--config", type=Path, help="JSON with DenseConfig fields (a previous result.json works)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one setting, e.g. --set grid=320 --set sizes=256,512")


def describe():
    """One line per setting with its default; used by --list-settings."""
    return "\n".join(f"{f.name} = {list(f.default) if isinstance(f.default, tuple) else f.default}" for f in fields(DenseConfig))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.set).to_json(), indent=2))


if __name__ == "__main__":
    main()
