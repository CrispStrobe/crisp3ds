"""Coarse-to-fine masked multi-view stereo over every registered view, with TSDF.

Differences from plane_sweep.py, each aimed at a measured failure on dark,
weakly textured turntable objects:

* grey values are contrast-normalised inside the mask, so the texture gate is
  relative to the object's own range instead of an absolute 8-bit variance;
* every registered view is used, so matching neighbours are a few degrees apart;
* a full sweep runs only at the coarsest level; finer levels search a narrow
  band around the smoothed coarser depth and warp along that surface;
* a silhouette hull bounds every hypothesis, and masks that single views got
  wrong are repaired from the multi-view consensus before the hull is final;
* thin parts get a depth candidate from the hull's front surface;
* cross-view depth agreement filters each level;
* depth is fused into a truncated signed distance volume instead of oriented
  points, so no per-pixel normals are needed.

Photo-derived cameras and masks only. No reference geometry is read. Every
tunable lives in dense_config.DenseConfig.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .dense_config import DenseConfig, add_arguments, build


def _gauss(x, sigma, torch, F):
    radius = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k[None, None, None])
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k[None, None, :, None])


class Hull:
    def __init__(self, occupied, origin, voxel, torch):
        self.shape = tuple(occupied.shape)
        self.flat = occupied.reshape(-1)
        self.origin = origin
        self.voxel = voxel
        self.limit = torch.tensor(self.shape, device=occupied.device)

    def contains(self, world):
        index = ((world - self.origin) / self.voxel).floor().long()
        inside = ((index >= 0) & (index < self.limit)).all(-1)
        nx, ny, nz = self.shape
        linear = (index[..., 0] * ny + index[..., 1]) * nz + index[..., 2]
        return inside & self.flat[linear.clamp(0, self.flat.numel() - 1)]

    def centres(self, stride=1):
        return (self.flat.reshape(self.shape).nonzero()[::stride].float() + 0.5) * self.voxel + self.origin


def carve(cameras, masks, origin, shape, voxel, torch, dev, candidate=None):
    """Count silhouette violations per voxel centre.

    A mask that stays clear of the image border shows the whole object, so a
    voxel projecting outside that image is a violation; otherwise it is neutral.
    With ``candidate`` (a coarser Hull) only voxels inside it are projected; the
    rest are reported as violating every view.
    """
    nx, ny, nz = shape
    start_value = 0 if candidate is None else len(cameras)
    violations = torch.full(shape, start_value, dtype=torch.int16, device=dev)
    ys = origin[1] + (torch.arange(ny, device=dev) + 0.5) * voxel
    zs = origin[2] + (torch.arange(nz, device=dev) + 0.5) * voxel
    whole = [not bool(m[0].any() | m[-1].any() | m[:, 0].any() | m[:, -1].any()) for m in masks]
    slab = max(1, 1_500_000 // (ny * nz))
    for start in range(0, nx, slab):
        xs = origin[0] + (torch.arange(start, min(nx, start + slab), device=dev) + 0.5) * voxel
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), -1)
        if candidate is None:
            points, where = grid.reshape(-1, 3), None
        else:
            where = candidate.contains(grid).reshape(-1).nonzero()[:, 0]
            if not len(where):
                continue
            points = grid.reshape(-1, 3)[where]
        count = torch.zeros(len(points), dtype=torch.int16, device=dev)
        for (R, t, k), m, full in zip(cameras, masks, whole):
            h, w = m.shape
            p = points @ R.T + t
            z = p[:, 2].clamp_min(1e-6)
            u = (p[:, 0] / z * k[0] + k[2] - 0.5).round().long()
            v = (p[:, 1] / z * k[1] + k[3] - 0.5).round().long()
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (p[:, 2] > 0)
            hit = m[v.clamp(0, h - 1), u.clamp(0, w - 1)]
            count += (~(inside & hit) if full else inside & ~hit).to(torch.int16)
        block = violations[start : start + slab]
        if where is None:
            block.copy_(count.reshape(block.shape))
        else:
            flat = block.reshape(-1).clone()
            flat[where] = count
            block.copy_(flat.reshape(block.shape))
    return violations


def ncc(ref, target, valid, window, min_variance, window_fill, torch, F):
    """Jointly masked ZNCC; −2 where unsupported. ref 1×1×H×W, others B×1×H×W."""
    r = (ref - 0.5) * valid
    t = (target - 0.5) * valid
    pooled = F.avg_pool2d(
        torch.cat((valid, r, t, r * (ref - 0.5), t * (target - 0.5), r * (target - 0.5)), 1),
        window, 1, window // 2,
    )
    n, sr, st, srr, stt, srt = pooled.unbind(1)
    d = n.clamp_min(1e-6)
    mr, mt = sr / d, st / d
    vr, vt = (srr / d - mr * mr).clamp_min(0), (stt / d - mt * mt).clamp_min(0)
    score = ((srt / d - mr * mt) / (vr * vt + 1e-12).sqrt()).clamp(-1, 1)
    okay = (valid[:, 0] > 0) & (n >= window_fill) & (vr >= min_variance) & (vt >= min_variance)
    return torch.where(okay, score, torch.full_like(score, -2))


def load_views(inputs):
    """Camera rows with absolute image/mask paths (relative ones resolve to inputs)."""
    inputs = Path(inputs)
    rows = json.loads((inputs / "cameras.json").read_text())["views"]
    for row in rows:
        for key in ("image", "mask"):
            path = Path(row[key])
            row[key] = str(path if path.is_absolute() else inputs / path)
    return rows


class Stereo:
    def __init__(self, inputs, device, config=None):
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        self.config = config = (config or DenseConfig()).validate()
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be explicit cpu, mps or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; no implicit CPU fallback")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; no implicit CPU fallback")
        self.dev = torch.device(device)
        self.rows = load_views(inputs)
        if len(self.rows) < config.neighbours + 1:
            raise ValueError("fewer views than neighbours + 1")
        self.sparse = np.load(Path(inputs) / "sparse_points.npy")
        self.voxel = None

        def ten(a):
            return torch.tensor(np.asarray(a), dtype=torch.float32, device=self.dev)

        self.cams = [(ten(r["rotation"]), ten(r["translation"]), ten(r["k"])) for r in self.rows]
        self.native_gray, self.native_mask, self.boxes = [], [], []
        pad = config.crop_padding
        for row in self.rows:
            rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
            mask = np.asarray(Image.open(row["mask"]).convert("L")) > 127
            if mask.shape != rgb.shape[:2] or not mask.any():
                raise ValueError("mask must be nonempty and match its image: " + row["name"])
            yy, xx = np.nonzero(mask)
            box = [xx.min() - pad, yy.min() - pad, xx.max() + pad + 1, yy.max() + pad + 1]
            gray = rgb.mean(2) / 255
            lo, hi = np.percentile(gray[mask], config.stretch_percentiles)
            self.native_gray.append(((gray - lo) / max(hi - lo, 1e-6)).astype(np.float32))
            self.native_mask.append(mask)
            self.boxes.append([int(b) for b in box])
        # One common canvas for every view: MPS recompiles each kernel per tensor
        # shape, which dominated run time with per-view crop sizes. Boxes may
        # extend past the image; PIL pads with zero, i.e. background.
        cw = max(b[2] - b[0] for b in self.boxes)
        ch = max(b[3] - b[1] for b in self.boxes)
        cw, ch = cw + cw % 2, ch + ch % 2
        self.boxes = [[(b[0] + b[2] - cw) // 2, (b[1] + b[3] - ch) // 2,
                       (b[0] + b[2] - cw) // 2 + cw, (b[1] + b[3] - ch) // 2 + ch] for b in self.boxes]
        self.longest = max(cw, ch)
        centers = np.array([-np.array(r["rotation"]).T @ np.array(r["translation"]) for r in self.rows])
        rays = centers - np.median(self.sparse, 0)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        self.angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1)))

    def neighbours(self, i, count):
        c = self.config
        order = [j for j in np.argsort(self.angles[i])
                 if j != i and c.minimum_angle <= self.angles[i, j] <= c.maximum_angle]
        if len(order) < 2:
            raise ValueError("view has fewer than two neighbours in the allowed angle range: " + self.rows[i]["name"])
        return order[:count]

    # ---- silhouette hull -------------------------------------------------
    def build_hull(self):
        """Coarse carve to find the object box, then a fine carve inside it."""
        torch, F, dev, c = self.torch, self.F, self.dev, self.config
        masks = []
        for m in self.native_mask:
            m = torch.tensor(m, device=dev)[None, None].float()
            if c.hull_dilate:
                m = F.max_pool2d(m, 2 * c.hull_dilate + 1, 1, c.hull_dilate)
            masks.append(m[0, 0] > 0)
        lo, hi = np.percentile(self.sparse, 1, 0), np.percentile(self.sparse, 99, 0)
        pad = 0.3 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad
        coarse = float((hi - lo).max() / 96)
        shape = tuple(int(math.ceil(s / coarse)) for s in (hi - lo))
        origin = torch.tensor(lo, dtype=torch.float32, device=dev)
        loose = max(c.hull_allowed, c.repair_loose, 16)
        occupied = carve(self.cams, masks, origin, shape, coarse, torch, dev) <= loose
        index = occupied.nonzero()
        if not len(index):
            raise ValueError("empty silhouette hull; check masks and cameras")
        low, high = index.min(0).values, index.max(0).values
        touching = bool(((low == 0) | (high == torch.tensor(shape, device=dev) - 1)).any())
        grown = (F.max_pool3d(occupied.cpu()[None, None].float(), 3, 1, 1)[0, 0] > 0).to(dev)
        candidate = Hull(grown, origin, coarse, torch)
        if self.voxel is None:
            self.voxel = float((high - low + 1).max()) * coarse / c.grid
        lo = lo + (low.cpu().numpy() - 1) * coarse
        hi = lo + ((high - low).cpu().numpy() + 3) * coarse
        shape = tuple(int(math.ceil(s / self.voxel)) for s in (hi - lo))
        origin = torch.tensor(lo, dtype=torch.float32, device=dev)
        self.violations = carve(self.cams, masks, origin, shape, self.voxel, torch, dev, candidate)
        occupied = self.violations <= c.hull_allowed
        self.hull = Hull(occupied, origin, self.voxel, torch)
        grown = F.max_pool3d(occupied.cpu()[None, None].float(), 3, 1, 1)[0, 0] > 0
        self.search_hull = Hull(grown.to(dev), origin, self.voxel, torch)
        points = self.hull.centres(7)
        self.bounds = []
        for R, t, _ in self.cams:
            z = (points @ R.T + t)[:, 2]
            self.bounds.append((float(z.min()) - 3 * self.voxel, float(z.max()) + 3 * self.voxel))
        return {"shape": list(shape), "voxel": self.voxel, "occupied": int(occupied.sum()), "coarse_touches_box": touching}

    def orbit_down(self):
        """Centre of the camera orbit and its normal, pointing from the cameras to the object side."""
        torch = self.torch
        centers = torch.stack([-R.T @ t for R, t, _ in self.cams])
        middle = centers.mean(0)
        down = torch.tensor(np.linalg.svd((centers - middle).cpu().numpy())[2][2], device=self.dev)
        if ((self.hull.centres(5) - middle) @ down).mean() < 0:
            down = -down
        return middle, down

    def repair_masks(self, ring=(5, 15)):
        """Add object pixels that single-view segmentation dropped.

        A pixel is added only when it lies in the projection of a loose hull
        (voxels that all but ``repair_loose`` views agree on) and is itself darker
        than the midpoint between this photo's object range and its backdrop. The
        loose hull alone would fill real gaps; the photo test rejects those.
        Assumes an object darker than its backdrop; a brighter object adds nothing.
        """
        torch, F, dev, c = self.torch, self.F, self.dev, self.config
        index = (self.violations <= c.repair_loose).nonzero()
        points = (index.float() + 0.5) * self.voxel + self.hull.origin
        # Contact shadows on the support are dark and view-consistent too. Keep
        # the repair away from the base: the orbit normal gives "down", and the
        # strict hull's lowest extent gives the support height.
        middle, down = self.orbit_down()
        strict = self.hull.centres(5)
        heights = ((strict - middle) @ down).sort().values
        base = float(heights[int(0.995 * (len(heights) - 1))])
        points = points[((points - middle) @ down) < base - c.repair_base_margin * self.voxel]
        added = []
        for n, ((R, t, k), mask, gray) in enumerate(zip(self.cams, self.native_mask, self.native_gray)):
            m = torch.tensor(mask, device=dev)
            g = torch.tensor(gray, device=dev)
            h, w = m.shape
            p = points @ R.T + t
            u = p[:, 0] / p[:, 2] * k[0] + k[2] - 0.5
            v = p[:, 1] / p[:, 2] * k[1] + k[3] - 0.5
            inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (p[:, 2] > 0)
            u, v = u[inside].floor().long(), v[inside].floor().long()
            cover = torch.zeros((h, w), dtype=torch.bool, device=dev)
            for du in (0, 1):
                for dv in (0, 1):
                    cover[v + dv, u + du] = True
            mf = m[None, None].float()
            near = F.max_pool2d(mf, 2 * ring[0] + 1, 1, ring[0])[0, 0] > 0
            far = F.max_pool2d(mf, 2 * ring[1] + 1, 1, ring[1])[0, 0] > 0
            backdrop = float(g[far & ~near].median())
            dark = g < 1 + 0.5 * (backdrop - 1)
            extra = cover & dark & ~m
            added.append(float(extra.sum() / m.sum()))
            self.native_mask[n] = (m | extra).cpu().numpy()
        return {"added_fraction_median": float(np.median(added)), "added_fraction_maximum": float(max(added))}

    # ---- pyramid level ---------------------------------------------------
    def build_level(self, size):
        torch, dev = self.torch, self.dev
        factor = min(1.0, size / self.longest)
        level = []
        for gray, mask, box, (R, t, k) in zip(self.native_gray, self.native_mask, self.boxes, self.cams):
            bw, bh = box[2] - box[0], box[3] - box[1]
            w, h = max(8, round(bw * factor)), max(8, round(bh * factor))
            g = np.asarray(Image.fromarray(gray).crop(box).resize((w, h), Image.Resampling.BILINEAR))
            m = np.asarray(Image.fromarray(mask.astype(np.float32)).crop(box).resize((w, h), Image.Resampling.BOX)) >= 0.5
            sx, sy = w / bw, h / bh
            kk = torch.stack((k[0] * sx, k[1] * sy, (k[2] - box[0]) * sx, (k[3] - box[1]) * sy))
            mt = torch.tensor(m, device=dev)
            level.append({"R": R, "t": t, "k": kk, "w": w, "h": h, "mask": mt,
                          "gm": torch.stack((torch.tensor(g, device=dev), mt.float()))[None]})
        self._rays = (None, None)
        return level

    def rays(self, view):
        """Pixel rays of one view; only the most recent grid is kept on the device."""
        if self._rays[0] is not view:
            torch, k = self.torch, view["k"]
            yy, xx = torch.meshgrid(torch.arange(view["h"], device=self.dev, dtype=torch.float32),
                                    torch.arange(view["w"], device=self.dev, dtype=torch.float32), indexing="ij")
            self._rays = (view, torch.stack(((xx + 0.5 - k[2]) / k[0], (yy + 0.5 - k[3]) / k[1], torch.ones_like(xx)), -1))
        return self._rays[1]

    def project(self, world, view):
        p = world @ view["R"].T + view["t"]
        z = p[..., 2]
        xy = p[..., :2] / z.clamp_min(1e-6)[..., None] * view["k"][:2] + view["k"][2:] - 0.5
        return xy, z

    def score(self, level, i, depth, nbrs, window):
        """Aggregate NCC of hypotheses depth[B,H,W] for reference i."""
        torch, F, c = self.torch, self.F, self.config
        v = level[i]
        world = (self.rays(v)[None] * depth[..., None] - v["t"]) @ v["R"]
        base = v["mask"][None] & self.search_hull.contains(world)
        # Best-N by elementwise min/max insertion: indexed max and scatter are
        # both very slow on MPS.
        best = [torch.full(depth.shape, -2.0, device=self.dev) for _ in range(c.best_of)]
        for j in nbrs:
            u = level[j]
            xy, z = self.project(world, u)
            grid = xy * xy.new_tensor([2 / (u["w"] - 1), 2 / (u["h"] - 1)]) - 1
            sample = F.grid_sample(u["gm"].expand(len(depth), -1, -1, -1), grid, align_corners=True)
            valid = base & (sample[:, 1] >= 0.999) & (z > 0)
            s = ncc(v["gm"][:, :1], sample[:, :1], valid[:, None].float(), window, c.min_variance, c.window_fill, torch, F)
            for rank in range(c.best_of):
                lower = torch.minimum(best[rank], s)
                best[rank] = torch.maximum(best[rank], s)
                s = lower
        total = sum(torch.where(k > -1.5, k, torch.zeros_like(k)) for k in best)
        count = sum((k > -1.5).float() for k in best)
        return torch.where(count >= 2, total / count.clamp_min(1), torch.full_like(total, -2))

    def _aggregate(self, volume, sigma):
        """Validity-weighted spatial blur of each hypothesis slice.

        In refine() the slices are offsets from a smooth surface, so this widens
        the support along that surface rather than fronto-parallel to the camera.
        Unsupported centres stay unsupported.
        """
        torch, F = self.torch, self.F
        if not sigma:
            return volume
        valid = (volume > -1.5).float()[:, None]
        num = _gauss(volume[:, None] * valid, sigma, torch, F)
        den = _gauss(valid, sigma, torch, F)
        out = torch.where((valid > 0) & (den > 0.3), num / den.clamp_min(1e-6), torch.full_like(num, -2))
        return out[:, 0]

    def _peak(self, volume):
        torch = self.torch
        best, index = volume.max(0)
        n = volume.shape[0]
        left = volume.gather(0, (index - 1).clamp_min(0)[None])[0]
        right = volume.gather(0, (index + 1).clamp_max(n - 1)[None])[0]
        interior = (index > 0) & (index < n - 1) & (left > -1.5) & (right > -1.5)
        curve = left - 2 * best + right
        offset = torch.where(interior & (curve < -1e-6), 0.5 * (left - right) / curve, torch.zeros_like(best))
        return best, index, offset.clamp(-0.5, 0.5), interior

    def sweep(self, level, i, nbrs, window, aggregate, min_score, chunk=16):
        torch, c = self.torch, self.config
        near, far = self.bounds[i]
        inverse = torch.linspace(1 / near, 1 / far, c.planes, device=self.dev)
        h, w = level[i]["h"], level[i]["w"]
        volume = torch.empty((c.planes, h, w), device=self.dev)
        for s in range(0, c.planes, chunk):
            d = (1 / inverse[s : s + chunk])[:, None, None].expand(-1, h, w)
            volume[s : s + chunk] = self.score(level, i, d, nbrs, window)
        best, index, offset, interior = self._peak(self._aggregate(volume, aggregate))
        step = (1 / far - 1 / near) / (c.planes - 1)
        depth = 1 / (1 / near + (index.float() + offset) * step)
        valid = (best >= min_score) & interior
        return torch.where(valid, depth, torch.zeros_like(depth)), abs(step)

    def refine(self, level, i, init, step, half, nbrs, window, aggregate, min_score, chunk=4):
        """Search ±half steps of inverse depth around a smooth init surface."""
        torch = self.torch
        base = 1 / init.clamp_min(1e-6)
        offsets = torch.arange(-half, half + 1, device=self.dev, dtype=torch.float32) * step
        volume = torch.empty((len(offsets), *init.shape), device=self.dev)
        for s in range(0, len(offsets), chunk):
            d = 1 / (base[None] + offsets[s : s + chunk, None, None]).clamp_min(1e-6)
            volume[s : s + chunk] = self.score(level, i, d, nbrs, window)
        best, index, offset, interior = self._peak(self._aggregate(volume, aggregate))
        depth = 1 / (base + (index.float() - half + offset) * step).clamp_min(1e-6)
        valid = (best >= min_score) & interior & (init > 0)
        return torch.where(valid, depth, torch.zeros_like(depth))

    def hull_front(self, level, i):
        """Depth at which each masked pixel's ray first enters the strict hull.

        Marched on a coarser pixel grid with two-voxel steps; the band search that
        follows covers far more than that error.
        """
        torch, F = self.torch, self.F
        v = level[i]
        stride = max(1, self.config.hull_front_stride)
        rays = self.rays(v)[stride // 2 :: stride, stride // 2 :: stride]
        near, far = self.bounds[i]
        samples = torch.linspace(near, far, int((far - near) / (2 * self.voxel)) + 1, device=self.dev)
        front = torch.zeros(rays.shape[:2], device=self.dev)
        for s in range(0, len(samples), 32):
            d = samples[s : s + 32]
            world = (rays[None] * d[:, None, None, None] - v["t"]) @ v["R"]
            inside = self.hull.contains(world)
            first = torch.where(inside, d[:, None, None].expand_as(inside), torch.full_like(inside, 1e9, dtype=torch.float32)).min(0).values
            front = torch.where((front == 0) & (first < 1e8), first, front)
        if stride > 1:
            # Nearest keeps depth edges; pixels whose coarse sample missed the hull get the local minimum.
            near_front = -F.max_pool2d(-torch.where(front > 0, front, torch.full_like(front, 1e9))[None, None], 3, 1, 1)
            front = torch.where(front > 0, front, torch.where(near_front[0, 0] < 1e8, near_front[0, 0], torch.zeros_like(front)))
            front = F.interpolate(front[None, None], size=(v["h"], v["w"]), mode="nearest")[0, 0]
        return torch.where(v["mask"], front, torch.zeros_like(front))

    def lookup(self, depth_map, xy):
        h, w = depth_map.shape
        ix, iy = xy[..., 0].round().long(), xy[..., 1].round().long()
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        value = depth_map[iy.clamp(0, h - 1), ix.clamp(0, w - 1)]
        return self.torch.where(inside, value, self.torch.zeros_like(value))

    def consistent(self, level, depths, tolerance, minimum):
        torch = self.torch
        out = []
        for i, v in enumerate(level):
            d = depths[i]
            world = (self.rays(v) * d[..., None] - v["t"]) @ v["R"]
            votes = torch.zeros_like(d)
            for j in self.neighbours(i, self.config.vote_neighbours):
                xy, z = self.project(world, level[j])
                other = self.lookup(depths[j], xy)
                votes += ((other > 0) & ((other - z).abs() <= tolerance * z)).float()
            out.append(torch.where((d > 0) & (votes >= minimum), d, torch.zeros_like(d)))
        return out

    def initial(self, depth, mask, shape, sigma):
        """Upsample, lightly smooth and hole-fill a depth map inside the mask."""
        torch, F = self.torch, self.F
        valid = (depth > 0).float()[None, None]
        d = depth[None, None]
        if tuple(depth.shape) != tuple(shape):
            d = F.interpolate(d * valid, size=shape, mode="bilinear", align_corners=False)
            valid = F.interpolate(valid, size=shape, mode="bilinear", align_corners=False)
            d = torch.where(valid > 0.5, d / valid.clamp_min(1e-6), torch.zeros_like(d))
            valid = (valid > 0.5).float()
        result = torch.zeros_like(d)
        filled = torch.zeros_like(valid, dtype=torch.bool)
        for s in (sigma, 2 * sigma, 4 * sigma):
            num, den = _gauss(d * valid, s, torch, F), _gauss(valid, s, torch, F)
            take = ~filled & (den > 0.05)
            result = torch.where(take, num / den.clamp_min(1e-6), result)
            filled |= take
        # Remaining large holes: the same normalised blur on an 8x coarser grid.
        small_d, small_v = F.avg_pool2d(d * valid, 8, ceil_mode=True), F.avg_pool2d(valid, 8, ceil_mode=True)
        for s in (1.0, 2.0, 4.0, 8.0):
            num, den = _gauss(small_d, s, torch, F), _gauss(small_v, s, torch, F)
            up = F.interpolate(torch.cat((num, den), 1), size=shape, mode="bilinear", align_corners=False)
            take = ~filled & (up[:, 1:] > 0.01)
            result = torch.where(take, up[:, :1] / up[:, 1:].clamp_min(1e-6), result)
            filled |= take
        return torch.where(mask & filled[0, 0], result[0, 0], torch.zeros_like(result[0, 0]))

    # ---- fusion ----------------------------------------------------------
    def tsdf(self, level, depths, rim_pixels=0):
        """Average truncated signed distance per hull voxel.

        Voxels a short way behind a measured surface get a weak inside vote.
        Without it the interior is unobserved and any smoothing of the field
        leaks free-space evidence through the surface.
        """
        torch, dev, c = self.torch, self.dev, self.config
        index = self.hull.flat.reshape(self.hull.shape).nonzero()
        points = (index.float() + 0.5) * self.voxel + self.hull.origin
        trunc = c.truncation_voxels * self.voxel
        total = torch.zeros(len(points), device=dev)
        weight = torch.zeros(len(points), device=dev)
        for v, depth in zip(level, depths):
            if rim_pixels:
                # A window that straddles the silhouette matches the apparent
                # contour, which triangulates behind the true surface and carves
                # thin parts from both sides. The hull is exact at the rim, so
                # leave the rim to it and fuse interior depth only.
                inner = -self.F.max_pool2d(-v["mask"][None, None].float(), 2 * rim_pixels + 1, 1, rim_pixels)[0, 0] > 0
                depth = torch.where(inner, depth, torch.zeros_like(depth))
            for s in range(0, len(points), 2_000_000):
                xy, z = self.project(points[s : s + 2_000_000], v)
                measured = self.lookup(depth, xy)
                sdf = measured - z
                seen = (measured > 0) & (z > 0)
                use = seen & (sdf > -trunc)
                behind = seen & ~use & (sdf > -c.behind_voxels * self.voxel)
                total[s : s + 2_000_000] += torch.where(use, (sdf / trunc).clamp(max=1), torch.zeros_like(sdf)) - c.behind_weight * behind.float()
                weight[s : s + 2_000_000] += use.float() + c.behind_weight * behind.float()
        # Support height: silhouettes cannot tell a flat base from a cone under it,
        # but no photo measures surface below the support. Take the lowest level
        # that still has well-supported measured surface.
        middle, down = self.orbit_down()
        surface = (weight >= 3) & ((total / weight.clamp_min(1e-6)).abs() < 0.5)
        support = None
        if int(surface.sum()) > 1000:
            heights = ((points[surface] - middle) @ down).sort().values
            support = float(heights[int(0.998 * (len(heights) - 1))])
        self.support = {"point": middle.cpu().numpy(), "down": down.cpu().numpy(), "height": support}
        return index.cpu().numpy().astype(np.int32), total.cpu().numpy(), weight.cpu().numpy(), trunc


def fused_field(stereo, index, total, weight):
    """Dense signed field on the device: fused evidence, hull interior where unobserved, outside elsewhere."""
    torch = stereo.torch
    field = torch.ones(stereo.hull.shape, device=stereo.dev)
    idx = torch.tensor(index.astype(np.int64), device=stereo.dev)
    t = torch.tensor(total, device=stereo.dev)
    w = torch.tensor(weight, device=stereo.dev)
    value = torch.where(w > 0, (t / w.clamp_min(1e-6)).clamp(-1, 1), torch.full_like(t, -1.0))
    field[idx[:, 0], idx[:, 1], idx[:, 2]] = value
    return field


def render_field(stereo, field, view, bounds, stride=2):
    """First outside-to-inside crossing of the field along each ray, on a coarser pixel grid."""
    torch = stereo.torch
    rays = stereo.rays(view)[stride // 2 :: stride, stride // 2 :: stride]
    near, far = bounds
    samples = torch.linspace(near, far, int((far - near) / stereo.voxel) + 1, device=stereo.dev)
    flat = field.reshape(-1)
    nx, ny, nz = field.shape
    limit = torch.tensor(field.shape, device=stereo.dev)
    depth = torch.zeros(rays.shape[:2], device=stereo.dev)
    previous = torch.ones(rays.shape[:2], device=stereo.dev)
    previous_d = torch.full(rays.shape[:2], near, device=stereo.dev)
    for s in range(0, len(samples), 32):
        d = samples[s : s + 32]
        world = (rays[None] * d[:, None, None, None] - view["t"]) @ view["R"]
        cell = ((world - stereo.hull.origin) / stereo.voxel).floor().long()
        inside = ((cell >= 0) & (cell < limit)).all(-1)
        linear = ((cell[..., 0] * ny + cell[..., 1]) * nz + cell[..., 2]).clamp(0, flat.numel() - 1)
        value = torch.where(inside, flat[linear], torch.ones_like(d)[:, None, None].expand_as(inside))
        before = torch.cat((previous[None], value[:-1]))
        before_d = torch.cat((previous_d[None], d[:-1, None, None].expand_as(value[:-1])))
        cross = (before > 0) & (value <= 0)
        here = before_d + (d[:, None, None] - before_d) * before / (before - value).clamp_min(1e-6)
        first = torch.where(cross, here, torch.full_like(here, 1e9)).min(0).values
        depth = torch.where((depth == 0) & (first < 1e8), first, depth)
        previous, previous_d = value[-1], d[-1].expand_as(previous_d)
    return depth


def preview(path, level, depths, picks):
    tiles = []
    for i in picks:
        d = depths[i].cpu().numpy()
        g = level[i]["gm"][0, 0].cpu().numpy()
        m = level[i]["mask"].cpu().numpy()
        v = d > 0
        lo, hi = (np.percentile(d[v], [1, 99]) if v.any() else (0, 1))
        gy, gx = np.gradient(np.where(v, d, np.nan))
        shade = 0.5 + 0.5 * (-gx - gy) / np.sqrt(gx * gx + gy * gy + ((hi - lo) / 80) ** 2)
        shade = np.where(np.isfinite(shade), shade, 0.12)
        photo = np.where(m, np.clip(g, 0, 1), 0.12)
        tile = np.hstack((photo, np.where(v, np.clip((d - lo) / (hi - lo + 1e-9), 0, 1), 0.12), shade))
        tiles.append(Image.fromarray((255 * tile).astype(np.uint8)).resize((1200, round(1200 * tile.shape[0] / tile.shape[1]))))
    sheet = Image.new("L", (1200, sum(t.height for t in tiles)))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(path)


def coverage(level, depths):
    values = [float(((d > 0) & v["mask"]).sum() / v["mask"].sum()) for v, d in zip(level, depths)]
    return {"median": float(np.median(values)), "minimum": float(min(values)), "maximum": float(max(values))}


def run(inputs, output, *, device="mps", config=None, reuse_depths=None, log=print):
    config = (config or DenseConfig()).validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"configuration": config.to_json(), "device": device, "levels": [],
              "reuse_depths": str(reuse_depths) if reuse_depths else None}
    stereo = Stereo(inputs, device, config)
    torch = stereo.torch
    report["torch_version"] = torch.__version__
    count = len(stereo.rows)
    report["views"] = count
    picks = [count // 7, count // 2 - 3, (4 * count) // 5]

    def sync():
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        t = time.monotonic()
        report["hull"] = {**stereo.build_hull(), "seconds": time.monotonic() - t}
        log("hull", report["hull"])
        if config.repair_masks:
            t = time.monotonic()
            report["mask_repair"] = stereo.repair_masks()
            report["hull_repaired"] = {**stereo.build_hull(), "seconds": time.monotonic() - t}
            log("repair", report["mask_repair"], report["hull_repaired"])
            (output / "masks-repaired").mkdir()
            for row, mask in zip(stereo.rows, stereo.native_mask):
                Image.fromarray((mask * 255).astype(np.uint8)).save(output / "masks-repaired" / (row["name"] + ".png"))
        sizes = [min(s, stereo.longest) for s in config.sizes]
        sizes = [s for n, s in enumerate(sizes) if s not in sizes[:n]]  # native cap may merge levels
        depths, step, coarser = None, None, None
        if reuse_depths is not None:
            saved = np.load(reuse_depths)
            depths = [torch.tensor(saved[f"depth_{i:03d}"], device=stereo.dev) for i in range(count)]
            level = stereo.build_level(sizes[-1])
            run_sizes = []
        else:
            run_sizes = sizes
        for li, size in enumerate(run_sizes):
            t = time.monotonic()
            level = stereo.build_level(size)
            window = config.level("windows", li)
            aggregate = config.level("aggregates", li)
            for p in range(1 if li == 0 else config.level("passes", li)):
                raw, front_wins = [], 0
                for i in range(count):
                    nbrs = stereo.neighbours(i, config.neighbours)
                    if li == 0:
                        d, step = stereo.sweep(level, i, nbrs, window, aggregate, config.min_score)
                    else:
                        shape = (level[i]["h"], level[i]["w"])
                        init = stereo.initial(depths[i], level[i]["mask"], shape, 1.5 if p == 0 else 1.0)
                        fine = step / (2 ** li) / (1 if p == 0 else 2)
                        half = config.level("band_first", li) if p == 0 else config.band_later
                        d = stereo.refine(level, i, init, fine, half, nbrs, window, aggregate, config.min_score)
                    if config.hull_front and li == min(config.hull_front_level, len(run_sizes) - 1) and p == 0:
                        # Thin parts are narrower than the coarse window, so they
                        # inherit the depth of whatever lies behind them. Test the
                        # hull's front surface too: a photo-consistent nearer
                        # surface occludes anything matched behind it.
                        fine = step / (2 ** li)
                        d2 = stereo.refine(level, i, stereo.hull_front(level, i), fine, config.level("band_first", li), nbrs,
                                           window, aggregate, max(config.min_score, config.hull_front_min_score))
                        nearer = (d2 > 0) & ((d == 0) | (d2 < d * (1 - config.hull_front_margin)))
                        front_wins += int(nearer.sum())
                        d = torch.where(nearer, d2, d)
                    raw.append(d)
                    if i % 10 == 9:
                        sync()
                t_c = time.monotonic()
                depths = stereo.consistent(level, raw, config.level("tolerances", li), config.level("min_votes", li))
                sync()
                row = {"size": size, "pass": p, "window": window, "working_size": [level[0]["w"], level[0]["h"]],
                       "raw_coverage": coverage(level, raw), "consistent_coverage": coverage(level, depths),
                       "hull_front_pixels": front_wins, "consistency_seconds": time.monotonic() - t_c,
                       "seconds": time.monotonic() - t}
                report["levels"].append(row)
                log(f"level {li} pass {p}: size {size}, coverage {row['consistent_coverage']['median']:.3f}, {row['seconds']:.1f}s")
            preview(output / f"depth-level-{li}.png", level, depths, picks)
            if config.fallback_level and li == len(run_sizes) - 1 and coarser is not None:
                # Keep the finest depth; where it failed its checks, fall back to
                # the previous level's consistent depth instead of leaving a hole.
                merged = []
                for v, fine_depth, coarse_depth in zip(level, depths, coarser):
                    up = torch.nn.functional.interpolate(coarse_depth[None, None], size=fine_depth.shape, mode="nearest")[0, 0]
                    merged.append(torch.where(fine_depth > 0, fine_depth, torch.where(v["mask"], up, torch.zeros_like(up))))
                report["fallback_coverage"] = coverage(level, merged)
                depths = merged
                preview(output / "depth-merged.png", level, depths, picks)
            coarser = depths
        t = time.monotonic()
        rim = round(config.rim_fraction * config.level("windows", len(sizes) - 1))
        index, total, weight, trunc = stereo.tsdf(level, depths, rim_pixels=rim)
        report["rim_pixels"] = rim
        report["fused_passes"] = []
        for n in range(config.fused_passes if step is not None else 0):
            # The fused surface is complete and averaged over many views. Matching
            # again in a narrow band around it fills holes with verified depth and
            # cannot drift far from the consensus.
            t_f = time.monotonic()
            field = fused_field(stereo, index, total, weight)
            li = len(sizes) - 1
            window, aggregate = config.level("windows", li), config.level("aggregates", li)
            raw = []
            for i in range(count):
                coarse = render_field(stereo, field, level[i], stereo.bounds[i])
                init = stereo.initial(coarse, level[i]["mask"], (level[i]["h"], level[i]["w"]), 1.0)
                raw.append(stereo.refine(level, i, init, step / (2 ** li) / 2, config.fused_band,
                                         stereo.neighbours(i, config.neighbours), window, aggregate, config.min_score))
                if i % 10 == 9:
                    sync()
            del field
            again = stereo.consistent(level, raw, config.level("tolerances", li), config.level("min_votes", li))
            depths = [torch.where(a > 0, a, d) for a, d in zip(again, depths)]
            index, total, weight, trunc = stereo.tsdf(level, depths, rim_pixels=rim)
            row = {"pass": n, "new_coverage": coverage(level, again), "merged_coverage": coverage(level, depths),
                   "seconds": time.monotonic() - t_f}
            report["fused_passes"].append(row)
            log(f"fused pass {n}: new {row['new_coverage']['median']:.3f}, merged {row['merged_coverage']['median']:.3f}, {row['seconds']:.1f}s")
        report["tsdf"] = {"hull_voxels": len(index), "observed_fraction": float((weight > 0).mean()), "seconds": time.monotonic() - t}
        log("tsdf", report["tsdf"])
    np.savez_compressed(output / "volume.npz", index=index, total=total.astype(np.float32),
                        weight=weight.astype(np.float32), shape=np.array(stereo.hull.shape),
                        origin=stereo.hull.origin.cpu().numpy(), voxel=np.float32(stereo.voxel), truncation=np.float32(trunc),
                        support_point=stereo.support["point"], support_down=stereo.support["down"],
                        support_height=np.float32(np.nan if stereo.support["height"] is None else stereo.support["height"]))
    np.savez_compressed(output / "depths.npz", **{f"depth_{i:03d}": d.cpu().numpy().astype(np.float32) for i, d in enumerate(depths)})
    report["seconds"] = time.monotonic() - started
    report["reference_used"] = False
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True, help="directory with cameras.json and sparse_points.npy")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--reuse-depths", type=Path, help="fuse saved final-level depths; skip stereo")
    add_arguments(parser)
    args = parser.parse_args()

    def log(*parts):
        print(*parts, flush=True)

    run(args.inputs, args.output, device=args.device, config=build(args.config, args.set),
        reuse_depths=args.reuse_depths, log=log)


if __name__ == "__main__":
    main()
