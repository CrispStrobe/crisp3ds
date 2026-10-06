//! Depth-map passes between matching steps: ports of `_gauss`, `Stereo.initial`,
//! `Stereo.lookup`, `Stereo.consistent`, `Stereo.hull_front` and the level
//! fallback merge. These are image-sized passes and run on the CPU in float32,
//! with PyTorch's conventions for interpolation, pooling and padding.

#![allow(clippy::needless_range_loop)]

use crate::hull::Hull;
use crate::inputs::{parallel_map, Plane};

use super::level::LevelView;

/// `torch.linspace(start, end, steps)` in float32 (symmetric evaluation from both ends).
pub fn linspace(start: f64, end: f64, steps: usize) -> Vec<f32> {
    let (start, end) = (start as f32, end as f32);
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    (0..steps).map(|i| if i < steps / 2 { start + step * i as f32 } else { end - step * (steps - i - 1) as f32 }).collect()
}

/// Normalised Gaussian taps of `_gauss`: radius `max(1, ceil(3 sigma))`, float32.
pub fn gauss_weights(sigma: f64) -> Vec<f32> {
    let radius = ((3.0 * sigma).ceil() as i64).max(1);
    let sigma = sigma as f32;
    let taps: Vec<f32> = (-radius..=radius).map(|i| (-0.5 * (i as f32 / sigma) * (i as f32 / sigma)).exp()).collect();
    let sum: f32 = taps.iter().sum();
    taps.into_iter().map(|t| t / sum).collect()
}

/// `_gauss`: separable blur along rows then columns with replicate padding.
pub fn gauss(source: &Plane<f32>, sigma: f64) -> Plane<f32> {
    let weights = gauss_weights(sigma);
    let radius = (weights.len() / 2) as i64;
    let (w, h) = (source.width, source.height);
    let mut rows = Plane::<f32>::new(w, h);
    for y in 0..h {
        let line = &source.data[y * w..(y + 1) * w];
        for x in 0..w {
            let mut sum = 0.0f32;
            for (n, weight) in weights.iter().enumerate() {
                let q = (x as i64 + n as i64 - radius).clamp(0, w as i64 - 1) as usize;
                sum += weight * line[q];
            }
            rows.data[y * w + x] = sum;
        }
    }
    let mut out = Plane::<f32>::new(w, h);
    for (n, weight) in weights.iter().enumerate() {
        for y in 0..h {
            let q = (y as i64 + n as i64 - radius).clamp(0, h as i64 - 1) as usize;
            let (target, from) = (&mut out.data[y * w..(y + 1) * w], &rows.data[q * w..(q + 1) * w]);
            for x in 0..w {
                target[x] += weight * from[x];
            }
        }
    }
    out
}

/// Source index and weight of `F.interpolate(mode="bilinear", align_corners=False)` along one axis.
fn bilinear_taps(input: usize, output: usize) -> Vec<(usize, usize, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|n| {
            let source = (scale * (n as f32 + 0.5) - 0.5).max(0.0);
            let first = (source as usize).min(input - 1);
            (first, (first + 1).min(input - 1), source - first as f32)
        })
        .collect()
}

/// `F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)`.
pub fn bilinear(source: &Plane<f32>, width: usize, height: usize) -> Plane<f32> {
    let (xs, ys) = (bilinear_taps(source.width, width), bilinear_taps(source.height, height));
    let mut out = Plane::<f32>::new(width, height);
    for (y, &(y0, y1, ty)) in ys.iter().enumerate() {
        let (top, bottom) = (&source.data[y0 * source.width..], &source.data[y1 * source.width..]);
        for (x, &(x0, x1, tx)) in xs.iter().enumerate() {
            let upper = (1.0 - tx) * top[x0] + tx * top[x1];
            let lower = (1.0 - tx) * bottom[x0] + tx * bottom[x1];
            out.data[y * width + x] = (1.0 - ty) * upper + ty * lower;
        }
    }
    out
}

/// Source index of `F.interpolate(mode="nearest")`: `floor(n * (input / output))` in float32.
pub fn nearest_index(n: usize, input: usize, output: usize) -> usize {
    (((n as f32) * (input as f32 / output as f32)).floor() as usize).min(input - 1)
}

/// `F.interpolate(x, size=(height, width), mode="nearest")`.
pub fn nearest(source: &Plane<f32>, width: usize, height: usize) -> Plane<f32> {
    let xs: Vec<usize> = (0..width).map(|x| nearest_index(x, source.width, width)).collect();
    let mut out = Plane::<f32>::new(width, height);
    for y in 0..height {
        let row = nearest_index(y, source.height, height) * source.width;
        for x in 0..width {
            out.data[y * width + x] = source.data[row + xs[x]];
        }
    }
    out
}

/// `F.avg_pool2d(x, 8, ceil_mode=True)`: partial windows at the far edges divide by their own size.
fn pool8(source: &Plane<f32>) -> Plane<f32> {
    let (w, h) = (source.width.div_ceil(8), source.height.div_ceil(8));
    let mut out = Plane::<f32>::new(w, h);
    for y in 0..h {
        let (y0, y1) = (y * 8, (y * 8 + 8).min(source.height));
        for x in 0..w {
            let (x0, x1) = (x * 8, (x * 8 + 8).min(source.width));
            let mut sum = 0.0f32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    sum += source.data[sy * source.width + sx];
                }
            }
            out.data[y * w + x] = sum / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }
    out
}

/// `Stereo.initial`: upsample, lightly smooth and hole-fill a depth map inside the mask.
pub fn initial(depth: &Plane<f32>, mask: &Plane<u8>, sigma: f64) -> Plane<f32> {
    let (w, h) = (mask.width, mask.height);
    let mut valid = Plane { width: depth.width, height: depth.height, data: depth.data.iter().map(|&d| (d > 0.0) as u8 as f32).collect() };
    // Depth is zero exactly where it is invalid, so `depth * valid` is the depth itself.
    let mut d = depth.clone();
    if (depth.width, depth.height) != (w, h) {
        d = bilinear(depth, w, h);
        let weight = bilinear(&valid, w, h);
        for (value, &v) in d.data.iter_mut().zip(&weight.data) {
            *value = if v > 0.5 { *value / v.max(1e-6) } else { 0.0 };
        }
        valid = Plane { width: w, height: h, data: weight.data.iter().map(|&v| (v > 0.5) as u8 as f32).collect() };
    }
    let mut result = vec![0.0f32; w * h];
    let mut filled = vec![false; w * h];
    let open = |filled: &[bool]| mask.data.iter().zip(filled).any(|(&m, &f)| m != 0 && !f);
    for s in [sigma, 2.0 * sigma, 4.0 * sigma] {
        if !open(&filled) {
            break; // every masked pixel has its value; later steps cannot change the result
        }
        let (num, den) = (gauss(&d, s), gauss(&valid, s));
        for p in 0..w * h {
            if !filled[p] && den.data[p] > 0.05 {
                result[p] = num.data[p] / den.data[p].max(1e-6);
                filled[p] = true;
            }
        }
    }
    if open(&filled) {
        // Remaining large holes: the same normalised blur on an 8x coarser grid.
        let (small_d, small_v) = (pool8(&d), pool8(&valid));
        for s in [1.0, 2.0, 4.0, 8.0] {
            if !open(&filled) {
                break;
            }
            let (num, den) = (bilinear(&gauss(&small_d, s), w, h), bilinear(&gauss(&small_v, s), w, h));
            for p in 0..w * h {
                if !filled[p] && den.data[p] > 0.01 {
                    result[p] = num.data[p] / den.data[p].max(1e-6);
                    filled[p] = true;
                }
            }
        }
    }
    for p in 0..w * h {
        if mask.data[p] == 0 || !filled[p] {
            result[p] = 0.0;
        }
    }
    Plane { width: w, height: h, data: result }
}

/// `Stereo.lookup`: the depth at the nearest pixel (round half to even), zero outside the map.
#[inline]
pub fn lookup(depth: &Plane<f32>, xy: [f32; 2]) -> f32 {
    let (x, y) = (xy[0].round_ties_even(), xy[1].round_ties_even());
    if x >= 0.0 && x < depth.width as f32 && y >= 0.0 && y < depth.height as f32 {
        depth.data[y as usize * depth.width + x as usize]
    } else {
        0.0
    }
}

/// `Stereo.consistent`: keeps a depth when at least `minimum` of the given
/// neighbour views hold a depth within `tolerance` (relative) of it at its projection.
pub fn consistent(level: &[LevelView], depths: &[Plane<f32>], neighbours: &[Vec<usize>], tolerance: f64, minimum: i64) -> Vec<Plane<f32>> {
    let tolerance = tolerance as f32;
    parallel_map(level.len(), |i| {
        let (view, depth) = (&level[i], &depths[i]);
        let mut out = Plane::<f32>::new(depth.width, depth.height);
        for y in 0..depth.height {
            for x in 0..depth.width {
                let d = depth.data[y * depth.width + x];
                if d <= 0.0 {
                    continue;
                }
                let world = view.unproject(x, y, d);
                let mut votes = 0i64;
                for &j in &neighbours[i] {
                    let (xy, z) = level[j].project(world);
                    let other = lookup(&depths[j], xy);
                    votes += (other > 0.0 && (other - z).abs() <= tolerance * z) as i64;
                }
                if votes >= minimum {
                    out.data[y * depth.width + x] = d;
                }
            }
        }
        out
    })
}

/// `Stereo.hull_front`: depth at which each masked pixel's ray first enters the
/// hull, marched with two-voxel steps on a pixel grid `stride` times coarser.
pub fn hull_front(view: &LevelView, hull: &Hull, bounds: (f64, f64), stride: usize) -> Plane<f32> {
    let stride = stride.max(1);
    let (near, far) = bounds;
    let samples = linspace(near, far, ((far - near) / (2.0 * hull.voxel)) as usize + 1);
    let xs: Vec<usize> = (stride / 2..view.width).step_by(stride).collect();
    let ys: Vec<usize> = (stride / 2..view.height).step_by(stride).collect();
    let mut front = Plane::<f32>::new(xs.len(), ys.len());
    for (row, &y) in ys.iter().enumerate() {
        for (column, &x) in xs.iter().enumerate() {
            for &d in &samples {
                if hull.contains(view.unproject(x, y, d)) {
                    front.data[row * xs.len() + column] = d;
                    break;
                }
            }
        }
    }
    if stride > 1 {
        // Pixels whose coarse sample missed the hull get the nearest front around them; nearest keeps depth edges.
        let (w, h) = (front.width, front.height);
        let mut spread = front.clone();
        for y in 0..h {
            for x in 0..w {
                if front.data[y * w + x] > 0.0 {
                    continue;
                }
                let mut lowest = f32::MAX;
                for qy in y.saturating_sub(1)..(y + 2).min(h) {
                    for qx in x.saturating_sub(1)..(x + 2).min(w) {
                        let value = front.data[qy * w + qx];
                        if value > 0.0 {
                            lowest = lowest.min(value);
                        }
                    }
                }
                if lowest < 1e8 {
                    spread.data[y * w + x] = lowest;
                }
            }
        }
        front = nearest(&spread, view.width, view.height);
    }
    for (value, &m) in front.data.iter_mut().zip(&view.mask.data) {
        if m == 0 {
            *value = 0.0;
        }
    }
    front
}

/// The level fallback: the fine depth where it exists, else the previous
/// level's depth (nearest-neighbour upsampled) inside the mask.
pub fn merge_fallback(view: &LevelView, fine: &Plane<f32>, coarse: &Plane<f32>) -> Plane<f32> {
    let up = nearest(coarse, fine.width, fine.height);
    let data = (0..fine.data.len())
        .map(|p| {
            if fine.data[p] > 0.0 {
                fine.data[p]
            } else if view.mask.data[p] != 0 {
                up.data[p]
            } else {
                0.0
            }
        })
        .collect();
    Plane { width: fine.width, height: fine.height, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: usize, height: usize, data: &[f32]) -> Plane<f32> {
        Plane { width, height, data: data.to_vec() }
    }

    #[test]
    fn linspace_and_weights_follow_torch() {
        assert_eq!(linspace(0.0, 1.0, 5), [0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(2.0, -1.0, 4), [2.0, 1.0, 0.0, -1.0]);
        // torch: _gauss kernel for sigma 1 has radius 3; centre tap 0.39905027
        let weights = gauss_weights(1.0);
        assert_eq!(weights.len(), 7);
        assert!((weights[3] - 0.399_050_27).abs() < 1e-6 && (weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(gauss_weights(0.2).len(), 3);
        // A constant image stays constant under replicate padding.
        let flat = plane(5, 4, &[2.0; 20]);
        assert!(gauss(&flat, 1.5).data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn interpolation_follows_torch() {
        // F.interpolate(torch.tensor([[[[0., 1., 2., 3.]]]]), size=(1, 8), mode="bilinear", align_corners=False)
        let row = plane(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(bilinear(&row, 8, 1).data, [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]);
        // ... size=(1, 2): plain sampling at 0.5 and 2.5, no antialiasing
        assert_eq!(bilinear(&row, 2, 1).data, [0.5, 2.5]);
        // mode="nearest": size 6 -> indices floor(n * 4 / 6) = 0, 0, 1, 2, 2, 3
        assert_eq!(nearest(&row, 6, 1).data, [0.0, 0.0, 1.0, 2.0, 2.0, 3.0]);
        // avg_pool2d(8, ceil_mode=True) of 1 x 10 ones-then-twos: the partial window divides by its own size
        let mut wide = vec![1.0f32; 8];
        wide.extend([2.0, 4.0]);
        assert_eq!(pool8(&plane(10, 1, &wide)).data, [1.0, 3.0]);
    }

    #[test]
    fn initial_fills_holes_inside_the_mask_only() {
        let (w, h) = (24, 16);
        let mut depth = Plane { width: w, height: h, data: vec![5.0f32; w * h] };
        let mut mask = Plane { width: w, height: h, data: vec![1u8; w * h] };
        for y in 6..9 {
            for x in 10..13 {
                depth.data[y * w + x] = 0.0;
            }
        }
        for y in 0..h {
            mask.data[y * w] = 0;
        }
        let out = initial(&depth, &mask, 1.5);
        for y in 0..h {
            assert_eq!(out.data[y * w], 0.0);
            for x in 1..w {
                assert!((out.data[y * w + x] - 5.0).abs() < 1e-4, "{x},{y}: {}", out.data[y * w + x]);
            }
        }
        // Upsampling a constant map keeps the constant.
        let small = Plane { width: w / 2, height: h / 2, data: vec![5.0f32; w * h / 4] };
        let up = initial(&small, &mask, 1.5);
        assert!((up.data[5 * w + 7] - 5.0).abs() < 1e-4 && up.data[5 * w] == 0.0);
        // Nothing to start from: nothing is invented.
        let empty = Plane::<f32>::new(w, h);
        assert!(initial(&empty, &mask, 1.5).data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_rounds_half_to_even() {
        let map = plane(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(lookup(&map, [0.5, 0.0]), 1.0);
        assert_eq!(lookup(&map, [1.5, 0.0]), 3.0);
        assert_eq!(lookup(&map, [1.4, 0.6]), 5.0);
        assert_eq!(lookup(&map, [2.6, 0.0]), 0.0);
        assert_eq!(lookup(&map, [-0.6, 0.0]), 0.0);
    }

    /// Agreement, hull front and fallback on the analytic sphere (CPU only).
    #[test]
    fn sphere_depths_agree_and_the_hull_front_finds_them() {
        use crate::inputs::Inputs;
        use crate::stereo::level::build_level;
        use crate::stereo::{options, synthetic};

        let root = synthetic::temporary("depth", 12, 96).unwrap();
        let overrides: Vec<String> = synthetic::SMALL.iter().map(|s| s.to_string()).collect();
        let config = options::build(None, &overrides).unwrap();
        let inputs = Inputs::load(&root.join("inputs"), &config).unwrap();
        crate::storage::remove_dir_all(&root).unwrap();
        let level = build_level(&inputs, 96);
        let exact: Vec<Plane<f32>> = level.iter().map(synthetic::sphere_depth).collect();
        let voters: Vec<Vec<usize>> = (0..12).map(|i| vec![(i + 1) % 12, (i + 11) % 12]).collect();

        // Exact depths agree between neighbours wherever both see the point; a wrong depth is dropped.
        let mut depths = exact.clone();
        let (w, h) = (level[0].width, level[0].height);
        let centre = (h / 2) * w + w / 2;
        depths[0].data[centre] *= 1.05;
        let agreed = consistent(&level, &depths, &voters, 0.006, 2);
        assert_eq!(agreed[0].data[centre], 0.0);
        assert_eq!(agreed[3].data[centre], exact[3].data[centre]);
        let kept = agreed[3].data.iter().filter(|&&d| d > 0.0).count() as f64 / exact[3].data.iter().filter(|&&d| d > 0.0).count() as f64;
        assert!(kept > 0.6 && kept < 1.0, "{kept}");
        assert!(agreed[3].data.iter().zip(&exact[3].data).all(|(&a, &e)| a == 0.0 || a == e));

        // A voxelised unit sphere as the hull: its front is the sphere's depth to within the march step.
        let (side, voxel) = (80usize, 0.03f64);
        let origin = [-(side as f32) * voxel as f32 / 2.0; 3];
        let mut flags = vec![false; side * side * side];
        for (linear, flag) in flags.iter_mut().enumerate() {
            let index = [linear / (side * side), (linear / side) % side, linear % side];
            let p = index.map(|i| (i as f64 + 0.5) * voxel + origin[0] as f64);
            *flag = p.iter().map(|v| v * v).sum::<f64>() <= 1.0;
        }
        let hull = Hull::from_flags(&flags, [side; 3], origin, voxel);
        let front = hull_front(&level[2], &hull, (4.5, 7.5), 2);
        let (mut checked, mut far) = (0, 0);
        for p in 0..w * h {
            if level[2].mask.data[p] == 0 {
                assert_eq!(front.data[p], 0.0);
            } else if exact[2].data[p] > 0.0 && front.data[p] > 0.0 {
                checked += 1;
                far += ((front.data[p] - exact[2].data[p]).abs() > 0.2) as usize;
            }
        }
        assert!(checked > 1000 && far * 20 < checked, "{far} of {checked}");

        // The fallback keeps fine depth, fills holes inside the mask from the coarser level only.
        let coarse_level = build_level(&inputs, 48);
        let coarse: Vec<Plane<f32>> = coarse_level.iter().map(synthetic::sphere_depth).collect();
        let mut fine = exact[2].clone();
        fine.data[centre] = 0.0;
        for (value, &m) in fine.data.iter_mut().zip(&level[2].mask.data) {
            if m == 0 {
                *value = 0.0;
            }
        }
        let merged = merge_fallback(&level[2], &fine, &coarse[2]);
        assert!((merged.data[centre] - exact[2].data[centre]).abs() < 0.05);
        assert!(merged.data.iter().zip(&fine.data).all(|(&m, &f)| f == 0.0 || m == f));
        assert!(merged.data.iter().zip(&level[2].mask.data).all(|(&m, &mask)| mask != 0 || m == 0.0));
    }
}
