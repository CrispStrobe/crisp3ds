//! Depth fusion into a truncated signed distance volume: port of `Stereo.tsdf`
//! from `scripts/turntable_mesh/multiscale_stereo.py`.
//!
//! Every hull voxel is projected into every view on the GPU; votes are added in
//! view order in float32, as the reference does. Voxels are processed as lists
//! in chunks and depth maps in batches that fit one storage binding, so the
//! default WebGPU limits are enough for any grid and view count.

use bytemuck::{Pod, Zeroable};

use crate::config::DenseConfig;
use crate::gpu::Gpu;
use crate::hull::{camera_rows, erode, Hull, VoxelList, CHUNK};
use crate::inputs::{Camera, Plane};
use crate::repair::orbit_down;
use crate::stereo::level::LevelView;

/// Support plane found by fusion: a point on the orbit axis, the direction
/// from the cameras to the object side, and the height of the lowest
/// well-supported measured surface along it (none without enough surface).
#[derive(Debug, Clone, Copy)]
pub struct Support {
    pub point: [f32; 3],
    pub down: [f32; 3],
    pub height: Option<f32>,
}

pub struct Fused {
    /// Linear indices of the hull voxels, in C order.
    pub indices: Vec<u32>,
    pub total: Vec<f32>,
    pub weight: Vec<f32>,
    pub truncation: f64,
    pub support: Support,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FuseParams {
    count: u32,
    views: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    pad: [u32; 3],
    truncation: f32,
    behind: f32,
    behind_weight: f32,
    free_weight: f32,
}

/// One view's vote for a point, as the kernel computes it: reference for tests.
pub fn vote(view: &LevelView, depth: &Plane<f32>, point: [f32; 3], truncation: f32, behind: f32, config: &DenseConfig) -> (f32, f32) {
    let (xy, z) = view.project(point);
    let (x, y) = (xy[0].round_ties_even(), xy[1].round_ties_even());
    let inside = x >= 0.0 && x < depth.width as f32 && y >= 0.0 && y < depth.height as f32;
    let measured = if inside { depth.at(x as usize, y as usize) } else { 0.0 };
    let sdf = measured - z;
    let seen = measured > 0.0 && z > 0.0;
    let near = seen && sdf > -truncation;
    let is_behind = seen && !near && sdf > -behind;
    let vote = if near {
        if sdf >= truncation {
            config.free_weight as f32
        } else {
            1.0
        }
    } else {
        0.0
    };
    let inside_vote = if is_behind { config.behind_weight as f32 } else { 0.0 };
    (vote * (sdf / truncation).min(1.0) - inside_vote, vote + inside_vote)
}

/// Depth with the silhouette rim removed: a window that straddles the
/// silhouette matches the apparent contour, so the rim is left to the hull.
pub fn without_rim(view: &LevelView, depth: &Plane<f32>, rim_pixels: usize) -> Plane<f32> {
    if rim_pixels == 0 {
        return depth.clone();
    }
    let inner = erode(&view.mask, rim_pixels);
    Plane {
        width: depth.width,
        height: depth.height,
        data: depth.data.iter().zip(&inner.data).map(|(&d, &m)| if m != 0 { d } else { 0.0 }).collect(),
    }
}

/// `Stereo.tsdf`: average truncated signed distance per hull voxel, plus the support height.
pub async fn tsdf(
    gpu: &Gpu,
    hull: &Hull,
    cameras: &[Camera],
    level: &[LevelView],
    depths: &[Plane<f32>],
    rim_pixels: usize,
    config: &DenseConfig,
) -> anyhow::Result<Fused> {
    let kernel = gpu.kernel("fuse", include_str!("shaders/fuse.wgsl"), "main").await?;
    let indices = hull.indices();
    let list = VoxelList::new(gpu, hull, &indices);
    let truncation = config.truncation_voxels * hull.voxel;
    let behind = config.behind_voxels * hull.voxel;

    // Depth maps in batches of whole views that fit one binding.
    let budget = (gpu.binding_budget() / 4) as usize;
    let mut batches: Vec<(u32, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
    let (mut rows, mut frames, mut values) = (Vec::<[f32; 4]>::new(), Vec::<[u32; 4]>::new(), Vec::<f32>::new());
    let mut flush = |rows: &mut Vec<[f32; 4]>, frames: &mut Vec<[u32; 4]>, values: &mut Vec<f32>| {
        if !frames.is_empty() {
            batches.push((
                frames.len() as u32,
                gpu.upload("fuse cameras", rows),
                gpu.upload("fuse frames", frames),
                gpu.upload("fuse depths", values),
            ));
            rows.clear();
            frames.clear();
            values.clear();
        }
    };
    for (view, depth) in level.iter().zip(depths) {
        anyhow::ensure!(
            (depth.width, depth.height) == (view.width, view.height),
            "depth map of {}x{} does not match the level size {}x{}",
            depth.width,
            depth.height,
            view.width,
            view.height
        );
        if !frames.is_empty() && values.len() + depth.data.len() > budget {
            flush(&mut rows, &mut frames, &mut values);
        }
        frames.push([depth.width as u32, depth.height as u32, values.len() as u32, 0]);
        rows.extend_from_slice(&camera_rows(&view.camera));
        values.extend_from_slice(&without_rim(view, depth, rim_pixels).data);
    }
    flush(&mut rows, &mut frames, &mut values);

    let totals = gpu.zeroed("fuse total", 4 * CHUNK as u64);
    let weights = gpu.zeroed("fuse weight", 4 * CHUNK as u64);
    let (mut total, mut weight) = (Vec::with_capacity(indices.len()), Vec::with_capacity(indices.len()));
    for (voxels, count) in &list.chunks {
        gpu.clear(&totals);
        gpu.clear(&weights);
        for (views, cameras, frames, values) in &batches {
            let params = gpu.uniform(
                "fuse params",
                &FuseParams {
                    count: *count as u32,
                    views: *views,
                    nx: hull.shape[0] as u32,
                    ny: hull.shape[1] as u32,
                    nz: hull.shape[2] as u32,
                    pad: [0; 3],
                    truncation: truncation as f32,
                    behind: behind as f32,
                    behind_weight: config.behind_weight as f32,
                    free_weight: config.free_weight as f32,
                },
            );
            gpu.run(&kernel, &[&params, cameras, frames, values, &list.axes, voxels, &totals, &weights], *count as u64).await?;
        }
        total.extend(gpu.read::<f32>(&totals, *count).await?);
        weight.extend(gpu.read::<f32>(&weights, *count).await?);
    }

    // Support height: silhouettes cannot tell a flat base from a cone under it,
    // but no photo measures surface below the support. Take the lowest level
    // that still has well-supported measured surface.
    let (middle, down) = orbit_down(cameras, hull);
    let axes = [hull.axis(0), hull.axis(1), hull.axis(2)];
    let mut heights: Vec<f32> = Vec::new();
    for (n, &linear) in indices.iter().enumerate() {
        if weight[n] >= 3.0 && (total[n] / weight[n].max(1e-6)).abs() < 0.5 {
            let [i, j, k] = hull.unravel(linear);
            let p = [axes[0][i], axes[1][j], axes[2][k]];
            heights.push((p[0] - middle[0]) * down[0] + (p[1] - middle[1]) * down[1] + (p[2] - middle[2]) * down[2]);
        }
    }
    let mut height = None;
    if heights.len() > 1000 {
        heights.sort_unstable_by(f32::total_cmp);
        height = Some(heights[(0.998 * (heights.len() - 1) as f64) as usize]);
    }
    Ok(Fused { indices, total, weight, truncation, support: Support { point: middle, down, height } })
}

/// Removes the skirt that contact shadow leaves around the foot of an object.
///
/// A mask made by thresholding takes the dark contact shadow on the support
/// for object. Seen from all around, such a flat patch carves to a thin slab
/// of hull lying on the support, and where no depth was measured on it the
/// surface stage keeps the hull. The support is known here (the orbit normal
/// and the lowest well-supported measured surface), so the hull is looked at
/// as columns standing on it: a column that rises no more than `band_voxels`
/// above the support, with no measured surface clearly above the support in it
/// or next to it, is shadow, not object. Its voxels are marked as free space.
/// Columns under real geometry are taller or carry measured surface and stay.
/// Returns the number of voxels changed; nothing happens without a support height.
pub fn remove_skirt(hull: &Hull, fused: &mut Fused, band_voxels: f64) -> usize {
    let Some(support) = fused.support.height else { return 0 };
    if band_voxels <= 0.0 || fused.indices.is_empty() {
        return 0;
    }
    let voxel = hull.voxel as f32;
    let (point, down) = (fused.support.point, fused.support.down);
    // Two directions in the support plane.
    let helper = if down[0].abs() <= down[1].abs() && down[0].abs() <= down[2].abs() {
        [1.0, 0.0, 0.0]
    } else if down[1].abs() <= down[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let unit = |a: [f32; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-20);
        a.map(|v| v / n)
    };
    let u = unit(cross(down, helper));
    let v = cross(down, u);
    let axes = [hull.axis(0), hull.axis(1), hull.axis(2)];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // Height above the support (positive upwards) and column of every hull voxel.
    let located: Vec<(i32, i32, f32)> = fused
        .indices
        .iter()
        .map(|&linear| {
            let [i, j, k] = hull.unravel(linear);
            let p = [axes[0][i] - point[0], axes[1][j] - point[1], axes[2][k] - point[2]];
            ((dot(p, u) / voxel).floor() as i32, (dot(p, v) / voxel).floor() as i32, support - dot(p, down))
        })
        .collect();
    let (a0, b0) = (located.iter().map(|l| l.0).min().unwrap() - 1, located.iter().map(|l| l.1).min().unwrap() - 1);
    let (wide, deep) =
        ((located.iter().map(|l| l.0).max().unwrap() - a0 + 2) as usize, (located.iter().map(|l| l.1).max().unwrap() - b0 + 2) as usize);
    let column = |l: &(i32, i32, f32)| (l.1 - b0) as usize * wide + (l.0 - a0) as usize;
    let mut top = vec![f32::MIN; wide * deep];
    let mut measured = vec![false; wide * deep];
    for (n, l) in located.iter().enumerate() {
        let c = column(l);
        top[c] = top[c].max(l.2);
        let (total, weight) = (fused.total[n], fused.weight[n]);
        if weight >= 3.0 && (total / weight).abs() < 0.5 && l.2 > 1.5 * voxel {
            measured[c] = true;
        }
    }
    // A column is judged with its eight neighbours: next to a tall or measured column it stays.
    let band = band_voxels as f32 * voxel;
    let mut shadow = vec![false; wide * deep];
    for y in 1..deep - 1 {
        for x in 1..wide - 1 {
            let mut flat = top[y * wide + x] > f32::MIN;
            for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let c = (y as i64 + dy) as usize * wide + (x as i64 + dx) as usize;
                flat &= top[c] <= band && !measured[c];
            }
            shadow[y * wide + x] = flat;
        }
    }
    let mut removed = 0;
    for (n, l) in located.iter().enumerate() {
        if shadow[column(l)] {
            let weight = fused.weight[n].max(1.0);
            fused.weight[n] = weight;
            fused.total[n] = weight;
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hull::build_hull;
    use crate::inputs::Inputs;
    use crate::stereo::level::build_level;
    use crate::stereo::{options, synthetic};

    /// GPU fusion of exact sphere depth against the scalar votes (`CRISP3DS_GPU_TESTS=1`).
    #[test]
    fn gpu_fusion_matches_scalar_votes_and_finds_the_sphere() {
        if !crate::gpu::tests_enabled() {
            return;
        }
        let root = synthetic::temporary("fusion", 24, 128).unwrap();
        let overrides: Vec<String> = synthetic::SMALL.iter().map(|s| s.to_string()).collect();
        let config = options::build(None, &overrides).unwrap();
        let inputs = Inputs::load(&root.join("inputs"), &config).unwrap();
        crate::storage::remove_dir_all(&root).unwrap();
        let gpu = Gpu::new().unwrap();
        let state = crate::gpu::block_on(build_hull(&gpu, &inputs, &config, &mut None)).unwrap();
        let level = build_level(&inputs, 128);
        let depths: Vec<Plane<f32>> = level.iter().map(synthetic::sphere_depth).collect();
        let rim = 4;
        let fused = crate::gpu::block_on(tsdf(&gpu, &state.hull, &inputs.cameras, &level, &depths, rim, &config)).unwrap();
        assert_eq!(fused.indices.len(), state.hull.count());
        let (truncation, behind) = (fused.truncation as f32, (config.behind_voxels * state.hull.voxel) as f32);
        let trimmed: Vec<Plane<f32>> = level.iter().zip(&depths).map(|(v, d)| without_rim(v, d, rim)).collect();
        let axes = [state.hull.axis(0), state.hull.axis(1), state.hull.axis(2)];
        let (mut checked, mut different, mut wrong_side) = (0, 0, 0);
        for n in (0..fused.indices.len()).step_by(23) {
            let [i, j, k] = state.hull.unravel(fused.indices[n]);
            let point = [axes[0][i], axes[1][j], axes[2][k]];
            let (mut total, mut weight) = (0.0f32, 0.0f32);
            for (view, depth) in level.iter().zip(&trimmed) {
                let (t, w) = vote(view, depth, point, truncation, behind, &config);
                total += t;
                weight += w;
            }
            checked += 1;
            if (total - fused.total[n]).abs() > 1e-3 || (weight - fused.weight[n]).abs() > 1e-3 {
                different += 1;
            }
            // Well-observed voxels clearly inside or outside the sphere carry the right sign.
            let radius = point.iter().map(|v| v * v).sum::<f32>().sqrt();
            if fused.weight[n] >= 3.0 {
                let mean = fused.total[n] / fused.weight[n];
                wrong_side += ((radius < 0.9 && mean > 0.0) || (radius > 1.1 && mean < 0.0)) as usize;
            }
        }
        assert!(checked > 3000, "{checked}");
        assert!(different * 500 <= checked, "{different} of {checked} voxels differ");
        assert_eq!(wrong_side, 0);
        // The ring looks down by 10 degrees; the support is the sphere's lowest measured level.
        assert!(fused.support.down[2] < -0.99, "{:?}", fused.support.down);
        let height = fused.support.height.expect("support height");
        let lowest = fused.support.point[2] - height;
        assert!(lowest < -0.5 && lowest > -1.1, "{lowest}");
    }

    /// A box standing on a plane with a thin skirt around it: the skirt goes, the box stays.
    #[test]
    fn a_flat_unmeasured_skirt_is_removed_and_the_object_stays() {
        let shape = [40usize, 40, 30];
        let mut flags = vec![false; 40 * 40 * 30];
        let at = |i: usize, j: usize, k: usize| (i * 40 + j) * 30 + k;
        // The support is the plane z = 4 (down is -z here); the box rises to z = 24, the skirt to z = 6.
        for i in 4..36 {
            for j in 4..36 {
                let in_box = (14..26).contains(&i) && (14..26).contains(&j);
                for k in 4..if in_box { 24 } else { 6 } {
                    flags[at(i, j, k)] = true;
                }
            }
        }
        let hull = Hull::from_flags(&flags, shape, [0.0; 3], 1.0);
        let indices = hull.indices();
        let (mut total, mut weight) = (vec![0.0f32; indices.len()], vec![0.0f32; indices.len()]);
        // Measured surface on the sides and top of the box.
        for (n, &linear) in indices.iter().enumerate() {
            let [i, j, k] = hull.unravel(linear);
            let on_box_surface = (14..26).contains(&i) && (14..26).contains(&j) && (i == 14 || i == 25 || j == 14 || j == 25 || k == 23);
            if on_box_surface {
                (total[n], weight[n]) = (0.0, 5.0);
            }
        }
        let support = Support { point: [20.0, 20.0, 40.0], down: [0.0, 0.0, -1.0], height: Some(36.0) };
        let mut fused = Fused { indices, total, weight, truncation: 3.0, support };
        let removed = remove_skirt(&hull, &mut fused, 4.0);
        let state = |i: usize, j: usize, k: usize| {
            let n = fused.indices.binary_search(&(at(i, j, k) as u32)).unwrap();
            (fused.total[n], fused.weight[n])
        };
        assert_eq!(state(6, 6, 4), (1.0, 1.0), "skirt far from the box is free space");
        assert_eq!(state(30, 20, 5), (1.0, 1.0));
        assert_eq!(state(20, 20, 4), (0.0, 0.0), "the unmeasured inside of the box is untouched");
        assert_eq!(state(14, 20, 10), (0.0, 5.0), "measured surface is untouched");
        assert_eq!(state(13, 20, 4), (0.0, 0.0), "the column next to the box stays");
        assert!(removed > 1500 && removed < 2 * 32 * 32, "{removed}");
        // Nothing is removed without a support height or with the setting at zero.
        let mut none = Fused {
            support: Support { height: None, ..support },
            ..Fused {
                indices: fused.indices.clone(),
                total: vec![0.0; fused.indices.len()],
                weight: vec![0.0; fused.indices.len()],
                truncation: 3.0,
                support,
            }
        };
        assert_eq!(remove_skirt(&hull, &mut none, 4.0), 0);
        none.support.height = Some(36.0);
        assert_eq!(remove_skirt(&hull, &mut none, 0.0), 0);
    }

    #[test]
    fn votes_follow_the_reference_rules() {
        let config = DenseConfig { free_weight: 0.5, ..DenseConfig::default() };
        let camera =
            Camera { rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], translation: [0.0; 3], k: [10.0, 10.0, 2.0, 2.0] };
        let view =
            LevelView { width: 4, height: 4, camera, gray: Plane::new(4, 4), mask: Plane { width: 4, height: 4, data: vec![1; 16] } };
        let mut depth = Plane::<f32>::new(4, 4);
        depth.data[2 * 4 + 2] = 2.0; // pixel (2, 2): rays through x, y in (0.0, 0.1)
        let at = |z: f32| vote(&view, &depth, [0.05 * z, 0.05 * z, z], 0.1, 0.4, &config);
        let near = |found: (f32, f32), expected: (f32, f32)| (found.0 - expected.0).abs() < 1e-5 && found.1 == expected.1;
        assert!(near(at(1.95), (0.5, 1.0))); // in front, inside the truncation band
        assert!(near(at(1.5), (0.5, 0.5))); // free space: weight free_weight, value clamped to 1
        assert!(near(at(2.05), (-0.5, 1.0))); // just behind the surface
        assert!(near(at(2.3), (-0.25, 0.25))); // weak inside vote
        assert!(near(at(2.5), (0.0, 0.0))); // too far behind
        assert_eq!(vote(&view, &depth, [-0.5, 0.0, 2.0], 0.1, 0.4, &config), (0.0, 0.0)); // outside the map
                                                                                          // The rim is left to the hull: eroding the full mask by one pixel keeps everything (outside does not erode).
        assert_eq!(without_rim(&view, &depth, 1), depth);
    }
}
