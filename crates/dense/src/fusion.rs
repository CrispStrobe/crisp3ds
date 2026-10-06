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
pub fn tsdf(
    gpu: &Gpu,
    hull: &Hull,
    cameras: &[Camera],
    level: &[LevelView],
    depths: &[Plane<f32>],
    rim_pixels: usize,
    config: &DenseConfig,
) -> anyhow::Result<Fused> {
    let kernel = gpu.kernel("fuse", include_str!("shaders/fuse.wgsl"), "main")?;
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
            gpu.run(&kernel, &[&params, cameras, frames, values, &list.axes, voxels, &totals, &weights], *count as u64)?;
        }
        total.extend(gpu.read::<f32>(&totals, *count)?);
        weight.extend(gpu.read::<f32>(&weights, *count)?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hull::build_hull;
    use crate::inputs::Inputs;
    use crate::stereo::level::build_level;
    use crate::stereo::{options, synthetic};

    /// Exact z-depth of the unit sphere at every level pixel.
    pub fn sphere_depth(view: &LevelView) -> Plane<f32> {
        let origin = view.camera.centre().map(|v| v as f64);
        let mut depth = Plane::<f32>::new(view.width, view.height);
        for y in 0..view.height {
            for x in 0..view.width {
                let at_one = view.unproject(x, y, 1.0);
                let d = [0, 1, 2].map(|a| at_one[a] as f64 - origin[a]);
                let (a, b) = (d.iter().map(|v| v * v).sum::<f64>(), d.iter().zip(&origin).map(|(d, o)| d * o).sum::<f64>());
                let c = origin.iter().map(|v| v * v).sum::<f64>() - 1.0;
                let discriminant = b * b - a * c;
                if discriminant > 0.0 {
                    depth.data[y * view.width + x] = ((-b - discriminant.sqrt()) / a) as f32;
                }
            }
        }
        depth
    }

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
        std::fs::remove_dir_all(&root).unwrap();
        let gpu = Gpu::new().unwrap();
        let state = build_hull(&gpu, &inputs, &config, &mut None).unwrap();
        let level = build_level(&inputs, 128);
        let depths: Vec<Plane<f32>> = level.iter().map(sphere_depth).collect();
        let rim = 4;
        let fused = tsdf(&gpu, &state.hull, &inputs.cameras, &level, &depths, rim, &config).unwrap();
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
