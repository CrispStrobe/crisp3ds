//! The scalar field whose zero level is the surface (port of `tsdf_hull_mesh.field`).
//!
//! Arithmetic follows the reference operation by operation, including where NumPy works in
//! `float32` and where in `float64`, so that both produce the same field up to the rounding
//! of the Gaussian sums.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;

use super::edt;
use super::gaussian::gaussian_filter;
use super::grid::{count, Dims};
use crate::config::DenseConfig;
use crate::npz::{Array, Npz};

/// Voxels of padding around the hull grid on every side.
pub const PAD: usize = 3;

/// The support plane the object stood on, as `multiscale_stereo` stores it.
pub struct Support {
    pub point: Array,
    pub down: Vec<f64>,
    pub height: f64,
}

/// Content of `volume.npz`: fused evidence at the voxels of the silhouette hull.
pub struct Volume {
    pub shape: Dims,
    /// Voxel indices of the hull voxels, three per voxel.
    pub index: Vec<i64>,
    pub total: Vec<f32>,
    pub weight: Vec<f32>,
    pub origin: Array,
    pub voxel: f64,
    /// Truncation distance in scene units.
    pub truncation: f64,
    pub support: Option<Support>,
}

impl Volume {
    pub fn read(path: &Path) -> Result<Self> {
        let npz = Npz::read(path)?;
        let shape = npz.get("shape")?.to_f64();
        ensure!(shape.len() == 3 && shape.iter().all(|&s| s >= 1.0 && s.fract() == 0.0), "volume shape must be three positive integers");
        let index = npz.get("index")?;
        ensure!(index.shape.len() == 2 && index.shape[1] == 3, "volume index must be N x 3, found {:?}", index.shape);
        let total = npz.get("total")?.to_f32();
        let weight = npz.get("weight")?.to_f32();
        ensure!(total.len() == index.shape[0] && weight.len() == index.shape[0], "total and weight must have one entry per index row");
        let origin = npz.get("origin")?.clone();
        ensure!(origin.len() == 3, "volume origin must have three entries");
        let support = if npz.contains("support_height") {
            let (point, down) = (npz.get("support_point")?.clone(), npz.get("support_down")?.to_f64());
            ensure!(point.len() == 3 && down.len() == 3, "support point and direction must have three entries");
            Some(Support { point, down, height: npz.get("support_height")?.scalar()? })
        } else {
            None
        };
        Ok(Volume {
            shape: [shape[0] as usize, shape[1] as usize, shape[2] as usize],
            index: index.to_i64().context("volume index")?,
            total,
            weight,
            origin,
            voxel: npz.get("voxel")?.scalar()?,
            truncation: npz.get("truncation")?.scalar()?,
            support,
        })
    }
}

/// What `field` reports besides the field itself.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldReport {
    pub flat_base_applied: bool,
    pub hull_fraction_below_support: Option<f64>,
    pub observed_hull_fraction: f64,
    pub extrapolated_hull_fraction: f64,
}

pub struct Field {
    /// Padded grid, negative inside.
    pub value: Vec<f32>,
    pub dims: Dims,
    pub report: FieldReport,
}

pub fn field(volume: &Volume, config: &DenseConfig) -> Result<Field> {
    let dims = volume.shape.map(|s| s + 2 * PAD);
    let total_voxels = count(dims);
    ensure!(total_voxels < u32::MAX as usize, "volume grid is too large");
    let truncation = volume.truncation / volume.voxel;
    ensure!(truncation.is_finite() && truncation > 0.0, "volume truncation and voxel size must be positive");
    let hull_count = volume.index.len() / 3;
    ensure!(hull_count > 0, "the volume has no hull voxels");
    let mut place = Vec::with_capacity(hull_count);
    for voxel in volume.index.as_chunks::<3>().0 {
        for axis in 0..3 {
            if voxel[axis] < 0 || voxel[axis] >= volume.shape[axis] as i64 {
                bail!("hull voxel {voxel:?} lies outside the volume shape {:?}", volume.shape);
            }
        }
        let [x, y, z] = [voxel[0] as usize + PAD, voxel[1] as usize + PAD, voxel[2] as usize + PAD];
        place.push(((x * dims[1] + y) * dims[2] + z) as u32);
    }

    // Silhouette prior: signed distance to the hull boundary in truncation units. A voxel
    // centre is half a voxel from the boundary between it and its neighbour.
    let mut hull = vec![false; total_voxels];
    for &p in &place {
        hull[p as usize] = true;
    }
    let mut value: Vec<f32> = {
        let outside = edt::squared_distance(|i| hull[i], dims);
        let inside = edt::squared_distance(|i| !hull[i], dims);
        outside
            .par_iter()
            .zip(&inside)
            .map(|(&out, &ins)| {
                let signed = f64::from(out).sqrt() - f64::from(ins).sqrt();
                ((signed + if signed > 0.0 { -0.5 } else { 0.5 }) / truncation).clamp(-1.0, 1.0) as f32
            })
            .collect()
    };
    drop(hull);

    // Averaged signed distance and a confidence that saturates at a few views.
    let minimum = config.mesh_minimum_weight as f32;
    let cap = config.mesh_confidence_cap as f32;
    let observed: Vec<bool> = volume.weight.iter().map(|&w| w >= minimum).collect();
    let mut weighted = vec![0f32; total_voxels];
    let mut confidence = vec![0f32; total_voxels];
    for (i, &p) in place.iter().enumerate() {
        if observed[i] {
            let average = volume.total[i] / volume.weight[i].max(1e-6);
            let c = volume.weight[i].min(cap) / cap;
            weighted[p as usize] = average * c;
            confidence[p as usize] = c;
        }
    }

    // Confidence-weighted smoothing where observed, then bounded extrapolation into nearby
    // unobserved hull voxels; what stays unfilled keeps the prior. A filled voxel never drops
    // below the prior (`max(value, prior)` in the reference; each voxel is filled at most
    // once, so `value` still holds its prior when it is filled).
    let mut filled = vec![false; hull_count];
    let mut work = vec![0f32; total_voxels];
    let mut numerator = vec![0f32; hull_count];
    for &sigma in std::iter::once(&config.mesh_smooth).chain(&config.mesh_fill_sigmas) {
        work.copy_from_slice(&weighted);
        gaussian_filter(&mut work, dims, sigma);
        for (slot, &p) in numerator.iter_mut().zip(&place) {
            *slot = work[p as usize];
        }
        work.copy_from_slice(&confidence);
        gaussian_filter(&mut work, dims, sigma);
        for (i, &p) in place.iter().enumerate() {
            let denominator = work[p as usize];
            if !filled[i] && denominator > 0.02 {
                let prior = value[p as usize];
                value[p as usize] = (numerator[i] / denominator).max(prior);
                filled[i] = true;
            }
        }
    }
    drop((weighted, confidence, work, numerator));

    let mut report = FieldReport {
        flat_base_applied: false,
        hull_fraction_below_support: None,
        observed_hull_fraction: observed.iter().filter(|&&o| o).count() as f64 / hull_count as f64,
        extrapolated_hull_fraction: filled.iter().zip(&observed).filter(|(&f, &o)| f && !o).count() as f64 / hull_count as f64,
    };

    if let Some(support) = volume.support.as_ref().filter(|s| config.mesh_flat_base && s.height.is_finite()) {
        // Signed distance to the support plane in truncation units, positive below it.
        let voxel = volume.voxel;
        let down = &support.down;
        let difference: Vec<f64> = if volume.origin.is_f32() && support.point.is_f32() {
            volume.origin.to_f32().iter().zip(support.point.to_f32()).map(|(&o, p)| f64::from(o - p)).collect()
        } else {
            volume.origin.to_f64().iter().zip(support.point.to_f64()).map(|(&o, p)| o - p).collect()
        };
        let offset = (difference[0] * down[0] + difference[1] * down[1] + difference[2] * down[2] - support.height) as f32;
        let axes: Vec<Vec<f32>> =
            (0..3).map(|a| (0..dims[a]).map(|i| ((i as f64 - PAD as f64 + 0.5) * voxel * down[a]) as f32).collect()).collect();
        let (voxel, margin, truncation) = (voxel as f32, config.mesh_base_margin as f32, truncation as f32);
        let plane = |x: usize, y: usize, z: usize| ((axes[0][x] + axes[1][y] + axes[2][z] + offset) / voxel - margin) / truncation;
        let plane = |x: usize, y: usize, z: usize| plane(x, y, z).clamp(-1.0, 1.0);
        let voxels = volume.index.as_chunks::<3>().0.iter();
        let below = voxels.filter(|v| plane(v[0] as usize + PAD, v[1] as usize + PAD, v[2] as usize + PAD) > 0.0).count();
        report.flat_base_applied = true;
        report.hull_fraction_below_support = Some(below as f64 / hull_count as f64);
        value.par_chunks_mut(dims[1] * dims[2]).enumerate().for_each(|(x, slab)| {
            for (y, row) in slab.chunks_exact_mut(dims[2]).enumerate() {
                for (z, v) in row.iter_mut().enumerate() {
                    *v = v.max(plane(x, y, z));
                }
            }
        });
    }

    // The maximum leaves creases that would give ambiguous cells.
    if config.mesh_final_smooth != 0.0 {
        gaussian_filter(&mut value, dims, config.mesh_final_smooth);
    }
    Ok(Field { value, dims, report })
}
