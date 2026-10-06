//! Settings of the dense pipeline, mirroring `scripts/turntable_mesh/dense_config.py`.
//!
//! The Python `DenseConfig` is the source of truth. `tests/fixtures/dense-config-defaults.json`
//! is its default record; a test here fails when the two drift apart. Meanings of the
//! fields are documented in the pipeline README and in `dense_config.SETTINGS`.

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DenseConfig {
    pub sizes: Vec<i64>,
    pub crop_padding: i64,
    pub stretch_percentiles: Vec<f64>,
    pub neighbours: i64,
    pub best_of: i64,
    pub minimum_angle: f64,
    pub maximum_angle: f64,
    pub planes: i64,
    pub windows: Vec<i64>,
    pub aggregates: Vec<f64>,
    pub passes: Vec<i64>,
    pub band_first: Vec<i64>,
    pub band_later: i64,
    pub min_score: f64,
    pub min_variance: f64,
    pub window_fill: f64,
    pub tolerances: Vec<f64>,
    pub min_votes: Vec<i64>,
    pub vote_neighbours: i64,
    pub grid: i64,
    pub hull_dilate: i64,
    pub hull_allowed: i64,
    pub repair_masks: bool,
    pub repair_loose: i64,
    pub repair_rounds: i64,
    pub repair_base_margin: f64,
    pub hull_front: bool,
    pub hull_front_level: i64,
    pub hull_front_min_score: f64,
    pub hull_front_margin: f64,
    pub hull_front_stride: i64,
    pub fused_passes: i64,
    pub fused_band: i64,
    pub fallback_level: bool,
    pub rim_fraction: f64,
    pub truncation_voxels: f64,
    pub behind_voxels: f64,
    pub behind_weight: f64,
    pub free_weight: f64,
    pub mesh_smooth: f64,
    pub mesh_fill_sigmas: Vec<f64>,
    pub mesh_final_smooth: f64,
    pub mesh_minimum_weight: f64,
    pub mesh_confidence_cap: f64,
    pub mesh_taubin_cycles: i64,
    pub mesh_flat_base: bool,
    pub mesh_base_margin: f64,
}

impl Default for DenseConfig {
    fn default() -> Self {
        DenseConfig {
            sizes: vec![256, 512, 1024],
            crop_padding: 24,
            stretch_percentiles: vec![1.0, 99.0],
            neighbours: 6,
            best_of: 3,
            minimum_angle: 3.0,
            maximum_angle: 40.0,
            planes: 96,
            windows: vec![7, 9, 11],
            aggregates: vec![1.0, 1.5, 2.0],
            passes: vec![1, 2, 1],
            band_first: vec![8, 8, 5],
            band_later: 5,
            min_score: 0.55,
            min_variance: 0.0001,
            window_fill: 0.6,
            tolerances: vec![0.006, 0.003, 0.002],
            min_votes: vec![2, 3, 3],
            vote_neighbours: 10,
            grid: 400,
            hull_dilate: 2,
            hull_allowed: 2,
            repair_masks: true,
            repair_loose: 0,
            repair_rounds: 2,
            repair_base_margin: 20.0,
            hull_front: true,
            hull_front_level: 1,
            hull_front_min_score: 0.6,
            hull_front_margin: 0.004,
            hull_front_stride: 2,
            fused_passes: 0,
            fused_band: 4,
            fallback_level: true,
            rim_fraction: 0.55,
            truncation_voxels: 3.0,
            behind_voxels: 12.0,
            behind_weight: 0.25,
            free_weight: 1.0,
            mesh_smooth: 1.0,
            mesh_fill_sigmas: vec![2.0, 4.0],
            mesh_final_smooth: 0.6,
            mesh_minimum_weight: 0.5,
            mesh_confidence_cap: 4.0,
            mesh_taubin_cycles: 5,
            mesh_flat_base: true,
            mesh_base_margin: 1.0,
        }
    }
}

impl DenseConfig {
    /// Reads a `config.json` (or a `result.json` carrying a `configuration` object).
    /// Unknown keys are ignored and missing ones take their defaults.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(&crate::storage::read_to_string(path)?)?;
        let value = value.get("configuration").cloned().unwrap_or(value);
        Ok(serde_json::from_value(value)?)
    }

    /// Per-level value; a shorter list repeats its last entry.
    pub fn level<T: Copy>(values: &[T], index: usize) -> T {
        values[index.min(values.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_python_reference() {
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/dense-config-defaults.json");
        let reference: serde_json::Value = serde_json::from_str(&crate::storage::read_to_string(fixture).unwrap()).unwrap();
        let ours = serde_json::to_value(DenseConfig::default()).unwrap();
        assert_eq!(ours, reference, "DenseConfig drifted from scripts/turntable_mesh/dense_config.py");
    }

    #[test]
    fn level_repeats_the_last_entry() {
        assert_eq!(DenseConfig::level(&[7, 9, 11], 1), 9);
        assert_eq!(DenseConfig::level(&[7, 9, 11], 5), 11);
    }
}
