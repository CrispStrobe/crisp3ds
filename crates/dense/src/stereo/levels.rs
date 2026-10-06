//! The coarse-to-fine level loop of `run` in `multiscale_stereo.py`: a full
//! sweep at the coarsest level, band refinements around the upsampled depth at
//! finer ones, the hull-front candidate for thin parts, cross-view agreement
//! after every pass, and the fallback merge at the end.

use std::path::Path;
use std::time::Instant;

use serde_json::{json, Value};

use crate::config::DenseConfig;
use crate::events::EventLog;
use crate::fusion::tsdf;
use crate::gpu::Gpu;
use crate::hull::HullState;
use crate::inputs::{parallel_map, round_half_even, Inputs, Plane};

use super::depth::{consistent, hull_front, initial, merge_fallback};
use super::level::{build_level, coverage, LevelView};
use super::matcher::Matcher;
use super::previews::depth_sheet;
use super::run::{log, Previews};

/// Everything the loop needs from the driver.
#[derive(Clone, Copy)]
pub struct LevelContext<'a> {
    pub gpu: &'a Gpu,
    pub inputs: &'a Inputs,
    pub state: &'a HullState,
    pub config: &'a DenseConfig,
    pub output: &'a Path,
    pub events: &'a EventLog,
    pub previews: &'a Previews<'a>,
    pub picks: &'a [usize],
}

/// Runs every level in `sizes`. Returns the finest level and its final depth
/// maps; level rows and the fallback coverage are added to `report`.
pub fn match_levels(context: &LevelContext, sizes: &[i64], report: &mut Value) -> anyhow::Result<(Vec<LevelView>, Vec<Plane<f32>>)> {
    let LevelContext { gpu, inputs, state, config, output, events, previews, picks } = *context;
    let count = inputs.count();
    let neighbours: Vec<Vec<usize>> =
        (0..count).map(|i| inputs.neighbours(i, config.neighbours as usize, config)).collect::<anyhow::Result<_>>()?;
    let voters: Vec<Vec<usize>> =
        (0..count).map(|i| inputs.neighbours(i, config.vote_neighbours as usize, config)).collect::<anyhow::Result<_>>()?;
    let mut matcher = Matcher::new(gpu, config, &state.search)?;
    let mut level: Vec<LevelView> = Vec::new();
    let mut depths: Vec<Plane<f32>> = Vec::new();
    let mut coarser: Option<Vec<Plane<f32>>> = None;
    // As in the reference, one step of inverse depth serves all views of the finer
    // levels: the step of the last view swept at the coarsest level.
    let mut step = 0.0f64;
    let mut rows: Vec<Value> = Vec::new();
    for (li, &size) in sizes.iter().enumerate() {
        let t = Instant::now();
        level = build_level(inputs, size);
        let buffers = matcher.upload(&level);
        let window = DenseConfig::level(&config.windows, li);
        let aggregate = DenseConfig::level(&config.aggregates, li);
        let passes = if li == 0 { 1 } else { DenseConfig::level(&config.passes, li) };
        let scale = 2f64.powi(li as i32);
        for p in 0..passes {
            let t_pass = Instant::now();
            let mut initial_seconds = 0.0;
            let mut raw: Vec<Plane<f32>> = Vec::with_capacity(count);
            let mut front_wins = 0usize;
            // Initial surfaces of a group of views at a time: CPU work in parallel, bounded memory.
            let mut inits: Vec<Plane<f32>> = Vec::new();
            let mut inits_from = 0usize;
            for i in 0..count {
                let mut d = if li == 0 {
                    let (d, view_step) =
                        matcher.sweep(&buffers, &level, i, &neighbours[i], state.bounds[i], window, aggregate, config.min_score)?;
                    step = view_step;
                    d
                } else {
                    if i >= inits_from + inits.len() {
                        let t_init = Instant::now();
                        inits_from = i;
                        let sigma = if p == 0 { 1.5 } else { 1.0 };
                        inits = parallel_map((count - i).min(16), |n| initial(&depths[i + n], &level[i + n].mask, sigma));
                        initial_seconds += t_init.elapsed().as_secs_f64();
                    }
                    let fine = step / scale / if p == 0 { 1.0 } else { 2.0 };
                    let half = if p == 0 { DenseConfig::level(&config.band_first, li) } else { config.band_later };
                    matcher.refine(
                        &buffers,
                        &level,
                        i,
                        &neighbours[i],
                        &inits[i - inits_from],
                        fine,
                        half,
                        window,
                        aggregate,
                        config.min_score,
                    )?
                };
                if config.hull_front && li == (config.hull_front_level as usize).min(sizes.len() - 1) && p == 0 {
                    // Thin parts are narrower than the coarse window, so they inherit the
                    // depth of whatever lies behind them. Test the hull's front surface
                    // too: a photo-consistent nearer surface occludes anything matched behind it.
                    let front = hull_front(&level[i], &state.hull, state.bounds[i], config.hull_front_stride as usize);
                    let half = DenseConfig::level(&config.band_first, li);
                    let minimum = config.min_score.max(config.hull_front_min_score);
                    let d2 = matcher.refine(&buffers, &level, i, &neighbours[i], &front, step / scale, half, window, aggregate, minimum)?;
                    let margin = (1.0 - config.hull_front_margin) as f32;
                    for (value, &candidate) in d.data.iter_mut().zip(&d2.data) {
                        if candidate > 0.0 && (*value == 0.0 || candidate < *value * margin) {
                            *value = candidate;
                            front_wins += 1;
                        }
                    }
                }
                raw.push(d);
                if i % 10 == 9 {
                    let done = (li as f64 + (p as f64 + (i + 1) as f64 / count as f64) / passes as f64) / sizes.len() as f64;
                    events.progress(
                        0.08 + 0.84 * done,
                        &format!("Matching level {} of {}, view {} of {}", li + 1, sizes.len(), i + 1, count),
                    )?;
                }
            }
            let t_c = Instant::now();
            let agreed =
                consistent(&level, &raw, &voters, DenseConfig::level(&config.tolerances, li), DenseConfig::level(&config.min_votes, li));
            let row = json!({
                "size": size, "pass": p, "window": window, "working_size": [level[0].width, level[0].height],
                "raw_coverage": coverage(&level, &raw), "consistent_coverage": coverage(&level, &agreed),
                "hull_front_pixels": front_wins, "consistency_seconds": t_c.elapsed().as_secs_f64(),
                "initial_seconds": initial_seconds, "pass_seconds": t_pass.elapsed().as_secs_f64(),
                "seconds": t.elapsed().as_secs_f64(),
            });
            log(format!(
                "level {li} pass {p}: size {size}, coverage {:.3}, {:.1}s",
                row["consistent_coverage"]["median"].as_f64().unwrap_or(0.0),
                row["seconds"].as_f64().unwrap_or(0.0)
            ));
            rows.push(row);
            depths = agreed;
        }
        let sheet = output.join(format!("depth-level-{li}.png"));
        depth_sheet(&sheet, &level, &depths, picks)?;
        events.artifact("depth_sheet", &sheet, &format!("Depth at level {} (photo, depth, shading)", li + 1), json!({"level": li}))?;
        events.metric(&format!("coverage_level_{li}"), rows.last().unwrap()["consistent_coverage"]["median"].as_f64().unwrap_or(0.0))?;
        if previews.enabled() && li + 1 < sizes.len() {
            let rim = round_half_even(config.rim_fraction * window as f64) as usize;
            let fused = tsdf(gpu, &state.hull, &inputs.cameras, &level, &depths, rim, config)?;
            previews.write(
                &format!("1{li}-level-{li}"),
                &format!("Surface after level {}", li + 1),
                &state.hull,
                &fused.indices,
                &fused.total,
                &fused.weight,
                Some(&fused.support),
                json!({"level": li}),
            )?;
        }
        if config.fallback_level && li + 1 == sizes.len() {
            if let Some(coarse) = &coarser {
                // Keep the finest depth; where it failed its checks, fall back to the
                // previous level's consistent depth instead of leaving a hole.
                depths = parallel_map(count, |i| merge_fallback(&level[i], &depths[i], &coarse[i]));
                report["fallback_coverage"] = coverage(&level, &depths);
                let sheet = output.join("depth-merged.png");
                depth_sheet(&sheet, &level, &depths, picks)?;
                events.artifact("depth_sheet", &sheet, "Final depth with level fallback", json!({"level": li}))?;
            }
        }
        coarser = Some(depths.clone());
    }
    report["levels"] = json!(rows);
    report["gpu_peak_binding_bytes"] = json!(gpu.peak_binding());
    Ok((level, depths))
}
