//! The stage driver: port of `run` in `multiscale_stereo.py`.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context};
use serde_json::{json, Value};

use crate::config::DenseConfig;
use crate::events::EventLog;
use crate::fusion::{tsdf, Fused, Support};
use crate::gpu::Gpu;
use crate::hull::{build_hull, Hull, HullState, VoxelList};
use crate::inputs::{parallel_map, round_half_even, Inputs, Plane};
use crate::npz::{self, Array, Data, Npz};
use crate::repair::{repair_masks, Coverer};

use super::level::{build_level, level_sizes};
use super::previews::colour_sheet;
use super::Arguments;

/// `index` rows (x, y, z) of the listed voxels as an N x 3 int32 array.
pub fn index_array(hull: &Hull, indices: &[u32]) -> Array {
    let mut data = Vec::with_capacity(indices.len() * 3);
    for &linear in indices {
        let [i, j, k] = hull.unravel(linear);
        data.extend_from_slice(&[i as i32, j as i32, k as i32]);
    }
    Array::new(&[indices.len(), 3], Data::I32(data))
}

/// Writes a volume file with the reference's member names, types and shapes.
#[allow(clippy::too_many_arguments)]
pub fn write_volume(
    path: &Path,
    compress: bool,
    hull: &Hull,
    indices: &[u32],
    total: &[f32],
    weight: &[f32],
    truncation: f64,
    support: Option<&Support>,
) -> anyhow::Result<()> {
    let f32s = |values: &[f32]| Array::new(&[values.len()], Data::F32(values.to_vec()));
    let index = index_array(hull, indices);
    let (total, weight) = (f32s(total), f32s(weight));
    let shape = Array::new(&[3], Data::I64(hull.shape.iter().map(|&s| s as i64).collect()));
    let origin = f32s(&hull.origin);
    let voxel = Array::scalar_f32(hull.voxel as f32);
    let truncation = Array::scalar_f32(truncation as f32);
    let mut arrays: Vec<(&str, &Array)> = vec![
        ("index", &index),
        ("total", &total),
        ("weight", &weight),
        ("shape", &shape),
        ("origin", &origin),
        ("voxel", &voxel),
        ("truncation", &truncation),
    ];
    let extra = support.map(|s| (f32s(&s.point), f32s(&s.down), Array::scalar_f32(s.height.unwrap_or(f32::NAN))));
    if let Some((point, down, height)) = &extra {
        arrays.extend([("support_point", point), ("support_down", down), ("support_height", height)]);
    }
    npz::write(path, &arrays, compress)
}

/// `depths.npz`: one `depth_NNN` float32 array (height x width) per view.
pub fn write_depths(path: &Path, depths: &[Plane<f32>]) -> anyhow::Result<()> {
    let arrays: Vec<(String, Array)> = depths
        .iter()
        .enumerate()
        .map(|(i, d)| (format!("depth_{i:03}"), Array::new(&[d.height, d.width], Data::F32(d.data.clone()))))
        .collect();
    let members: Vec<(&str, &Array)> = arrays.iter().map(|(name, array)| (name.as_str(), array)).collect();
    npz::write(path, &members, true)
}

pub fn read_depths(path: &Path, count: usize) -> anyhow::Result<Vec<Plane<f32>>> {
    let archive = Npz::read(path)?;
    (0..count)
        .map(|i| {
            let array = archive.get(&format!("depth_{i:03}"))?;
            if array.shape.len() != 2 {
                bail!("depth_{i:03} is not an image");
            }
            Ok(Plane { width: array.shape[1], height: array.shape[0], data: array.to_f32() })
        })
        .collect()
}

fn log(text: impl AsRef<str>) {
    use std::io::Write;
    println!("{}", text.as_ref());
    let _ = std::io::stdout().flush();
}

/// Volumes the driver meshes coarsely while matching continues, renamed into place when complete.
struct Previews<'a> {
    directory: Option<std::path::PathBuf>,
    events: &'a EventLog,
    config: &'a DenseConfig,
}

impl Previews<'_> {
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        name: &str,
        label: &str,
        hull: &Hull,
        indices: &[u32],
        total: &[f32],
        weight: &[f32],
        support: Option<&Support>,
        meta: Value,
    ) -> anyhow::Result<()> {
        let Some(directory) = &self.directory else { return Ok(()) };
        let (partial, target) = (directory.join(format!("{name}.partial")), directory.join(format!("{name}.npz")));
        write_volume(&partial, false, hull, indices, total, weight, self.config.truncation_voxels * hull.voxel, support)?;
        std::fs::rename(&partial, &target)?;
        self.events.artifact("preview_volume", &target, label, meta)
    }

    fn hull(&self, name: &str, label: &str, hull: &Hull) -> anyhow::Result<()> {
        if self.directory.is_none() {
            return Ok(());
        }
        let indices = hull.indices();
        let zeros = vec![0.0f32; indices.len()];
        self.write(name, label, hull, &indices, &zeros, &zeros, None, json!({}))
    }
}

fn hull_report(state: &HullState, since: Instant) -> anyhow::Result<Value> {
    let mut report = serde_json::to_value(&state.report)?;
    report["seconds"] = json!(since.elapsed().as_secs_f64());
    Ok(report)
}

pub fn run(arguments: &Arguments, config: &DenseConfig) -> anyhow::Result<Value> {
    let output = &arguments.output;
    if output.exists() {
        bail!("output directory exists: {}", output.display());
    }
    std::fs::create_dir_all(output).with_context(|| output.display().to_string())?;
    let started = Instant::now();
    let events = EventLog::new(arguments.events.as_deref(), "stereo");
    let gpu = Gpu::new()?;
    let mut report = json!({
        "configuration": serde_json::to_value(config)?,
        "device": format!("wgpu: {}", gpu.describe()),
        "engine": concat!("crisp3ds-dense ", env!("CARGO_PKG_VERSION")),
        "levels": [],
        "reuse_depths": arguments.reuse_depths.as_ref().map(|p| p.display().to_string()),
    });
    let t = Instant::now();
    let mut inputs = Inputs::load(&arguments.inputs, config)?;
    let count = inputs.count();
    report["views"] = json!(count);
    report["load_seconds"] = json!(t.elapsed().as_secs_f64());
    let picks = [count / 7, (count / 2).saturating_sub(3), (4 * count) / 5];
    let previews = Previews { directory: arguments.previews.then(|| output.join("preview")), events: &events, config };
    if let Some(directory) = &previews.directory {
        std::fs::create_dir(directory)?;
    }

    let t = Instant::now();
    let mut voxel = None;
    let mut state = build_hull(&gpu, &inputs, config, &mut voxel)?;
    report["hull"] = hull_report(&state, t)?;
    log(format!("hull {}", report["hull"]));
    events.progress(0.03, "Silhouette hull built")?;
    previews.hull("00-hull", "Silhouette hull", &state.hull)?;

    if arguments.only.as_deref() == Some("hull") {
        // Diagnostic: the hull and what `Stereo.__init__` derived, for parity checks.
        let indices = state.hull.indices();
        let zeros = vec![0.0f32; indices.len()];
        let truncation = config.truncation_voxels * state.hull.voxel;
        write_volume(&output.join("volume.npz"), true, &state.hull, &indices, &zeros, &zeros, truncation, None)?;
        report["bounds"] = json!(state.bounds);
        report["boxes"] = json!(inputs.boxes);
        report["longest"] = json!(inputs.longest);
        report["repair_loose"] = json!(inputs.repair_loose);
        report["gray_crc32"] = json!(inputs.gray.iter().map(|g| crc32fast::hash(bytemuck::cast_slice(&g.data))).collect::<Vec<_>>());
        let neighbours: Vec<Vec<usize>> =
            (0..count).map(|i| inputs.neighbours(i, config.vote_neighbours as usize, config)).collect::<anyhow::Result<_>>()?;
        report["neighbours"] = json!(neighbours);
        return finish(report, output, started);
    }

    if config.repair_masks {
        let t = Instant::now();
        let before: Vec<Plane<u8>> = picks.iter().map(|&i| inputs.masks[i].clone()).collect();
        let mut rounds = Vec::new();
        for _ in 0..config.repair_rounds {
            rounds.push(repair_masks(&gpu, &mut inputs, &state, config)?);
            state = build_hull(&gpu, &inputs, config, &mut voxel)?;
            report["hull_repaired"] = hull_report(&state, t)?;
        }
        let added: f64 = rounds.iter().map(|r| r.added_fraction_median).sum();
        report["mask_repair"] = json!({"loose_views": inputs.repair_loose, "rounds": rounds, "added_fraction_median": added});
        log(format!("repair {} {}", report["mask_repair"], report["hull_repaired"]));
        std::fs::create_dir(output.join("masks-repaired"))?;
        let written: Vec<anyhow::Result<()>> = parallel_map(count, |n| {
            let mask = &inputs.masks[n];
            let pixels: Vec<u8> = mask.data.iter().map(|&m| m * 255).collect();
            let path = output.join("masks-repaired").join(format!("{}.png", inputs.rows[n].name));
            image::GrayImage::from_raw(mask.width as u32, mask.height as u32, pixels).expect("mask size").save(&path)?;
            Ok(())
        });
        written.into_iter().collect::<anyhow::Result<()>>()?;
        let sheet = output.join("mask-repair.png");
        colour_sheet(&sheet, &inputs, &picks, |i| {
            let old = &before[picks.iter().position(|&p| p == i).unwrap()];
            let new = &inputs.masks[i];
            let added = new.data.iter().zip(&old.data).map(|(&n, &o)| (n != 0 && o == 0) as u8).collect();
            vec![(Plane { width: new.width, height: new.height, data: added }, [0.1, 0.9, 0.2])]
        })?;
        events.artifact("mask_repair_sheet", &sheet, "Mask repair (added pixels in green)", json!({}))?;
        events.metric("mask_added_fraction_median", added)?;
        previews.hull("01-hull-repaired", "Silhouette hull after mask repair", &state.hull)?;
    }
    if arguments.only.as_deref() == Some("repair") {
        return finish(report, output, started);
    }

    {
        let list = VoxelList::new(&gpu, &state.hull, &state.hull.indices());
        let coverer = Coverer::new(&gpu)?;
        let mut covers = Vec::new();
        for &i in &picks {
            covers.push(coverer.cover(&list, &inputs.cameras[i], inputs.masks[i].width, inputs.masks[i].height)?);
        }
        let sheet = output.join("hull-vs-mask.png");
        colour_sheet(&sheet, &inputs, &picks, |i| {
            let cover = &covers[picks.iter().position(|&p| p == i).unwrap()];
            let mask = &inputs.masks[i];
            let plane = |data: Vec<u8>| Plane { width: mask.width, height: mask.height, data };
            let uncovered = mask.data.iter().zip(&cover.data).map(|(&m, &c)| (m != 0 && c == 0) as u8).collect();
            let outside = mask.data.iter().zip(&cover.data).map(|(&m, &c)| (m == 0 && c != 0) as u8).collect();
            vec![(plane(uncovered), [1.0, 0.1, 0.1]), (plane(outside), [0.1, 0.5, 1.0])]
        })?;
        events.artifact("hull_mask_sheet", &sheet, "Hull against masks (red: mask not covered, blue: hull outside mask)", json!({}))?;
    }
    events.progress(0.08, "Masks and hull ready")?;

    let sizes = level_sizes(&config.sizes, inputs.longest);
    let (level, depths) = match &arguments.reuse_depths {
        Some(path) => (build_level(&inputs, *sizes.last().unwrap()), read_depths(path, count)?),
        None => bail!("matching is not ported yet; pass --reuse-depths"),
    };

    let t = Instant::now();
    let rim = round_half_even(config.rim_fraction * DenseConfig::level(&config.windows, sizes.len() - 1) as f64) as usize;
    let fused: Fused = tsdf(&gpu, &state.hull, &inputs.cameras, &level, &depths, rim, config)?;
    report["rim_pixels"] = json!(rim);
    report["fused_passes"] = json!([]);
    let observed = fused.weight.iter().filter(|&&w| w > 0.0).count() as f64 / fused.weight.len().max(1) as f64;
    report["tsdf"] = json!({"hull_voxels": fused.indices.len(), "observed_fraction": observed, "seconds": t.elapsed().as_secs_f64()});
    log(format!("tsdf {}", report["tsdf"]));
    events.progress(0.97, "Depth fused")?;

    let t = Instant::now();
    write_volume(
        &output.join("volume.npz"),
        true,
        &state.hull,
        &fused.indices,
        &fused.total,
        &fused.weight,
        fused.truncation,
        Some(&fused.support),
    )?;
    write_depths(&output.join("depths.npz"), &depths)?;
    report["write_seconds"] = json!(t.elapsed().as_secs_f64());
    finish(report, output, started)
}

fn finish(mut report: Value, output: &Path, started: Instant) -> anyhow::Result<Value> {
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    report["reference_used"] = json!(false);
    std::fs::write(output.join("result.json"), serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(report)
}
