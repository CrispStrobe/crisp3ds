//! The stage driver: port of `run` in `multiscale_stereo.py`.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context};
use serde_json::{json, Value};

use crate::config::DenseConfig;
use crate::events::EventLog;
use crate::gpu::Gpu;
use crate::hull::{build_hull, Hull};
use crate::inputs::Inputs;
use crate::npz::{self, Array, Data};

use super::Arguments;

/// Support plane found by fusion: a point on the orbit axis, the direction
/// from the cameras to the object side, and the height of the lowest measured surface.
#[derive(Debug, Clone, Copy)]
pub struct Support {
    pub point: [f32; 3],
    pub down: [f32; 3],
    pub height: Option<f32>,
}

/// `index` rows (x, y, z) of the hull's occupied voxels as an N x 3 int32 array.
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

fn log(text: impl AsRef<str>) {
    use std::io::Write;
    println!("{}", text.as_ref());
    let _ = std::io::stdout().flush();
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
    let inputs = Inputs::load(&arguments.inputs, config)?;
    report["views"] = json!(inputs.count());
    report["load_seconds"] = json!(t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut voxel = None;
    let state = build_hull(&gpu, &inputs, config, &mut voxel)?;
    report["hull"] = serde_json::to_value(&state.report)?;
    report["hull"]["seconds"] = json!(t.elapsed().as_secs_f64());
    log(format!("hull {}", report["hull"]));
    events.progress(0.03, "Silhouette hull built")?;

    if arguments.only.as_deref() == Some("hull") {
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
            (0..inputs.count()).map(|i| inputs.neighbours(i, config.vote_neighbours as usize, config)).collect::<anyhow::Result<_>>()?;
        report["neighbours"] = json!(neighbours);
    }
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    report["reference_used"] = json!(false);
    std::fs::write(output.join("result.json"), serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(report)
}
