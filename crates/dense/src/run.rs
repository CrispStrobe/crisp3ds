//! A whole run in one process: port of `scripts/turntable_mesh/dense_pipeline.py`
//! for the `--inputs` and `--scene`/`--prepared`/`--raw-masks` starting points.
//!
//! Stages run in order on the calling thread: `inputs` (only from a scene),
//! `stereo`, `mesh`, `check`. Preview volumes written while matching runs are
//! meshed coarsely on a worker thread. The run directory follows
//! `docs/ENGINE-CONTRACT.md`: `config.json`, `events.jsonl` (always closed by
//! `run_finished` once `run_started` was written), `pipeline.json`, a `cancel`
//! file that stops the run. An embedding application calls [`run`] on a worker
//! thread with an event observer and a cancel flag.
//!
//! Not here: masks and cameras from plain photos (SAM, AliceVision) and scoring
//! against a reference scan; those stay in the Python driver.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::check;
use crate::config::DenseConfig;
use crate::control::{Control, Stopped};
use crate::events::{EventLog, Observer, SCHEMA};
use crate::gpu::Gpu;
use crate::inputs::load_views;
use crate::render::Rgb;
use crate::scene::{self, read_gray};
use crate::stereo;

/// What to run. Either `inputs`, or all of `scene`, `prepared` and `raw_masks`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunOptions {
    /// Fresh run directory.
    pub output: PathBuf,
    /// Existing inputs directory (`cameras.json`, `sparse_points.npy`).
    pub inputs: Option<PathBuf>,
    /// AliceVision `.sfm` with poses and one `radialk3` intrinsic.
    pub scene: Option<PathBuf>,
    /// Undistorted images named `<viewId>.png`.
    pub prepared: Option<PathBuf>,
    /// 0/255 masks named like the source photos.
    pub raw_masks: Option<PathBuf>,
    /// JSON with settings (a `config.json` or a previous `result.json`).
    pub config: Option<PathBuf>,
    /// `key=value` overrides, applied after `config`.
    pub overrides: Vec<String>,
    /// Settings by name, as `GET /api/settings` lists them; applied after `overrides`.
    pub settings: serde_json::Map<String, Value>,
    /// Worker threads of the surface stage.
    pub threads: usize,
    /// Seconds the stereo stage may take.
    pub stereo_timeout: f64,
    pub minimum_free_gib: f64,
    /// `depths.npz` of an earlier run on the same inputs: skip matching, fuse only.
    pub reuse_depths: Option<PathBuf>,
    /// Mesh intermediate surfaces while matching runs.
    pub live_previews: bool,
    /// Marching-cubes step of live preview meshes.
    pub preview_step: usize,
    /// Run the photo check.
    pub check: bool,
    /// Write the check's overlay and preview sheets.
    pub preview: bool,
    /// Keep `stereo/volume.npz` for re-meshing.
    pub keep_volume: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            output: PathBuf::new(),
            inputs: None,
            scene: None,
            prepared: None,
            raw_masks: None,
            config: None,
            overrides: Vec::new(),
            settings: serde_json::Map::new(),
            threads: 2,
            stereo_timeout: 3600.0,
            minimum_free_gib: 2.0,
            reuse_depths: None,
            live_previews: true,
            preview_step: 2,
            check: true,
            preview: true,
            keep_volume: false,
        }
    }
}

impl RunOptions {
    /// The resolved settings: defaults, `config`, `overrides`, `settings`.
    pub fn configuration(&self) -> anyhow::Result<DenseConfig> {
        let mut overrides = self.overrides.clone();
        for (key, value) in &self.settings {
            let text = match value {
                Value::Array(items) => items.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            overrides.push(format!("{key}={text}"));
        }
        stereo::options::build(self.config.as_deref(), &overrides)
    }
}

/// Contact sheet of evenly spaced photos with their mask outlines (`input_sheet`).
pub fn input_sheet(inputs: &Path, path: &Path) -> anyhow::Result<()> {
    const COUNT: usize = 8;
    const COLUMNS: usize = 4;
    let rows = load_views(inputs)?;
    let shown = COUNT.min(rows.len());
    let picks: Vec<usize> = (0..shown).map(|n| (n as f64 * (rows.len() as f64 / shown as f64)) as usize).collect();
    let tiles: Vec<anyhow::Result<Rgb>> = crate::inputs::parallel_map(picks.len(), |n| {
        let row = &rows[picks[n]];
        let mut photo = Rgb::open(Path::new(&row.image))?;
        let mask = read_gray(Path::new(&row.mask))?;
        if (mask.width, mask.height) != (photo.width, photo.height) {
            bail!("photo and mask sizes differ for {}", row.name);
        }
        // Mask pixels with a background neighbour, drawn about two pixels wide.
        let (w, h) = (mask.width, mask.height);
        let inside = |x: i64, y: i64| x >= 0 && y >= 0 && x < w as i64 && y < h as i64 && mask.data[y as usize * w + x as usize] > 127;
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                if inside(x, y) && !(inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1)) {
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        if x + dx < w as i64 && y + dy < h as i64 {
                            photo.set((x + dx) as usize, (y + dy) as usize, [60, 220, 60]);
                        }
                    }
                }
            }
        }
        let height = (crate::inputs::round_half_even(420.0 * photo.height as f64 / photo.width as f64) as usize).max(1);
        Ok(photo.resize(420, height))
    });
    let tiles: Vec<Rgb> = tiles.into_iter().collect::<anyhow::Result<_>>()?;
    let height = tiles.first().map(|t| t.height).ok_or_else(|| anyhow!("no views"))?;
    let mut sheet = Rgb::filled(420 * COLUMNS, height * tiles.len().div_ceil(COLUMNS), [30; 3]);
    for (n, tile) in tiles.iter().enumerate() {
        let clipped = tile.crop(0, 0, tile.width, tile.height.min(height));
        sheet.paste(&clipped, (n % COLUMNS) * 420, (n / COLUMNS) * height);
    }
    sheet.save(path)
}

struct Driver {
    output: PathBuf,
    events: EventLog,
    report: Value,
    started: Instant,
    cancel: Option<Arc<AtomicBool>>,
    finished: bool,
}

impl Driver {
    fn control(&self, stage: &str, timeout: Option<f64>) -> anyhow::Result<Control> {
        let deadline = timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds.max(0.0)));
        Control::new(self.cancel.clone(), Some(self.output.join("cancel")), deadline, Some(&self.output.join(format!("{stage}.log"))))
    }

    fn write_report(&self) -> anyhow::Result<()> {
        std::fs::write(self.output.join("pipeline.json"), serde_json::to_string_pretty(&self.report)? + "\n")?;
        Ok(())
    }

    fn finish(&mut self, status: &str) -> anyhow::Result<()> {
        self.finished = true;
        self.report["status"] = json!(status);
        self.report["seconds"] = json!(self.started.elapsed().as_secs_f64());
        self.write_report()?;
        let cancelled = Control::new(self.cancel.clone(), Some(self.output.join("cancel")), None, None)?.cancelled();
        let outcome = if status == "complete" {
            "complete"
        } else if cancelled {
            "cancelled"
        } else {
            "failed"
        };
        self.events.emit("run_finished", json!({"status": outcome, "seconds": self.report["seconds"]}))
    }

    /// One stage: events around it, its entry in `pipeline.json`, its log file.
    fn stage<T>(
        &mut self,
        name: &str,
        command: Vec<String>,
        timeout: Option<f64>,
        work: impl FnOnce(&EventLog, &Control) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        println!("[{name}] ...");
        let events = self.events.stage(name);
        events.emit("stage_started", json!({}))?;
        let started = Instant::now();
        let control = self.control(name, timeout)?;
        let result = control.check().and_then(|_| work(&events, &control));
        let seconds = started.elapsed().as_secs_f64();
        let stopped = result.as_ref().err().and_then(|e| e.downcast_ref::<Stopped>().copied());
        if let Err(error) = &result {
            control.log(format!("{error:#}"));
        }
        self.report["stages"][name] = json!({
            "command": command,
            "exit_code": match (&result, stopped) { (Ok(_), _) => json!(0), (_, Some(_)) => Value::Null, _ => json!(1) },
            "timed_out": stopped == Some(Stopped::Deadline),
            "cancelled": stopped == Some(Stopped::Cancelled),
            "seconds": seconds,
            "log": self.output.join(format!("{name}.log")).display().to_string(),
        });
        self.write_report()?;
        match result {
            Ok(value) => {
                events.emit("stage_finished", json!({"seconds": seconds}))?;
                println!("[{name}] {seconds:.1}s");
                Ok(value)
            }
            Err(error) => {
                let message = match stopped {
                    Some(Stopped::Cancelled) => "cancelled on request".to_string(),
                    Some(Stopped::Deadline) => format!("deadline: {error:#}"),
                    None => format!("{error:#}"),
                };
                events.emit("error", json!({"message": message}))?;
                self.finish(&format!("failed in {name}"))?;
                Err(error.context(format!("stage {name} failed")))
            }
        }
    }
}

fn text(path: &Path) -> String {
    path.display().to_string()
}

/// Runs the pipeline into the fresh directory `options.output` and returns the
/// content of `pipeline.json`. Every event is appended to `events.jsonl` there
/// and passed to `observer`; setting `cancel` (or creating the `cancel` file)
/// stops the run at the next view while matching, or between stages otherwise.
pub fn run(options: &RunOptions, observer: Option<Observer>, cancel: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
    let config = options.configuration()?;
    if options.output.as_os_str().is_empty() {
        bail!("an output directory is required");
    }
    let output = std::path::absolute(&options.output)?;
    if output.exists() {
        bail!("output directory exists: {}", output.display());
    }
    let from_scene = options.scene.is_some() && options.prepared.is_some() && options.raw_masks.is_some();
    if options.inputs.is_none() && !from_scene {
        bail!("give inputs, or all of scene, prepared and raw_masks");
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let parent = output.parent().filter(|p| p.exists()).map(Path::to_path_buf).unwrap_or(std::env::current_dir()?);
        let free = fs4::available_space(&parent).with_context(|| parent.display().to_string())? as f64 / (1u64 << 30) as f64;
        if free < options.minimum_free_gib {
            bail!("only {free:.1} GiB free; need {} (see minimum_free_gib)", options.minimum_free_gib);
        }
    }
    std::fs::create_dir_all(&output).with_context(|| output.display().to_string())?;
    // Event paths are relative to the run directory as the file system names it.
    let output = std::fs::canonicalize(&output)?;
    std::fs::write(output.join("config.json"), serde_json::to_string_pretty(&config)? + "\n")?;

    // Preview volumes announced by the stereo stage go to the meshing thread.
    let (sender, receiver) = mpsc::channel::<Option<(PathBuf, String)>>();
    let release = sender.clone();
    let sender = Mutex::new(sender);
    let run_directory = output.clone();
    let watching: Observer = Arc::new(move |event: &Value| {
        if event["type"] == "artifact" && event["kind"] == "preview_volume" {
            if let (Some(path), Some(label)) = (event["path"].as_str(), event["label"].as_str()) {
                let _ = sender.lock().unwrap_or_else(|p| p.into_inner()).send(Some((run_directory.join(path), label.to_string())));
            }
        } else if event["type"] == "run_finished" {
            let _ = sender.lock().unwrap_or_else(|p| p.into_inner()).send(None);
        }
        if let Some(observer) = &observer {
            observer(event);
        }
    });
    let events = EventLog::for_run(&output, Some(watching));
    let mut driver = Driver {
        output: output.clone(),
        events: events.clone(),
        report: json!({"output": text(&output), "device": "wgpu", "stages": {}, "status": "running"}),
        started: Instant::now(),
        cancel,
        finished: false,
    };
    let source = options.inputs.as_ref().or(options.scene.as_ref()).expect("checked above");
    let source_name = source.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();

    let outcome = std::thread::scope(|scope| {
        let preview_events = events.stage("stereo");
        let (preview_config, step, threads) = (config.clone(), options.preview_step.max(1), options.threads.max(1));
        let mesher = scope.spawn(move || {
            while let Ok(Some((volume, label))) = receiver.recv() {
                let target = volume.with_extension("");
                let result = crate::mesh::run(&volume, &target, &preview_config, step, &preview_events, Some(&label), threads);
                let log = match &result {
                    Ok(report) => serde_json::to_string_pretty(report).unwrap_or_default(),
                    Err(error) => format!("{error:#}"),
                };
                let _ = std::fs::write(volume.with_extension("log"), log + "\n");
            }
        });
        let outcome = stages(options, &config, &output, &mut driver, source_name);
        if let Err(error) = &outcome {
            if !driver.finished {
                // Anything outside a stage must still close the event log.
                let _ = events.emit("error", json!({"message": format!("{error:#}")}));
                let _ = driver.finish("failed before or between stages");
            }
        }
        // Release the meshing thread; outstanding previews finish first, they are small.
        let _ = release.send(None);
        let _ = mesher.join();
        outcome
    });
    let previews = output.join("stereo/preview");
    if let Ok(entries) = std::fs::read_dir(&previews) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "npz") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    outcome?;
    Ok(driver.report)
}

fn stages(options: &RunOptions, config: &DenseConfig, output: &Path, driver: &mut Driver, source_name: String) -> anyhow::Result<()> {
    let gpu = Gpu::new();
    let device = match &gpu {
        Ok(gpu) => format!("wgpu: {}", gpu.describe()),
        Err(_) => "wgpu".to_string(),
    };
    driver.report["device"] = json!(device);
    driver.events.emit(
        "run_started",
        json!({"schema": SCHEMA, "configuration": serde_json::to_value(config)?, "device": device, "inputs": source_name}),
    )?;
    let inputs = match &options.inputs {
        Some(inputs) => {
            let inputs = std::path::absolute(inputs)?;
            if !inputs.join("cameras.json").is_file() {
                bail!("{} has no cameras.json; it is not an inputs directory", inputs.display());
            }
            inputs
        }
        None => {
            let (scene, prepared, raw_masks) = (
                options.scene.clone().expect("checked by run"),
                options.prepared.clone().expect("checked by run"),
                options.raw_masks.clone().expect("checked by run"),
            );
            let inputs = output.join("inputs");
            let command =
                ["crisp3ds-dense", "inputs", "--scene", &text(&scene), "--prepared", &text(&prepared), "--raw-masks", &text(&raw_masks)]
                    .map(String::from)
                    .to_vec();
            driver.stage("inputs", command, Some(600.0), |_, control| {
                let report = scene::run(&scene, &prepared, &raw_masks, &inputs)?;
                control.log(report.to_string());
                Ok(())
            })?;
            inputs
        }
    };
    if input_sheet(&inputs, &output.join("input-sheet.png")).is_ok() {
        driver.events.stage("inputs").artifact("input_sheet", &output.join("input-sheet.png"), "Photos with mask outlines", json!({}))?;
    }
    driver.report["native_stages"] =
        json!(if options.inputs.is_some() { vec!["stereo", "mesh", "check"] } else { vec!["inputs", "stereo", "mesh", "check"] });

    let gpu = gpu?;
    let arguments = stereo::Arguments {
        inputs: inputs.clone(),
        output: output.join("stereo"),
        reuse_depths: options.reuse_depths.clone(),
        previews: options.live_previews,
        ..Default::default()
    };
    let mut command = vec!["crisp3ds-dense".to_string(), "stereo".to_string(), "--inputs".to_string(), text(&inputs)];
    if let Some(depths) = &options.reuse_depths {
        command.extend(["--reuse-depths".to_string(), text(depths)]);
    }
    driver.stage("stereo", command, Some(options.stereo_timeout), |events, control| {
        stereo::run::run_with(&arguments, config, &gpu, events, control).map(|_| ())
    })?;
    drop(gpu);

    let volume = output.join("stereo/volume.npz");
    let command = ["crisp3ds-dense", "mesh", "--volume", &text(&volume)].map(String::from).to_vec();
    let threads = options.threads.max(1);
    let mesh = driver.stage("mesh", command, Some(900.0), |events, control| {
        let report = crate::mesh::run(&volume, &output.join("mesh"), config, 1, events, None, threads)?;
        control.log(serde_json::to_string_pretty(&report)?);
        Ok(report)
    })?;

    if options.check {
        let stl = output.join("mesh/mesh.stl");
        let repaired = output.join("stereo/masks-repaired");
        let command = ["crisp3ds-dense", "check", "--inputs", &text(&inputs), "--mesh", &text(&stl)].map(String::from).to_vec();
        let report = driver.stage("check", command, Some(900.0), |events, control| {
            let options = check::Options {
                inputs: &inputs,
                mesh: &stl,
                output: &output.join("check"),
                repaired_masks: Some(&repaired),
                preview_views: if options.preview { 3 } else { 0 },
                check_views: 24,
            };
            let report = check::run(&options, events)?;
            control.log(serde_json::to_string_pretty(&report)?);
            Ok(report)
        })?;
        driver.report["photo_check"] = report;
    }
    if !options.keep_volume {
        std::fs::remove_file(&volume)?;
    }
    driver.report["mesh"] = json!(text(&output.join("mesh/mesh.stl")));
    driver.report["closed"] = json!(mesh.closed);
    driver.report["triangles"] = json!(mesh.triangles);
    driver.report["genus"] = json!(mesh.genus);
    driver.report["reference_used"] = json!(false);
    driver.report["physical_scale_established"] = json!(false);
    driver.events.stage("mesh").artifact("report", &output.join("mesh/result.json"), "Mesh report", json!({}))?;
    driver.finish("complete")
}

pub const USAGE: &str = "usage: crisp3ds-dense run --output DIR (--inputs DIR | --scene FILE --prepared DIR --raw-masks DIR) \
[--config FILE] [--set KEY=VALUE]... [--threads N] [--stereo-timeout SECONDS] [--minimum-free-gib G] [--reuse-depths FILE] \
[--no-live-previews] [--preview-step N] [--skip-check] [--no-preview] [--keep-volume]";

/// Options from the command line of `dense_pipeline.py` (the parts this driver covers).
pub fn parse(arguments: &[String]) -> anyhow::Result<RunOptions> {
    let mut options = RunOptions::default();
    let mut rest = arguments.iter();
    while let Some(flag) = rest.next() {
        let (flag, inline) = match flag.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (flag.as_str(), None),
        };
        let mut value = || inline.clone().or_else(|| rest.next().cloned()).ok_or_else(|| anyhow!("{flag} needs a value\n{USAGE}"));
        let number = |text: String| text.parse::<f64>().map_err(|_| anyhow!("{flag} needs a number, found {text:?}"));
        match flag {
            "--output" => options.output = PathBuf::from(value()?),
            "--inputs" => options.inputs = Some(PathBuf::from(value()?)),
            "--scene" => options.scene = Some(PathBuf::from(value()?)),
            "--prepared" => options.prepared = Some(PathBuf::from(value()?)),
            "--raw-masks" => options.raw_masks = Some(PathBuf::from(value()?)),
            "--config" => options.config = Some(PathBuf::from(value()?)),
            "--set" => options.overrides.push(value()?),
            "--threads" => options.threads = number(value()?)? as usize,
            "--stereo-timeout" => options.stereo_timeout = number(value()?)?,
            "--minimum-free-gib" => options.minimum_free_gib = number(value()?)?,
            "--reuse-depths" => options.reuse_depths = Some(PathBuf::from(value()?)),
            "--preview-step" => options.preview_step = number(value()?)? as usize,
            "--no-live-previews" => options.live_previews = false,
            "--skip-check" => options.check = false,
            "--no-preview" => options.preview = false,
            "--keep-volume" => options.keep_volume = true,
            // Accepted for command lines written for the Python driver; they select nothing here.
            "--device" | "--python" | "--torch-python" | "--native" => drop(value()?),
            "--photos" | "--calibration" | "--photos-option" | "--photos-timeout" => {
                bail!("{flag}: masks and cameras from plain photos need SAM 2.1 and AliceVision; use scripts/turntable_mesh/dense_pipeline.py for that step")
            }
            "--reference" => bail!("--reference: scoring against a scan is done by scripts/turntable_mesh/scan_evaluate.py; run it on mesh/mesh.stl afterwards"),
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    if options.output.as_os_str().is_empty() {
        bail!("--output is required\n{USAGE}");
    }
    Ok(options)
}

/// `crisp3ds-dense run ...`: prints the summary the Python driver prints.
pub fn main(arguments: &[String]) -> anyhow::Result<()> {
    if arguments.iter().any(|a| a == "--list-settings") {
        let defaults = serde_json::to_value(DenseConfig::default())?;
        for (key, value) in defaults.as_object().into_iter().flatten() {
            println!("{key} = {value}");
        }
        return Ok(());
    }
    let report = run(&parse(arguments)?, None, None)?;
    let summary: serde_json::Map<String, Value> =
        ["status", "mesh", "closed", "triangles", "genus", "seconds"].iter().map(|k| (k.to_string(), report[*k].clone())).collect();
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_from_the_command_line_and_from_json() {
        let words = "--output run --inputs in --set grid=96 --no-live-previews --threads 3 --device mps --keep-volume --set=sizes=64,128";
        let options = parse(&words.split(' ').map(String::from).collect::<Vec<_>>()).unwrap();
        assert_eq!((options.output.clone(), options.inputs.clone()), (PathBuf::from("run"), Some(PathBuf::from("in"))));
        assert!(!options.live_previews && options.keep_volume && options.check && options.threads == 3);
        let config = options.configuration().unwrap();
        assert_eq!((config.grid, config.sizes), (96, vec![64, 128]));
        for refused in ["--photos x", "--reference scan.ply", "--bogus", "--inputs in"] {
            assert!(parse(&refused.split(' ').map(String::from).collect::<Vec<_>>()).is_err(), "{refused}");
        }
        let json =
            r#"{"output": "run", "inputs": "in", "settings": {"grid": 320, "sizes": [256, 512], "repair_masks": false}, "check": false}"#;
        let options: RunOptions = serde_json::from_str(json).unwrap();
        let config = options.configuration().unwrap();
        assert_eq!((config.grid, config.sizes, config.repair_masks, options.check), (320, vec![256, 512], false, false));
        assert!(serde_json::from_str::<RunOptions>(r#"{"output": "run", "typo": 1}"#).is_err());
        let invalid: RunOptions = serde_json::from_str(r#"{"output": "run", "inputs": "in", "settings": {"windows": [4]}}"#).unwrap();
        assert!(invalid.configuration().is_err());
    }

    #[test]
    fn refuses_before_anything_is_written() {
        let root = std::env::temp_dir().join(format!("crisp3ds-run-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let options = RunOptions { output: root.clone(), ..Default::default() };
        assert!(run(&options, None, None).is_err());
        assert!(!root.exists());
        let options = RunOptions { output: root.clone(), inputs: Some(root.join("in")), minimum_free_gib: 1e9, ..Default::default() };
        assert!(run(&options, None, None).unwrap_err().to_string().contains("GiB free"));
        assert!(!root.exists());
    }

    /// A directory without a camera table fails after `run_started`, and the log is still closed (no GPU needed).
    #[test]
    fn a_failure_between_stages_closes_the_event_log() {
        let root = std::env::temp_dir().join(format!("crisp3ds-run-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("in")).unwrap();
        let options = RunOptions { output: root.join("run"), inputs: Some(root.join("in")), ..Default::default() };
        assert!(run(&options, None, None).is_err());
        let log = std::fs::read_to_string(root.join("run/events.jsonl")).unwrap();
        let types: Vec<String> =
            log.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["type"].as_str().unwrap().to_string()).collect();
        assert_eq!(types, ["run_started", "error", "run_finished"]);
        let last: Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
        assert_eq!(last["status"], "failed");
        let pipeline: Value = serde_json::from_str(&std::fs::read_to_string(root.join("run/pipeline.json")).unwrap()).unwrap();
        assert_eq!(pipeline["status"], "failed before or between stages");
        assert!(root.join("run/config.json").is_file());
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn kinds(events: &[Value], kind: &str) -> usize {
        events.iter().filter(|e| e["type"] == "artifact" && e["kind"] == kind).count()
    }

    /// The whole pipeline on the analytic sphere, then a cancelled run (CRISP3DS_GPU_TESTS=1).
    #[test]
    fn gpu_run_completes_and_can_be_cancelled() {
        if !crate::gpu::tests_enabled() {
            return;
        }
        use crate::stereo::synthetic;
        let root = synthetic::temporary("run", 24, 128).unwrap();
        let overrides: Vec<String> = synthetic::SMALL.iter().map(|s| s.to_string()).collect();
        let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let observer: Observer = Arc::new(move |event: &Value| sink.lock().unwrap().push(event.clone()));
        let options =
            RunOptions { output: root.join("run"), inputs: Some(root.join("inputs")), overrides: overrides.clone(), ..Default::default() };
        let report = run(&options, Some(observer), None).unwrap();
        assert_eq!(report["status"], "complete");
        assert_eq!(report["closed"], true);
        assert!(report["triangles"].as_u64().unwrap() > 10_000);
        assert!(report["photo_check"]["silhouette_iou_input_masks"]["median"].as_f64().unwrap() > 0.9);
        for stage in ["stereo", "mesh", "check"] {
            assert_eq!(report["stages"][stage]["exit_code"], 0, "{stage}");
            assert!(Path::new(report["stages"][stage]["log"].as_str().unwrap()).is_file());
        }
        let run_directory = std::fs::canonicalize(root.join("run")).unwrap();
        for file in
            ["config.json", "pipeline.json", "events.jsonl", "input-sheet.png", "mesh/mesh.stl", "check/preview.png", "stereo/depths.npz"]
        {
            assert!(run_directory.join(file).is_file(), "{file}");
        }
        assert!(!run_directory.join("stereo/volume.npz").exists());
        assert!(std::fs::read_dir(run_directory.join("stereo/preview"))
            .unwrap()
            .flatten()
            .all(|e| e.path().extension().is_none_or(|x| x != "npz")));
        let events = seen.lock().unwrap().clone();
        assert_eq!(std::fs::read_to_string(run_directory.join("events.jsonl")).unwrap().lines().count(), events.len());
        assert_eq!((events[0]["type"].as_str(), events[0]["schema"].as_str()), (Some("run_started"), Some(SCHEMA)));
        assert_eq!(
            (events.last().unwrap()["type"].as_str(), events.last().unwrap()["status"].as_str()),
            (Some("run_finished"), Some("complete"))
        );
        let stages: Vec<(String, String)> = events
            .iter()
            .filter(|e| e["type"] == "stage_started" || e["type"] == "stage_finished")
            .map(|e| (e["type"].as_str().unwrap().to_string(), e["stage"].as_str().unwrap().to_string()))
            .collect();
        let expected = [
            ("stage_started", "stereo"),
            ("stage_finished", "stereo"),
            ("stage_started", "mesh"),
            ("stage_finished", "mesh"),
            ("stage_started", "check"),
            ("stage_finished", "check"),
        ];
        assert_eq!(stages, expected.map(|(a, b)| (a.to_string(), b.to_string())));
        for (kind, count) in [
            ("input_sheet", 1),
            ("mask_repair_sheet", 1),
            ("hull_mask_sheet", 1),
            ("depth_sheet", 3),
            ("preview_volume", 3),
            ("preview_mesh", 3),
            ("final_mesh", 1),
            ("photo_overlay", 1),
            ("preview_render", 1),
            ("report", 2),
        ] {
            assert_eq!(kinds(&events, kind), count, "{kind}");
        }
        for event in events.iter().filter(|e| e["type"] == "artifact" && e["kind"] != "preview_volume") {
            assert!(run_directory.join(event["path"].as_str().unwrap()).is_file(), "{event}");
        }

        // Cancelled from the observer as soon as matching reports progress.
        let cancel = Arc::new(AtomicBool::new(false));
        let (flag, seen) = (cancel.clone(), Arc::new(Mutex::new(Vec::<Value>::new())));
        let sink = seen.clone();
        let observer: Observer = Arc::new(move |event: &Value| {
            if event["type"] == "progress" && event["message"].as_str().is_some_and(|m| m.starts_with("Matching")) {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            sink.lock().unwrap().push(event.clone());
        });
        let options = RunOptions { output: root.join("cancelled"), inputs: Some(root.join("inputs")), overrides, ..Default::default() };
        let error = run(&options, Some(observer), Some(cancel)).unwrap_err();
        assert_eq!(error.downcast_ref::<Stopped>(), Some(&Stopped::Cancelled));
        let events = seen.lock().unwrap().clone();
        assert_eq!(events.last().unwrap()["status"], "cancelled");
        assert!(events.iter().any(|e| e["type"] == "error" && e["message"] == "cancelled on request" && e["stage"] == "stereo"));
        let pipeline: Value = serde_json::from_str(&std::fs::read_to_string(root.join("cancelled/pipeline.json")).unwrap()).unwrap();
        assert_eq!(pipeline["status"], "failed in stereo");
        assert_eq!(pipeline["stages"]["stereo"]["cancelled"], true);
        assert!(pipeline["stages"]["stereo"]["exit_code"].is_null());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
