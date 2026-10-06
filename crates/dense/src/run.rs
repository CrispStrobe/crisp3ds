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
use std::sync::{Arc, Mutex};

use web_time::{Duration, Instant};

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
    /// Folder of turntable photos: the photos stage (masks, cameras) runs first
    /// and its scene is the inputs directory. Not in a browser.
    pub photos: Option<PathBuf>,
    /// Command-line options of the photos stage (`crisp3ds-dense photos --help`):
    /// `--calibration FILE`, `--masks PROVIDER`, `--cameras PROVIDER` and the
    /// providers' own options, one word per element.
    pub photo_options: Vec<String>,
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
            photos: None,
            photo_options: Vec::new(),
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
        // Mask pixels with a background neighbour, drawn wide enough to survive the reduction to the tile.
        let (w, h) = (mask.width, mask.height);
        let inside = |x: i64, y: i64| x >= 0 && y >= 0 && x < w as i64 && y < h as i64 && mask.data[y as usize * w + x as usize] > 127;
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                if inside(x, y) && !(inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1)) {
                    for (dx, dy) in (-1..=2).flat_map(|dy| (-1..=2).map(move |dx| (dx, dy))) {
                        if x + dx >= 0 && y + dy >= 0 && x + dx < w as i64 && y + dy < h as i64 {
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

/// Preview volumes waiting to be meshed coarsely. With threads a worker takes
/// them as they are announced; without (wasm32) the driver meshes them itself
/// at the next stage boundary.
#[derive(Default)]
struct PreviewQueue {
    state: Mutex<PreviewState>,
    changed: std::sync::Condvar,
}

#[derive(Default)]
struct PreviewState {
    waiting: std::collections::VecDeque<(PathBuf, String)>,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    meshing: bool,
    closed: bool,
    /// Set when the run failed or was cancelled: waiting volumes are dropped.
    abandoned: bool,
}

struct PreviewMesher {
    queue: Arc<PreviewQueue>,
    events: EventLog,
    config: DenseConfig,
    step: usize,
    threads: usize,
}

impl PreviewMesher {
    fn lock(&self) -> std::sync::MutexGuard<'_, PreviewState> {
        self.queue.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn mesh(&self, volume: &Path, label: &str) {
        let result = crate::mesh::run(volume, &volume.with_extension(""), &self.config, self.step, &self.events, Some(label), self.threads);
        let log = match &result {
            Ok(report) => serde_json::to_string_pretty(report).unwrap_or_default(),
            Err(error) => format!("{error:#}"),
        };
        let _ = crate::storage::write(volume.with_extension("log"), log + "\n");
        // The volume has served its purpose; it is large, and without a disk it occupies memory.
        let _ = crate::storage::remove_file(volume);
    }

    /// The worker: meshes volumes until the queue is closed and empty.
    #[cfg(not(target_arch = "wasm32"))]
    fn work(&self) {
        loop {
            let next = {
                let mut state = self.lock();
                loop {
                    if state.abandoned {
                        state.waiting.clear();
                    }
                    if let Some(next) = state.waiting.pop_front() {
                        state.meshing = true;
                        break Some(next);
                    }
                    if state.closed {
                        break None;
                    }
                    state = self.queue.changed.wait(state).unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            };
            let Some((volume, label)) = next else { return };
            self.mesh(&volume, &label);
            self.lock().meshing = false;
            self.queue.changed.notify_all();
        }
    }

    /// Returns when no preview is waiting or being meshed. `abandon` drops the waiting ones.
    fn settle(&self, abandon: bool) {
        if abandon {
            let mut state = self.lock();
            state.abandoned = true;
            state.waiting.clear();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.queue.changed.notify_all();
            let mut state = self.lock();
            while state.meshing || !state.waiting.is_empty() {
                state = self.queue.changed.wait(state).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
        #[cfg(target_arch = "wasm32")]
        loop {
            let next = self.lock().waiting.pop_front();
            let Some((volume, label)) = next else { break };
            self.mesh(&volume, &label);
        }
    }

    fn close(&self) {
        self.lock().closed = true;
        self.queue.changed.notify_all();
    }
}

struct Driver {
    output: PathBuf,
    events: EventLog,
    report: Value,
    started: Instant,
    cancel: Option<Arc<AtomicBool>>,
    finished: bool,
    previews: Arc<PreviewMesher>,
}

/// A stage that has started: its events, its control and its clock.
struct Stage {
    name: String,
    command: Vec<String>,
    events: EventLog,
    control: Control,
    started: Instant,
}

impl Driver {
    fn write_report(&self) -> anyhow::Result<()> {
        crate::storage::write(self.output.join("pipeline.json"), serde_json::to_string_pretty(&self.report)? + "\n")?;
        Ok(())
    }

    /// Closes the run: `pipeline.json`, then `run_finished` as the last event (previews are settled first).
    fn finish(&mut self, status: &str) -> anyhow::Result<()> {
        self.finished = true;
        self.previews.settle(status != "complete");
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

    /// Starts a stage: `stage_started`, its log file, its deadline.
    fn begin(&mut self, name: &str, command: Vec<String>, timeout: Option<f64>) -> anyhow::Result<Stage> {
        println!("[{name}] ...");
        let events = self.events.stage(name);
        events.emit("stage_started", json!({}))?;
        let deadline = timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds.max(0.0)));
        let log = self.output.join(format!("{name}.log"));
        let control = Control::new(self.cancel.clone(), Some(self.output.join("cancel")), deadline, Some(&log))?;
        Ok(Stage { name: name.to_string(), command, events, control, started: Instant::now() })
    }

    /// Ends a stage with its result: its entry in `pipeline.json`, `stage_finished` or `error` and the end of the run.
    fn end<T>(&mut self, stage: Stage, result: anyhow::Result<T>) -> anyhow::Result<T> {
        let Stage { name, command, events, control, started } = stage;
        let seconds = started.elapsed().as_secs_f64();
        let stopped = result.as_ref().err().and_then(|e| e.downcast_ref::<Stopped>().copied());
        if let Err(error) = &result {
            control.log(format!("{error:#}"));
        }
        self.report["stages"][&name] = json!({
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
///
/// Blocks the calling thread until the run is over; call it on a worker thread.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(options: &RunOptions, observer: Option<Observer>, cancel: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
    crate::gpu::block_on(run_async(options, observer, cancel))
}

/// [`run`] as a future, for hosts that cannot block (a browser). Paths may
/// name files of the in-memory tree (`crate::storage`); in a browser they all do.
pub async fn run_async(options: &RunOptions, observer: Option<Observer>, cancel: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
    let config = options.configuration()?;
    if options.output.as_os_str().is_empty() {
        bail!("an output directory is required");
    }
    let output = crate::storage::absolute(&options.output)?;
    if crate::storage::exists(&output) {
        bail!("output directory exists: {}", output.display());
    }
    let from_scene = options.scene.is_some() && options.prepared.is_some() && options.raw_masks.is_some();
    if options.inputs.is_none() && !from_scene && options.photos.is_none() {
        bail!("give photos, or inputs, or all of scene, prepared and raw_masks");
    }
    if options.photos.is_some() && (cfg!(target_arch = "wasm32") || crate::storage::is_memory(&output)) {
        bail!("the photos stage starts external programs and needs a run directory on disk");
    }
    #[cfg(not(target_arch = "wasm32"))]
    let photo_stage = match &options.photos {
        Some(photos) => Some(photo_stage_options(options, photos, &output)?),
        None => None,
    };
    #[cfg(not(target_arch = "wasm32"))]
    if !crate::storage::is_memory(&output) {
        let parent = output.parent().filter(|p| crate::storage::exists(p)).map(Path::to_path_buf).unwrap_or(std::env::current_dir()?);
        let free = fs4::available_space(&parent).with_context(|| parent.display().to_string())? as f64 / (1u64 << 30) as f64;
        if free < options.minimum_free_gib {
            bail!("only {free:.1} GiB free; need {} (see minimum_free_gib)", options.minimum_free_gib);
        }
    }
    crate::storage::create_dir_all(&output).with_context(|| output.display().to_string())?;
    // Event paths are relative to the run directory as the file system names it.
    let output = crate::storage::canonicalize(&output)?;
    crate::storage::write(output.join("config.json"), serde_json::to_string_pretty(&config)? + "\n")?;

    // Preview volumes announced by the stereo stage are queued for coarse meshing.
    let queue = Arc::new(PreviewQueue::default());
    let (announced, run_directory) = (queue.clone(), output.clone());
    // Without a second thread a volume is meshed as soon as it is announced, so that it does not stay in memory.
    let inline: Arc<std::sync::OnceLock<Arc<PreviewMesher>>> = Arc::new(std::sync::OnceLock::new());
    let meshing_inline = inline.clone();
    let watching: Observer = Arc::new(move |event: &Value| {
        if event["type"] == "artifact" && event["kind"] == "preview_volume" {
            if let (Some(path), Some(label)) = (event["path"].as_str(), event["label"].as_str()) {
                if let Some(mesher) = meshing_inline.get() {
                    if let Some(observer) = &observer {
                        observer(event);
                    }
                    mesher.mesh(&run_directory.join(path), label);
                    return;
                }
                let mut state = announced.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if !state.abandoned {
                    state.waiting.push_back((run_directory.join(path), label.to_string()));
                }
                drop(state);
                announced.changed.notify_all();
            }
        }
        if let Some(observer) = &observer {
            observer(event);
        }
    });
    let events = EventLog::for_run(&output, Some(watching));
    let mesher = Arc::new(PreviewMesher {
        queue,
        events: events.stage("stereo"),
        config: config.clone(),
        step: options.preview_step.max(1),
        threads: options.threads.max(1),
    });
    if cfg!(target_arch = "wasm32") {
        let _ = inline.set(mesher.clone());
    }
    let mut driver = Driver {
        output: output.clone(),
        events: events.clone(),
        report: json!({"output": text(&output), "device": "wgpu", "stages": {}, "status": "running"}),
        started: Instant::now(),
        cancel,
        finished: false,
        previews: mesher.clone(),
    };
    let source = options.photos.as_ref().or(options.inputs.as_ref()).or(options.scene.as_ref()).expect("checked above");
    let source_name = source.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();

    // With threads, previews are meshed on a second one while matching continues.
    #[cfg(not(target_arch = "wasm32"))]
    let worker = {
        let mesher = mesher.clone();
        std::thread::Builder::new().name("crisp3ds-previews".into()).spawn(move || mesher.work())?
    };
    #[cfg(not(target_arch = "wasm32"))]
    let outcome = stages(options, photo_stage.as_ref(), &config, &output, &mut driver, source_name).await;
    #[cfg(target_arch = "wasm32")]
    let outcome = stages(options, &config, &output, &mut driver, source_name).await;
    let outcome = close(outcome, &events, &mut driver);
    mesher.close();
    #[cfg(not(target_arch = "wasm32"))]
    let _ = worker.join();
    for entry in crate::storage::list(output.join("stereo/preview")).unwrap_or_default() {
        if entry.extension().is_some_and(|e| e == "npz") {
            let _ = crate::storage::remove_file(entry);
        }
    }
    outcome?;
    Ok(driver.report)
}

/// Anything that failed outside a stage must still close the event log.
fn close(outcome: anyhow::Result<()>, events: &EventLog, driver: &mut Driver) -> anyhow::Result<()> {
    if let Err(error) = &outcome {
        if !driver.finished {
            let _ = events.emit("error", json!({"message": format!("{error:#}")}));
            let _ = driver.finish("failed before or between stages");
        }
    }
    outcome
}

/// The photos stage's options for this run: its command line with the run's directories filled in.
/// Resolved before anything is written, so that a bad option or a missing program fails early.
#[cfg(not(target_arch = "wasm32"))]
fn photo_stage_options(options: &RunOptions, photos: &Path, output: &Path) -> anyhow::Result<crate::photos::options::Options> {
    use crate::photos::{cameras::camera_provider, masks::mask_provider, options::resolve};
    let mut arguments: Vec<String> =
        ["--threads", &options.threads.to_string(), "--minimum-free-gib", &options.minimum_free_gib.to_string()].map(String::from).to_vec();
    arguments.extend(options.photo_options.iter().cloned());
    for (flag, value) in [("--photos", photos), ("--output", &output.join("frontend")), ("--events", &output.join("events.jsonl"))] {
        arguments.extend([flag.to_string(), text(value)]);
    }
    let resolved = resolve(&arguments, &|name| std::env::var(name).ok()).context("photos stage options")?;
    if resolved.stop_after_masks {
        bail!("--stop-after masks leaves no scene to reconstruct; use the photos command for that");
    }
    mask_provider(&resolved).check(&resolved)?;
    camera_provider(&resolved).check(&resolved)?;
    Ok(resolved)
}

async fn stages(
    options: &RunOptions,
    #[cfg(not(target_arch = "wasm32"))] photo_stage: Option<&crate::photos::options::Options>,
    config: &DenseConfig,
    output: &Path,
    driver: &mut Driver,
    source_name: String,
) -> anyhow::Result<()> {
    let gpu = Gpu::request().await;
    let device = match &gpu {
        Ok(gpu) => format!("wgpu: {}", gpu.describe()),
        Err(_) => "wgpu".to_string(),
    };
    driver.report["device"] = json!(device);
    driver.events.emit(
        "run_started",
        json!({"schema": SCHEMA, "configuration": serde_json::to_value(config)?, "device": device, "inputs": source_name}),
    )?;

    // From plain photos: masks and cameras first. That stage writes its own `masks` and `cameras` stage events.
    #[cfg(not(target_arch = "wasm32"))]
    let from_photos = match photo_stage {
        Some(photo_options) => {
            println!("[photos] ...");
            let started = Instant::now();
            let events_path = output.join("events.jsonl");
            let finished = crate::photos::run::run(photo_options, &driver.events, &events_path, driver.cancel.clone());
            let seconds = started.elapsed().as_secs_f64();
            let cancelled = Control::new(driver.cancel.clone(), Some(output.join("cancel")), None, None)?.cancelled();
            let code = finished.as_ref().map(|f| f.code).unwrap_or(1);
            driver.report["stages"]["photos"] = json!({
                "command": std::iter::once("crisp3ds-dense photos".to_string()).chain(options.photo_options.iter().cloned()).collect::<Vec<_>>(),
                "exit_code": if cancelled { Value::Null } else { json!(code) },
                "timed_out": false, "cancelled": cancelled, "seconds": seconds,
                "log": text(&output.join("frontend/logs")),
            });
            driver.write_report()?;
            if code != 0 {
                let reason = match &finished {
                    Ok(finished) => {
                        format!("photos stage ended with status {} ({})", finished.report["status"], finished.report["reasons"])
                    }
                    Err(error) => format!("{error:#}"),
                };
                if cancelled {
                    driver.events.stage("cameras").emit("error", json!({"message": "cancelled on request"}))?;
                } else if finished.is_err() {
                    driver.events.stage("cameras").emit("error", json!({"message": reason}))?;
                }
                driver.finish("failed in photos")?;
                bail!("stage photos failed: {reason}");
            }
            println!("[photos] {seconds:.1}s");
            Some(output.join("frontend/inputs"))
        }
        None => None,
    };
    #[cfg(target_arch = "wasm32")]
    let from_photos: Option<PathBuf> = None;
    let given = from_photos.as_ref().or(options.inputs.as_ref());
    let inputs = match given {
        Some(inputs) => {
            let inputs = crate::storage::absolute(inputs)?;
            if !crate::storage::is_file(inputs.join("cameras.json")) {
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
            let stage = driver.begin("inputs", command, Some(600.0))?;
            let result = stage.control.check().and_then(|_| scene::run(&scene, &prepared, &raw_masks, &inputs));
            if let Ok(report) = &result {
                stage.control.log(report.to_string());
            }
            driver.end(stage, result)?;
            inputs
        }
    };
    if input_sheet(&inputs, &output.join("input-sheet.png")).is_ok() {
        driver.events.stage("inputs").artifact("input_sheet", &output.join("input-sheet.png"), "Photos with mask outlines", json!({}))?;
    }
    driver.report["native_stages"] = json!(if from_photos.is_some() {
        vec!["photos", "stereo", "mesh", "check"]
    } else if options.inputs.is_some() {
        vec!["stereo", "mesh", "check"]
    } else {
        vec!["inputs", "stereo", "mesh", "check"]
    });

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
    let stage = driver.begin("stereo", command, Some(options.stereo_timeout))?;
    let result = match stage.control.check() {
        Ok(()) => stereo::run::run_with(&arguments, config, &gpu, &stage.events, &stage.control).await.map(|_| ()),
        Err(stopped) => Err(stopped),
    };
    driver.end(stage, result)?;
    drop(gpu);
    // Without a second thread the previews are meshed here.
    #[cfg(target_arch = "wasm32")]
    driver.previews.settle(false);

    let volume = output.join("stereo/volume.npz");
    let command = ["crisp3ds-dense", "mesh", "--volume", &text(&volume)].map(String::from).to_vec();
    let threads = options.threads.max(1);
    let stage = driver.begin("mesh", command, Some(900.0))?;
    let result = crate::mesh::run_with(&volume, &output.join("mesh"), config, 1, &stage.events, None, threads, &stage.control);
    if let Ok(report) = &result {
        stage.control.log(serde_json::to_string_pretty(report)?);
    }
    let mesh = driver.end(stage, result)?;

    if options.check {
        let stl = output.join("mesh/mesh.stl");
        let repaired = output.join("stereo/masks-repaired");
        let command = ["crisp3ds-dense", "check", "--inputs", &text(&inputs), "--mesh", &text(&stl)].map(String::from).to_vec();
        let stage = driver.begin("check", command, Some(900.0))?;
        let check_options = check::Options {
            inputs: &inputs,
            mesh: &stl,
            output: &output.join("check"),
            repaired_masks: Some(&repaired),
            preview_views: if options.preview { 3 } else { 0 },
            check_views: 24,
        };
        let result = check::run_with(&check_options, &stage.events, &stage.control);
        if let Ok(report) = &result {
            stage.control.log(serde_json::to_string_pretty(report)?);
        }
        driver.report["photo_check"] = driver.end(stage, result)?;
    }
    if !options.keep_volume {
        crate::storage::remove_file(&volume)?;
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

/// How a run may start on this platform, for a front end that builds its form
/// from data: `{"schema", "platform", "start_points": [...]}`. Each start point
/// has `id`, `label`, `meaning`, `fields` (`key` is the field of [`RunOptions`];
/// `kind` is `inputs`, `folder` or `file`) and `providers`: per module of the
/// photos stage the implementations this build can run, with `id`, `label`,
/// `meaning`, `platforms`, `external` programs, `license`, and the `option`
/// that selects it in `photo_options`.
pub fn describe() -> Value {
    let field =
        |key: &str, label: &str, kind: &str, help: &str| json!({"key": key, "label": label, "kind": kind, "required": true, "help": help});
    let mut points = vec![
        json!({
            "id": "inputs", "label": "Inputs folder",
            "meaning": "A folder with cameras.json, the undistorted photos and their masks.",
            "fields": [field("inputs", "Inputs folder", "inputs", "A folder with cameras.json, the photos and their masks.")],
            "providers": [],
        }),
        json!({
            "id": "scene", "label": "Camera solution, images and masks",
            "meaning": "An AliceVision solution with its prepared images and one raw mask per photo.",
            "fields": [
                field("scene", "Camera solution (.sfm)", "file", "AliceVision SfM file with poses and one radialk3 lens."),
                field("prepared", "Prepared images", "folder", "Undistorted images named <viewId>.png."),
                field("raw_masks", "Raw masks", "folder", "One 0/255 mask per source photo, named like the photo."),
            ],
            "providers": [],
        }),
    ];
    let table = crate::photos::providers::listing();
    if cfg!(not(target_arch = "wasm32")) {
        let choice = |module: &str, label: &str| {
            let options: Vec<Value> = table[module]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|row| row["available"] == true)
                .map(|row| {
                    json!({
                        "id": row["name"], "label": row["name"], "meaning": row["summary"], "option": row["selector"],
                        "platforms": row["platforms"], "external": row["external"], "license": row["license"],
                    })
                })
                .collect();
            let default = table[module].as_array().into_iter().flatten().find(|row| row["default"] == true && row["available"] == true);
            json!({"module": module, "label": label, "options": options, "default": default.map(|row| row["name"].clone())})
        };
        points.push(json!({
            "id": "photos", "label": "Turntable photos",
            "meaning": "A folder of photos of an object on a turntable and the calibration of the lens; masks and cameras are made first.",
            "fields": [
                field("photos", "Photos", "folder", "One photo per turntable position, named in capture order."),
                field("calibration", "Lens calibration (.json)", "file", "Focal length, principal point and radial distortion of the lens."),
            ],
            "providers": [choice("masks", "Masks"), choice("cameras", "Cameras")],
        }));
    }
    json!({"schema": "crisp3ds_start_points_v1", "platform": table["platform"], "start_points": points})
}

pub const USAGE: &str = "usage: crisp3ds-dense run --output DIR (--photos DIR --calibration JSON [--masks PROVIDER] [--cameras PROVIDER] \
[options of `crisp3ds-dense photos`] | --inputs DIR | --scene FILE --prepared DIR --raw-masks DIR) \
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
            "--device" | "--torch-python" | "--native" | "--photos-timeout" => drop(value()?),
            "--photos" => options.photos = Some(PathBuf::from(value()?)),
            "--photos-option" => options.photo_options.push(value()?),
            "--reference" => bail!("--reference: scoring against a scan is done by scripts/turntable_mesh/scan_evaluate.py; run it on mesh/mesh.stl afterwards"),
            // Everything else belongs to the photos stage (`crisp3ds-dense photos --help`); it checks the names.
            other => {
                options.photo_options.push(match &inline {
                    Some(value) => format!("{other}={value}"),
                    None => other.to_string(),
                });
                if inline.is_none() && rest.clone().next().is_some_and(|next| !next.starts_with("--")) {
                    options.photo_options.push(rest.next().cloned().expect("peeked"));
                }
            }
        }
    }
    if options.photos.is_none() && !options.photo_options.is_empty() {
        bail!("unknown argument {} (options of the photos stage need --photos)\n{USAGE}", options.photo_options[0]);
    }
    if options.output.as_os_str().is_empty() {
        bail!("--output is required\n{USAGE}");
    }
    Ok(options)
}

/// `crisp3ds-dense run ...`: prints the summary the Python driver prints.
#[cfg(not(target_arch = "wasm32"))]
pub fn main(arguments: &[String]) -> anyhow::Result<()> {
    if arguments.iter().any(|a| a == "--describe") {
        println!("{}", serde_json::to_string_pretty(&describe())?);
        return Ok(());
    }
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
        let from_photos = parse(
            &"--output run --photos shots --calibration lens.json --masks threshold --colmap-masks=off --no-sam-multimask --threads 3"
                .split(' ')
                .map(String::from)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(from_photos.photos, Some(PathBuf::from("shots")));
        assert_eq!(
            from_photos.photo_options,
            ["--calibration", "lens.json", "--masks", "threshold", "--colmap-masks=off", "--no-sam-multimask"]
        );
        assert_eq!(from_photos.threads, 3);
        for refused in ["--reference scan.ply", "--output run --inputs in --bogus", "--inputs in"] {
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
    fn start_points_are_described_as_the_form_reads_them() {
        let described = describe();
        let points = described["start_points"].as_array().unwrap();
        let ids: Vec<&str> = points.iter().map(|p| p["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["inputs", "scene", "photos"]);
        for point in points {
            assert!(point["label"].is_string() && !point["fields"].as_array().unwrap().is_empty());
            for field in point["fields"].as_array().unwrap() {
                assert!(["inputs", "folder", "file"].contains(&field["kind"].as_str().unwrap()));
                // Every field is a field of RunOptions or an option of the photos stage.
                let key = field["key"].as_str().unwrap();
                assert!(
                    serde_json::to_value(key).is_ok()
                        && ["inputs", "scene", "prepared", "raw_masks", "photos", "calibration"].contains(&key)
                );
            }
        }
        let providers = points[2]["providers"].as_array().unwrap();
        assert_eq!(providers.iter().map(|c| c["module"].as_str().unwrap()).collect::<Vec<_>>(), ["masks", "cameras"]);
        for choice in providers {
            let options = choice["options"].as_array().unwrap();
            assert!(options.iter().any(|o| o["id"] == choice["default"]));
            assert!(options.iter().all(|o| o["license"].is_string() && o["option"].as_str().unwrap().starts_with("--")));
        }
    }

    #[test]
    fn refuses_before_anything_is_written() {
        let root = std::env::temp_dir().join(format!("crisp3ds-run-refuse-{}", std::process::id()));
        let _ = crate::storage::remove_dir_all(&root);
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
        let _ = crate::storage::remove_dir_all(&root);
        crate::storage::create_dir_all(root.join("in")).unwrap();
        let options = RunOptions { output: root.join("run"), inputs: Some(root.join("in")), ..Default::default() };
        assert!(run(&options, None, None).is_err());
        let log = crate::storage::read_to_string(root.join("run/events.jsonl")).unwrap();
        let types: Vec<String> =
            log.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["type"].as_str().unwrap().to_string()).collect();
        assert_eq!(types, ["run_started", "error", "run_finished"]);
        let last: Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
        assert_eq!(last["status"], "failed");
        let pipeline: Value = serde_json::from_str(&crate::storage::read_to_string(root.join("run/pipeline.json")).unwrap()).unwrap();
        assert_eq!(pipeline["status"], "failed before or between stages");
        assert!(root.join("run/config.json").is_file());
        crate::storage::remove_dir_all(&root).unwrap();
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
        let run_directory = crate::storage::canonicalize(root.join("run")).unwrap();
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
        assert_eq!(crate::storage::read_to_string(run_directory.join("events.jsonl")).unwrap().lines().count(), events.len());
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

        // The same run entirely in the in-memory tree: same result, nothing on disk.
        let memory = PathBuf::from(format!("mem:/run-test-{}", std::process::id()));
        synthetic::write(&memory.join("inputs"), 24, 128).unwrap();
        let options = RunOptions {
            output: memory.join("run"),
            inputs: Some(memory.join("inputs")),
            overrides: overrides.clone(),
            ..Default::default()
        };
        let in_memory = run(&options, None, None).unwrap();
        assert_eq!(in_memory["triangles"], report["triangles"]);
        assert_eq!(in_memory["photo_check"]["silhouette_iou_input_masks"], report["photo_check"]["silhouette_iou_input_masks"]);
        for file in ["run/events.jsonl", "run/pipeline.json", "run/mesh/mesh.stl", "run/check/preview.png", "run/stereo.log"] {
            assert!(crate::storage::is_file(memory.join(file)), "{file}");
        }
        assert!(!Path::new("mem:").exists(), "a memory run must not touch the disk");
        assert_eq!(
            crate::storage::read_to_string(memory.join("run/events.jsonl")).unwrap().lines().count(),
            crate::storage::read_to_string(run_directory.join("events.jsonl")).unwrap().lines().count()
        );
        crate::storage::remove_dir_all(&memory).unwrap();
        assert!(crate::storage::memory_files(&memory).is_empty());

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
        let pipeline: Value = serde_json::from_str(&crate::storage::read_to_string(root.join("cancelled/pipeline.json")).unwrap()).unwrap();
        assert_eq!(pipeline["status"], "failed in stereo");
        assert_eq!(pipeline["stages"]["stereo"]["cancelled"], true);
        assert!(pipeline["stages"]["stereo"]["exit_code"].is_null());
        crate::storage::remove_dir_all(&root).unwrap();
    }
}
