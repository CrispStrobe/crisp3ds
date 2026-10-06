//! Orchestration of the photo front stage (`run` of `photos_to_inputs.py`):
//! two stages in the event log, `masks` then `cameras`; image processing and
//! bookkeeping in this process, AliceVision (and optionally SAM) as bounded
//! child processes; the gates decide the exit code.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};

use crate::control::Stopped;
use crate::events::EventLog;

use super::audit::{audit_scene, Policy};
use super::calibration::{calibrated_scene, load_calibration, scale_calibration, scene_size};
use super::options::{ExternalCommand, MaskMode, Options, SCHEMA};
use super::process::{bounded, log_tail};
use super::ring::{decide_gates, ring_statistics, scene_cameras};
use super::staging::{capture_name, list_photos, open_photo, save_mask};
use super::{cleanup, contrast, sheets, staging, util};

pub const INTERMEDIATES: [&str; 4] = ["work/photos", "work/contrast", "work/features", "work/matches"];

/// The report (`frontend.json`) and the exit code: 0 complete, 2 cameras rejected by a gate, 1 a step failed.
pub struct Finished {
    pub report: Value,
    pub code: u8,
}

struct Run<'a> {
    options: &'a Options,
    out: &'a Path,
    events: EventLog,
    cancel_file: PathBuf,
    cancel: Option<Arc<AtomicBool>>,
    commands: Vec<ExternalCommand>,
    report: Value,
    started: Instant,
}

/// A handful of progress events per step, not one per photo: the fraction is rounded to two digits and reported in steps of 0.05.
struct Ticker<'a> {
    events: &'a EventLog,
    low: f64,
    high: f64,
    last: f64,
    message: &'a str,
}

impl Ticker<'_> {
    fn tick(&mut self, completion: f64) {
        let fraction = ((self.low + (self.high - self.low) * completion.clamp(0.0, 1.0)) * 100.0).round() / 100.0;
        if fraction >= self.last + 0.05 {
            self.last = fraction;
            let _ = self.events.progress(fraction, self.message);
        }
    }
}

fn count_files(folder: &Path, extension: &str) -> usize {
    std::fs::read_dir(folder)
        .map(|entries| entries.flatten().filter(|e| e.path().extension().is_some_and(|x| x == extension)).count())
        .unwrap_or(0)
}

fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

/// Masks made elsewhere, brought to the names and form the cleanup reads.
fn import_masks(source: &Path, target: &Path, map: &Value, watch: &mut dyn FnMut(usize) -> anyhow::Result<()>) -> anyhow::Result<()> {
    std::fs::create_dir_all(target)?;
    let (width, height) = (map["width"].as_u64().unwrap_or(0) as u32, map["height"].as_u64().unwrap_or(0) as u32);
    for (index, row) in map["photos"].as_array().ok_or_else(|| anyhow!("photo-map.json has no photos"))?.iter().enumerate() {
        let (capture, original) = (capture_name(index), row["source"].as_str().unwrap_or_default());
        let stem = Path::new(original).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let candidates =
            [capture.clone(), format!("{capture}.png"), original.to_string(), format!("{original}.png"), format!("{stem}.png")];
        let found =
            candidates.iter().map(|name| source.join(name)).find(|path| path.extension().is_some_and(|e| e == "png") && path.is_file());
        let Some(path) = found else {
            bail!("--masks {}: no mask for photo {original} (looked for {})", source.display(), candidates.join(", "));
        };
        let gray = image::open(&path).with_context(|| path.display().to_string())?.to_luma8();
        if (gray.width(), gray.height()) != (width, height) {
            bail!("{}: mask is {}x{}, the photos are {width}x{height}", path.display(), gray.width(), gray.height());
        }
        let data: Vec<u8> = gray.into_raw().into_iter().map(|v| (v > 127) as u8).collect();
        if !data.contains(&1) {
            bail!("{}: mask is empty", path.display());
        }
        save_mask(&target.join(format!("{capture}.png")), &crate::inputs::Plane { width: width as usize, height: height as usize, data })?;
        watch(index + 1)?;
    }
    Ok(())
}

/// Audit, ring statistics and the gate decision; writes the three JSON files and returns the gates.
fn audit(options: &Options, out: &Path) -> anyhow::Result<Value> {
    let sfm = out.join("sfm");
    let scene = util::read_json(&sfm.join("final.sfm"))?;
    let expected = util::read_json(&sfm.join("expected-calibration.json"))?;
    let list = |value: &Value| -> Vec<f64> { value.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default() };
    let (pixels, k) = (list(&expected["pixels"]), list(&expected["k"]));
    if pixels.len() != 4 || k.len() != 3 {
        bail!("expected-calibration.json is malformed");
    }
    let gates = &options.gates;
    let names: Vec<String> = (0..options.photo_count).map(capture_name).collect();
    let policy = Policy {
        expected_names: Some(&names),
        expected_calibration: Some(([pixels[0], pixels[1], pixels[2], pixels[3]], [k[0], k[1], k[2]])),
        minimum_coverage: gates.minimum_registered_fraction,
        minimum_observations_per_view: gates.minimum_observations_per_view,
        maximum_reprojection_p95: gates.maximum_view_reprojection_p95_pixels,
    };
    let audit = audit_scene(&sfm.join("final.sfm"), &policy)?;
    util::write_json(&sfm.join("camera-audit.json"), &audit, 1)?;
    let ring = ring_statistics(&scene_cameras(&scene)?, gates.duplicate_step_deg);
    util::write_json(&sfm.join("ring-sanity.json"), &ring, 1)?;
    let limits = gates.to_json();
    let (passed, reasons) = decide_gates(&audit, &ring, &limits);
    let worst = audit["per_view"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|view| view["reprojection_pixels"]["p95"].as_f64())
        .fold(None, |best: Option<f64>, p95| Some(best.map_or(p95, |b| b.max(p95))));
    let result = json!({
        "passed": passed, "reasons": reasons, "limits": limits, "registered_views": audit["registered_cameras"],
        "input_photos": audit["input_images"], "missing": audit["missing_names"], "landmarks": audit["landmarks"],
        "observations": audit["observations"], "reprojection_pixels": audit["reprojection_pixels"],
        "worst_view_reprojection_p95_pixels": worst, "positive_depth_fraction": audit["positive_depth_fraction"], "ring": ring,
    });
    util::write_json(&sfm.join("gates.json"), &result, 1)?;
    Ok(result)
}

impl Run<'_> {
    fn save(&mut self) -> anyhow::Result<()> {
        self.report["seconds"] = json!(self.started.elapsed().as_secs_f64());
        util::write_json(&self.out.join("frontend.json"), &self.report, 1)
    }

    fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Relaxed)) || self.cancel_file.exists()
    }

    fn warn(&mut self, text: String) {
        if let Some(warnings) = self.report["warnings"].as_array_mut() {
            warnings.push(json!(text));
        }
    }

    fn step_count(&self) -> usize {
        self.report["steps"].as_array().map(Vec::len).unwrap_or(0)
    }

    /// Free space floor, the opening progress event and the log file of a step.
    fn begin(&self, stage: &str, name: &str, low: f64, message: &str) -> anyhow::Result<PathBuf> {
        let free = fs4::available_space(self.out).with_context(|| self.out.display().to_string())? as f64 / (1u64 << 30) as f64;
        if free < self.options.minimum_free_gib {
            bail!("{name}: only {free:.1} GiB free; need {} (--minimum-free-gib)", util::python_float(self.options.minimum_free_gib));
        }
        if self.cancelled() {
            bail!("{name} failed (cancelled)");
        }
        self.events.stage(stage).progress(low, message)?;
        println!("[{stage}] {message} ...");
        Ok(self.out.join("logs").join(format!("{:02}-{name}.log", self.step_count() + 1)))
    }

    fn record(&mut self, stage: &str, name: &str, mut entry: Value) -> anyhow::Result<()> {
        entry["name"] = json!(name);
        entry["stage"] = json!(stage);
        if let Some(steps) = self.report["steps"].as_array_mut() {
            steps.push(entry);
        }
        self.save()
    }

    /// One external tool as a bounded process. `counter` estimates its completion between 0 and 1.
    #[allow(clippy::too_many_arguments)]
    fn external(
        &mut self,
        stage: &str,
        name: &str,
        timeout: u64,
        low: f64,
        high: f64,
        message: &str,
        counter: Option<&dyn Fn() -> f64>,
    ) -> anyhow::Result<()> {
        let log = self.begin(stage, name, low, message)?;
        let entry = self.commands.iter().find(|c| c.name == name).cloned().ok_or_else(|| anyhow!("no command for step {name}"))?;
        let threads = self.options.threads.to_string();
        let mut environment: Vec<(String, String)> = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
            .iter()
            .map(|n| (n.to_string(), threads.clone()))
            .collect();
        environment.extend(entry.environment.iter().cloned());
        let directory =
            if name == "sam" { self.options.repository.clone().unwrap_or_else(|| self.out.to_path_buf()) } else { self.out.to_path_buf() };
        let staged = self.events.stage(stage);
        let mut ticker = Ticker { events: &staged, low, high, last: low, message };
        let outcome = bounded(&entry.command, &log, Duration::from_secs(timeout), &environment, &directory, &mut || {
            if let Some(counter) = counter {
                ticker.tick(counter());
            }
            self.cancelled().then_some(true)
        })?;
        self.record(stage, name, outcome.to_json())?;
        if !outcome.succeeded() {
            let reason = if outcome.cancelled {
                "cancelled".to_string()
            } else if outcome.timed_out {
                format!("deadline of {timeout} s")
            } else {
                format!("exit code {}", outcome.exit_code.map(|c| c.to_string()).unwrap_or_else(|| "none".into()))
            };
            bail!("{name} failed ({reason}): {}", log_tail(&log, 600));
        }
        self.events.stage(stage).progress(high, &format!("{message}: done"))?;
        Ok(())
    }

    /// One step done in this process. `work` gets a callback to report finished
    /// items (of `total`), which also enforces the deadline and the cancel request.
    #[allow(clippy::too_many_arguments)]
    fn internal<T>(
        &mut self,
        stage: &str,
        name: &str,
        timeout: u64,
        low: f64,
        high: f64,
        message: &str,
        total: usize,
        work: impl FnOnce(&mut dyn FnMut(usize) -> anyhow::Result<()>) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let log = self.begin(stage, name, low, message)?;
        let started = Instant::now();
        let staged = self.events.stage(stage);
        let mut ticker = Ticker { events: &staged, low, high, last: low, message };
        let (cancel, cancel_file) = (self.cancel.clone(), self.cancel_file.clone());
        let result = work(&mut |done| {
            if cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Relaxed)) || cancel_file.exists() {
                return Err(Stopped::Cancelled.into());
            }
            if started.elapsed() > Duration::from_secs(timeout) {
                return Err(Stopped::Deadline.into());
            }
            ticker.tick(done as f64 / total.max(1) as f64);
            Ok(())
        });
        let stopped = result.as_ref().err().and_then(|e| e.downcast_ref::<Stopped>().copied());
        let text = match &result {
            Ok(_) => "done\n".to_string(),
            Err(error) => format!("{error:#}\n"),
        };
        std::fs::write(&log, &text).with_context(|| log.display().to_string())?;
        let entry = json!({
            "command": ["crisp3ds-dense", "photos", "(in process)", name],
            "exit_code": if stopped.is_some() { Value::Null } else { json!(result.is_err() as u8) },
            "timed_out": stopped == Some(Stopped::Deadline), "cancelled": stopped == Some(Stopped::Cancelled),
            "seconds": started.elapsed().as_secs_f64(), "log": log.to_string_lossy(),
        });
        self.record(stage, name, entry)?;
        let value = result.map_err(|error| match stopped {
            Some(Stopped::Cancelled) => anyhow!("{name} failed (cancelled)"),
            Some(Stopped::Deadline) => anyhow!("{name} failed (deadline of {timeout} s)"),
            None => anyhow!("{name} failed: {error:#}"),
        })?;
        self.events.stage(stage).progress(high, &format!("{message}: done"))?;
        Ok(value)
    }

    fn masks_stage(&mut self) -> anyhow::Result<()> {
        let (options, out) = (self.options, self.out);
        let (count, small, threads) = (options.photo_count, options.timeouts.small, options.threads);
        let stage_started = Instant::now();
        self.events.stage("masks").emit("stage_started", json!({}))?;
        // The share of the stage each step gets depends on whether a network runs in between.
        let segmenting = options.mask_mode != MaskMode::Dark;
        let (coarse_end, cleanup_start, cleanup_end) =
            if options.mask_mode == MaskMode::Sam { (0.08, 0.85, 0.92) } else { (0.45, 0.55, 0.85) };
        let photos = list_photos(&options.photos)?;
        let photo_map =
            self.internal("masks", "coarse", small, 0.0, coarse_end, "Reading photos, coarse dark-object masks", count, |watch| {
                staging::step_coarse(out, &photos, &options.envelope, options.dark_threshold, threads, watch)
            })?;
        let touching: Vec<&str> = photo_map["photos"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row["touches_envelope"] == true)
            .filter_map(|row| row["capture"].as_str())
            .collect();
        if let Some(first) = touching.first() {
            self.warn(format!(
                "coarse object region touches the envelope in {} photos (first {first}): object clipped by the frame or by --envelope",
                touching.len()
            ));
        }
        let segmented = match options.mask_mode {
            MaskMode::Dark => out.join("work/coarse-masks"),
            MaskMode::Sam => {
                let produced = out.join("work/sam/masks");
                let counter = || count_files(&produced, "png") as f64 / count as f64;
                self.external("masks", "sam", options.timeouts.sam, coarse_end, cleanup_start, "Segmenting with SAM 2.1", Some(&counter))?;
                produced
            }
            MaskMode::External => {
                let (source, target) =
                    (options.masks.clone().ok_or_else(|| anyhow!("--masks is missing"))?, out.join("work/external-masks"));
                self.internal("masks", "import-masks", small, coarse_end, cleanup_start, "Reading supplied masks", count, |watch| {
                    import_masks(&source, &target, &photo_map, watch)
                })?;
                target
            }
        };
        let cleaned = out.join("work/mask-cleanup");
        self.internal("masks", "cleanup", small, cleanup_start, cleanup_end, "Filling small dark holes", count, |watch| {
            cleanup::run(&out.join("work/photos"), &segmented, &cleaned, options.hole_cleanup_budget, threads, watch)
        })?;
        let masks =
            self.internal("masks", "publish-masks", small, cleanup_end, 1.0, "Writing masks and contact sheet", count, |watch| {
                sheets::step_publish_masks(out, count, &cleaned.join("masks"), watch)
            })?;
        let mut summary = masks.clone();
        if let Some(object) = summary.as_object_mut() {
            object.remove("views");
        }
        summary["mode"] = json!(options.mask_mode.name());
        self.report["masks"] = summary;
        let dropping = masks["views_dropping_over_3_percent_of_coarse"].as_array().map(Vec::len).unwrap_or(0);
        if dropping > 0 && segmenting {
            let who = if options.mask_mode == MaskMode::Sam { "SAM dropped" } else { "the supplied masks leave out" };
            self.warn(format!("{who} more than 3% of the coarse dark region in {dropping} photos (thin parts?)"));
        }
        let staged = self.events.stage("masks");
        staged.emit("metric", json!({"name": "mask_area_median_pixels", "value": masks["median_area_pixels"]}))?;
        let fraction = round_to(masks["median_area_fraction"].as_f64().unwrap_or(0.0), 5);
        staged.emit("metric", json!({"name": "mask_area_median_fraction", "value": fraction}))?;
        staged.artifact("mask_sheet", &out.join("mask-contact-sheet.png"), "Masks on photos (red: dark pixels left out)", json!({}))?;
        staged.emit("stage_finished", json!({"seconds": stage_started.elapsed().as_secs_f64()}))?;
        Ok(())
    }

    /// Returns the exit code: 0 when the cameras passed, 2 when a gate rejected them.
    fn cameras_stage(&mut self) -> anyhow::Result<u8> {
        let (options, out) = (self.options, self.out);
        let (count, timeouts, threads) = (options.photo_count, &options.timeouts, options.threads);
        let small = timeouts.small;
        let stage_started = Instant::now();
        let staged = self.events.stage("cameras");
        staged.emit("stage_started", json!({}))?;
        self.internal("cameras", "contrast", small, 0.0, 0.04, "Contrast images for feature detection", count, |watch| {
            let target = out.join("work/contrast");
            std::fs::create_dir_all(&target)?;
            let work = |index: usize| -> anyhow::Result<()> {
                let name = capture_name(index);
                let photo = open_photo(&out.join("work/photos").join(&name))?;
                let (width, height) = (photo.width(), photo.height());
                let mut rgb = photo.rgb.into_raw();
                contrast::contrast_image(&mut rgb, width, height, options.contrast_gamma, options.clahe_clip, options.clahe_grid);
                util::save_rgb(&target.join(&name), width, height, rgb)
            };
            util::parallel(count, threads, work, watch).map(|_| ())
        })?;
        self.external("cameras", "cameraInit-uncalibrated", small, 0.04, 0.05, "Listing views", None)?;
        let features = out.join("work/features");
        let extracted = || count_files(&features, "feat") as f64 / count as f64;
        self.external("cameras", "featureExtraction", timeouts.features, 0.05, 0.30, "Detecting features", Some(&extracted))?;
        let scene = util::read_json(&out.join("sfm/cameraInit-uncalibrated.sfm"))?;
        let (width, height) = scene_size(&scene)?;
        let scaled = scale_calibration(&load_calibration(&options.calibration)?, width, height)?;
        util::write_json(&out.join("sfm/calibrated-input.sfm"), &calibrated_scene(&scene, &scaled)?, 2)?;
        let expected = json!({"pixels": [scaled.fx, scaled.fy, scaled.cx, scaled.cy], "k": scaled.k});
        util::write_json(&out.join("sfm/expected-calibration.json"), &expected, 1)?;
        let mut lens = scaled.to_json();
        lens["width"] = json!(width);
        lens["height"] = json!(height);
        self.report["lens"] = lens;
        self.external("cameras", "cameraInit", small, 0.30, 0.31, "Applying the declared lens", None)?;
        self.external("cameras", "imageMatching", small, 0.31, 0.32, "Choosing image pairs", None)?;
        self.external("cameras", "featureMatching", timeouts.matching, 0.32, 0.60, "Matching features", None)?;
        self.external("cameras", "globalSfM", timeouts.sfm, 0.60, 0.82, "Recovering cameras (global SfM, locked lens)", None)?;
        let gates = self.internal("cameras", "audit", small, 0.82, 0.86, "Checking cameras", 1, |_| audit(options, out))?;
        self.report["gates"] = gates.clone();
        let ring = &gates["ring"];
        let degenerate = ring["degenerate"].as_bool().unwrap_or(true);
        let rounded = |value: &Value, digits: i32| value.as_f64().map(|v| json!(round_to(v, digits))).unwrap_or(Value::Null);
        let metrics = [
            ("registered_views", gates["registered_views"].clone()),
            ("input_photos", gates["input_photos"].clone()),
            ("reprojection_median_pixels", rounded(&gates["reprojection_pixels"]["median"], 4)),
            ("reprojection_p95_pixels", rounded(&gates["reprojection_pixels"]["p95"], 4)),
            ("ring_radius_spread_percent", if degenerate { Value::Null } else { rounded(&ring["radius_spread_percent"], 3) }),
            ("ring_largest_gap_deg", if degenerate { Value::Null } else { rounded(&ring["largest_angular_gap_deg"], 2) }),
        ];
        for (name, value) in metrics {
            if !value.is_null() {
                staged.emit("metric", json!({"name": name, "value": value}))?;
            }
        }
        let duplicates: Vec<String> =
            ring["duplicate_pose_pairs"].as_array().into_iter().flatten().map(|pair| format!("[{}, {}]", pair[0], pair[1])).collect();
        if !degenerate && !duplicates.is_empty() {
            self.warn(format!("consecutive photos with the same pose (turntable did not move?): [{}]", duplicates.join(", ")));
        }
        if gates["passed"] != true {
            let reasons: Vec<String> =
                gates["reasons"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from).collect();
            self.report["status"] = json!("failed");
            self.report["reasons"] = json!(reasons);
            self.save()?;
            staged.artifact("report", &out.join("frontend.json"), "Front stage report (cameras rejected)", json!({}))?;
            staged.emit("error", json!({"message": format!("cameras rejected: {}", reasons.join("; "))}))?;
            return Ok(2);
        }
        let registered = gates["registered_views"].as_u64().unwrap_or(0).max(1) as f64;
        let prepared = out.join("native-prepared");
        let undistorted = || count_files(&prepared, "png") as f64 / registered;
        self.external("cameras", "prepareDenseScene", timeouts.prepare, 0.86, 0.95, "Undistorting photos", Some(&undistorted))?;
        self.internal("cameras", "verify-prepared", small, 0.95, 0.96, "Checking undistorted photos", 1, |_| {
            sheets::step_verify_prepared(out)
        })?;
        self.internal("cameras", "inputs", small, 0.96, 0.98, "Writing dense inputs", 1, |_| {
            crate::scene::run(&out.join("sfm/final.sfm"), &prepared, &out.join("masks"), &out.join("inputs"))
        })?;
        let overlay = self.internal("cameras", "overlay", small, 0.98, 1.0, "Sparse points on photos", registered as usize, |watch| {
            sheets::step_overlay(out, watch)
        })?;
        self.report["sparse_overlay"] = overlay;
        staged.artifact("sparse_overlay", &out.join("sparse-overlay.png"), "Sparse points on undistorted photos and masks", json!({}))?;
        self.remove_intermediates(&INTERMEDIATES, true);
        let shown = |path: PathBuf| path.to_string_lossy().to_string();
        self.report["outputs"] = json!({
            "inputs": shown(out.join("inputs")), "masks": shown(out.join("masks")), "scene": shown(out.join("sfm/final.sfm")),
            "prepared": shown(prepared.clone()),
        });
        self.report["status"] = json!("complete");
        self.save()?;
        staged.artifact("report", &out.join("frontend.json"), "Front stage report", json!({}))?;
        staged.emit("stage_finished", json!({"seconds": stage_started.elapsed().as_secs_f64()}))?;
        Ok(0)
    }

    fn remove_intermediates(&mut self, names: &[&str], scene_paths_dangle: bool) {
        let mut deleted = Vec::new();
        if !self.options.keep_intermediates {
            for name in names {
                let _ = std::fs::remove_dir_all(self.out.join(name));
                deleted.push(*name);
            }
            if scene_paths_dangle {
                self.warn(
                    "sfm/final.sfm view paths point at deleted contrast images; downstream stages use native-prepared/ and inputs/ instead"
                        .to_string(),
                );
            }
        }
        self.report["intermediates_deleted"] = json!(deleted);
    }
}

/// Runs the stage into the fresh directory `options.output`. Events go to
/// `events` under the stages `masks` and `cameras` (no `run_started` or
/// `run_finished`: those belong to whatever drives the run). A `cancel` file
/// next to the event log, or `cancel` set, stops the run.
pub fn run(options: &Options, events: &EventLog, events_path: &Path, cancel: Option<Arc<AtomicBool>>) -> anyhow::Result<Finished> {
    let out = options.output.as_path();
    if out.exists() {
        bail!("output exists: {}", out.display());
    }
    for name in ["logs", "sfm/extra", "work/matches", "work/features"] {
        std::fs::create_dir_all(out.join(name)).with_context(|| out.join(name).display().to_string())?;
    }
    let configuration = options.to_json();
    util::write_json(&out.join("frontend-config.json"), &configuration, 1)?;
    let environment = |name: &str| std::env::var(name).ok();
    let report = json!({
        "schema": SCHEMA, "status": "running", "reasons": [], "warnings": [], "configuration": configuration,
        "events": events_path.to_string_lossy(), "steps": [], "gates": null, "masks": null,
        "reference_geometry_used": false, "supplied_poses_used": false, "depth_used": false,
    });
    let mut run = Run {
        options,
        out,
        events: events.clone(),
        cancel_file: events_path.parent().unwrap_or(Path::new(".")).join("cancel"),
        cancel,
        commands: Vec::new(),
        report,
        started: Instant::now(),
    };
    let mut stage = "masks";
    let outcome = (|| -> anyhow::Result<u8> {
        run.commands = options.build_commands(&environment)?;
        run.masks_stage()?;
        if options.stop_after_masks {
            run.remove_intermediates(&INTERMEDIATES[..1], false);
            run.report["outputs"] = json!({"masks": out.join("masks").to_string_lossy()});
            run.report["status"] = json!("complete");
            run.report["stopped_after"] = json!("masks");
            run.save()?;
            return Ok(0);
        }
        stage = "cameras";
        run.cameras_stage()
    })();
    let code = match outcome {
        Ok(code) => code,
        Err(error) => {
            let text = format!("{error:#}");
            run.report["status"] = json!("failed");
            run.report["reasons"] = json!([text]);
            run.save()?;
            let skip = text.chars().count().saturating_sub(600);
            events.stage(stage).emit("error", json!({"message": text.chars().skip(skip).collect::<String>()}))?;
            1
        }
    };
    Ok(Finished { report: run.report, code })
}

/// The few lines the command prints at the end.
pub fn summary(report: &Value) -> Value {
    let gates = &report["gates"];
    json!({
        "status": report["status"], "reasons": report["reasons"], "warnings": report["warnings"],
        "registered_views": gates["registered_views"], "reprojection_pixels": gates["reprojection_pixels"],
        "seconds": report["seconds"],
        "report": Path::new(report["configuration"]["output"].as_str().unwrap_or_default()).join("frontend.json").to_string_lossy(),
    })
}
