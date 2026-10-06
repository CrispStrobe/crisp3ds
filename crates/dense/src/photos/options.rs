//! Command line and resolved configuration of the photo front stage, and the
//! external commands it runs (`parser`, `resolve`, `alicevision_command`,
//! `sensor_database` and `build_commands` of `photos_to_inputs.py`).
//!
//! AliceVision stays an external program. It is reached in one of three ways:
//! an install prefix containing `bin/aliceVision_TOOL` (run directly, with the
//! library path set here), an executable wrapper called as `wrapper TOOL
//! args`, or a `.py` wrapper run with `--python`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail};
use serde_json::{json, Value};

use super::coarse::Threshold;
use super::staging::{list_photos, PHOTO_SUFFIXES};

pub const SCHEMA: &str = "crisp3ds_photos_to_inputs_v1";
/// The AliceVision tools of the camera recovery, and the one of the dense build.
pub const SPARSE_TOOLS: [&str; 5] = ["cameraInit", "featureExtraction", "imageMatching", "featureMatching", "globalSfM"];
pub const DENSE_TOOLS: [&str; 1] = ["prepareDenseScene"];

/// Where the object masks come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    /// Coarse dark-object mask, then the dark-hole cleanup. No network.
    Dark,
    /// SAM 2.1 through `scripts/turntable_mesh/segment.py` in an external interpreter, then the cleanup.
    Sam,
    /// Masks made elsewhere (`--masks DIR`), then the cleanup.
    External,
}

impl MaskMode {
    pub fn name(self) -> &'static str {
        match self {
            MaskMode::Dark => "dark",
            MaskMode::Sam => "sam",
            MaskMode::External => "external",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timeouts {
    pub small: u64,
    pub sam: u64,
    pub features: u64,
    pub matching: u64,
    pub sfm: u64,
    pub prepare: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gates {
    pub minimum_registered_fraction: f64,
    pub minimum_observations_per_view: i64,
    pub maximum_view_reprojection_p95_pixels: f64,
    pub minimum_positive_depth_fraction: f64,
    pub maximum_radius_spread_percent: f64,
    pub maximum_out_of_plane_percent: f64,
    pub maximum_angular_gap_deg: f64,
    pub maximum_reversed_steps: i64,
    pub maximum_optical_axis_miss_percent: f64,
    pub duplicate_step_deg: f64,
}

impl Gates {
    pub fn to_json(&self) -> Value {
        json!({
            "minimum_registered_fraction": self.minimum_registered_fraction,
            "minimum_observations_per_view": self.minimum_observations_per_view,
            "maximum_view_reprojection_p95_pixels": self.maximum_view_reprojection_p95_pixels,
            "minimum_positive_depth_fraction": self.minimum_positive_depth_fraction,
            "maximum_radius_spread_percent": self.maximum_radius_spread_percent,
            "maximum_out_of_plane_percent": self.maximum_out_of_plane_percent,
            "maximum_angular_gap_deg": self.maximum_angular_gap_deg,
            "maximum_reversed_steps": self.maximum_reversed_steps,
            "maximum_optical_axis_miss_percent": self.maximum_optical_axis_miss_percent,
            "duplicate_step_deg": self.duplicate_step_deg,
        })
    }
}

/// One AliceVision build: where it is and what its tools need in their environment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tools {
    /// Install prefix, executable wrapper or `.py` wrapper.
    pub location: PathBuf,
    /// Extra library directories after `<prefix>/lib` (prefix form only).
    pub library_path: Vec<PathBuf>,
    /// Extra environment variables (prefix form only).
    pub environment: Vec<(String, String)>,
}

/// The resolved options: flags, then environment variables, then defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub photos: PathBuf,
    pub photo_count: usize,
    pub calibration: PathBuf,
    pub output: PathBuf,
    pub events: Option<PathBuf>,
    pub python: Option<String>,
    pub alicevision: Option<Tools>,
    pub alicevision_dense: Option<Tools>,
    pub sensor_database: Option<PathBuf>,
    pub alicevision_memory_gib: f64,
    pub mask_mode: MaskMode,
    pub masks: Option<PathBuf>,
    pub sam_python: Option<String>,
    pub sam_source: Option<PathBuf>,
    pub sam_checkpoint: Option<PathBuf>,
    pub sam_config: Option<String>,
    pub sam_pythonpath: Vec<String>,
    pub repository: Option<PathBuf>,
    pub device: String,
    pub threads: usize,
    pub minimum_free_gib: f64,
    pub keep_intermediates: bool,
    pub stop_after_masks: bool,
    pub envelope: String,
    pub dark_threshold: Threshold,
    pub sam_multimask: bool,
    pub sam_preserve_holes: bool,
    pub sam_automatic_cues: bool,
    pub hole_cleanup_budget: f64,
    pub contrast_gamma: f64,
    pub clahe_clip: f64,
    pub clahe_grid: usize,
    pub initial_field_of_view: f64,
    pub describer_types: String,
    pub describer_preset: String,
    pub matching_method: String,
    pub random_seed: i64,
    pub sfm_option: Vec<String>,
    pub timeouts: Timeouts,
    pub gates: Gates,
}

pub const USAGE: &str = "\
usage: crisp3ds-dense photos --photos DIR --calibration JSON --output DIR [--events PATH] [options]

From a folder of turntable photos to the dense inputs directory: object masks,
camera recovery with a locked, declared lens (AliceVision, an external
program), quality gates, undistorted photos and inputs/. Exit code 0: complete;
2: cameras rejected by a gate (or a usage error); 1: a step failed.
Port of scripts/turntable_mesh/photos_to_inputs.py; see docs/PHOTOS-TO-INPUTS.md
and crates/dense/README.md.

tools (flag, then environment variable):
  --alicevision PATH              install prefix with bin/aliceVision_*, or a wrapper script [CRISP3DS_ALICEVISION]
  --alicevision-dense PATH        prefix or wrapper with prepareDenseScene [CRISP3DS_ALICEVISION_DENSE, else --alicevision]
  --alicevision-library-path L    extra library directories for a prefix, separated like PATH [CRISP3DS_ALICEVISION_LIBRARY_PATH]
  --alicevision-dense-library-path L                                           [CRISP3DS_ALICEVISION_DENSE_LIBRARY_PATH]
  --alicevision-env K=V           extra variable for the tools of a prefix; repeat [CRISP3DS_ALICEVISION_ENV: K=V,K=V]
  --alicevision-dense-env K=V                                                  [CRISP3DS_ALICEVISION_DENSE_ENV]
  --sensor-database FILE          cameraSensors.db [CRISP3DS_ALICEVISION_SENSOR_DB, else <prefix>/share/aliceVision/]
  --alicevision-memory-gib N      memory hint for the tools (4)
  --python EXE                    only for a .py wrapper [CRISP3DS_PYTHON]
masks:
  --mask-mode dark|sam            dark: threshold, largest dark region, hole cleanup (no network); sam: SAM 2.1 through
                                  scripts/turntable_mesh/segment.py in an external interpreter [CRISP3DS_MASK_MODE] (dark)
  --masks DIR                     use masks made elsewhere instead of segmenting: one 8-bit PNG per photo, white object,
                                  named capture_NNNN.png in capture order or like the photo; the hole cleanup still runs
  --envelope SPEC                 where the object can be: auto (whole frame) or x0,y0,x1,y1 in pixels, fractions if all <= 1
  --dark-threshold N|otsu         grey level below which a pixel is object (70)
  --hole-cleanup-budget F         largest share of the foreground the dark-hole fill may add (0.02)
  --sam-python EXE --sam-source DIR --sam-checkpoint FILE [--sam-config NAME] [--sam-pythonpath LIST] --repository DIR
                                  for --mask-mode sam [CRISP3DS_SAM_PYTHON, _SAM_SOURCE, _SAM_CHECKPOINT, _SAM_CONFIG,
                                  _SAM_PYTHONPATH, CRISP3DS_REPOSITORY]
  --[no-]sam-multimask --[no-]sam-preserve-holes --[no-]sam-automatic-cues   segment.py flags (on)
  --device mps|cpu|cuda           SAM device (mps)
machine:
  --threads N (2)   --minimum-free-gib N (10, checked before every step)   --keep-intermediates
  --stop-after masks              write the masks and stop; needs no AliceVision
camera recovery:
  --contrast-gamma G (0.5; 1 disables)   --clahe-clip C (2.0; 0 disables)   --clahe-grid N (8)
  --initial-field-of-view DEG (45)   --describer-types T (sift)   --describer-preset P (normal)
  --matching-method M (Exhaustive)   --random-seed N (0)   --sfm-option ARG (repeat per token)
deadlines in seconds (the whole process group is stopped):
  --sam-timeout 1800  --features-timeout 1800  --matching-timeout 1800  --sfm-timeout 900  --prepare-timeout 600
  --small-step-timeout 600
quality gates (a failed gate means exit code 2 and no dense inputs):
  --minimum-registered-fraction 0.8   --minimum-observations-per-view 20   --maximum-view-reprojection-p95 4.0
  --minimum-positive-depth-fraction 0.999   --maximum-radius-spread-percent 5   --maximum-out-of-plane-percent 5
  --maximum-angular-gap-deg 30   --maximum-reversed-steps 0   --maximum-optical-axis-miss-percent 25
  --duplicate-step-deg 0.5 (warning only)";

fn absolute(path: &str) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| PathBuf::from(path))
}

fn split_paths(text: &str) -> Vec<String> {
    std::env::split_paths(text).map(|p| p.to_string_lossy().to_string()).filter(|p| !p.is_empty()).collect()
}

fn pair(text: &str) -> anyhow::Result<(String, String)> {
    let (name, value) = text.split_once('=').ok_or_else(|| anyhow!("expected NAME=VALUE, found {text:?}"))?;
    if name.is_empty() {
        bail!("expected NAME=VALUE, found {text:?}");
    }
    Ok((name.to_string(), value.to_string()))
}

/// Parses the command line. `environment` looks up a variable (injected for tests).
pub fn resolve(arguments: &[String], environment: &dyn Fn(&str) -> Option<String>) -> anyhow::Result<Options> {
    let mut values: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    const SWITCHES: [&str; 4] = ["keep-intermediates", "sam-multimask", "sam-preserve-holes", "sam-automatic-cues"];
    const VALUED: &[&str] = &[
        "photos",
        "calibration",
        "output",
        "events",
        "python",
        "alicevision",
        "alicevision-dense",
        "alicevision-library-path",
        "alicevision-dense-library-path",
        "alicevision-env",
        "alicevision-dense-env",
        "sensor-database",
        "alicevision-memory-gib",
        "mask-mode",
        "masks",
        "sam-python",
        "sam-source",
        "sam-checkpoint",
        "sam-config",
        "sam-pythonpath",
        "repository",
        "device",
        "threads",
        "minimum-free-gib",
        "stop-after",
        "envelope",
        "dark-threshold",
        "hole-cleanup-budget",
        "contrast-gamma",
        "clahe-clip",
        "clahe-grid",
        "initial-field-of-view",
        "describer-types",
        "describer-preset",
        "matching-method",
        "random-seed",
        "sfm-option",
        "sam-timeout",
        "features-timeout",
        "matching-timeout",
        "sfm-timeout",
        "prepare-timeout",
        "small-step-timeout",
        "minimum-registered-fraction",
        "minimum-observations-per-view",
        "maximum-view-reprojection-p95",
        "minimum-positive-depth-fraction",
        "maximum-radius-spread-percent",
        "maximum-out-of-plane-percent",
        "maximum-angular-gap-deg",
        "maximum-reversed-steps",
        "maximum-optical-axis-miss-percent",
        "duplicate-step-deg",
    ];
    let mut index = 0;
    while index < arguments.len() {
        let word = &arguments[index];
        let Some(flag) = word.strip_prefix("--") else { bail!("unexpected argument {word:?}") };
        let (name, inline) = match flag.split_once('=') {
            Some((name, value)) => (name, Some(value.to_string())),
            None => (flag, None),
        };
        if let Some(switch) = SWITCHES.iter().find(|s| name == **s || name.strip_prefix("no-") == Some(**s)) {
            if inline.is_some() {
                bail!("--{name} takes no value");
            }
            values.insert(switch.to_string(), vec![(!name.starts_with("no-")).to_string()]);
        } else if VALUED.contains(&name) {
            let value = match inline {
                Some(value) => value,
                None => {
                    index += 1;
                    arguments.get(index).cloned().ok_or_else(|| anyhow!("--{name} needs a value"))?
                }
            };
            let slot = values.entry(name.to_string()).or_default();
            if !matches!(name, "sfm-option" | "alicevision-env" | "alicevision-dense-env") {
                slot.clear();
            }
            slot.push(value);
        } else {
            bail!("unknown option --{name}");
        }
        index += 1;
    }
    let text = |name: &str| values.get(name).and_then(|v| v.last()).cloned();
    let pick =
        |name: &str, variable: &str| text(name).filter(|v| !v.is_empty()).or_else(|| environment(variable).filter(|v| !v.is_empty()));
    let number = |name: &str, default: f64| -> anyhow::Result<f64> {
        match text(name) {
            None => Ok(default),
            Some(value) => {
                value.trim().parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| anyhow!("--{name} needs a number, got {value:?}"))
            }
        }
    };
    let integer = |name: &str, default: i64| -> anyhow::Result<i64> {
        match text(name) {
            None => Ok(default),
            Some(value) => value.trim().parse::<i64>().map_err(|_| anyhow!("--{name} needs an integer, got {value:?}")),
        }
    };
    let switch = |name: &str, default: bool| text(name).map(|v| v == "true").unwrap_or(default);
    let required = |name: &str| text(name).ok_or_else(|| anyhow!("--{name} is required"));

    let photos = absolute(&required("photos")?);
    let calibration = absolute(&required("calibration")?);
    let output = absolute(&required("output")?);
    if !photos.is_dir() {
        bail!("--photos must be a directory");
    }
    let photo_count = list_photos(&photos)?.len();
    if !(3..=96).contains(&photo_count) {
        bail!("found {photo_count} photos ({}); the camera audit supports 3 to 96", PHOTO_SUFFIXES.join(", "));
    }
    let stop_after_masks = match text("stop-after").as_deref() {
        None => false,
        Some("masks") => true,
        Some(other) => bail!("--stop-after {other}: only 'masks' is offered"),
    };
    let masks = text("masks").map(|p| absolute(&p));
    let mask_mode = match (masks.is_some(), pick("mask-mode", "CRISP3DS_MASK_MODE").as_deref()) {
        (true, _) => MaskMode::External,
        (false, None | Some("dark")) => MaskMode::Dark,
        (false, Some("sam")) => MaskMode::Sam,
        (false, Some(other)) => bail!("--mask-mode {other}: expected dark or sam"),
    };
    if masks.as_ref().is_some_and(|m| !m.is_dir()) {
        bail!("--masks must be a directory");
    }
    let python = pick("python", "CRISP3DS_PYTHON");
    let tools = |flag: &str, variable: &str, fallback: Option<&Tools>| -> anyhow::Result<Option<Tools>> {
        let Some(location) = pick(flag, variable) else { return Ok(fallback.cloned()) };
        let library_flag = format!("{flag}-library-path");
        let library_path = pick(&library_flag, &format!("{variable}_LIBRARY_PATH")).map(|v| split_paths(&v)).unwrap_or_default();
        let mut extra = Vec::new();
        match values.get(&format!("{flag}-env")) {
            Some(pairs) => {
                for item in pairs {
                    extra.push(pair(item)?);
                }
            }
            None => {
                for item in environment(&format!("{variable}_ENV")).unwrap_or_default().split(',').filter(|v| !v.is_empty()) {
                    extra.push(pair(item)?);
                }
            }
        }
        Ok(Some(Tools {
            location: absolute(&location),
            library_path: library_path.into_iter().map(PathBuf::from).collect(),
            environment: extra,
        }))
    };
    let alicevision = tools("alicevision", "CRISP3DS_ALICEVISION", None)?;
    let alicevision_dense = tools("alicevision-dense", "CRISP3DS_ALICEVISION_DENSE", alicevision.as_ref())?;
    let (sam_python, sam_source, sam_checkpoint) = (
        pick("sam-python", "CRISP3DS_SAM_PYTHON"),
        pick("sam-source", "CRISP3DS_SAM_SOURCE"),
        pick("sam-checkpoint", "CRISP3DS_SAM_CHECKPOINT"),
    );
    let repository = pick("repository", "CRISP3DS_REPOSITORY");
    let mut missing = Vec::new();
    if alicevision.is_none() && !stop_after_masks {
        missing.push("--alicevision / CRISP3DS_ALICEVISION");
    }
    if mask_mode == MaskMode::Sam {
        for (name, value) in [
            ("--sam-python / CRISP3DS_SAM_PYTHON", &sam_python),
            ("--sam-source / CRISP3DS_SAM_SOURCE", &sam_source),
            ("--sam-checkpoint / CRISP3DS_SAM_CHECKPOINT", &sam_checkpoint),
            ("--repository / CRISP3DS_REPOSITORY (the checkout with scripts/turntable_mesh/segment.py)", &repository),
        ] {
            if value.is_none() {
                missing.push(name);
            }
        }
    }
    if !missing.is_empty() {
        bail!("missing tool locations: {}", missing.join("; "));
    }
    let device = text("device").unwrap_or_else(|| "mps".into());
    if !["mps", "cpu", "cuda"].contains(&device.as_str()) {
        bail!("--device {device}: expected mps, cpu or cuda");
    }
    let threads = integer("threads", 2)?;
    let clahe_grid = integer("clahe-grid", 8)?;
    if !(1..=256).contains(&threads) || !(1..=64).contains(&clahe_grid) {
        bail!("--threads must be 1..256 and --clahe-grid 1..64");
    }
    let seconds = |name: &str, default: i64| -> anyhow::Result<u64> {
        let value = integer(name, default)?;
        if value < 1 {
            bail!("--{name} must be at least one second");
        }
        Ok(value as u64)
    };
    let options = Options {
        photos,
        photo_count,
        calibration,
        output,
        events: text("events").map(|p| absolute(&p)),
        python,
        alicevision,
        alicevision_dense,
        sensor_database: pick("sensor-database", "CRISP3DS_ALICEVISION_SENSOR_DB").map(PathBuf::from),
        alicevision_memory_gib: number("alicevision-memory-gib", 4.0)?,
        mask_mode,
        masks,
        sam_python,
        sam_source: sam_source.map(|p| absolute(&p)),
        sam_checkpoint: sam_checkpoint.map(|p| absolute(&p)),
        sam_config: pick("sam-config", "CRISP3DS_SAM_CONFIG"),
        sam_pythonpath: pick("sam-pythonpath", "CRISP3DS_SAM_PYTHONPATH").map(|v| split_paths(&v)).unwrap_or_default(),
        repository: repository.map(|p| absolute(&p)),
        device,
        threads: threads as usize,
        minimum_free_gib: number("minimum-free-gib", 10.0)?,
        keep_intermediates: switch("keep-intermediates", false),
        stop_after_masks,
        envelope: text("envelope").unwrap_or_else(|| "auto".into()),
        dark_threshold: Threshold::parse(&text("dark-threshold").unwrap_or_else(|| "70".into()))?,
        sam_multimask: switch("sam-multimask", true),
        sam_preserve_holes: switch("sam-preserve-holes", true),
        sam_automatic_cues: switch("sam-automatic-cues", true),
        hole_cleanup_budget: super::cleanup::validate_budget(number("hole-cleanup-budget", 0.02)?)?,
        contrast_gamma: number("contrast-gamma", 0.5)?,
        clahe_clip: number("clahe-clip", 2.0)?,
        clahe_grid: clahe_grid as usize,
        initial_field_of_view: number("initial-field-of-view", 45.0)?,
        describer_types: text("describer-types").unwrap_or_else(|| "sift".into()),
        describer_preset: text("describer-preset").unwrap_or_else(|| "normal".into()),
        matching_method: text("matching-method").unwrap_or_else(|| "Exhaustive".into()),
        random_seed: integer("random-seed", 0)?,
        sfm_option: values.get("sfm-option").cloned().unwrap_or_default(),
        timeouts: Timeouts {
            small: seconds("small-step-timeout", 600)?,
            sam: seconds("sam-timeout", 1800)?,
            features: seconds("features-timeout", 1800)?,
            matching: seconds("matching-timeout", 1800)?,
            sfm: seconds("sfm-timeout", 900)?,
            prepare: seconds("prepare-timeout", 600)?,
        },
        gates: Gates {
            minimum_registered_fraction: number("minimum-registered-fraction", 0.8)?,
            minimum_observations_per_view: integer("minimum-observations-per-view", 20)?,
            maximum_view_reprojection_p95_pixels: number("maximum-view-reprojection-p95", 4.0)?,
            minimum_positive_depth_fraction: number("minimum-positive-depth-fraction", 0.999)?,
            maximum_radius_spread_percent: number("maximum-radius-spread-percent", 5.0)?,
            maximum_out_of_plane_percent: number("maximum-out-of-plane-percent", 5.0)?,
            maximum_angular_gap_deg: number("maximum-angular-gap-deg", 30.0)?,
            maximum_reversed_steps: integer("maximum-reversed-steps", 0)?,
            maximum_optical_axis_miss_percent: number("maximum-optical-axis-miss-percent", 25.0)?,
            duplicate_step_deg: number("duplicate-step-deg", 0.5)?,
        },
    };
    super::coarse::validate_envelope(&options.envelope)?;
    super::calibration::load_calibration(&options.calibration)?;
    if !options.stop_after_masks {
        options.check_tools()?;
    }
    Ok(options)
}

fn executable(prefix: &Path, tool: &str) -> PathBuf {
    prefix.join("bin").join(format!("aliceVision_{tool}{}", std::env::consts::EXE_SUFFIX))
}

/// `(command, extra environment)` for one AliceVision tool.
///
/// A prefix gets `ALICEVISION_ROOT` and the library path (`<prefix>/lib`, the
/// extra directories, then whatever the variable already held) under the names
/// the dynamic loaders of macOS and Linux read, and `<prefix>/bin` and
/// `<prefix>/lib` in front of `PATH` on Windows.
#[allow(clippy::type_complexity)]
pub fn alicevision_command(
    tools: &Tools,
    tool: &str,
    arguments: &[String],
    python: Option<&str>,
    environment: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<(Vec<String>, Vec<(String, String)>)> {
    let location = tools.location.to_string_lossy().to_string();
    let mut command;
    let mut extra = Vec::new();
    if tools.location.is_dir() {
        command = vec![executable(&tools.location, tool).to_string_lossy().to_string()];
        extra.push(("ALICEVISION_ROOT".to_string(), location));
        let mut libraries = vec![tools.location.join("lib")];
        libraries.extend(tools.library_path.iter().cloned());
        let join = |first: &[PathBuf], name: &str| -> anyhow::Result<String> {
            let inherited = environment(name).filter(|v| !v.is_empty());
            let all = first.iter().cloned().chain(inherited.iter().flat_map(std::env::split_paths));
            Ok(std::env::join_paths(all).map_err(|e| anyhow!("library path: {e}"))?.to_string_lossy().to_string())
        };
        for name in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"] {
            extra.push((name.to_string(), join(&libraries, name)?));
        }
        if cfg!(windows) {
            let mut first = vec![tools.location.join("bin")];
            first.extend(libraries.iter().cloned());
            extra.push(("PATH".to_string(), join(&first, "PATH")?));
        }
        extra.extend(tools.environment.iter().cloned());
    } else if tools.location.extension().is_some_and(|e| e == "py") {
        let python = python
            .ok_or_else(|| anyhow!("{location} is a Python wrapper; give --python / CRISP3DS_PYTHON, or an install prefix instead"))?;
        command = vec![python.to_string(), location, tool.to_string()];
    } else {
        command = vec![location, tool.to_string()];
    }
    command.extend(arguments.iter().cloned());
    Ok((command, extra))
}

/// `cameraSensors.db`: the explicit file, else below the prefix (for a wrapper: below `prefix/` next to it).
pub fn sensor_database(location: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(explicit) = explicit {
        return explicit.to_path_buf();
    }
    let prefix = if location.is_dir() { location.to_path_buf() } else { location.parent().unwrap_or(Path::new(".")).join("prefix") };
    prefix.join("share").join("aliceVision").join("cameraSensors.db")
}

/// One external step: its name, command line and the variables added to the inherited environment.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalCommand {
    pub name: &'static str,
    pub command: Vec<String>,
    pub environment: Vec<(String, String)>,
}

impl Options {
    /// The record written to `frontend-config.json` and into the report (the reference's keys, plus the native ones).
    pub fn to_json(&self) -> Value {
        let tools = |tools: &Option<Tools>| tools.as_ref().map(|t| t.location.to_string_lossy().to_string());
        let extras = |tools: &Option<Tools>| {
            tools.as_ref().map(|t| {
                json!({
                    "library_path": t.library_path,
                    "environment": t.environment.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                })
            })
        };
        let mut record = json!({
            "schema": SCHEMA, "engine": concat!("crisp3ds-dense ", env!("CARGO_PKG_VERSION")),
            "photos": self.photos, "photo_count": self.photo_count, "calibration": self.calibration, "output": self.output,
            "python": self.python, "alicevision": tools(&self.alicevision), "alicevision_dense": tools(&self.alicevision_dense),
            "alicevision_direct": extras(&self.alicevision), "alicevision_dense_direct": extras(&self.alicevision_dense),
            "sensor_database": self.sensor_database, "alicevision_memory_gib": self.alicevision_memory_gib,
            "mask_mode": self.mask_mode.name(), "masks": self.masks,
            "device": self.device, "threads": self.threads, "minimum_free_gib": self.minimum_free_gib,
            "envelope": self.envelope, "dark_threshold": self.dark_threshold.to_json(),
            "hole_cleanup_budget": self.hole_cleanup_budget,
            "gates": self.gates.to_json(), "keep_intermediates": self.keep_intermediates, "stop_after_masks": self.stop_after_masks,
        });
        // In two parts: one literal of this size exceeds the macro recursion limit.
        let more = json!({
            "sam_python": self.sam_python, "sam_source": self.sam_source, "sam_checkpoint": self.sam_checkpoint,
            "sam_config": self.sam_config, "sam_pythonpath": self.sam_pythonpath, "repository": self.repository,
            "sam_multimask": self.sam_multimask, "sam_preserve_holes": self.sam_preserve_holes,
            "sam_automatic_cues": self.sam_automatic_cues,
            "contrast_gamma": self.contrast_gamma, "clahe_clip": self.clahe_clip, "clahe_grid": self.clahe_grid,
            "initial_field_of_view": self.initial_field_of_view, "describer_types": self.describer_types,
            "describer_preset": self.describer_preset, "matching_method": self.matching_method,
            "random_seed": self.random_seed, "sfm_option": self.sfm_option,
            "timeouts": {"small": self.timeouts.small, "sam": self.timeouts.sam, "features": self.timeouts.features,
                         "matching": self.timeouts.matching, "sfm": self.timeouts.sfm, "prepare": self.timeouts.prepare},
        });
        if let (Some(record), Value::Object(more)) = (record.as_object_mut(), more) {
            record.extend(more);
        }
        record
    }

    /// Fails with a clear message when AliceVision is not where the options say.
    pub fn check_tools(&self) -> anyhow::Result<()> {
        for (tools, names, flag) in
            [(&self.alicevision, &SPARSE_TOOLS[..], "--alicevision"), (&self.alicevision_dense, &DENSE_TOOLS[..], "--alicevision-dense")]
        {
            let Some(tools) = tools else { bail!("missing tool locations: {flag}") };
            let location = &tools.location;
            if location.is_dir() {
                let absent: Vec<String> =
                    names.iter().filter(|tool| !executable(location, tool).is_file()).map(|tool| format!("aliceVision_{tool}")).collect();
                if !absent.is_empty() {
                    bail!(
                        "{flag} {}: no bin/{} there. AliceVision is an external program this stage calls; \
                         give its install prefix or a wrapper script (docs/PHOTOS-TO-INPUTS.md)",
                        location.display(),
                        absent.join(", bin/")
                    );
                }
            } else if !location.is_file() {
                bail!(
                    "{flag} {}: not found. AliceVision is an external program this stage calls; \
                     give its install prefix or a wrapper script (docs/PHOTOS-TO-INPUTS.md)",
                    location.display()
                );
            } else if location.extension().is_some_and(|e| e == "py") && self.python.is_none() {
                bail!("{flag} {} is a Python wrapper; give --python / CRISP3DS_PYTHON, or an install prefix instead", location.display());
            }
        }
        Ok(())
    }

    /// Every external command of a run, in the order they run.
    pub fn build_commands(&self, environment: &dyn Fn(&str) -> Option<String>) -> anyhow::Result<Vec<ExternalCommand>> {
        let out = &self.output;
        let (work, sfm) = (out.join("work"), out.join("sfm"));
        let path = |p: PathBuf| p.to_string_lossy().to_string();
        let text = |s: &str| s.to_string();
        let threads = self.threads.to_string();
        let resource = [
            text("--maxCoresAvailable"),
            threads.clone(),
            text("--maxMemoryAvailable"),
            ((self.alicevision_memory_gib * (1u64 << 30) as f64) as i64).to_string(),
        ];
        let mut commands = Vec::new();
        if self.mask_mode == MaskMode::Sam {
            let need = |value: Option<String>, name: &str| value.ok_or_else(|| anyhow!("--mask-mode sam needs {name}"));
            let repository = self.repository.clone().ok_or_else(|| anyhow!("--mask-mode sam needs --repository"))?;
            let mut sam = vec![
                need(self.sam_python.clone(), "--sam-python")?,
                text("-m"),
                text("scripts.turntable_mesh.segment"),
                text("--images"),
                path(work.join("photos")),
                text("--coarse-masks"),
                path(work.join("coarse-masks")),
                text("--output"),
                path(work.join("sam")),
                text("--source"),
                need(self.sam_source.clone().map(path), "--sam-source")?,
                text("--checkpoint"),
                need(self.sam_checkpoint.clone().map(path), "--sam-checkpoint")?,
                text("--device"),
                self.device.clone(),
                text("--views"),
                self.photo_count.to_string(),
            ];
            if let Some(config) = &self.sam_config {
                sam.extend([text("--model-config"), config.clone()]);
            }
            for (flag, on) in [
                ("--multimask", self.sam_multimask),
                ("--preserve-holes", self.sam_preserve_holes),
                ("--automatic-cues", self.sam_automatic_cues),
            ] {
                if on {
                    sam.push(text(flag));
                }
            }
            let pythonpath = std::env::join_paths(std::iter::once(path(repository)).chain(self.sam_pythonpath.iter().cloned()))
                .map_err(|e| anyhow!("--sam-pythonpath: {e}"))?;
            let environment = vec![
                (text("PYTHONPATH"), pythonpath.to_string_lossy().to_string()),
                (text("PYTORCH_ENABLE_MPS_FALLBACK"), text("0")),
                (text("OPENBLAS_NUM_THREADS"), threads.clone()),
                (text("VECLIB_MAXIMUM_THREADS"), threads.clone()),
            ];
            commands.push(ExternalCommand { name: "sam", command: sam, environment });
        }
        if self.stop_after_masks {
            return Ok(commands);
        }
        let sparse = self.alicevision.as_ref().ok_or_else(|| anyhow!("missing tool locations: --alicevision / CRISP3DS_ALICEVISION"))?;
        let dense = self.alicevision_dense.as_ref().unwrap_or(sparse);
        let mut av = |name: &'static str, tool: &str, arguments: Vec<String>, tools: &Tools| -> anyhow::Result<()> {
            let arguments: Vec<String> = arguments.into_iter().chain(resource.iter().cloned()).collect();
            let (command, mut extra) = alicevision_command(tools, tool, &arguments, self.python.as_deref(), environment)?;
            // The tools thread through OpenMP; keep the BLAS underneath single-threaded.
            extra.extend([(text("OPENBLAS_NUM_THREADS"), text("1")), (text("VECLIB_MAXIMUM_THREADS"), text("1"))]);
            commands.push(ExternalCommand { name, command, environment: extra });
            Ok(())
        };
        let (features, matches) = (path(work.join("features")), path(work.join("matches")));
        let database = path(sensor_database(&sparse.location, self.sensor_database.as_deref()));
        let uncalibrated = path(sfm.join("cameraInit-uncalibrated.sfm"));
        let initial = path(sfm.join("cameraInit.sfm"));
        let (pairs, last) = (path(sfm.join("pairs.txt")), path(sfm.join("final.sfm")));
        let words = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        av(
            "cameraInit-uncalibrated",
            "cameraInit",
            words(&[
                "--imageFolder",
                &path(work.join("contrast")),
                "--output",
                &uncalibrated,
                "--sensorDatabase",
                &database,
                "--defaultFieldOfView",
                &super::util::python_float(self.initial_field_of_view),
                "--groupCameraFallback",
                "global",
            ]),
            sparse,
        )?;
        av(
            "featureExtraction",
            "featureExtraction",
            words(&[
                "--input",
                &uncalibrated,
                "--output",
                &features,
                "--describerTypes",
                &self.describer_types,
                "--describerPreset",
                &self.describer_preset,
                "--forceCpuExtraction",
                "true",
                "--masksFolder",
                &path(out.join("masks")),
                "--maskExtension",
                "png",
                "--maskInvert",
                "false",
                "--maxThreads",
                &threads,
            ]),
            sparse,
        )?;
        av(
            "cameraInit",
            "cameraInit",
            words(&[
                "--input",
                &path(sfm.join("calibrated-input.sfm")),
                "--output",
                &initial,
                "--sensorDatabase",
                &database,
                "--groupCameraFallback",
                "global",
            ]),
            sparse,
        )?;
        av(
            "imageMatching",
            "imageMatching",
            words(&["--input", &initial, "--featuresFolders", &features, "--output", &pairs, "--method", &self.matching_method]),
            sparse,
        )?;
        av(
            "featureMatching",
            "featureMatching",
            words(&[
                "--input",
                &initial,
                "--featuresFolders",
                &features,
                "--imagePairsList",
                &pairs,
                "--output",
                &matches,
                "--describerTypes",
                &self.describer_types,
                "--randomSeed",
                &self.random_seed.to_string(),
            ]),
            sparse,
        )?;
        let mut global = words(&[
            "--input",
            &initial,
            "--featuresFolders",
            &features,
            "--matchesFolders",
            &matches,
            "--output",
            &last,
            "--outputViewsAndPoses",
            &path(sfm.join("views.sfm")),
            "--extraInfoFolder",
            &path(sfm.join("extra")),
            "--lockAllIntrinsics",
            "true",
            "--randomSeed",
            &self.random_seed.to_string(),
        ]);
        global.extend(self.sfm_option.iter().cloned());
        av("globalSfM", "globalSfM", global, sparse)?;
        av(
            "prepareDenseScene",
            "prepareDenseScene",
            words(&["--input", &last, "--output", &path(out.join("native-prepared")), "--outputFileType", "png", "--evCorrection", "0"]),
            dense,
        )?;
        Ok(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("crisp3ds-options-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(folder.join("photos")).unwrap();
        for n in [1, 2, 10] {
            std::fs::write(folder.join(format!("photos/thing_{n}_rgb.png")), b"").unwrap();
        }
        std::fs::write(folder.join("av.py"), b"").unwrap();
        folder
    }

    fn example() -> String {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/turntable_mesh/calibrations/3dlf-pro.json").to_string()
    }

    fn arguments(folder: &Path, more: &[&str]) -> Vec<String> {
        let mut words: Vec<String> = [
            "--photos",
            &folder.join("photos").to_string_lossy(),
            "--calibration",
            &example(),
            "--output",
            &folder.join("out").to_string_lossy(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        words.extend(more.iter().map(|s| s.to_string()));
        words
    }

    fn none(_: &str) -> Option<String> {
        None
    }

    fn after<'a>(command: &'a [String], flag: &str) -> &'a str {
        &command[command.iter().position(|w| w == flag).unwrap_or_else(|| panic!("{flag} missing in {command:?}")) + 1]
    }

    fn named<'a>(commands: &'a [ExternalCommand], name: &str) -> &'a ExternalCommand {
        commands.iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no command {name}"))
    }

    #[test]
    fn wrapper_and_prefix_forms() {
        let folder = scratch("forms");
        let wrapper = Tools { location: folder.join("av.py"), ..Tools::default() };
        let arguments = ["--input".to_string(), "3".to_string()];
        let (command, extra) = alicevision_command(&wrapper, "globalSfM", &arguments, Some("/sci/python"), &none).unwrap();
        assert_eq!(command, ["/sci/python", &folder.join("av.py").to_string_lossy(), "globalSfM", "--input", "3"]);
        assert!(extra.is_empty());
        assert!(alicevision_command(&wrapper, "globalSfM", &arguments, None, &none).unwrap_err().to_string().contains("--python"));
        let plain = Tools { location: folder.join("av"), ..Tools::default() };
        assert_eq!(
            alicevision_command(&plain, "cameraInit", &[], None, &none).unwrap().0,
            [&*folder.join("av").to_string_lossy(), "cameraInit"]
        );
        let prefix = Tools {
            location: folder.clone(),
            library_path: vec![PathBuf::from("/opt/extra/lib")],
            environment: vec![("ACPP_VISIBILITY_MASK".to_string(), "omp".to_string())],
        };
        let inherited = |name: &str| (name == "DYLD_LIBRARY_PATH").then(|| "/already/there".to_string());
        let (command, extra) = alicevision_command(&prefix, "cameraInit", &["-x".to_string()], None, &inherited).unwrap();
        assert_eq!(command, [&*executable(&folder, "cameraInit").to_string_lossy(), "-x"]);
        let value = |name: &str| extra.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()).unwrap();
        assert_eq!(value("ALICEVISION_ROOT"), folder.to_string_lossy());
        let expected: Vec<PathBuf> = vec![folder.join("lib"), "/opt/extra/lib".into(), "/already/there".into()];
        assert_eq!(std::env::split_paths(&value("DYLD_LIBRARY_PATH")).collect::<Vec<_>>(), expected);
        assert_eq!(std::env::split_paths(&value("LD_LIBRARY_PATH")).collect::<Vec<_>>(), expected[..2]);
        assert_eq!(value("ACPP_VISIBILITY_MASK"), "omp");
        assert_eq!(sensor_database(&folder, None), folder.join("share/aliceVision/cameraSensors.db"));
        assert_eq!(sensor_database(&folder.join("av.py"), None), folder.join("prefix/share/aliceVision/cameraSensors.db"));
        assert_eq!(sensor_database(&folder, Some(Path::new("/else/db"))), Path::new("/else/db"));
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn defaults_reproduce_the_hand_driven_commands() {
        let folder = scratch("defaults");
        let wrapper = folder.join("av.py").to_string_lossy().to_string();
        let options = resolve(&arguments(&folder, &["--alicevision", &wrapper, "--python", "/sci/python"]), &none).unwrap();
        let out = folder.join("out");
        let shown = |p: PathBuf| p.to_string_lossy().to_string();
        assert_eq!((options.photo_count, options.dark_threshold, options.envelope.as_str()), (3, Threshold::Level(70), "auto"));
        assert_eq!(options.mask_mode, MaskMode::Dark);
        let commands = options.build_commands(&none).unwrap();
        assert_eq!(
            commands.iter().map(|c| c.name).collect::<Vec<_>>(),
            [
                "cameraInit-uncalibrated",
                "featureExtraction",
                "cameraInit",
                "imageMatching",
                "featureMatching",
                "globalSfM",
                "prepareDenseScene"
            ]
        );
        let listing = &named(&commands, "cameraInit-uncalibrated").command;
        assert_eq!(listing[..3], ["/sci/python".to_string(), wrapper.clone(), "cameraInit".to_string()]);
        assert_eq!(after(listing, "--imageFolder"), shown(out.join("work/contrast")));
        assert_eq!(after(listing, "--defaultFieldOfView"), "45.0");
        assert_eq!(after(listing, "--sensorDatabase"), shown(folder.join("prefix/share/aliceVision/cameraSensors.db")));
        let features = &named(&commands, "featureExtraction").command;
        assert_eq!(features[..3], ["/sci/python".to_string(), wrapper.clone(), "featureExtraction".to_string()]);
        for (flag, value) in [
            ("--describerTypes", "sift".to_string()),
            ("--describerPreset", "normal".to_string()),
            ("--forceCpuExtraction", "true".to_string()),
            ("--masksFolder", shown(out.join("masks"))),
            ("--maskExtension", "png".to_string()),
            ("--maxThreads", "2".to_string()),
            ("--maxCoresAvailable", "2".to_string()),
            ("--maxMemoryAvailable", (4u64 << 30).to_string()),
        ] {
            assert_eq!(after(features, flag), value, "{flag}");
        }
        assert_eq!(after(&named(&commands, "cameraInit").command, "--input"), shown(out.join("sfm/calibrated-input.sfm")));
        assert_eq!(after(&named(&commands, "imageMatching").command, "--method"), "Exhaustive");
        assert_eq!(after(&named(&commands, "featureMatching").command, "--imagePairsList"), shown(out.join("sfm/pairs.txt")));
        let sfm = &named(&commands, "globalSfM").command;
        assert_eq!(after(sfm, "--lockAllIntrinsics"), "true");
        assert_eq!(after(sfm, "--randomSeed"), "0");
        assert_eq!(after(sfm, "--output"), shown(out.join("sfm/final.sfm")));
        assert_eq!(after(sfm, "--extraInfoFolder"), shown(out.join("sfm/extra")));
        let blas = named(&commands, "globalSfM").environment.iter().find(|(k, _)| k == "OPENBLAS_NUM_THREADS").unwrap();
        assert_eq!(blas.1, "1");
        let prepare = &named(&commands, "prepareDenseScene").command;
        assert_eq!(after(prepare, "--outputFileType"), "png");
        assert_eq!(after(prepare, "--evCorrection"), "0");
        assert_eq!(prepare[1], wrapper); // falls back to --alicevision
        let record = options.to_json();
        assert_eq!(
            (record["schema"].as_str(), record["mask_mode"].as_str(), record["dark_threshold"].as_u64()),
            (Some(SCHEMA), Some("dark"), Some(70))
        );
        assert_eq!(record["gates"]["minimum_positive_depth_fraction"], 0.999);
        assert_eq!(record["timeouts"]["sfm"], 900);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn options_reach_the_commands() {
        let folder = scratch("reach");
        std::fs::create_dir_all(folder.join("dense")).unwrap();
        let (wrapper, dense) =
            (folder.join("av.py").to_string_lossy().to_string(), folder.join("dense-wrapper").to_string_lossy().to_string());
        std::fs::write(&dense, b"").unwrap();
        let variables = |name: &str| match name {
            "CRISP3DS_ALICEVISION" => Some(wrapper.clone()),
            "CRISP3DS_PYTHON" => Some("/sci/python".to_string()),
            "CRISP3DS_SAM_PYTHON" => Some("/sam/python".to_string()),
            "CRISP3DS_SAM_SOURCE" => Some("/sam/source".to_string()),
            "CRISP3DS_REPOSITORY" => Some("/repo".to_string()),
            _ => None,
        };
        let more = [
            "--mask-mode",
            "sam",
            "--sam-checkpoint",
            "/sam/tiny.pt",
            "--device",
            "cpu",
            "--threads=3",
            "--no-sam-multimask",
            "--sam-config",
            "configs/x.yaml",
            "--sam-pythonpath",
            "/a:/b",
            "--hole-cleanup-budget",
            "0.05",
            "--matching-method",
            "Sequential",
            "--random-seed",
            "7",
            "--alicevision-dense",
            &dense,
            "--dark-threshold",
            "otsu",
            "--envelope",
            "0.1,0.1,0.9,0.9",
            "--sfm-option",
            "--a",
            "--sfm-option=b",
            "--keep-intermediates",
        ];
        let options = resolve(&arguments(&folder, &more), &variables).unwrap();
        let commands = options.build_commands(&none).unwrap();
        let sam = &named(&commands, "sam").command;
        assert_eq!(sam[..3], ["/sam/python", "-m", "scripts.turntable_mesh.segment"]);
        assert_eq!((after(sam, "--device"), after(sam, "--views"), after(sam, "--model-config")), ("cpu", "3", "configs/x.yaml"));
        assert_eq!(sam[sam.len() - 2..], ["--preserve-holes", "--automatic-cues"]);
        if cfg!(unix) {
            let pythonpath = &named(&commands, "sam").environment.iter().find(|(k, _)| k == "PYTHONPATH").unwrap().1;
            assert_eq!(pythonpath, "/repo:/a:/b");
        }
        assert_eq!(after(&named(&commands, "imageMatching").command, "--method"), "Sequential");
        assert_eq!(after(&named(&commands, "featureMatching").command, "--randomSeed"), "7");
        let sfm = &named(&commands, "globalSfM").command;
        assert_eq!(after(sfm, "--maxCoresAvailable"), "3");
        assert_eq!(sfm[sfm.len() - 6..sfm.len() - 4], ["--a", "b"]);
        assert_eq!(named(&commands, "prepareDenseScene").command[..2], [dense.clone(), "prepareDenseScene".to_string()]); // a non-.py wrapper runs directly
        assert_eq!(
            (options.dark_threshold, options.envelope.as_str(), options.hole_cleanup_budget),
            (Threshold::Otsu, "0.1,0.1,0.9,0.9", 0.05)
        );
        assert!(options.keep_intermediates && !options.sam_multimask && options.sam_preserve_holes);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn invalid_requests_are_refused_before_anything_runs() {
        let folder = scratch("invalid");
        let wrapper = folder.join("av.py").to_string_lossy().to_string();
        let base = ["--alicevision", wrapper.as_str(), "--python", "/sci/python"];
        let with = |more: &[&str]| resolve(&arguments(&folder, &[&base[..], more].concat()), &none);
        assert!(with(&[]).is_ok());
        for (more, expected) in [
            (&["--dark-threshold", "300"][..], "--dark-threshold"),
            (&["--envelope", "1,2,3"][..], "envelope"),
            (&["--hole-cleanup-budget", "0.2"][..], "finite"),
            (&["--mask-mode", "magic"][..], "expected dark or sam"),
            (&["--mask-mode", "sam"][..], "missing tool locations: --sam-python"),
            (&["--masks", "/no/such/masks"][..], "--masks must be a directory"),
            (&["--threads", "two"][..], "needs an integer"),
            (&["--bogus", "1"][..], "unknown option --bogus"),
            (&["--stop-after", "cameras"][..], "only 'masks'"),
            (&["--sfm-timeout", "0"][..], "at least one second"),
        ] {
            let error = with(more).unwrap_err().to_string();
            assert!(error.contains(expected), "{more:?}: {error}");
        }
        std::fs::write(folder.join("bad.json"), "{\"fx\": 1}").unwrap();
        assert!(with(&["--calibration", &folder.join("bad.json").to_string_lossy()])
            .unwrap_err()
            .to_string()
            .contains("unknown calibration format"));
        std::fs::create_dir_all(folder.join("two")).unwrap();
        for name in ["a.png", "b.png", "notes.txt"] {
            std::fs::write(folder.join("two").join(name), b"").unwrap();
        }
        assert!(with(&["--photos", &folder.join("two").to_string_lossy()]).unwrap_err().to_string().contains("found 2 photos"));
        // AliceVision is external: a missing prefix, an incomplete prefix and a wrapper without interpreter are named.
        let missing = resolve(&arguments(&folder, &[]), &none).unwrap_err().to_string();
        assert_eq!(missing, "missing tool locations: --alicevision / CRISP3DS_ALICEVISION");
        let absent = resolve(&arguments(&folder, &["--alicevision", "/no/such/prefix"]), &none).unwrap_err().to_string();
        assert!(absent.contains("not found. AliceVision is an external program"), "{absent}");
        std::fs::create_dir_all(folder.join("prefix/bin")).unwrap();
        let prefix = folder.join("prefix").to_string_lossy().to_string();
        let incomplete = resolve(&arguments(&folder, &["--alicevision", &prefix]), &none).unwrap_err().to_string();
        assert!(incomplete.contains("no bin/aliceVision_cameraInit, bin/aliceVision_featureExtraction"), "{incomplete}");
        assert!(resolve(&arguments(&folder, &["--alicevision", &wrapper]), &none).unwrap_err().to_string().contains("Python wrapper"));
        // The masks alone need no AliceVision.
        let masks_only = resolve(&arguments(&folder, &["--stop-after", "masks"]), &none).unwrap();
        assert!(masks_only.build_commands(&none).unwrap().is_empty());
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
