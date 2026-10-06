//! The external programs behind some providers of the photos start (AliceVision, COLMAP,
//! SAM 2.1 in a Python interpreter): whether they are there, what they say about
//! themselves, and how they are handed to the crate.
//!
//! The web view never names an executable. Where the programs are comes from the saved
//! settings (or the environment variables the crate reads); a start request may only
//! choose providers and their tuning options.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use crate::config::Tools;

pub const ALICEVISION_PROGRAMS: [&str; 5] = [
    "aliceVision_cameraInit",
    "aliceVision_featureExtraction",
    "aliceVision_imageMatching",
    "aliceVision_featureMatching",
    "aliceVision_globalSfM",
];

/// Options of the photos stage that say where things are or what to run. They are set by
/// the shell from the request's own fields and the saved settings, never by option tokens
/// that came from the page.
const RESERVED: [&str; 20] = [
    "--photos",
    "--output",
    "--events",
    "--calibration",
    "--masks",
    "--cameras",
    "--stop-after",
    "--python",
    "--alicevision",
    "--alicevision-library-path",
    "--alicevision-env",
    "--alicevision-sensor-database",
    "--colmap",
    "--sam-python",
    "--sam-source",
    "--sam-checkpoint",
    "--sam-pythonpath",
    "--sam-repository",
    "--list-providers",
    "--markers-mat",
];

/// True inside the macOS App Sandbox, where the app cannot start programs from the disk.
pub fn sandboxed() -> bool {
    std::env::var_os("APP_SANDBOX_CONTAINER_ID").is_some()
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths).map(|folder| folder.join(exe(name))).find(|candidate| candidate.is_file())
}

fn colmap_program(tools: &Tools) -> Option<PathBuf> {
    if tools.colmap.value.is_empty() {
        on_path("colmap")
    } else {
        Some(PathBuf::from(&tools.colmap.value))
    }
}

/// Why a provider cannot be used right now, or `None` when it can. Cheap: only looks at
/// the file system. `repo` is the crisp3ds checkout, which `external-sam` needs for its script.
pub fn unavailable(module: &str, provider: &str, tools: &Tools, repo: &str, sandboxed: bool) -> Option<String> {
    let external = matches!((module, provider), ("cameras", "alicevision") | ("cameras", "colmap") | ("masks", "external-sam"));
    if !external {
        return None;
    }
    if sandboxed {
        return Some("This edition of the app runs in the App Sandbox and cannot start other programs.".into());
    }
    match (module, provider) {
        ("cameras", "alicevision") => {
            let location = Path::new(&tools.alicevision.value);
            if tools.alicevision.value.is_empty() {
                Some("AliceVision is not set up. Name its folder under Tools.".into())
            } else if location.is_dir() {
                let missing: Vec<&str> = ALICEVISION_PROGRAMS.into_iter().filter(|name| !location.join("bin").join(exe(name)).is_file()).collect();
                (!missing.is_empty()).then(|| format!("The AliceVision folder has no bin/{}.", missing.join(", bin/")))
            } else if location.is_file() {
                None
            } else {
                Some(format!("{} does not exist.", location.display()))
            }
        }
        ("cameras", "colmap") => match colmap_program(tools) {
            None => Some("COLMAP is not set up. Name its program under Tools.".into()),
            Some(program) if !program.is_file() => Some(format!("{} does not exist.", program.display())),
            Some(_) => None,
        },
        _ => {
            let mut missing = Vec::new();
            for (what, value, folder) in [
                ("the Python interpreter", &tools.sam_python.value, false),
                ("the SAM 2 source folder", &tools.sam_source.value, true),
                ("the SAM 2.1 checkpoint", &tools.sam_checkpoint.value, false),
            ] {
                let path = Path::new(value);
                // An interpreter may be a bare command name found on PATH.
                let there = if folder { path.is_dir() } else { path.is_file() || (!value.is_empty() && !value.contains(std::path::MAIN_SEPARATOR) && on_path(value).is_some()) };
                if value.is_empty() || !there {
                    missing.push(what);
                }
            }
            if !Path::new(repo).join("scripts/turntable_mesh/segment.py").is_file() {
                missing.push("the crisp3ds source folder (for scripts/turntable_mesh/segment.py)");
            }
            (!missing.is_empty()).then(|| format!("SAM is not set up: {} missing. See Tools.", missing.join(", ")))
        }
    }
}

/// Adds `available` and `reason` to every provider option of the crate's start points.
pub fn annotate(start_points: &mut Value, tools: &Tools, repo: &str, sandboxed: bool) {
    for point in start_points.as_array_mut().into_iter().flatten() {
        for choice in point["providers"].as_array_mut().into_iter().flatten() {
            let module = choice["module"].as_str().unwrap_or_default().to_string();
            let mut first_available: Option<Value> = None;
            for option in choice["options"].as_array_mut().into_iter().flatten() {
                let reason = unavailable(&module, option["id"].as_str().unwrap_or_default(), tools, repo, sandboxed);
                option["available"] = json!(reason.is_none());
                option["reason"] = json!(reason);
                if reason.is_none() && first_available.is_none() {
                    first_available = Some(option["id"].clone());
                }
            }
            // Keep the crate's default when it can run; else the first provider that can.
            let default_ok = choice["options"].as_array().into_iter().flatten().any(|option| option["id"] == choice["default"] && option["available"] == true);
            if !default_ok {
                if let Some(id) = first_available {
                    choice["default"] = id;
                }
            }
        }
    }
}

/// The location flags for the chosen providers, from the settings.
pub fn location_flags(masks: &str, cameras: &str, tools: &Tools, repo: &str) -> Vec<String> {
    let mut flags = Vec::new();
    let mut push = |flag: &str, value: &str| {
        if !value.is_empty() {
            flags.push(flag.to_string());
            flags.push(value.to_string());
        }
    };
    if cameras == "alicevision" {
        push("--alicevision", &tools.alicevision.value);
        push("--alicevision-library-path", &tools.alicevision_library_path.value);
    }
    if cameras == "colmap" {
        if let Some(program) = colmap_program(tools) {
            push("--colmap", &program.to_string_lossy());
        }
    }
    if masks == "external-sam" {
        push("--sam-python", &tools.sam_python.value);
        push("--sam-source", &tools.sam_source.value);
        push("--sam-checkpoint", &tools.sam_checkpoint.value);
        push("--sam-repository", repo);
    }
    flags
}

/// Refuses option tokens from a start request that name a place or a program.
pub fn check_tokens(tokens: &[String]) -> Result<(), String> {
    for token in tokens {
        if token.contains('\0') {
            return Err("photo_options: invalid character".into());
        }
        let name = token.split('=').next().unwrap_or_default();
        if RESERVED.contains(&name) {
            return Err(format!("photo_options: {name} cannot be set by a run request; it comes from the form's own fields and the Tools settings"));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Checked {
    pub ok: bool,
    /// One line: the version, or what is wrong.
    pub summary: String,
    /// What was run and the first lines it printed.
    pub detail: String,
}

fn failed(summary: impl Into<String>) -> Checked {
    Checked { ok: false, summary: summary.into(), detail: String::new() }
}

/// Runs a program for at most `limit` and returns its exit success and combined output.
fn output_of(command: &mut Command, limit: Duration) -> Result<(bool, String), String> {
    let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|error| error.to_string())?;
    let (mut stdout, mut stderr) = (child.stdout.take(), child.stderr.take());
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        for stream in [stdout.as_mut().map(|s| s as &mut dyn Read), stderr.as_mut().map(|s| s as &mut dyn Read)].into_iter().flatten() {
            let mut bytes = Vec::new();
            let _ = stream.take(64 * 1024).read_to_end(&mut bytes);
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
        text
    });
    let deadline = Instant::now() + limit;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(30)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let text = reader.join().unwrap_or_default();
    match status {
        Some(status) => Ok((status.success(), text)),
        None => Err(format!("no answer within {} seconds", limit.as_secs())),
    }
}

fn first_lines(text: &str, count: usize) -> String {
    text.lines().map(str::trim).filter(|line| !line.is_empty()).take(count).collect::<Vec<_>>().join("\n")
}

/// Asks a tool about itself. `tool` is `alicevision`, `colmap` or `sam`.
///
/// This is the shell's own check (do the programs exist, does one of them start and
/// answer); the crate has no availability check of its providers yet.
pub fn check(tool: &str, tools: &Tools, repo: &str) -> Checked {
    if sandboxed() {
        return failed("This edition of the app runs in the App Sandbox and cannot start other programs.");
    }
    let limit = Duration::from_secs(20);
    match tool {
        "alicevision" => {
            if let Some(reason) = unavailable("cameras", "alicevision", tools, repo, false) {
                return failed(reason);
            }
            let location = PathBuf::from(&tools.alicevision.value);
            let mut command = if location.is_dir() {
                let mut command = Command::new(location.join("bin").join(exe("aliceVision_cameraInit")));
                let mut libraries = vec![location.join("lib")];
                libraries.extend(std::env::split_paths(&tools.alicevision_library_path.value));
                let joined = std::env::join_paths(libraries).unwrap_or_default();
                command.env("DYLD_LIBRARY_PATH", &joined).env("LD_LIBRARY_PATH", &joined);
                command
            } else {
                let mut command = Command::new(&location);
                command.arg("aliceVision_cameraInit");
                command
            };
            // AliceVision programs have no version option. Asking for help makes one start, load
            // its libraries and print its usage; the version is read from the library names.
            let shown = format!("{} --help", location.join("bin/aliceVision_cameraInit").display());
            match output_of(command.arg("--help"), limit) {
                Ok((_, text)) => {
                    let answered = text.contains("AliceVision cameraInit") || text.contains("Usage:");
                    let version = alicevision_version(&location.join("lib"));
                    let useful: Vec<&str> = text.lines().filter(|line| !line.contains("[debug]")).collect();
                    Checked {
                        ok: answered,
                        summary: match (answered, version) {
                            (true, Some(version)) => format!("AliceVision {version} starts."),
                            (true, None) => "AliceVision starts.".into(),
                            (false, _) => first_lines(&useful.join("\n"), 1),
                        },
                        detail: format!("{shown}\n{}", first_lines(&useful.join("\n"), if answered { 3 } else { 6 })),
                    }
                }
                Err(error) => Checked { ok: false, summary: format!("AliceVision could not be started: {error}"), detail: shown },
            }
        }
        "colmap" => {
            let Some(program) = colmap_program(tools) else {
                return failed("COLMAP is not set up, and no colmap program is on the PATH.");
            };
            if !program.is_file() {
                return failed(format!("{} does not exist.", program.display()));
            }
            let shown = format!("{} help", program.display());
            match output_of(Command::new(&program).arg("help"), limit) {
                Ok((ok, text)) => {
                    let version = text.lines().map(str::trim).find(|line| line.starts_with("COLMAP")).map(str::to_string);
                    let fine = ok || version.is_some();
                    Checked { ok: fine, summary: version.unwrap_or_else(|| first_lines(&text, 1)), detail: format!("{shown}\n{}", first_lines(&text, 4)) }
                }
                Err(error) => Checked { ok: false, summary: format!("COLMAP could not be started: {error}"), detail: shown },
            }
        }
        "sam" => {
            if let Some(reason) = unavailable("masks", "external-sam", tools, repo, false) {
                return failed(reason);
            }
            let script = "import sys; sys.path.insert(0, sys.argv[1]); import torch, sam2; print('PyTorch', torch.__version__, '| SAM 2 importable | mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())";
            let shown = format!("{} -c \"import torch, sam2\"", tools.sam_python.value);
            match output_of(Command::new(&tools.sam_python.value).args(["-c", script, &tools.sam_source.value]), Duration::from_secs(60)) {
                Ok((true, text)) => Checked { ok: true, summary: first_lines(&text, 1), detail: shown },
                Ok((false, text)) => Checked {
                    ok: false,
                    summary: text.lines().map(str::trim).filter(|line| !line.is_empty()).last().unwrap_or("The interpreter failed.").to_string(),
                    detail: format!("{shown}\n{}", first_lines(&text, 8)),
                },
                Err(error) => Checked { ok: false, summary: format!("The interpreter could not be started: {error}"), detail: shown },
            }
        }
        _ => failed("unknown tool"),
    }
}

#[derive(Debug, Serialize, PartialEq)]
pub struct Calibration {
    /// Shown in the list: the file name, and the camera or lens it names if it says so.
    pub label: String,
    pub path: String,
}

/// Lens calibration files (`crisp3ds_lens_calibration_v1`) in the usual places.
/// The version in the name of AliceVision's system library: `libaliceVision_system.3.4.dylib`,
/// `libaliceVision_system.so.3.4.0`. The longest one found.
fn alicevision_version(lib: &Path) -> Option<String> {
    let mut best: Option<String> = None;
    for entry in std::fs::read_dir(lib).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("libaliceVision_system.").or_else(|| name.strip_prefix("aliceVision_system.")) else { continue };
        let version: Vec<&str> = rest.split('.').filter(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit())).collect();
        if version.is_empty() {
            continue;
        }
        let version = version.join(".");
        if best.as_ref().is_none_or(|known| version.len() > known.len()) {
            best = Some(version);
        }
    }
    best
}

pub fn calibrations(folders: &[PathBuf]) -> Vec<Calibration> {
    let mut found = Vec::new();
    for folder in folders {
        let Ok(entries) = std::fs::read_dir(folder) else { continue };
        let mut files: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|e| e == "json")).collect();
        files.sort();
        for path in files {
            // Calibration files are small; anything large is something else.
            if std::fs::metadata(&path).map(|meta| meta.len() > 256 * 1024).unwrap_or(true) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(value) = serde_json::from_str::<Value>(&text) else { continue };
            if value["schema"] != "crisp3ds_lens_calibration_v1" {
                continue;
            }
            let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
            let about = ["camera", "lens", "name", "description"].iter().find_map(|key| value[*key].as_str()).unwrap_or_default();
            let label = if about.is_empty() { name } else { format!("{name} ({about})") };
            let path = std::fs::canonicalize(&path).unwrap_or(path).to_string_lossy().into_owned();
            if !found.iter().any(|known: &Calibration| known.path == path) {
                found.push(Calibration { label, path });
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Source, Value as Setting};

    #[test]
    fn the_alicevision_version_is_read_from_its_library_names() {
        let lib = std::env::temp_dir().join(format!("studio-av-{}", std::process::id()));
        std::fs::create_dir_all(&lib).unwrap();
        assert_eq!(alicevision_version(&lib), None);
        for name in ["libaliceVision_system.dylib", "libaliceVision_system.3.dylib", "libaliceVision_system.3.4.dylib", "libaliceVision_sfm.9.9.9.dylib"] {
            std::fs::write(lib.join(name), "").unwrap();
        }
        assert_eq!(alicevision_version(&lib).as_deref(), Some("3.4"));
        std::fs::write(lib.join("libaliceVision_system.so.3.4.1"), "").unwrap();
        assert_eq!(alicevision_version(&lib).as_deref(), Some("3.4.1"));
        std::fs::remove_dir_all(&lib).unwrap();
        assert_eq!(alicevision_version(&lib), None);
    }

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("crisp3ds-studio-tools-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::canonicalize(folder).unwrap()
    }

    fn tools() -> Tools {
        let empty = || Setting { value: String::new(), source: Source::Default };
        Tools {
            alicevision: empty(),
            alicevision_library_path: empty(),
            colmap: Setting { value: "/definitely/not/colmap".into(), source: Source::Setting },
            sam_python: empty(),
            sam_source: empty(),
            sam_checkpoint: empty(),
        }
    }

    fn fake_alicevision(root: &Path, programs: &[&str]) -> PathBuf {
        let prefix = root.join("av");
        std::fs::create_dir_all(prefix.join("bin")).unwrap();
        for name in programs {
            std::fs::write(prefix.join("bin").join(exe(name)), "").unwrap();
        }
        prefix
    }

    #[test]
    fn providers_inside_the_crate_are_always_available() {
        for (module, provider) in [("masks", "threshold"), ("masks", "import"), ("cameras", "import")] {
            assert_eq!(unavailable(module, provider, &tools(), "", false), None);
            assert_eq!(unavailable(module, provider, &tools(), "", true), None);
        }
    }

    #[test]
    fn external_providers_say_what_is_missing() {
        let root = scratch("missing");
        let mut tools = tools();
        assert!(unavailable("cameras", "alicevision", &tools, "", false).unwrap().contains("not set up"));
        tools.alicevision.value = fake_alicevision(&root, &ALICEVISION_PROGRAMS[..3]).to_string_lossy().into_owned();
        let reason = unavailable("cameras", "alicevision", &tools, "", false).unwrap();
        assert!(reason.contains("aliceVision_featureMatching") && reason.contains("aliceVision_globalSfM"), "{reason}");
        assert!(!reason.contains("aliceVision_cameraInit"));
        tools.alicevision.value = fake_alicevision(&root, &ALICEVISION_PROGRAMS).to_string_lossy().into_owned();
        assert_eq!(unavailable("cameras", "alicevision", &tools, "", false), None);
        tools.alicevision.value = root.join("gone").to_string_lossy().into_owned();
        assert!(unavailable("cameras", "alicevision", &tools, "", false).unwrap().contains("does not exist"));

        assert!(unavailable("cameras", "colmap", &tools, "", false).unwrap().contains("does not exist"));
        let reason = unavailable("masks", "external-sam", &tools, "", false).unwrap();
        for part in ["Python interpreter", "source folder", "checkpoint", "segment.py"] {
            assert!(reason.contains(part), "{reason}");
        }
    }

    #[test]
    fn nothing_external_is_available_in_the_sandbox() {
        let root = scratch("sandbox");
        let mut tools = tools();
        tools.alicevision.value = fake_alicevision(&root, &ALICEVISION_PROGRAMS).to_string_lossy().into_owned();
        assert!(unavailable("cameras", "alicevision", &tools, "", true).unwrap().contains("App Sandbox"));
    }

    #[test]
    fn start_points_are_annotated_and_the_default_moves_to_something_that_runs() {
        let mut points = json!([{
            "id": "photos",
            "providers": [
                {"module": "masks", "default": "external-sam", "options": [{"id": "threshold"}, {"id": "import"}, {"id": "external-sam"}]},
                {"module": "cameras", "default": "alicevision", "options": [{"id": "alicevision"}, {"id": "colmap"}, {"id": "import"}]},
            ],
        }, {"id": "inputs", "providers": []}]);
        annotate(&mut points, &tools(), "", false);
        let masks = &points[0]["providers"][0];
        assert_eq!(masks["default"], "threshold");
        assert_eq!(masks["options"][0]["available"], true);
        assert_eq!(masks["options"][2]["available"], false);
        assert!(masks["options"][2]["reason"].as_str().unwrap().contains("SAM is not set up"));
        let cameras = &points[0]["providers"][1];
        assert_eq!(cameras["default"], "import");
        assert_eq!(cameras["options"][0]["available"], false);

        let root = scratch("annotate");
        let mut with = tools();
        with.alicevision.value = fake_alicevision(&root, &ALICEVISION_PROGRAMS).to_string_lossy().into_owned();
        let mut again = json!([{"providers": [{"module": "cameras", "default": "alicevision", "options": [{"id": "alicevision"}, {"id": "import"}]}]}]);
        annotate(&mut again, &with, "", false);
        assert_eq!(again[0]["providers"][0]["default"], "alicevision");
        assert_eq!(again[0]["providers"][0]["options"][0]["reason"], Value::Null);
    }

    #[test]
    fn locations_come_from_the_settings_for_the_chosen_providers_only() {
        let mut tools = tools();
        tools.alicevision.value = "/opt/av".into();
        tools.alicevision_library_path.value = "/opt/homebrew/lib".into();
        tools.sam_python.value = "/venv/bin/python".into();
        assert_eq!(
            location_flags("threshold", "alicevision", &tools, "/src"),
            ["--alicevision", "/opt/av", "--alicevision-library-path", "/opt/homebrew/lib"]
        );
        assert_eq!(location_flags("threshold", "colmap", &tools, "/src"), ["--colmap", "/definitely/not/colmap"]);
        assert_eq!(location_flags("external-sam", "import", &tools, "/src"), ["--sam-python", "/venv/bin/python", "--sam-repository", "/src"]);
        assert!(location_flags("threshold", "import", &tools, "/src").is_empty());
    }

    #[test]
    fn a_run_request_cannot_name_programs_or_places() {
        let ok: Vec<String> = ["--threshold-level", "otsu", "--alicevision-describer-preset", "high", "--no-sam-multimask", "--colmap-matching=ring"].map(String::from).to_vec();
        assert_eq!(check_tokens(&ok), Ok(()));
        for bad in ["--alicevision", "--colmap", "--sam-python", "--python", "--output", "--photos", "--events", "--masks", "--stop-after", "--alicevision-env", "--colmap=/bin/sh", "--sam-repository=/tmp"] {
            assert!(check_tokens(&[bad.to_string(), "x".to_string()]).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn checks_report_instead_of_failing() {
        let checked = check("colmap", &tools(), "");
        assert!(!checked.ok && checked.summary.contains("does not exist"));
        assert!(!check("alicevision", &tools(), "").ok);
        assert!(!check("sam", &tools(), "").ok);
        assert!(!check("nonsense", &tools(), "").ok);
    }

    #[cfg(unix)]
    #[test]
    fn a_tool_that_answers_is_reported_with_its_version_and_one_that_hangs_is_stopped() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch("answers");
        let program = root.join("colmap");
        std::fs::write(&program, "#!/bin/sh\necho 'COLMAP 3.11.1 -- Structure-from-Motion and Multi-View Stereo'\necho '(Commit abc on 2026-01-01 without CUDA)'\n").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut tools = tools();
        tools.colmap.value = program.to_string_lossy().into_owned();
        let checked = check("colmap", &tools, "");
        assert!(checked.ok, "{checked:?}");
        assert!(checked.summary.starts_with("COLMAP 3.11.1"));

        let (ok, text) = output_of(Command::new("sh").args(["-c", "echo out; echo err 1>&2; exit 3"]), Duration::from_secs(5)).unwrap();
        assert!(!ok && text.contains("out") && text.contains("err"));
        let hanging = output_of(Command::new("sleep").arg("30"), Duration::from_millis(200));
        assert!(hanging.unwrap_err().contains("no answer"));
    }

    #[test]
    fn calibration_files_are_found_by_their_schema() {
        let root = scratch("calibrations");
        std::fs::write(root.join("3dlf-pro.json"), r#"{"schema": "crisp3ds_lens_calibration_v1", "camera": "3DLF Pro"}"#).unwrap();
        std::fs::write(root.join("plain.json"), r#"{"schema": "crisp3ds_lens_calibration_v1"}"#).unwrap();
        std::fs::write(root.join("other.json"), r#"{"schema": "something else"}"#).unwrap();
        std::fs::write(root.join("broken.json"), "{").unwrap();
        std::fs::write(root.join("notes.txt"), "x").unwrap();
        let found = calibrations(&[root.clone(), root.join("missing"), root.clone()]);
        assert_eq!(found.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(), ["3dlf-pro.json (3DLF Pro)", "plain.json"]);
        assert_eq!(found[0].path, root.join("3dlf-pro.json").to_string_lossy());
    }
}
