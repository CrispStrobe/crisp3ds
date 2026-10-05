//! What the shell needs to know to start the engine, where each value comes from, and
//! how it is stored. Nothing here starts a process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What the user saved. An empty string means "not set: use what is found".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Checkout of the crisp3ds repository (the folder that contains `scripts/turntable_mesh`).
    pub repo: String,
    /// Interpreter for the engine and the NumPy/SciPy/OpenCV stages.
    pub python: String,
    /// Interpreter with PyTorch, for the stereo stage.
    pub torch_python: String,
    pub runs_dir: String,
    pub data_dir: String,
    /// `auto`, `mps`, `cuda` or `cpu`.
    pub device: String,
}

/// Where a resolved value came from, shown next to it on the settings screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Saved on the settings screen.
    Setting,
    /// An environment variable.
    Environment,
    /// Found by looking around (a checkout above the app, a `.venv` in it).
    Found,
    /// Nothing better known.
    Default,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Value {
    pub value: String,
    pub source: Source,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolved {
    pub repo: Value,
    pub python: Value,
    pub torch_python: Value,
    pub runs_dir: Value,
    pub data_dir: Value,
    pub device: Value,
}

pub const ENGINE_MODULE: &str = "scripts/turntable_mesh/engine_server.py";

/// Everything outside the saved file that resolution looks at; injected so it can be tested.
pub struct Surroundings {
    pub env: HashMap<String, String>,
    /// Folders from which to look upwards for a checkout (the executable's, the working directory).
    pub search_from: Vec<PathBuf>,
    /// The app's own data folder, for the default runs and data folders.
    pub app_data: PathBuf,
}

impl Surroundings {
    pub fn real(app_data: PathBuf) -> Self {
        let mut search_from = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            search_from.push(exe);
        }
        if let Ok(cwd) = std::env::current_dir() {
            search_from.push(cwd);
        }
        Surroundings { env: std::env::vars().collect(), search_from, app_data }
    }
}

pub fn is_checkout(folder: &Path) -> bool {
    folder.join(ENGINE_MODULE).is_file()
}

fn find_checkout(from: &[PathBuf]) -> Option<PathBuf> {
    from.iter().flat_map(|start| start.ancestors()).find(|folder| is_checkout(folder)).map(Path::to_path_buf)
}

/// The device the engine is told to use when the setting is `auto`.
pub fn auto_device() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "mps"
    } else {
        "cpu"
    }
}

fn default_python() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

fn venv_python(repo: &Path) -> Option<PathBuf> {
    let candidate = if cfg!(windows) { repo.join(".venv/Scripts/python.exe") } else { repo.join(".venv/bin/python") };
    candidate.is_file().then_some(candidate)
}

fn set(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

pub fn resolve(config: &Config, around: &Surroundings) -> Resolved {
    let env = |name: &str| around.env.get(name).and_then(|value| set(value));
    let pick = |saved: &str, variable: Option<&str>, found: Option<String>, fallback: String| -> Value {
        if let Some(value) = set(saved) {
            return Value { value, source: Source::Setting };
        }
        if let Some(value) = variable.and_then(env) {
            return Value { value, source: Source::Environment };
        }
        match found {
            Some(value) => Value { value, source: Source::Found },
            None => Value { value: fallback, source: Source::Default },
        }
    };
    let text = |path: PathBuf| path.to_string_lossy().into_owned();

    let repo = pick(&config.repo, Some("CRISP3DS_REPO"), find_checkout(&around.search_from).map(text), String::new());
    let python = pick(
        &config.python,
        Some("CRISP3DS_PYTHON"),
        venv_python(Path::new(&repo.value)).map(text),
        default_python().to_string(),
    );
    // Without a word about PyTorch, one interpreter is assumed to have everything.
    let torch_python = pick(&config.torch_python, Some("CRISP3DS_TORCH_PYTHON"), None, python.value.clone());
    let runs_dir = pick(&config.runs_dir, Some("CRISP3DS_RUNS_DIR"), None, text(around.app_data.join("runs")));
    let data_dir = pick(&config.data_dir, Some("CRISP3DS_DATA_DIR"), None, text(around.app_data.join("data")));
    let device = match set(&config.device).filter(|device| device != "auto") {
        Some(value) => Value { value, source: Source::Setting },
        None => Value { value: auto_device().to_string(), source: Source::Default },
    };
    Resolved { repo, python, torch_python, runs_dir, data_dir, device }
}

/// Checks what can be checked before starting anything, with a sentence the user can act on.
pub fn problems(resolved: &Resolved) -> Option<String> {
    if resolved.repo.value.is_empty() {
        return Some(
            "The Crisp3DS source folder is not known. Choose the folder of your crisp3ds checkout (the one that contains \
             scripts/turntable_mesh)."
                .into(),
        );
    }
    if !is_checkout(Path::new(&resolved.repo.value)) {
        return Some(format!(
            "{} is not a crisp3ds checkout: {} is missing in it.",
            resolved.repo.value, ENGINE_MODULE
        ));
    }
    if !matches!(resolved.device.value.as_str(), "mps" | "cuda" | "cpu") {
        return Some(format!("\"{}\" is not a device. Use auto, mps, cuda or cpu.", resolved.device.value));
    }
    None
}

pub fn load(path: &Path) -> Config {
    std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

pub fn save(path: &Path, config: &Config) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Could not create {}: {error}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(config).map_err(|error| error.to_string())?;
    std::fs::write(path, text + "\n").map_err(|error| format!("Could not write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("crisp3ds-studio-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    fn checkout(name: &str) -> PathBuf {
        let repo = scratch(name);
        std::fs::create_dir_all(repo.join("scripts/turntable_mesh")).unwrap();
        std::fs::write(repo.join(ENGINE_MODULE), "").unwrap();
        repo
    }

    fn around(env: &[(&str, &str)], search_from: Vec<PathBuf>) -> Surroundings {
        Surroundings {
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            search_from,
            app_data: PathBuf::from("/appdata"),
        }
    }

    #[test]
    fn saved_settings_win_over_environment_and_discovery() {
        let repo = checkout("order");
        let config = Config { repo: "/saved/repo".into(), python: "/saved/python".into(), ..Config::default() };
        let env = [("CRISP3DS_REPO", "/env/repo"), ("CRISP3DS_PYTHON", "/env/python"), ("CRISP3DS_TORCH_PYTHON", "/env/torch")];
        let resolved = resolve(&config, &around(&env, vec![repo.join("apps/studio")]));
        assert_eq!(resolved.repo, Value { value: "/saved/repo".into(), source: Source::Setting });
        assert_eq!(resolved.python, Value { value: "/saved/python".into(), source: Source::Setting });
        assert_eq!(resolved.torch_python, Value { value: "/env/torch".into(), source: Source::Environment });
    }

    #[test]
    fn environment_wins_over_discovery() {
        let repo = checkout("env");
        let resolved = resolve(&Config::default(), &around(&[("CRISP3DS_REPO", "/env/repo")], vec![repo.clone()]));
        assert_eq!(resolved.repo, Value { value: "/env/repo".into(), source: Source::Environment });
    }

    #[test]
    fn a_checkout_above_the_app_is_found() {
        let repo = checkout("found");
        let deep = repo.join("apps/studio/src-tauri/target/debug/crisp3ds-studio");
        let resolved = resolve(&Config::default(), &around(&[], vec![deep]));
        assert_eq!(resolved.repo, Value { value: repo.to_string_lossy().into_owned(), source: Source::Found });
        assert_eq!(problems(&resolved), None);
    }

    #[test]
    fn defaults_when_nothing_is_known() {
        let resolved = resolve(&Config::default(), &around(&[("CRISP3DS_PYTHON", "  ")], vec![PathBuf::from("/nowhere/app")]));
        assert_eq!(resolved.repo.value, "");
        assert_eq!(resolved.python, Value { value: default_python().into(), source: Source::Default });
        assert_eq!(resolved.torch_python.value, resolved.python.value);
        assert_eq!(resolved.runs_dir.value, PathBuf::from("/appdata").join("runs").to_string_lossy());
        assert_eq!(resolved.data_dir.value, PathBuf::from("/appdata").join("data").to_string_lossy());
        assert_eq!(resolved.device.value, auto_device());
        assert!(problems(&resolved).unwrap().contains("not known"));
    }

    #[test]
    fn the_torch_interpreter_follows_the_other_one_unless_named() {
        let config = Config { python: "/venv/bin/python".into(), ..Config::default() };
        assert_eq!(resolve(&config, &around(&[], vec![])).torch_python.value, "/venv/bin/python");
    }

    #[test]
    fn device_auto_and_explicit() {
        let auto = Config { device: "auto".into(), ..Config::default() };
        assert_eq!(resolve(&auto, &around(&[], vec![])).device.value, auto_device());
        let cpu = Config { device: "cpu".into(), ..Config::default() };
        assert_eq!(resolve(&cpu, &around(&[], vec![])).device, Value { value: "cpu".into(), source: Source::Setting });
    }

    #[test]
    fn a_folder_that_is_not_a_checkout_and_a_bad_device_are_named() {
        let empty = scratch("not-a-checkout");
        let config = Config { repo: empty.to_string_lossy().into_owned(), ..Config::default() };
        assert!(problems(&resolve(&config, &around(&[], vec![]))).unwrap().contains("is not a crisp3ds checkout"));
        let repo = checkout("bad-device");
        let config = Config { repo: repo.to_string_lossy().into_owned(), device: "gpu".into(), ..Config::default() };
        assert!(problems(&resolve(&config, &around(&[], vec![]))).unwrap().contains("not a device"));
    }

    #[test]
    fn settings_survive_a_round_trip_and_damaged_files_give_defaults() {
        let file = scratch("roundtrip").join("nested/config.json");
        let config = Config { repo: "/r".into(), device: "cpu".into(), ..Config::default() };
        save(&file, &config).unwrap();
        assert_eq!(load(&file), config);
        std::fs::write(&file, "{ not json").unwrap();
        assert_eq!(load(&file), Config::default());
        std::fs::write(&file, r#"{"repo": "/x", "later_field": 1}"#).unwrap();
        assert_eq!(load(&file).repo, "/x");
    }
}
