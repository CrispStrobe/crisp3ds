//! Crisp3DS Studio as an app.
//!
//! Desktop: the shell starts the engine (`scripts.turntable_mesh.engine_server`) as a child
//! process on a free localhost port with a random token, tells the web view where it is,
//! and takes it down on exit. Python and PyTorch are NOT bundled: the shell runs the
//! interpreters it is configured with, in a crisp3ds checkout.
//!
//! Mobile: no engine is started. The app is the same web front end, opening on its
//! connection screen (an engine elsewhere, or the demo recording).
//!
//! The web view gets no Tauri permissions beyond calling the commands below.

mod config;
#[cfg(desktop)]
mod engine;

use serde::Serialize;
use tauri::Manager;

#[derive(Serialize)]
struct ShellInfo {
    /// False on phones and tablets, where no engine can be started.
    can_run_engine: bool,
    os: &'static str,
    version: &'static str,
    /// The device `auto` stands for on this computer.
    auto_device: &'static str,
}

#[tauri::command]
fn shell_info() -> ShellInfo {
    ShellInfo {
        can_run_engine: cfg!(desktop),
        os: std::env::consts::OS,
        version: env!("CARGO_PKG_VERSION"),
        auto_device: config::auto_device(),
    }
}

#[cfg(desktop)]
mod desktop {
    use std::path::PathBuf;

    use serde::Serialize;
    use tauri::{AppHandle, Manager, State};
    use tauri_plugin_dialog::DialogExt;

    use crate::config::{self, Config, Resolved, Surroundings};
    use crate::engine::{Engine, Status};

    #[derive(Serialize)]
    pub struct Settings {
        saved: Config,
        /// What will actually be used, and where each value comes from.
        resolved: Resolved,
        /// A sentence when the engine cannot be started with these values.
        problem: Option<String>,
        file: String,
    }

    fn config_file(app: &AppHandle) -> Result<PathBuf, String> {
        app.path().app_config_dir().map(|folder| folder.join("config.json")).map_err(|error| error.to_string())
    }

    fn surroundings(app: &AppHandle) -> Result<Surroundings, String> {
        app.path().app_data_dir().map(Surroundings::real).map_err(|error| error.to_string())
    }

    fn settings(app: &AppHandle) -> Result<Settings, String> {
        let file = config_file(app)?;
        let saved = config::load(&file);
        let resolved = config::resolve(&saved, &surroundings(app)?);
        let problem = config::problems(&resolved);
        Ok(Settings { saved, resolved, problem, file: file.to_string_lossy().into_owned() })
    }

    pub fn start(app: &AppHandle) {
        match settings(app) {
            Ok(settings) => app.state::<Engine>().restart(settings.resolved),
            Err(error) => eprintln!("crisp3ds studio: {error}"),
        }
    }

    #[tauri::command]
    pub fn get_settings(app: AppHandle) -> Result<Settings, String> {
        settings(&app)
    }

    #[tauri::command]
    pub fn save_settings(app: AppHandle, config: Config) -> Result<Settings, String> {
        config::save(&config_file(&app)?, &config)?;
        settings(&app)
    }

    #[tauri::command]
    pub fn engine_status(engine: State<'_, Engine>) -> Status {
        engine.status()
    }

    #[tauri::command]
    pub fn restart_engine(app: AppHandle) {
        start(&app);
    }

    /// Opens the native picker. `kind` is `folder` or `file`. Returns the chosen path or null.
    #[tauri::command]
    pub async fn pick_path(app: AppHandle, kind: String, title: String, start: Option<String>) -> Option<String> {
        let mut dialog = app.dialog().file().set_title(title);
        if let Some(start) = start.map(PathBuf::from).filter(|path| path.is_dir()) {
            dialog = dialog.set_directory(start);
        }
        let chosen = if kind == "file" { dialog.blocking_pick_file() } else { dialog.blocking_pick_folder() };
        chosen.and_then(|path| path.into_path().ok()).map(|path| path.to_string_lossy().into_owned())
    }
}

/// Debug builds only: lets a test script that the shell injected report back on stderr.
#[cfg(debug_assertions)]
#[tauri::command]
fn autopilot_log(line: String) {
    eprintln!("[autopilot] {line}");
}

/// Debug builds only: quits the app the way the menu does, so the exit path can be tested.
#[cfg(debug_assertions)]
#[tauri::command]
fn autopilot_quit(app: tauri::AppHandle) {
    app.exit(0);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();

    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_dialog::init()).manage(engine::Engine::default()).setup(|app| {
        engine::install_signal_handlers();
        desktop::start(app.handle());
        Ok(())
    });

    // Debug builds only: CRISP3DS_STUDIO_AUTOPILOT=<file.js> runs that script in the window
    // once the page has loaded, so the real app can be driven end to end without a human.
    #[cfg(debug_assertions)]
    let builder = builder.on_page_load(|webview, payload| {
        if payload.event() != tauri::webview::PageLoadEvent::Finished {
            return;
        }
        if let Some(script) = std::env::var_os("CRISP3DS_STUDIO_AUTOPILOT").and_then(|file| std::fs::read_to_string(file).ok()) {
            let _ = webview.eval(script);
        }
    });

    #[cfg(all(desktop, debug_assertions))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        shell_info,
        desktop::get_settings,
        desktop::save_settings,
        desktop::engine_status,
        desktop::restart_engine,
        desktop::pick_path,
        autopilot_log,
        autopilot_quit
    ]);
    #[cfg(all(desktop, not(debug_assertions)))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        shell_info,
        desktop::get_settings,
        desktop::save_settings,
        desktop::engine_status,
        desktop::restart_engine,
        desktop::pick_path
    ]);
    #[cfg(all(mobile, debug_assertions))]
    let builder = builder.invoke_handler(tauri::generate_handler![shell_info, autopilot_log, autopilot_quit]);
    #[cfg(all(mobile, not(debug_assertions)))]
    let builder = builder.invoke_handler(tauri::generate_handler![shell_info]);

    let app = builder.build(tauri::generate_context!()).expect("error while building Crisp3DS Studio");
    app.run(|_handle, _event| {
        #[cfg(desktop)]
        if let tauri::RunEvent::Exit = _event {
            // Blocks until the engine and its process group are gone.
            _handle.state::<engine::Engine>().stop();
        }
    });
}
