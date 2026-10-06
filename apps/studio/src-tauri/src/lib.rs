//! Crisp3DS Studio as an app.
//!
//! Three ways to get a reconstruction, in the order the app prefers them:
//!
//! 1. **Built-in engine** (`native_engine`, module `native`): `crates/dense` runs inside
//!    this process on a worker thread. No Python, no child process. This is the default
//!    wherever the crate is linked, including the sandboxed Mac App Store variant.
//! 2. **External Python engine** (`local_engine`, module `engine`, desktop only): the shell
//!    starts `scripts.turntable_mesh.engine_server` with an interpreter the user names and
//!    hands its address and token to the web view. Kept for the photos start and for
//!    comparisons. Python is not bundled.
//! 3. An engine elsewhere over HTTP, recorded runs, the demo: plain web app, no shell code.
//!
//! The web view gets no Tauri permissions beyond calling the commands below. Every path it
//! can name is relative to the runs or the data folder and checked to stay inside.

mod config;
#[cfg(local_engine)]
mod engine;
#[cfg(native_engine)]
mod native;
#[cfg(native_engine)]
mod tools;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager};

use config::{Config, Resolved, Surroundings};

#[allow(dead_code)]
const UNAVAILABLE: &str = "This build of the app does not include that.";

#[derive(Serialize)]
struct ShellInfo {
    /// The reconstruction is built into this app.
    native_engine: bool,
    /// This app can start the external Python engine (desktop builds outside the sandbox).
    can_run_engine: bool,
    /// Native folder and file pickers exist.
    can_pick_paths: bool,
    /// The app runs in the macOS App Sandbox: no other programs, only folders the user picked.
    sandboxed: bool,
    os: &'static str,
    version: &'static str,
    /// The device `auto` stands for when the Python engine is used on this computer.
    auto_device: &'static str,
}

#[tauri::command]
fn shell_info() -> ShellInfo {
    ShellInfo {
        native_engine: cfg!(native_engine),
        can_run_engine: cfg!(local_engine),
        can_pick_paths: cfg!(desktop),
        sandboxed: std::env::var_os("APP_SANDBOX_CONTAINER_ID").is_some(),
        os: std::env::consts::OS,
        version: env!("CARGO_PKG_VERSION"),
        auto_device: config::auto_device(),
    }
}

// ---------------------------------------------------------------------------------------------
// Settings: where the folders and (for the Python engine) the interpreters are

#[derive(Serialize)]
struct Settings {
    saved: Config,
    /// What will actually be used, and where each value comes from.
    resolved: Resolved,
    /// Where the external programs of the photos start are (AliceVision, COLMAP, SAM).
    tools: config::Tools,
    /// A sentence when the external Python engine cannot be started with these values.
    problem: Option<String>,
    file: String,
}

fn config_file(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    app.path().app_config_dir().map(|folder| folder.join("config.json")).map_err(|error| error.to_string())
}

fn settings(app: &AppHandle) -> Result<Settings, String> {
    let file = config_file(app)?;
    let saved = config::load(&file);
    let around = app.path().app_data_dir().map(Surroundings::real).map_err(|error| error.to_string())?;
    let resolved = config::resolve(&saved, &around);
    let problem = config::problems(&resolved);
    let tools = config::resolve_tools(&saved, &around);
    Ok(Settings { saved, resolved, tools, problem, file: file.to_string_lossy().into_owned() })
}

/// Points the built-in engine at the configured folders.
fn configure_native(app: &AppHandle) -> Result<(), String> {
    #[cfg(native_engine)]
    {
        let settings = settings(app)?;
        let engine = app.state::<std::sync::Arc<native::Native>>();
        engine.configure_tools(settings.tools.clone(), settings.resolved.repo.value.clone());
        engine.configure(std::path::Path::new(&settings.resolved.runs_dir.value), std::path::Path::new(&settings.resolved.data_dir.value))?;
    }
    #[cfg(not(native_engine))]
    let _ = app;
    Ok(())
}

#[tauri::command]
fn get_settings(app: AppHandle) -> Result<Settings, String> {
    settings(&app)
}

#[tauri::command]
fn save_settings(app: AppHandle, config: Config) -> Result<Settings, String> {
    config::save(&config_file(&app)?, &config)?;
    configure_native(&app)?;
    settings(&app)
}

/// Opens the native picker. `kind` is `folder` or `file`. Returns the chosen path or null.
/// What was picked may afterwards be used as the input of a run of the built-in engine.
#[tauri::command]
async fn pick_path(app: AppHandle, kind: String, title: String, start: Option<String>) -> Result<Option<String>, String> {
    #[cfg(desktop)]
    {
        use tauri_plugin_dialog::DialogExt;
        let mut dialog = app.dialog().file().set_title(title);
        if let Some(start) = start.map(std::path::PathBuf::from).filter(|path| path.is_dir()) {
            dialog = dialog.set_directory(start);
        }
        let chosen = if kind == "file" { dialog.blocking_pick_file() } else { dialog.blocking_pick_folder() };
        let path = chosen.and_then(|path| path.into_path().ok());
        #[cfg(native_engine)]
        if let Some(path) = &path {
            app.state::<std::sync::Arc<native::Native>>().grant(path);
        }
        Ok(path.map(|path| path.to_string_lossy().into_owned()))
    }
    #[cfg(not(desktop))]
    {
        let _ = (app, kind, title, start);
        Err(UNAVAILABLE.into())
    }
}

// ---------------------------------------------------------------------------------------------
// External Python engine (optional)

#[tauri::command]
fn engine_status(app: AppHandle) -> Result<Value, String> {
    #[cfg(local_engine)]
    {
        serde_json::to_value(app.state::<engine::Engine>().status()).map_err(|error| error.to_string())
    }
    #[cfg(not(local_engine))]
    {
        let _ = app;
        Err(UNAVAILABLE.into())
    }
}

/// Starts (or restarts) the Python engine with the current settings. It is not started
/// until somebody asks: the built-in engine is the default.
#[tauri::command]
fn restart_engine(app: AppHandle) -> Result<(), String> {
    #[cfg(local_engine)]
    {
        let settings = settings(&app)?;
        app.state::<engine::Engine>().restart(settings.resolved);
        Ok(())
    }
    #[cfg(not(local_engine))]
    {
        let _ = app;
        Err(UNAVAILABLE.into())
    }
}

#[tauri::command]
fn stop_engine(app: AppHandle) -> Result<(), String> {
    #[cfg(local_engine)]
    {
        let engine = app.state::<engine::Engine>().inner().clone();
        std::thread::spawn(move || engine.stop());
    }
    let _ = app;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Built-in engine: the same operations as the HTTP engine of docs/ENGINE-CONTRACT.md

#[cfg(native_engine)]
fn built_in(app: &AppHandle) -> std::sync::Arc<native::Native> {
    app.state::<std::sync::Arc<native::Native>>().inner().clone()
}

/// `GET /api/health`, plus what the form needs to know about this engine.
#[tauri::command]
async fn native_health(app: AppHandle) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        let (runs, data) = built_in(&app).folders();
        Ok(serde_json::json!({
            "schema": crisp3ds_dense::events::SCHEMA,
            "device": "wgpu",
            "can_start_runs": true,
            "runs_dir": runs.to_string_lossy(),
            "data_dir": data.to_string_lossy(),
            "start_points": built_in(&app).start_points(),
            "sandboxed": tools::sandboxed(),
        }))
    }
    #[cfg(not(native_engine))]
    {
        let _ = app;
        Err(UNAVAILABLE.into())
    }
}

/// `GET /api/settings`.
#[tauri::command]
fn native_settings() -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        Ok(native::settings_schema())
    }
    #[cfg(not(native_engine))]
    Err(UNAVAILABLE.into())
}

/// `GET /api/runs`.
#[tauri::command]
async fn native_runs(app: AppHandle) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        Ok(serde_json::json!({ "runs": built_in(&app).list_runs() }))
    }
    #[cfg(not(native_engine))]
    {
        let _ = app;
        Err(UNAVAILABLE.into())
    }
}

/// `POST /api/runs`. Returns `{"id"}`; an invalid request is an error with the engine's sentence.
#[tauri::command]
async fn native_start(app: AppHandle, body: Value) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        let body: native::StartBody = serde_json::from_value(body).map_err(|error| format!("JSON body required: {error}"))?;
        let id = built_in(&app).start(body)?;
        Ok(serde_json::json!({ "id": id }))
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, body);
        Err(UNAVAILABLE.into())
    }
}

/// `GET /api/runs/<id>/events?since=N`.
#[tauri::command]
async fn native_events(app: AppHandle, id: String, since: usize) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        serde_json::to_value(built_in(&app).events(&id, since)?).map_err(|error| error.to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, id, since);
        Err(UNAVAILABLE.into())
    }
}

/// `POST /api/runs/<id>/cancel`.
#[tauri::command]
async fn native_cancel(app: AppHandle, id: String) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        built_in(&app).cancel(&id)?;
        Ok(serde_json::json!({ "id": id, "cancel_requested": true }))
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, id);
        Err(UNAVAILABLE.into())
    }
}

/// `GET /api/runs/<id>/files/<path>`: the file's bytes, not as JSON.
#[tauri::command]
async fn native_file(app: AppHandle, id: String, path: String) -> Result<tauri::ipc::Response, String> {
    #[cfg(native_engine)]
    {
        let file = built_in(&app).file(&id, &path)?;
        std::fs::read(&file).map(tauri::ipc::Response::new).map_err(|_| "not found".to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, id, path);
        Err(UNAVAILABLE.into())
    }
}

/// `HEAD` of the same: the size in bytes.
#[tauri::command]
async fn native_file_size(app: AppHandle, id: String, path: String) -> Result<u64, String> {
    #[cfg(native_engine)]
    {
        let file = built_in(&app).file(&id, &path)?;
        std::fs::metadata(&file).map(|meta| meta.len()).map_err(|_| "not found".to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, id, path);
        Err(UNAVAILABLE.into())
    }
}

/// `GET /api/data?path=<relative>`.
#[tauri::command]
async fn native_data(app: AppHandle, path: String) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        serde_json::to_value(built_in(&app).list_data(&path)?).map_err(|error| error.to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, path);
        Err(UNAVAILABLE.into())
    }
}

/// Asks an external program of the photos start about itself: `alicevision`, `colmap` or `sam`.
/// Uses the saved settings; returns `{ok, summary, detail}`.
#[tauri::command]
async fn check_tool(app: AppHandle, tool: String) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        let settings = settings(&app)?;
        serde_json::to_value(tools::check(&tool, &settings.tools, &settings.resolved.repo.value)).map_err(|error| error.to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = (app, tool);
        Err(UNAVAILABLE.into())
    }
}

/// Lens calibration files found next to the app: in the crisp3ds checkout and in the data
/// folder. What is listed may be used as a run's calibration.
#[tauri::command]
async fn native_calibrations(app: AppHandle) -> Result<Value, String> {
    #[cfg(native_engine)]
    {
        let settings = settings(&app)?;
        let data = std::path::PathBuf::from(&settings.resolved.data_dir.value);
        let mut folders = vec![data.clone(), data.join("calibrations")];
        if !settings.resolved.repo.value.is_empty() {
            folders.push(std::path::Path::new(&settings.resolved.repo.value).join("scripts/turntable_mesh/calibrations"));
        }
        let found = tools::calibrations(&folders);
        let engine = built_in(&app);
        for calibration in &found {
            engine.grant(std::path::Path::new(&calibration.path));
        }
        serde_json::to_value(found).map_err(|error| error.to_string())
    }
    #[cfg(not(native_engine))]
    {
        let _ = app;
        Err(UNAVAILABLE.into())
    }
}

// ---------------------------------------------------------------------------------------------

/// Debug builds only: lets a test script that the shell injected report back on stderr.
#[cfg(debug_assertions)]
#[tauri::command]
fn autopilot_log(line: String) {
    eprintln!("[autopilot] {line}");
}

/// Debug builds only: quits the app the way the menu does, so the exit path can be tested.
#[cfg(debug_assertions)]
#[tauri::command]
fn autopilot_quit(app: AppHandle, after_ms: Option<u64>) {
    // Timed here, not in the page: a web view that is not visible may not run its timers.
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(after_ms.unwrap_or(0)));
        app.exit(0);
    });
}

/// Debug builds only: a pause timed here, because a web view that is not visible (another
/// window in front, a locked screen) slows or stops its own timers.
#[cfg(debug_assertions)]
#[tauri::command]
async fn autopilot_sleep(ms: u64) {
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

/// Debug builds only: memory of this process in bytes (resident now, and its peak).
#[cfg(debug_assertions)]
#[tauri::command]
fn autopilot_memory() -> (u64, u64) {
    #[cfg(unix)]
    {
        // SAFETY: getrusage fills the struct it is given.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        // macOS reports bytes, Linux kilobytes.
        let peak = usage.ru_maxrss as u64 * if cfg!(target_os = "macos") || cfg!(target_os = "ios") { 1 } else { 1024 };
        (0, peak)
    }
    #[cfg(not(unix))]
    (0, 0)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();

    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_dialog::init());
    #[cfg(local_engine)]
    let builder = builder.manage(engine::Engine::default());
    #[cfg(native_engine)]
    let builder = builder.manage(std::sync::Arc::new(native::Native::default()));

    let builder = builder.setup(|app| {
        #[cfg(local_engine)]
        engine::install_signal_handlers();
        if let Err(error) = configure_native(app.handle()) {
            eprintln!("crisp3ds studio: {error}");
        }
        // Debug builds only: CRISP3DS_STUDIO_AUTORUN=<start request as JSON> starts that run on
        // the built-in engine without the window taking part, reports on stderr and quits.
        // For measuring the engine inside the app when no screen is available to drive the UI.
        #[cfg(all(debug_assertions, native_engine))]
        if let Some(body) = std::env::var("CRISP3DS_STUDIO_AUTORUN").ok().and_then(|text| serde_json::from_str::<native::StartBody>(&text).ok()) {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let engine = built_in(&handle);
                let began = std::time::Instant::now();
                match engine.start(body) {
                    Ok(id) => {
                        eprintln!("[autorun] started {id}");
                        let mut seen = 0;
                        loop {
                            let page = engine.events(&id, seen).unwrap_or(native::EventPage { events: Vec::new(), next: seen });
                            for event in &page.events {
                                if event["type"] == "artifact" || event["type"] == "stage_finished" || event["type"] == "error" || event["type"] == "run_finished" {
                                    let what = event["label"].as_str().or(event["message"].as_str()).or(event["status"].as_str()).unwrap_or("");
                                    eprintln!("[autorun] {:6.1} s  {} {} {what}", began.elapsed().as_secs_f64(), event["type"].as_str().unwrap_or(""), event["stage"].as_str().unwrap_or(""));
                                }
                            }
                            seen = page.next;
                            if engine.running() == 0 {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(200));
                        }
                        eprintln!("[autorun] ended after {:.1} s; peak resident memory {} MB", began.elapsed().as_secs_f64(), autopilot_memory().1 / 1_000_000);
                    }
                    Err(error) => eprintln!("[autorun] refused: {error}"),
                }
                handle.exit(0);
            });
        }
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

    #[cfg(debug_assertions)]
    let builder = builder.invoke_handler(tauri::generate_handler![
        shell_info,
        get_settings,
        save_settings,
        pick_path,
        engine_status,
        restart_engine,
        stop_engine,
        native_health,
        native_settings,
        native_runs,
        native_start,
        native_events,
        native_cancel,
        native_file,
        native_file_size,
        native_data,
        native_calibrations,
        check_tool,
        autopilot_log,
        autopilot_quit,
        autopilot_sleep,
        autopilot_memory
    ]);
    #[cfg(not(debug_assertions))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        shell_info,
        get_settings,
        save_settings,
        pick_path,
        engine_status,
        restart_engine,
        stop_engine,
        native_health,
        native_settings,
        native_runs,
        native_start,
        native_events,
        native_cancel,
        native_file,
        native_file_size,
        native_data,
        native_calibrations,
        check_tool
    ]);

    let app = builder.build(tauri::generate_context!()).expect("error while building Crisp3DS Studio");
    app.run(|_handle, _event| {
        if let tauri::RunEvent::Exit = _event {
            // Runs of the built-in engine are asked to stop and given a moment; whatever is
            // still busy then has its log closed, because the process ends next.
            #[cfg(native_engine)]
            built_in(_handle).shutdown(std::time::Duration::from_secs(8));
            // Blocks until the Python engine and its process group are gone.
            #[cfg(local_engine)]
            _handle.state::<engine::Engine>().stop();
        }
    });
}
