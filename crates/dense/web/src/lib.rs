//! Browser bindings of the dense pipeline: WebAssembly for the CPU passes,
//! WebGPU for the kernels, an in-memory file tree instead of a disk.
//!
//! The host (a page or, better, a worker) puts the input files into the tree,
//! starts a run with a JSON options string whose paths point into the tree,
//! receives every event as an object through a callback, and reads output files
//! out of the tree, for example when an `artifact` event names one. See
//! `crisp3ds-dense.js` for a wrapper that does this from a map of files.
//!
//! Everything runs on the calling JavaScript thread. GPU work is awaited, so the
//! thread returns to its event loop at every readback (once per view while
//! matching); the CPU passes between them do not yield.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crisp3ds_dense::events::Observer;
use crisp3ds_dense::run::{run_async, RunOptions};
use crisp3ds_dense::storage;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Event callbacks of the runs in progress, by run number.
    static CALLBACKS: RefCell<BTreeMap<u32, js_sys::Function>> = const { RefCell::new(BTreeMap::new()) };
    static NEXT: RefCell<u32> = const { RefCell::new(1) };
    /// Supplies files the host keeps in JavaScript memory.
    static SOURCE: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// Version of the engine.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Name, group, meaning, kind and default of every setting: `{"settings": [...]}` as JSON text.
#[wasm_bindgen(js_name = settingsSchema)]
pub fn settings_schema() -> String {
    crisp3ds_dense::config::SETTINGS_SCHEMA.to_string()
}

/// The default settings as JSON text.
#[wasm_bindgen(js_name = defaultSettings)]
pub fn default_settings() -> String {
    serde_json::to_string(&crisp3ds_dense::config::DenseConfig::default()).unwrap_or_default()
}

/// Stores a file in the in-memory tree.
#[wasm_bindgen(js_name = putFile)]
pub fn put_file(path: &str, bytes: Vec<u8>) -> Result<(), JsError> {
    storage::write_owned(path, bytes).map_err(|e| JsError::new(&e.to_string()))
}

/// A file of the in-memory tree, or undefined.
#[wasm_bindgen(js_name = getFile)]
pub fn get_file(path: &str) -> Option<Vec<u8>> {
    storage::read(path).ok()
}

/// Every file under a directory of the tree: `[[path, bytes], ...]` as JSON text.
#[wasm_bindgen(js_name = listFiles)]
pub fn list_files(directory: &str) -> String {
    serde_json::to_string(&storage::memory_files(directory)).unwrap_or_default()
}

/// Registers `source(path, probe)`: called for files that are not in the tree.
/// It returns a `Uint8Array` with the file (or, when `probe` is true, any
/// truthy value if the file exists) and `undefined` otherwise. Large inputs can
/// then stay in JavaScript memory instead of WebAssembly memory. `undefined`
/// or `null` removes the source.
#[wasm_bindgen(js_name = setFileSource)]
pub fn set_file_source(source: Option<js_sys::Function>) {
    let present = source.is_some();
    SOURCE.with(|slot| *slot.borrow_mut() = source);
    storage::set_memory_source(present.then(|| {
        Box::new(|path: &str, probe: bool| {
            let function = SOURCE.with(|slot| slot.borrow().clone())?;
            let value = function.call2(&JsValue::NULL, &JsValue::from_str(path), &JsValue::from_bool(probe)).ok()?;
            if probe {
                return value.is_truthy().then(Vec::new);
            }
            value.is_instance_of::<js_sys::Uint8Array>().then(|| js_sys::Uint8Array::from(value).to_vec())
        }) as storage::Source
    }));
}

/// Removes a directory of the tree with everything in it.
#[wasm_bindgen(js_name = removeTree)]
pub fn remove_tree(directory: &str) {
    let _ = storage::remove_dir_all(directory);
}

/// Bytes held by the files of the tree.
#[wasm_bindgen(js_name = storedBytes)]
pub fn stored_bytes() -> f64 {
    storage::memory_bytes() as f64
}

/// One run. Create it, keep it to cancel, call `start` once.
#[wasm_bindgen]
pub struct Run {
    cancel: Arc<AtomicBool>,
}

impl Default for Run {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Run {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Run {
        Run { cancel: Arc::new(AtomicBool::new(false)) }
    }

    /// Asks the run to stop; it ends with status `cancelled` at the next view or stage boundary.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Runs the pipeline. `options` is JSON text with the fields of `RunOptions`
    /// (`output` and `inputs` are paths in the tree). `on_event` receives every
    /// event as an object, the same objects as the lines of `events.jsonl`.
    /// Resolves with the content of `pipeline.json` as JSON text; rejects with
    /// the error message when the run fails or is cancelled.
    pub async fn start(&self, options: String, on_event: Option<js_sys::Function>) -> Result<String, JsError> {
        let options: RunOptions = serde_json::from_str(&options).map_err(|e| JsError::new(&format!("options: {e}")))?;
        let id = NEXT.with(|next| {
            let mut next = next.borrow_mut();
            *next += 1;
            *next
        });
        if let Some(callback) = on_event {
            CALLBACKS.with(|callbacks| callbacks.borrow_mut().insert(id, callback));
        }
        // The callback lives in thread-local storage: the observer itself must be shareable between threads.
        let observer: Observer = Arc::new(move |event: &serde_json::Value| {
            let callback = CALLBACKS.with(|callbacks| callbacks.borrow().get(&id).cloned());
            if let Some(callback) = callback {
                if let Ok(object) = js_sys::JSON::parse(&event.to_string()) {
                    let _ = callback.call1(&JsValue::NULL, &object);
                }
            }
        });
        let result = run_async(&options, Some(observer), Some(self.cancel.clone())).await;
        CALLBACKS.with(|callbacks| callbacks.borrow_mut().remove(&id));
        match result {
            Ok(report) => Ok(report.to_string()),
            Err(error) => Err(JsError::new(&format!("{error:#}"))),
        }
    }
}
