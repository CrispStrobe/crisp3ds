//! ggml backend of the `sam` provider (cargo feature `sam-ggml`): SAM 2.1 in
//! CrispEmbed's ggml engine (`src/sam2.cpp`), reached through its C API in the
//! shared library `libcrispembed-sam2` (CrispEmbed's CMake option
//! `CRISPEMBED_SAM2_SHARED`). The library is opened when the first model is
//! loaded (`--sam-runtime`, else `CRISPEMBED_SAM2_LIB`), so building this
//! crate compiles and links no C++.
//!
//! The model is a GGUF file (`models/convert-sam2-to-gguf.py` in CrispEmbed,
//! or `cstr/sam2.1-hiera-tiny-GGUF` on Hugging Face). Prepared image in, mask
//! logits out: everything around the network stays in this crate, as for ONNX.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail};

use super::backend::{BackendOptions, ModelInfo, Prediction, SamBackend};

/// Environment variable naming the library when `--sam-runtime` is not given.
pub const LIBRARY_VARIABLE: &str = "CRISPEMBED_SAM2_LIB";

type Init = unsafe extern "C" fn(*const c_char, c_int) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Size = unsafe extern "C" fn(*const c_void) -> c_int;
type Name = unsafe extern "C" fn(*const c_void) -> *const c_char;
type SetImage = unsafe extern "C" fn(*mut c_void, *const f32) -> c_int;
type Predict = unsafe extern "C" fn(*mut c_void, *const f32, *const c_int, c_int, *mut f32, *mut f32) -> c_int;

/// The C API of `libcrispembed-sam2`.
struct Api {
    init: Init,
    free: Free,
    image_size: Size,
    mask_size: Size,
    backend: Name,
    set_image_f32: SetImage,
    predict: Predict,
    path: PathBuf,
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

/// The library to open: `--sam-runtime`, else the environment variable.
pub fn library_path(runtime: Option<&Path>) -> Option<PathBuf> {
    runtime.map(Path::to_path_buf).or_else(|| std::env::var_os(LIBRARY_VARIABLE).map(PathBuf::from))
}

fn api(runtime: Option<&Path>) -> anyhow::Result<&'static Api> {
    let loaded = API.get_or_init(|| {
        let Some(path) = library_path(runtime) else {
            return Err(format!("libcrispembed-sam2 is not given (--sam-runtime FILE or {LIBRARY_VARIABLE})"));
        };
        // The library stays loaded for the life of the process: contexts may outlive any one caller.
        let library = unsafe { libloading::Library::new(&path) }.map_err(|e| format!("{}: {e}", path.display()))?;
        let library: &'static libloading::Library = Box::leak(Box::new(library));
        macro_rules! symbol {
            ($name:literal) => {
                unsafe { library.get($name) }.map(|s| *s).map_err(|e| format!("{}: {e}", path.display()))?
            };
        }
        Ok(Api {
            init: symbol!(b"crispembed_sam2_init\0"),
            free: symbol!(b"crispembed_sam2_free\0"),
            image_size: symbol!(b"crispembed_sam2_image_size\0"),
            mask_size: symbol!(b"crispembed_sam2_mask_size\0"),
            backend: symbol!(b"crispembed_sam2_backend\0"),
            set_image_f32: symbol!(b"crispembed_sam2_set_image_f32\0"),
            predict: symbol!(b"crispembed_sam2_predict\0"),
            path,
        })
    });
    loaded.as_ref().map_err(|reason| anyhow!("{reason}"))
}

pub struct GgmlBackend {
    api: &'static Api,
    ctx: *mut c_void,
    image_size: usize,
    mask_size: usize,
    description: String,
}

impl GgmlBackend {
    pub fn open(model: &ModelInfo, options: &BackendOptions) -> anyhow::Result<Self> {
        let api = api(options.runtime.as_deref())?;
        match options.accelerator.as_str() {
            // The engine picks the best GPU backend the library was built with unless told otherwise.
            "cpu" => std::env::set_var("SAM2_FORCE_CPU", "1"),
            "gpu" => std::env::remove_var("SAM2_FORCE_CPU"),
            other => bail!("--sam-accelerator {other}: the ggml backend offers cpu and gpu"),
        }
        let path = CString::new(model.encoder.to_string_lossy().as_bytes()).map_err(|_| anyhow!("model path contains a NUL byte"))?;
        let ctx = unsafe { (api.init)(path.as_ptr(), options.threads.max(1) as c_int) };
        if ctx.is_null() {
            bail!("{}: libcrispembed-sam2 could not load this model", model.encoder.display());
        }
        let (image_size, mask_size) = unsafe { ((api.image_size)(ctx) as usize, (api.mask_size)(ctx) as usize) };
        if (image_size, mask_size) != (model.image_size, model.mask_size) {
            unsafe { (api.free)(ctx) };
            bail!(
                "the GGUF is for {image_size} x {image_size} input and {mask_size} x {mask_size} masks, not {} and {}",
                model.image_size,
                model.mask_size
            );
        }
        let name = unsafe { CStr::from_ptr((api.backend)(ctx)) }.to_string_lossy().to_string();
        let description = format!("CrispEmbed ggml ({}), {name}, {} threads", api.path.display(), options.threads.max(1));
        Ok(GgmlBackend { api, ctx, image_size, mask_size, description })
    }
}

impl Drop for GgmlBackend {
    fn drop(&mut self) {
        unsafe { (self.api.free)(self.ctx) };
    }
}

impl SamBackend for GgmlBackend {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn set_image(&mut self, input: &[f32]) -> anyhow::Result<()> {
        let side = self.image_size;
        if input.len() != 3 * side * side {
            bail!("the network input must hold 3 x {side} x {side} values");
        }
        match unsafe { (self.api.set_image_f32)(self.ctx, input.as_ptr()) } {
            0 => Ok(()),
            status => bail!("SAM image encoder (ggml) failed ({status})"),
        }
    }

    fn predict(&mut self, points: &[[f32; 2]], labels: &[i64]) -> anyhow::Result<Prediction> {
        if points.is_empty() || points.len() != labels.len() {
            bail!("prompts need one label per point");
        }
        let flat: Vec<f32> = points.iter().flatten().copied().collect();
        let labels: Vec<c_int> = labels.iter().map(|&l| l as c_int).collect();
        let mut logits = vec![0f32; 4 * self.mask_size * self.mask_size];
        let mut scores = vec![0f32; 4];
        let status = unsafe {
            (self.api.predict)(self.ctx, flat.as_ptr(), labels.as_ptr(), labels.len() as c_int, logits.as_mut_ptr(), scores.as_mut_ptr())
        };
        if status != 0 {
            bail!("SAM mask decoder (ggml) failed ({status})");
        }
        Ok(Prediction { logits, scores })
    }
}
