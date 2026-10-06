//! Photo front stage: from a folder of turntable photos to the dense inputs
//! directory. Port of `scripts/turntable_mesh/photos_to_inputs.py` and of the
//! modules it calls (`silhouette_cleanup.py`, `alicevision_cameras.py`,
//! `mve_full/prepare.py`'s contrast profile).
//!
//! | Module | Content |
//! | --- | --- |
//! | `options` | command line, resolved configuration, AliceVision command lines |
//! | `staging`, `coarse` | capture order, `capture_NNNN.png` copies, coarse dark-object masks |
//! | `cleanup` | dark-hole cleanup of the masks |
//! | `sheets` | published masks, statistics, mask contact sheet, sparse overlay |
//! | `contrast` | gamma and CLAHE contrast images (OpenCV's 8-bit Lab and CLAHE) |
//! | `calibration` | lens file, scaling, the locked AliceVision intrinsic |
//! | `audit`, `ring` | camera audit, ring statistics, gate decision |
//! | `process` | bounded child processes |
//! | `run` | the two stages, events, `frontend.json` |
//!
//! Not ported: AliceVision (an external MPL-2.0 program, called as a child
//! process) and SAM 2.1 (a PyTorch network; `--mask-mode sam` runs the
//! reference `segment.py` in an external interpreter, `--masks DIR` takes masks
//! made elsewhere).

pub mod audit;
pub mod calibration;
pub mod cleanup;
pub mod coarse;
pub mod contrast;
pub mod options;
// External tools are child processes and the free-space floor needs a file system: not in a browser.
#[cfg(not(target_arch = "wasm32"))]
pub mod process;
pub mod ring;
#[cfg(not(target_arch = "wasm32"))]
pub mod run;
pub mod sheets;
pub mod staging;
pub mod util;

/// Entry point of the `photos` subcommand.
#[cfg(not(target_arch = "wasm32"))]
pub fn command(arguments: &[String]) -> std::process::ExitCode {
    use std::process::ExitCode;
    if arguments.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", options::USAGE);
        return ExitCode::SUCCESS;
    }
    let usage = |error: anyhow::Error| {
        eprintln!("crisp3ds-dense photos: {error:#}\n(crisp3ds-dense photos --help lists the options)");
        ExitCode::from(2)
    };
    let options = match options::resolve(arguments, &|name| std::env::var(name).ok()) {
        Ok(options) => options,
        Err(error) => return usage(error),
    };
    if options.output.exists() {
        return usage(anyhow::anyhow!("output exists: {}", options.output.display()));
    }
    let events_path = options.events.clone().unwrap_or_else(|| options.output.join("events.jsonl"));
    let events = crate::events::EventLog::new(Some(&events_path), "cameras");
    match run::run(&options, &events, &events_path, None) {
        Ok(finished) => {
            println!("{}", serde_json::to_string_pretty(&run::summary(&finished.report)).unwrap_or_default());
            ExitCode::from(finished.code)
        }
        Err(error) => {
            eprintln!("crisp3ds-dense photos: {error:#}");
            ExitCode::from(1)
        }
    }
}
