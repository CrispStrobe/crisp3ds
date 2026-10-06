//! Command line of the native dense pipeline. Subcommands mirror the Python modules.

use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        Some("defaults") => {
            println!("{}", serde_json::to_string_pretty(&crisp3ds_dense::config::DenseConfig::default()).unwrap());
            ExitCode::SUCCESS
        }
        Some("mesh") => crisp3ds_dense::mesh::command(&arguments[1..]),
        #[cfg(not(target_arch = "wasm32"))]
        Some("photos") => crisp3ds_dense::photos::command(&arguments[1..]),
        Some("--version") => {
            println!("crisp3ds-dense {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some(name @ ("stereo" | "inputs" | "check" | "run" | "synthetic")) => {
            let result = match name {
                "stereo" => crisp3ds_dense::stereo::main(&arguments[1..]),
                "check" => crisp3ds_dense::check::main(&arguments[1..]),
                "run" => crisp3ds_dense::run::main(&arguments[1..]),
                "synthetic" => crisp3ds_dense::stereo::synthetic::main(&arguments[1..]),
                _ => crisp3ds_dense::scene::main(&arguments[1..]),
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("crisp3ds-dense {name}: {error:#}");
                    ExitCode::from(1)
                }
            }
        }
        _ => {
            eprintln!("usage: crisp3ds-dense <run|photos|inputs|stereo|mesh|check|synthetic|defaults|--version> ...   (see crates/dense/README.md)");
            ExitCode::from(2)
        }
    }
}
