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
        Some("--version") => {
            println!("crisp3ds-dense {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("stereo") => match crisp3ds_dense::stereo::main(&arguments[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("crisp3ds-dense stereo: {error:#}");
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!("usage: crisp3ds-dense <defaults|--version>   (stages are being ported; see crates/dense/README.md)");
            ExitCode::from(2)
        }
    }
}
