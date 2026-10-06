#!/bin/sh
# Builds the browser package into crates/dense/web/pkg.
#
# Needs the wasm32 target and wasm-bindgen-cli of the version in Cargo.lock:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.129 --locked
set -eu
cd "$(dirname "$0")"
cargo build --release --locked --target wasm32-unknown-unknown
target="${CARGO_TARGET_DIR:-target}"
wasm-bindgen --target web --out-dir pkg "$target/wasm32-unknown-unknown/release/crisp3ds_dense_web.wasm"
ls -l pkg
