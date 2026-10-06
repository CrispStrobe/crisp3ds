//! Native port of the all-view dense pipeline.
//!
//! The Python package `scripts/turntable_mesh` is the reference implementation.
//! Every stage here reads and writes the same files as its Python counterpart
//! (inputs directory, `depths.npz`, `volume.npz`, `mesh.stl`, `events.jsonl`,
//! `config.json`), so stages can be swapped one at a time and compared.

#[cfg(feature = "capi")]
pub mod capi;
pub mod check;
pub mod config;
pub mod control;
pub mod events;
pub mod fusion;
pub mod gpu;
pub mod hull;
pub mod inputs;
pub mod mesh;
pub mod npz;
// The photo front end drives external programs; it has no place in a browser build.
#[cfg(not(target_arch = "wasm32"))]
pub mod photos;
pub mod render;
pub mod repair;
pub mod run;
pub mod scene;
pub mod stereo;
pub mod stl;
pub mod storage;
