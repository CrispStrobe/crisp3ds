//! Native port of the all-view dense pipeline.
//!
//! The Python package `scripts/turntable_mesh` is the reference implementation.
//! Every stage here reads and writes the same files as its Python counterpart
//! (inputs directory, `depths.npz`, `volume.npz`, `mesh.stl`, `events.jsonl`,
//! `config.json`), so stages can be swapped one at a time and compared.

pub mod check;
pub mod config;
pub mod events;
pub mod fusion;
pub mod gpu;
pub mod hull;
pub mod inputs;
pub mod mesh;
pub mod npz;
pub mod render;
pub mod repair;
pub mod scene;
pub mod stereo;
pub mod stl;
