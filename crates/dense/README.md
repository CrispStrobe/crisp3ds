# crisp3ds-dense: the native dense pipeline

A Rust port of `scripts/turntable_mesh`, so the reconstruction runs without
Python: GPU stages as WebGPU compute shaders through `wgpu` (Metal, Vulkan,
DirectX 12, and WebGPU in browsers), the rest as plain Rust. The Python package
stays the reference implementation until every stage here reproduces it.

Status: skeleton. Configuration mirror and event log exist; stages are being
ported in this order: surface extraction, hull and fusion, matching, inputs.

## Rules of the port

- **Same files.** Every stage reads and writes what its Python counterpart
  does: the inputs directory (`cameras.json`, masks, `sparse_points.npy`),
  `depths.npz`, `volume.npz`, `mesh/mesh.stl`, `result.json`, `events.jsonl`,
  `config.json`. A run may therefore mix Python and native stages, which is
  how each native stage is checked.
- **Same settings.** `DenseConfig` mirrors `dense_config.py`; a test fails when
  the defaults drift (`tests/fixtures/dense-config-defaults.json`).
- **Same events.** `docs/ENGINE-CONTRACT.md` is binding for the native stages.
- **Parity before promotion.** A native stage replaces the Python one only
  when, on the synthetic scene and on real objects, its output matches within
  stated tolerances and the scanner scores (`scan_evaluate.py`) are unchanged
  within 0.003.
- **No reference geometry** is ever read by a stage.

## Layout

| Path | Content |
| --- | --- |
| `src/config.rs` | Settings mirror |
| `src/events.rs` | Event log writer |
| `src/npz.rs`, `src/stl.rs` | Array and mesh file formats |
| `src/mesh/` | Surface extraction (port of `tsdf_hull_mesh.py`) |
| `src/gpu/`, `src/shaders/` | `wgpu` device handling and WGSL kernels |
| `src/inputs.rs`, `src/arrays.rs` | Inputs directory as `Stereo.__init__` prepares it; standalone `.npy` |
| `src/hull.rs`, `src/repair.rs`, `src/fusion.rs`, `src/stereo/` | Ports of `multiscale_stereo.py` |
| `src/main.rs` | `crisp3ds-dense <stage>` command line |

## Build and test

```sh
cd crates/dense
cargo test
cargo run --release -- defaults
```

License: AGPL-3.0-only, like the rest of the repository. Dependencies and their
licenses are listed here as they are added.

| Crate | License | Use |
| --- | --- | --- |
| anyhow | MIT OR Apache-2.0 | errors |
| serde, serde_json | MIT OR Apache-2.0 | configuration and events |
| wgpu | MIT OR Apache-2.0 | WebGPU compute (Metal, Vulkan, DirectX 12, browsers) |
| bytemuck | Zlib OR Apache-2.0 OR MIT | plain-data casts for GPU buffers |
| pollster | Apache-2.0 OR MIT | blocking on `wgpu` futures |
| image (png, jpeg only) | MIT OR Apache-2.0 | photo and mask decoding, PNG writing |
