# crisp3ds-dense: the native dense pipeline

A Rust port of `scripts/turntable_mesh`, so the reconstruction runs without
Python: GPU stages as WebGPU compute shaders through `wgpu` (Metal, Vulkan,
DirectX 12, and WebGPU in browsers), the rest as plain Rust. The Python package
stays the reference implementation until every stage here reproduces it.

Status: surface extraction (`mesh`) and dense stereo (`stereo`) are ported; see
their sections below for measured parity. Not ported: the photo front end
(masks and cameras from plain photos), the photo check and the scan evaluator.

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

Formatting follows `rustfmt.toml` (`cargo fmt --check`), lints are
`cargo clippy --all-targets -- -D warnings`; `.github/workflows/dense-native.yml`
runs both and the tests on Linux, macOS and Windows. Needs Rust 1.88 or newer.

## Surface extraction: `crisp3ds-dense mesh`

Status: ported, at parity with `python -m scripts.turntable_mesh.tsdf_hull_mesh`
on the four test objects (table below). Not yet called by `dense_pipeline.py`.

```sh
crisp3ds-dense mesh --volume stereo/volume.npz --output mesh \
    [--config config.json] [--set key=value ...] [--step N] \
    [--events events.jsonl] [--label TEXT] [--threads N]
```

Same behaviour as the Python module: refuses an existing output directory,
writes `mesh.stl` (binary, outward normals, largest component) and
`result.json` (same keys, same order), prints the report, and appends the
`final_mesh` artifact event with stage `mesh`, or `preview_mesh` with stage
`stereo` when `--step` is above 1. `--threads` (default 2) is the only addition.

| Step | Implementation (`src/mesh/`) |
| --- | --- |
| Hull prior | `edt.rs`: exact Euclidean distance transform, the separable lower-envelope algorithm of Felzenszwalb and Huttenlocher on integer squared distances; the square root equals `scipy.ndimage.distance_transform_edt` |
| Smoothing, extrapolation, final blur | `gaussian.rs`: separable, kernel cut at 4 sigma, `reflect` boundary, each line summed in double precision in SciPy's order and stored as `float32` between axes |
| Field | `field.rs`: the reference's arithmetic operation by operation, `float32` where NumPy works in `float32` |
| Iso-surface | `cubes.rs`: marching cubes whose cell boundaries are drawn face by face, ambiguous faces resolved by the asymptotic decider; one vertex per crossed grid edge |
| Mesh | `surface.rs`: largest component (union-find), Taubin smoothing (lambda 0.5, mu -0.53, uniform weights), edge counts, genus, signed volume |

Why the surface is a closed 2-manifold. The reference uses scikit-image's
Lewiner tables. Here a cell's triangles are derived from what the surface does
on the cell's six faces: marching-squares segments between the crossings on the
face's edges, where a face with two diagonally opposite inside corners is
decided by the sign of the bilinear interpolant at its saddle point. That
decision uses the four corner values of the face only, so the two cells sharing
a face draw the same segments. Each crossing ends exactly one segment on each
of its two faces, so the segments of a cell form closed rings; a ring is filled
with a fan whose diagonals never join two crossings on one face (a centre
vertex is added in the 140 of 656 table entries where no such fan exists, all
of them cells with an ambiguous face). Every mesh edge is therefore either
private to one cell and shared by two triangles of a fan, or a face segment
with exactly one triangle on either side. The tests check this exhaustively
for all 256 corner configurations and 64 face decisions, and on random fields.
Not ported: Lewiner's tests for tunnels through the interior of a cell, so the
genus can differ in a few cells (table).

Other deviations from the reference: sums are taken in a different order
(kernel normalisation, signed volume; the preview's signed volume is summed in
double instead of single precision); vertices are interpolated linearly without
scikit-image's `FLT_EPSILON` weights; among components with equally many
triangles the one with the lowest vertex index is kept; an empty hull or a
field without zero crossing is reported as an error; the event's keys are
written in alphabetical order and its path is relative only when the mesh lies
under the directory of the event log.

Parity on real volumes (hull grid about 400 voxels, re-fused from saved depths
with the default settings; Apple M-series, 2 threads; distances between the two
meshes on 200 000 area-weighted samples each way, in voxels; scanner scores
from `scan_evaluate.py` against the independent scans, which no stage reads):

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| Triangles, Python / native | 1 044 840 / 1 044 840 | 772 208 / 772 200 | 991 246 / 991 234 | 354 354 / 354 346 |
| Closed, both | yes | yes | yes | yes |
| Components (discarded faces), both | 8 (2416) | 10 (520) | 21 (4072) | 4 (124) |
| Genus, Python / native | 3 / 3 | 5 / 3 | 16 / 14 | 14 / 13 |
| Signed volume, relative difference | 1.6e-6 | 5e-7 | 6.1e-6 | 8.9e-6 |
| Distance Python to native: median / p99 / max | 0.003 / 0.049 / 0.29 | 0.005 / 0.065 / 0.26 | 0.006 / 0.076 / 0.56 | 0.005 / 0.069 / 0.60 |
| Distance native to Python: median / p99 / max | 0.003 / 0.048 / 0.26 | 0.005 / 0.065 / 0.27 | 0.006 / 0.075 / 0.32 | 0.005 / 0.068 / 0.37 |
| F1 `all` at 0.5 / 1 / 2 %, Python | 0.904 / 0.939 / 0.955 | 0.921 / 0.970 / 0.987 | 0.769 / 0.925 / 0.981 | 0.790 / 0.904 / 0.938 |
| F1 `all` at 0.5 / 1 / 2 %, native | 0.903 / 0.939 / 0.955 | 0.921 / 0.970 / 0.987 | 0.771 / 0.925 / 0.981 | 0.789 / 0.903 / 0.938 |
| F1 `above_margin`, Python | 0.964 / 0.995 / 1.000 | 0.951 / 0.996 / 1.000 | 0.820 / 0.960 / 0.994 | 0.827 / 0.945 / 0.970 |
| F1 `above_margin`, native | 0.963 / 0.995 / 1.000 | 0.951 / 0.996 / 1.000 | 0.822 / 0.961 / 0.994 | 0.826 / 0.945 / 0.970 |
| Largest F1 difference | 0.0007 | 0.0005 | 0.0018 | 0.0007 |
| Wall time, Python / native | 32.8 s / 7.4 s | 34.4 s / 7.5 s | 30.3 s / 8.5 s | 11.9 s / 3.3 s |
| Peak memory, Python / native | 3.3 GiB / 1.2 GiB | 3.0 GiB / 0.72 GiB | 2.4 GiB / 0.74 GiB | 1.0 GiB / 0.24 GiB |

`hull_fraction_below_support`, `observed_hull_fraction` and
`extrapolated_hull_fraction` are identical to all printed digits on the four
objects. Preview path (`--step 2`, Dragon): both closed, 245 442 / 245 436
triangles, genus 12 / 11, distance median 0.001, p99 0.19, maximum 1.05 voxels
(cells are two voxels wide there), 22.9 s / 6.2 s.

## Dense stereo: `crisp3ds-dense stereo`

Port of `python -m scripts.turntable_mesh.multiscale_stereo`: inputs, silhouette
hull, mask repair, coarse-to-fine matching, cross-view agreement and TSDF
fusion. Not yet called by `dense_pipeline.py`.

```sh
crisp3ds-dense stereo --inputs DIR --output DIR [--config config.json] \
    [--set key=value ...] [--reuse-depths depths.npz] [--events events.jsonl] \
    [--previews] [--device NAME]
```

Same behaviour as the Python module: refuses an existing output directory;
writes `masks-repaired/`, `mask-repair.png`, `hull-vs-mask.png`,
`depth-level-N.png`, `depth-merged.png`, `depths.npz`, `volume.npz` and
`result.json`; with `--previews` writes `preview/*.npz` (renamed into place when
complete); appends the same progress, metric and artifact events. `--device` is
accepted and ignored (the adapter is chosen by `wgpu`); `result.json` carries
`device` (adapter and backend) and `engine` instead of `torch_version`, and a
few extra timings. `fused_passes` above 0 is refused (not ported; the default
is 0).

| Step | Where | Runs on |
| --- | --- | --- |
| Photos, masks, grey stretch, common canvas, neighbours | `inputs.rs` | CPU, 4 threads |
| Silhouette carving (`carve`, `build_hull`) | `hull.rs`, `shaders/carve.wgsl` | GPU over candidate voxel lists; grid geometry and candidate test on the CPU |
| Mask repair (`orbit_down`, `cover`, `repair_masks`) | `repair.rs`, `shaders/cover.wgsl` | GPU projection, CPU photo test |
| Pyramid levels (`build_level`) | `stereo/level.rs` | CPU; Pillow's BILINEAR and BOX reductions reimplemented |
| Matching (`score`, `ncc`, `_aggregate`, `_peak`, `sweep`, `refine`) | `stereo/matcher.rs`, `shaders/warp.wgsl`, `ncc_rows.wgsl`, `ncc_columns.wgsl`, `aggregate.wgsl`, `peak.wgsl` | GPU; the cost volume stays on the device |
| `initial`, `consistent`, `hull_front`, level fallback | `stereo/depth.rs` | CPU, float32, PyTorch's interpolation and pooling conventions |
| Fusion (`tsdf`) and support height | `fusion.rs`, `shaders/fuse.wgsl` | GPU over the hull's voxel list |
| Level loop, sheets, files, events | `stereo/levels.rs`, `stereo/run.rs`, `stereo/previews.rs` | CPU |

GPU use is plain WebGPU: WGSL compute shaders, storage and uniform buffers, no
features, the default limits. The device is requested with `Limits::default()`,
so a kernel that needs more than 8 storage buffers per stage, 128 MiB per
binding or 65 535 workgroups per dimension fails validation here as it would in
a browser. Work is chunked to fit: voxels in lists of 2 M, masks and depth maps
in batches of whole views below 112 MiB, hypotheses in chunks whose six row
sums per pixel fit one binding. The largest binding of a run is reported as
`gpu_peak_binding_bytes`. A level whose whole cost volume (hypotheses x pixels
x 4 bytes) exceeds 112 MiB is refused; with the default settings that is a
canvas above about 2.6 megapixels at the finest level. Raised limits are not
requested even where the adapter offers them.

Tests: `cargo test` runs everything that needs no GPU, including validation of
every shader with naga. `CRISP3DS_GPU_TESTS=1 cargo test` adds the kernels
against scalar implementations and the whole stage on an analytic sphere
(`stereo/synthetic.rs`, a port of `synthetic_scene.py`) with the thresholds of
`test_multiscale_stereo.py`. `CRISP3DS_GPU_FALLBACK=1` asks for a software
adapter.

Deliberate differences from the reference:

- Window sums of the matching score are accumulated row by row in float32
  instead of through `avg_pool2d`; bilinear samples, projections and votes are
  computed per element in WGSL rather than as tensor operations. Results differ
  in the last bits and, at thresholds and pixel boundaries, in single decisions.
- `orbit_down` takes the orbit normal from a double-precision eigenvector of
  the camera centres' scatter matrix instead of a single-precision SVD.
- `hull_front`, `initial`, `consistent` and the fallback merge run on the CPU.
- Preview sheets are resampled in float and rounded (Pillow resamples 8-bit
  data in fixed point); shading values above 1 saturate instead of wrapping.
- `result.json` keys are in alphabetical order.

Kept as in the reference although probably unintended there: all views of the
finer levels search with the inverse-depth step of the last view swept at the
coarsest level (`step` is overwritten per view in the level-0 loop).

## Dependencies

License: AGPL-3.0-only, like the rest of the repository. Dependencies and their
licenses are listed here as they are added.

| Crate | License | Use |
| --- | --- | --- |
| anyhow | MIT OR Apache-2.0 | errors |
| serde, serde_json | MIT OR Apache-2.0 | configuration and events |
| rayon | MIT OR Apache-2.0 | worker threads of the grid passes |
| miniz_oxide | MIT OR Zlib OR Apache-2.0 | deflate for `.npz` members |
| crc32fast | MIT OR Apache-2.0 | `.npz` member checksums |
| wgpu | MIT OR Apache-2.0 | WebGPU compute (Metal, Vulkan, DirectX 12, browsers) |
| bytemuck | Zlib OR Apache-2.0 OR MIT | plain-data casts for GPU buffers |
| pollster | Apache-2.0 OR MIT | blocking on `wgpu` futures |
| image (png, jpeg only) | MIT OR Apache-2.0 | photo and mask decoding, PNG writing |
