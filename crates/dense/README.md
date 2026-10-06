# crisp3ds-dense: the native dense pipeline

A Rust port of `scripts/turntable_mesh`, so the reconstruction runs without
Python: GPU stages as WebGPU compute shaders through `wgpu` (Metal, Vulkan,
DirectX 12, and WebGPU in browsers), the rest as plain Rust. The Python package
stays the reference implementation until every stage here reproduces it.

Status: the whole pipeline from recovered cameras to a checked STL runs
natively, as one command (`run`), as a library call, or through a C interface.
Each stage is also a subcommand (`inputs`, `stereo`, `mesh`, `check`) that reads
and writes the files of its Python counterpart; measured parity is listed per
stage below. Not ported: masks and cameras from plain photos (SAM 2.1,
AliceVision) and the scan evaluator, which stay in Python.

Where this crate sits among the provider-based stages (masks, cameras,
undistortion) and which platforms run what is described in
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

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
| `src/run.rs`, `src/control.rs` | The run driver (port of `dense_pipeline.py`): library API and `run` command; cancel flag, deadlines, stage logs |
| `src/capi.rs`, `include/crisp3ds_dense.h` | C interface of the run driver (feature `capi`) |
| `src/scene.rs` | Inputs directory from an AliceVision scene (port of `dense_all_views_inputs.py`) |
| `src/check.rs`, `src/render.rs` | Photo check and its sheets (ports of `mesh_photo_check.py`, `stl_compare_render.py`) |
| `src/inputs.rs` | Inputs directory as `Stereo.__init__` prepares it |
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

## A whole run: `crisp3ds-dense run`, `crisp3ds_dense::run`, C interface

Port of `scripts/turntable_mesh/dense_pipeline.py` for two of its three
starting points, in one process and without Python:

```sh
crisp3ds-dense run --output RUN --inputs DIR                                   # an inputs directory
crisp3ds-dense run --output RUN --scene final.sfm --prepared DIR --raw-masks DIR
    [--config FILE] [--set key=value ...] [--threads N] [--stereo-timeout S]
    [--minimum-free-gib G] [--reuse-depths depths.npz] [--no-live-previews]
    [--preview-step N] [--skip-check] [--no-preview] [--keep-volume]
```

`--photos`/`--calibration` (masks and cameras from plain photos) and
`--reference` (scoring against a scan) are refused with a pointer to the Python
tools. `--device`, `--python`, `--torch-python` and `--native` are accepted and
ignored, so a command line written for the Python driver works.

The run directory is the one `docs/ENGINE-CONTRACT.md` describes: `config.json`,
`events.jsonl` (`run_started`, stage events, artifacts, `run_finished`; once
`run_started` is written `run_finished` always follows, also on failure, with an
`error` event before it), `pipeline.json` with the Python driver's keys,
`input-sheet.png`, `stereo/`, `mesh/`, `check/`, one `<stage>.log` per stage.
Stages run on the calling thread; preview volumes are meshed on a worker thread
while matching continues, and their `preview_mesh` events arrive when they are
done. Creating a file named `cancel` in the run directory stops the run: while
matching at the next view, otherwise at the next stage boundary (surface
extraction and the check are not interrupted; they take seconds). The run then
ends with status `cancelled`. `stereo/volume.npz` is deleted at the end unless
`--keep-volume`. `scripts/turntable_mesh/export_replay.py` accepts the result.

Against the Python driver (all stages in Python, Torch on MPS) on the same
machine: on the synthetic sphere and on the Bunny started from its scene, the
event logs have the same stage sequence, the same artifact kinds and counts per
stage (Bunny: 4 depth sheets, 4 preview volumes, 4 preview meshes, 3 reports,
one of each other kind) and the same metric names, and `pipeline.json` has the
same keys. Synthetic: 33 792 triangles from both, 21.0 s / 1.3 s. Bunny:
1 046 354 / 1 047 200 triangles, both closed, silhouette IoU median 0.9667 /
0.9667, scanner F1 `all` 0.905 / 0.941 / 0.955 against 0.905 / 0.941 / 0.956 and
`above_margin` 0.964 / 0.995 / 1.000 against 0.963 / 0.995 / 1.000 (largest
difference 0.0011), 1006 s / 193 s with both measured while other jobs loaded
the machine; a native run from the inputs directory alone took 95 s (stereo 78,
surface 12, check 5) under similar load.

As a library:

```rust
use std::sync::{atomic::AtomicBool, Arc};
use crisp3ds_dense::run::{run, RunOptions};

let options = RunOptions { output: "runs/bunny".into(), inputs: Some("bunny/inputs".into()), ..Default::default() };
let cancel = Arc::new(AtomicBool::new(false));           // set from any thread to stop the run
let observer = Arc::new(|event: &serde_json::Value| {     // every event, on the thread that emits it
    println!("{}", event["type"]);
});
let report = run(&options, Some(observer), Some(cancel))?;   // blocks; call it on a worker thread
```

`RunOptions` deserialises from JSON with the same field names (unknown fields
are an error); `settings` takes values by name as a settings form produces them.
`run` returns the content of `pipeline.json`, or the error that stopped the run
(`crisp3ds_dense::control::Stopped` when cancelled or past a deadline). The
observer is called from the run's thread and from the preview thread, so it must
be `Send + Sync` and should return quickly; a Tauri command would forward each
event to the window from there. One run uses one GPU device; start runs one at a
time.

From C, Swift, Kotlin or C++ (`include/crisp3ds_dense.h`):

```sh
cargo rustc --release --lib --features capi --crate-type cdylib      # or staticlib
```

```c
char *error = NULL;
Crisp3dsRun *run = crisp3ds_run_start("{\"output\": \"runs/bunny\", \"inputs\": \"bunny/inputs\"}", &error);
for (;;) {
    char *state = crisp3ds_run_poll(run);    /* {"finished", "status", "events": [...], "report", "error"} */
    /* ... show the new events; stop when "finished" is true ... */
    crisp3ds_string_free(state);
}
crisp3ds_run_cancel(run);                    /* optional, from any thread */
crisp3ds_run_free(run);                      /* cancels if still running, waits, releases */
```

The run executes on a thread the library starts; `poll` never blocks. This is
the only module with `unsafe` code beyond byte casts (it dereferences the
caller's pointers).

## Inputs from a scene: `crisp3ds-dense inputs`

```sh
crisp3ds-dense inputs --scene final.sfm --prepared DIR --raw-masks DIR --output DIR
```

Port of `dense_all_views_inputs.py`: one camera table for all registered views
of an AliceVision scene with one shared `radialk3` intrinsic, pointing at the
prepared images in place; raw photo masks remapped through the lens
undistortion with nearest sampling; the sparse points as `sparse_points.npy`.
The undistortion map follows OpenCV's `initUndistortRectifyMap` (double
precision, the camera matrix inverted by cofactors, coordinates accumulated
along each row, stored as float32) and `remap` with `INTER_NEAREST` (round half
to even, zero outside).

Parity with the Python module (OpenCV 4.10) on the Bunny and the Armadillo, 73
views each: every number of `cameras.json` identical, `sparse_points.npy`
identical, all 147 466 935 mask pixels identical. The translation is
accumulated with fused multiply-adds, which is what NumPy's matrix product does
on Apple silicon; a NumPy build that does not may differ in the last bit of a
translation.

## Photo check: `crisp3ds-dense check`

```sh
crisp3ds-dense check --inputs DIR --mesh mesh.stl --output DIR \
    [--repaired-masks DIR] [--preview-views N] [--check-views N] [--events events.jsonl]
```

Port of `mesh_photo_check.py` with the renderer of `stl_compare_render.py`:
silhouette intersection-over-union of the projected mesh against the input
masks and the repaired masks on evenly spaced views, `photo-overlay.png`,
`preview.png` (photos above, z-buffered flat-shaded mesh below), `result.json`
and the events. The silhouette is drawn as `cv2.fillPoly` draws a list of
triangles with four fractional bits: every edge as an 8-connected line between
the rounded vertex pixels, then an even-odd scanline fill of the whole edge
collection with span ends rounded to the nearest pixel.

Parity on the Bunny (mesh of 1 055 216 triangles, 26 views): the drawn
silhouettes are pixel-identical to OpenCV's on the views compared, and all IoU
numbers (median, minimum, maximum against input and repaired masks) equal the
Python module's to the last printed digit. The sheets show the same crops,
scales, colours and shading; they are resampled in float instead of OpenCV's
fixed point, and the two labels of the preview are set in a built-in 5x7 font
instead of OpenCV's Hershey font.

## Surface extraction: `crisp3ds-dense mesh`

Status: ported, at parity with `python -m scripts.turntable_mesh.tsdf_hull_mesh`
on the four test objects (table below). `dense_pipeline.py --native` calls it.

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
fusion. `dense_pipeline.py --native` calls it.

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

Parity with the Python stage (Torch 2.7 on MPS, Apple M1; native on the same
GPU through Metal; default settings unless noted; 73 or 71 photos of
1749 x 1155).

Inputs and hull. Grey images, canvas boxes, neighbour lists and pyramid levels
(grey, mask, camera; three sizes) are bit-identical on the Bunny in all 73
views. Synthetic sphere: identical hull voxel set. Bunny without mask repair:
12 826 963 voxels on both sides, 7 only in Python and 7 only in native, each
with a projection within 0.0003 pixels of a rounding boundary. Depth bounds per
view then differ by up to 3.6e-4 relative, because the reference takes them
from every seventh hull voxel and the lists are shifted against each other;
with identical hulls they agree to float32 rounding.

Mask repair and fusion from the same saved depths (`--reuse-depths`):

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| Repaired mask pixels that differ (of 147 M; pixels added by repair) | 19 (714 488) | 15 (731 575) | 187 (779 416) | 94 (509 033) |
| Hull voxels, Python / native | 13 560 339 / 13 560 383 | 5 628 084 / 5 628 109 | 6 785 506 / 6 785 672 | 1 794 135 / 1 794 147 |
| Voxels only in Python / only in native | 7 / 51 | 4 / 29 | 7 / 173 | 6 / 18 |
| `weight` identical on common voxels; largest difference | 99.992 %; 2 | 99.984 %; 1 | 99.979 %; 3 | 99.972 %; 2 |
| `total` absolute difference, p99.9 / maximum | 0.00028 / 2.0 | 0.00037 / 1.7 | 0.00027 / 3.0 | 0.00037 / 2.0 |
| Largest F1 difference after the Python mesher | 0.0003 | 0.0001 | 0.0010 | 0.0003 |

With identical masks (Bunny, repair off) fusion alone gives: `weight` identical
in 99.993 % of voxels, largest difference 1 (one nearest-pixel lookup or one
truncation threshold decided the other way; the median distance of those
voxels' nearest lookup from a rounding boundary is 0.00009 pixels), `total`
p99.9 0.00025. Support height and observed fraction agree to all printed digits.

Whole stage, each side with its own matching:

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| Consistent coverage per pass, Python | 0.554 / 0.820 / 0.810 / 0.710 | 0.499 / 0.773 / 0.776 / 0.703 | 0.633 / 0.753 / 0.765 / 0.648 | 0.600 / 0.794 / 0.734 / 0.585 |
| Consistent coverage per pass, native | 0.556 / 0.821 / 0.809 / 0.711 | 0.498 / 0.772 / 0.776 / 0.701 | 0.634 / 0.752 / 0.764 / 0.648 | 0.598 / 0.794 / 0.735 / 0.589 |
| Largest coverage difference | 0.0013 | 0.0019 | 0.0012 | 0.0042 |
| Hull-front pixels, Python / native | 933 637 / 933 652 | 924 985 / 918 072 | 790 169 / 787 561 | 283 605 / 286 843 |
| Final depth valid in both, of valid in Python (median over views) | 97.3 % | 95.4 % | 96.1 % | 95.6 % |
| Relative depth difference where both valid, median / p95 | 5.0e-5 / 1.0e-3 | 1.1e-4 / 1.4e-3 | 6.3e-5 / 1.5e-3 | 8.0e-5 / 1.4e-3 |
| F1 `all` at 0.5 / 1 / 2 %, Python | 0.905 / 0.941 / 0.955 | 0.922 / 0.971 / 0.987 | 0.772 / 0.926 / 0.981 | 0.789 / 0.903 / 0.938 |
| F1 `all` at 0.5 / 1 / 2 %, native | 0.905 / 0.941 / 0.956 | 0.923 / 0.971 / 0.987 | 0.770 / 0.927 / 0.981 | 0.789 / 0.904 / 0.938 |
| Largest F1 difference (`all` and `above_margin`) | 0.0007 | 0.0008 | 0.0020 | 0.0005 |
| Largest GPU binding | 110.6 MiB | 110.9 MiB | 111.3 MiB | 104.3 MiB |
| Stage time, Python / native | 588 s / 78 s | 475 s / 58 s | 505 s / 66 s | 306 s / 40 s |

On the synthetic sphere, where the hulls are identical, the two stages agree
far more closely: coverage per pass within 0.0001, depth valid in both for at
least 99.9 % of each view's pixels, relative depth difference median 9e-8 and
p99 4e-6. On real objects the hulls differ in a few dozen voxels, so depth
bounds and with them the sweep planes are not the same.

Time per step on the Bunny, seconds, Python / native: hull
33.5 / 1.6; two repair rounds with their hulls 127 / 8.3; level 256 sweep
79.5 / 6.5; level 512 first pass with hull-front candidate 135.9 / 14.7;
level 512 second pass 41.6 / 5.2; level 876 121.9 / 17.8 (of which 4.5 on the
CPU for the initial surfaces and 0.9 for agreement); fusion 20.8 / 0.9;
everything else (loading, sheets, writing the files) 27.8 / 23.2, of which the
native stage spent 0.8 on loading and 21.8 on writing `volume.npz` and
`depths.npz` with single-threaded deflate. Members are now deflated in 4 MiB
pieces on four threads (same level, files a few bytes per piece larger, read by
NumPy and any inflater); writing took 12.8 s in a later run on a machine loaded
by other jobs. Storing without compression was not chosen: `depths.npz` and
`volume.npz` would grow from about 150 MB to about 480 MB on the Bunny, and the
depth file is kept with every run.

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
| fs4 | MIT OR Apache-2.0 | free disk space before a run |
