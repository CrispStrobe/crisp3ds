# crisp3ds-dense: the native dense pipeline

A Rust port of `scripts/turntable_mesh`, so the reconstruction runs without
Python: GPU stages as WebGPU compute shaders through `wgpu` (Metal, Vulkan,
DirectX 12, and WebGPU in browsers), the rest as plain Rust. The Python package
stays the reference implementation until every stage here reproduces it.

Status: the whole pipeline from recovered cameras to a checked STL runs
natively, as one command (`run`), as a library call, or through a C interface.
Each stage is also a subcommand (`inputs`, `stereo`, `mesh`, `check`) that reads
and writes the files of its Python counterpart; measured parity is listed per
stage below. The step before it, from plain photos to the scene, is native too
(`photos`): its providers start AliceVision or COLMAP as external programs,
and SAM 2.1 remains an external PyTorch program. Not ported: the scan
evaluator, a development tool that stays in Python.

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
| `src/storage.rs` | Files on disk or in an in-memory tree (`mem:` paths; everything in a browser) |
| `web/` | Browser package: `wasm-bindgen` bindings, JavaScript wrapper, worker, test page and its headless test |
| `src/run.rs`, `src/control.rs` | The run driver (port of `dense_pipeline.py`): library API and `run` command; cancel flag, deadlines, stage logs |
| `src/capi.rs`, `include/crisp3ds_dense.h` | C interface of the run driver (feature `capi`) |
| `src/scene.rs` | Inputs directory from an AliceVision scene (port of `dense_all_views_inputs.py`) |
| `src/check.rs`, `src/render.rs` | Photo check and its sheets (ports of `mesh_photo_check.py`, `stl_compare_render.py`) |
| `src/inputs.rs` | Inputs directory as `Stereo.__init__` prepares it |
| `src/hull.rs`, `src/repair.rs`, `src/fusion.rs`, `src/stereo/` | Ports of `multiscale_stereo.py` |
| `src/photos/` | From photos to the scene: mask and camera providers, gates, undistortion (port of `photos_to_inputs.py`) |
| `tools/colmap_pycolmap.py` | Stand-in for the `colmap` executable through the PyCOLMAP wheel (development aid) |
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
`run_async` is the same as a future, for hosts that must not block.
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

## From photos to the scene: `crisp3ds-dense photos`

Native form of `python -m scripts.turntable_mesh.photos_to_inputs`, built
around providers (`docs/ARCHITECTURE.md`, `docs/PHOTOS-TO-INPUTS.md`): one for
the masks, one for the cameras; staging, hole cleanup, contrast images, gates,
undistortion, the scene, sheets and events are done in this crate for all of
them. No Python is needed with `--masks threshold` or `import:DIR` and
`--cameras alicevision` (install prefix), `colmap` (executable) or
`import:PATH`.

```sh
crisp3ds-dense photos --photos DIR --calibration lens.json --output DIR [--events events.jsonl] \
    --masks threshold|import:DIR|external-sam --cameras alicevision|colmap|import:PATH [options]
crisp3ds-dense photos --list-providers
```

`--list-providers` prints each provider's platforms (desktop, mobile, wasm),
the external programs it starts and its license as JSON. Same gates, exit
codes (0 complete, 2 cameras rejected, 1 a step failed), events (`masks`,
`cameras`; `mask_sheet`, `sparse_overlay`, `report`) and layout as the
reference, except that the scene's undistorted photos are in `inputs/images/`
(there is no `native-prepared/`) and that `prepareDenseScene` is not called.

| Step | Where (`src/photos/`) | Reference |
| --- | --- | --- |
| Capture order, `capture_NNNN.png` copies, coarse dark-object masks | `staging.rs`, `coarse.rs` | `step_coarse`, `coarse_mask`, `otsu_threshold`; components labelled like `scipy.ndimage.label` |
| Mask providers `threshold`, `import`, `external-sam` | `masks.rs` | `segment.py` is started as a child process for `external-sam` |
| Dark-hole cleanup, published masks, contact sheet | `cleanup.rs`, `sheets.rs` | `silhouette_cleanup.py`, `step_publish_masks` |
| Contrast images | `contrast.rs` | `contrast_image`: OpenCV's 8-bit RGB/Lab conversions and CLAHE, reimplemented |
| Lens file, scaling, locked AliceVision intrinsic | `calibration.rs` | `load_calibration`, `scale_calibration`, `calibrated_scene` |
| Camera providers `alicevision`, `colmap`, `import` | `cameras.rs`, `process.rs` | `build_commands`, `bounded`; COLMAP is new |
| Neutral solution; readers for `.sfm` and COLMAP models | `solution.rs` | |
| Audit, ring statistics, gates | `audit.rs`, `ring.rs` | `alicevision_cameras.py`, `ring_statistics`, `decide_gates` |
| Undistortion and scene | `scene_writer.rs` (map and mask remap from `src/scene.rs`) | `prepareDenseScene` and `dense_all_views_inputs.py` |
| Sparse overlay, stages, `frontend.json` | `sheets.rs`, `run.rs` | `step_overlay`, `run` |

Parity with the reference, measured on the 3DLF photos (73 per object,
1749 x 1155; OpenCV 4.10, SciPy, Pillow on the reference side):

- **Masks.** Coarse masks plus cleanup: identical to the Python functions in
  all 73 masks of the Bunny and of the Armadillo (0 of 147 M pixels).
  `masks-report.json` is identical on the Dragon and Lucy.
- **Contrast images.** Identical to `contrast_image` (gamma 0.5, CLAHE 2.0 on
  an 8 x 8 grid) in all 292 photos of the four objects: 0 of 442 M channel
  values per object. The two colour conversions are identical to
  `cv2.cvtColor` for all 2^24 input triples in both directions
  (`tests/fixtures/dense-native/make_photos_fixtures.py --exhaustive`). They are
  OpenCV's integer algorithms (`RGB2Lab_b`, `Lab2RGBinteger`) with tables
  computed here; one entry of the cube-root table (324) lands on a rounding tie
  in single precision that OpenCV resolves the other way and is set explicitly.
- **Audit and gates.** `camera-audit.json`, `ring-sanity.json` and `gates.json`
  of the Dragon and Lucy scenes agree with the files the Python stage wrote to
  2.6e-13 relative in every number; the sparse-overlay fractions are equal.
- **Scene.** Importing the existing `final.sfm` of the four objects reproduces
  the existing inputs exactly: cameras (`k`, rotation, translation), sparse
  points and all remapped masks.
- **Undistorted photos** against AliceVision's `prepareDenseScene` (13 photos
  per object): mean absolute difference 0.007 to 0.010 grey levels, 99 % of the
  values identical, at most 0.06 % differ by more than one level, largest
  difference 13 to 17 at dark-to-light edges. Both are bilinear; AliceVision
  converts to linear light first, interpolates in floating point and converts
  back, this crate interpolates the 8-bit values. Bunny with the same cameras
  and masks, native dense stages: F1 `all` 0.9049 / 0.9404 / 0.9554 against
  0.9049 / 0.9412 / 0.9560 with AliceVision's images, `above_margin` 0.9641 /
  0.9948 / 0.9998 against 0.9629 / 0.9946 / 0.9998 (largest difference 0.0012).
- **Sheets** are pictures for people: tiles are reduced with the crate's
  bicubic resampler (Pillow Lanczos, OpenCV bilinear in the reference) and
  labelled with the built-in font, which has no `_` or `.`.

Time for everything that is not a provider, Bunny, Python / native: 70 s / 7 s
(coarse masks 7.1 / 0.7, cleanup 14.2 / 1.4, masks and sheet 4.6 / 1.0,
contrast images 12.6 / 1.7, audit 2.0 / 0.02, undistortion and scene 27.3
(`prepareDenseScene`, check, inputs) / 1.3, overlay 2.0 / 0.4; 4 threads).

### Masks without a network

Scanner F1 (`scan_evaluate.py`, evaluation only) of the native dense stages on
three mask sets with the same cameras: SAM 2.1 masks (the existing ones), the
`threshold` provider at the fixed level 70 and at Otsu's level per photo
(104 to 125 on these photos). `all` / `above_margin` at 0.5 % of the diagonal;
in brackets at 1 % and 2 %.

| | SAM 2.1 | threshold 70 | threshold Otsu |
| --- | --- | --- | --- |
| Bunny `all` | 0.905 (0.941, 0.956) | 0.892 (0.934, 0.952) | 0.891 (0.925, 0.947) |
| Bunny `above_margin` | 0.963 (0.995, 1.000) | 0.957 (0.995, 1.000) | 0.962 (0.994, 1.000) |
| Armadillo `all` | 0.923 (0.971, 0.987) | 0.901 (0.962, 0.984) | 0.907 (0.957, 0.972) |
| Armadillo `above_margin` | 0.952 (0.996, 1.000) | 0.930 (0.989, 0.999) | 0.950 (0.997, 1.000) |
| Dragon `all` | 0.785 (0.928, 0.981) | 0.786 (0.914, 0.972) | 0.766 (0.895, 0.956) |
| Dragon `above_margin` | 0.834 (0.963, 0.994) | 0.851 (0.969, 0.995) | 0.848 (0.967, 0.994) |
| Lucy `all` | 0.790 (0.904, 0.938) | 0.807 (0.923, 0.956) | 0.818 (0.936, 0.962) |
| Lucy `above_margin` | 0.826 (0.945, 0.970) | 0.845 (0.967, 0.991) | 0.862 (0.985, 1.000) |
| Silhouette IoU against own masks (median), B / A / D / L | 0.967 / 0.935 / 0.949 / 0.881 | 0.951 / 0.935 / 0.947 / 0.909 | 0.967 / 0.942 / 0.951 / 0.928 |
| Genus, B / A / D / L | 11 / 2 / 22 / 14 | 4 / 19 / 12 / 3 | 16 / 4 / 24 / 3 |

Reading, with the preview sheets: at level 70 lit upward faces are brighter
than the threshold and are cut away by the silhouette hull (the top of the
Armadillo's shell is sliced off; genus 19). Otsu's level includes them and
brings the Bunny and the Armadillo to SAM's score above the support (within
0.002), and the Dragon and Lucy above it (SAM drops horns, hands and wing
tips in runs of photos; the threshold keeps them). The price is at the base:
the contact shadow is dark, belongs to the largest dark region and becomes a
ragged skirt around the feet, which costs 0.014 to 0.019 in `all` on three
objects. Thin parts are no worse than with SAM on any sheet. Because of the
base, `external-sam` remains the default mask provider and `threshold` (Otsu)
is the provider without a network.

### Camera providers through the dense stages

A provider is judged by the reconstruction it leads to. `colmap` against
`alicevision`, same SAM masks (imported), native dense stages, scanner F1 at
0.5 % (1 %, 2 %). COLMAP ran through the PyCOLMAP 3.11 stand-in
(`tools/colmap_pycolmap.py`): default SIFT, exhaustive matching, masks on, one
shared `FULL_OPENCV` camera fixed to the declared lens.

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| Registered, `colmap` | 73 of 73 | 73 of 73 | 73 of 73 | 73 of 73 |
| Sparse points, `alicevision` / `colmap` | 9 087 / 2 813 | 6 677 / 5 240 | 5 619 / 3 360 | 3 266 / 2 143 |
| Reprojection median, p95 (px), `alicevision` | 0.42, 1.45 | not recorded | 0.52, 1.88 | 0.63, 2.31 |
| Reprojection median, p95 (px), `colmap` | 0.44, 1.92 | 0.42, 1.82 | 0.54, 2.16 | 0.61, 2.33 |
| Ring radius spread, `colmap` | 0.75 % | 0.44 % | 0.38 % | 0.79 % |
| Camera recovery (features, matching, SfM), `alicevision` | 1 021 s | not recorded | 494 s | 297 s |
| Camera recovery, `colmap` | 64 s | 105 s | 182 s | 133 s |
| F1 `all`, `alicevision` | 0.905 (0.941, 0.956) | 0.923 (0.971, 0.987) | 0.785 (0.928, 0.981) | 0.790 (0.904, 0.938) |
| F1 `all`, `colmap` | 0.905 (0.941, 0.956) | 0.924 (0.971, 0.987) | 0.795 (0.927, 0.982) | 0.828 (0.945, 0.968) |
| F1 `above_margin`, `alicevision` | 0.963 (0.995, 1.000) | 0.952 (0.996, 1.000) | 0.834 (0.963, 0.994) | 0.826 (0.945, 0.970) |
| F1 `above_margin`, `colmap` | 0.965 (0.996, 1.000) | 0.952 (0.996, 1.000) | 0.844 (0.961, 0.994) | 0.865 (0.986, 0.999) |

The AliceVision rows are the existing camera solutions with the native dense
stages on the existing inputs; their times are those of the Python stage's
runs (the Bunny's on a busier machine). COLMAP times were measured while
another job used the machine, with 4 threads against AliceVision's 2. One run
per object: neither program is repeatable, and the second Bunny run with
COLMAP that was planned was not made (the disk reached its floor). Not tried:
`--colmap-masks off`, sequential or ring matching, the COLMAP executable.

### The Bunny from its photos

`crisp3ds-dense photos` on the 73 Bunny photos, then the native dense stages
and the scanner evaluator. AliceVision 3.4 local build, 2 threads.

| | Python stage (run 220) | native, prefix called directly, `--masks threshold` | native, `av.py` wrapper, `--masks external-sam` |
| --- | --- | --- | --- |
| Masks | SAM 2.1 | Otsu threshold, median area 321 051 px | SAM 2.1 through the reference script: identical to the existing masks (0 of 147 M pixels) |
| Registered | 73 of 73 | 73 of 73 | 73 of 73 |
| Sparse points, observations | 8 638, 43 205 | 8 581, 42 690 | 8 788, 44 483 |
| Reprojection median, p95, max (px) | 0.416, 1.444, 18.6 | 0.414, 1.436, 17.7 | 0.423, 1.482, 28.6 |
| Ring radius spread, out of plane, largest gap | 0.48 %, 0.41 %, 6.08 deg | 0.48 %, 0.61 %, 6.12 deg | 0.41 %, 0.53 %, 6.07 deg |
| Duplicate pose pair reported | 13, 14 | 13, 14 | 13, 14 |
| Seconds: masks (coarse, SAM, cleanup, sheet) | 7.1, 81.8, 14.2, 4.6 | 2.2, none, 3.4, 1.8 | 1.9, 74.5, 3.4, 1.5 |
| Seconds: contrast images | 12.6 | 9.0 | 4.4 |
| Seconds: features, matching, global SfM | 152.8, 549.6, 318.4 | 262.5, 595.9, 318.4 | 104.6, 348.8, 266.5 |
| Seconds: audit; undistortion and scene; overlay | 2.0; 27.3; 2.0 | 0.02; 2.5; 0.5 | stopped before the audit |
| Gates | passed | passed | not reached: the run stopped at the free-space floor (7.4 GiB free, floor 8); the numbers above are from the reference audit of its `final.sfm` |
| F1 `all` at 0.5 / 1 / 2 % | 0.905 / 0.941 / 0.956 (cameras of run 218) | 0.896 / 0.929 / 0.947 | not run |
| F1 `above_margin` | 0.963 / 0.995 / 1.000 | 0.965 / 0.995 / 1.000 | not run |

The machine was shared with other jobs during both native runs, so the times
of the external steps say little; the steps done in this crate are 5 to 10
times faster than their Python counterparts. Command line of the middle
column:

```sh
crisp3ds-dense photos --photos <bunny>/rgb --calibration scripts/turntable_mesh/calibrations/3dlf-pro.json \
    --output <run>/frontend --events <run>/events.jsonl --threads 2 \
    --masks threshold --cameras alicevision \
    --alicevision .local-tools/alicevision-local/prefix --alicevision-library-path /opt/homebrew/lib
```

### Tests

`cargo test` covers calibration scaling, envelope and threshold logic,
component labelling, cleanup rules, CLAHE and the colour conversions against
values written by OpenCV (`tests/fixtures/dense-native/opencv-contrast.json`,
produced by `make_photos_fixtures.py` next to it), the audit, ring statistics
and gate decisions (the cases of `test_photos_to_inputs.py`,
`test_silhouette_cleanup.py` and `test_alicevision_cameras.py`), the readers
for `.sfm` and COLMAP text and binary models, command lines of the three
external programs, bounded processes (deadline, cancel, process group), and the
whole stage on a synthetic capture with threshold masks and imported cameras.
`CRISP3DS_ALICEVISION_TESTS=1` with `CRISP3DS_ALICEVISION` (and, for a build
against Homebrew, `CRISP3DS_ALICEVISION_LIBRARY_PATH=/opt/homebrew/lib`) adds a
test that starts the real `cameraInit`.

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

A later native run of the whole pipeline on the Bunny without live previews
(load average of the shared machine 6 to 7) took 51.8 s: loading 0.5, hull 1.1,
repair with its hulls 6.0, level 256 4.5, level 512 first pass 10.0, second pass
3.1, level 876 10.3, fusion 0.8, writing the files 7.3, the rest of the stereo
stage (sheets) 0.8, surface 4.8, check 2.4.

## Portability

Measured parity is from Metal on an Apple M1. Beyond that, the CI workflow runs
the six GPU tests (every kernel against its scalar implementation, the stereo
stage and the whole `run` on the analytic sphere, including cancellation) on
two software WebGPU adapters, with the same default limits: Mesa llvmpipe
through Vulkan on Linux (12 s) and the Microsoft Basic Render Driver (WARP)
through DirectX 12 on Windows (62 s). Both pass. The jobs are marked
non-blocking because software rasterisers on shared runners can be missing or
slow. No real Vulkan or DirectX GPU and no browser has run the kernels yet, and
no real object has been reconstructed on anything but Metal.

The same code runs in a browser; see the next section.

## In the browser: `crates/dense/web`

The dense stages (`stereo`, `mesh`, `check`, and `inputs` from a scene) as
WebAssembly with the kernels on WebGPU. Nothing in the library blocks or needs a
disk there:

- **Storage** (`src/storage.rs`). Every stage reads and writes through one
  module. A path whose first component is `mem:` names a file in an in-memory
  tree on any platform; in a browser every path does. The layout of a run
  directory and the events are the same. A host can also register a source that
  hands over files it keeps elsewhere (the browser package keeps the photos in
  JavaScript memory and passes them in one at a time).
- **GPU** (`src/gpu/`). Device creation, kernel compilation, dispatch and
  readback are `async`. Native callers drive them with a blocking executor
  (`run`, `stereo::run::run`); a browser awaits `run_async`. In a browser,
  errors of dispatches are collected by the device's error callback and
  reported at the next readback, because awaiting an error scope per dispatch
  costs a round trip through the event loop each time.
- **Threads and time.** On wasm32 the CPU passes are plain loops, preview
  volumes are meshed on the spot instead of on a second thread, and time comes
  from `web-time`. The thread returns to its event loop at every readback, once
  per view while matching, so events are delivered and a cancel takes effect
  there; the CPU passes in between do not yield. Run the engine in a worker.

Build and test (the native build has no wasm dependencies; the bindings are a
crate of their own):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
sh crates/dense/web/build.sh                       # writes web/pkg (2.5 MB of WebAssembly)
cargo run --release -- synthetic --output web/scene # from crates/dense: a small test scene
cd crates/dense/web && npm install && npx playwright install chromium
node test/run.mjs --scene scene --output browser-run \
    --options '{"overrides":["sizes=64,128","grid=96","planes=48","neighbours=4","best_of=2","vote_neighbours=4","min_votes=2,2","crop_padding=6","hull_dilate=1","windows=5,7","aggregates=1,1"]}'
```

`index.html` with `main.js` and `worker.js` is a test page without a framework:
it fetches `files.json` and the files it lists, runs them in a worker and shows
status text and the final triangle count. `test/run.mjs` serves an inputs
directory to that page in headless Chromium and writes the events, the kept
output files, the adapter, the limits and memory figures; `--software` asks
Chromium for its software adapter (SwiftShader).

```js
import { createRun, settingsSchema } from "./crisp3ds-dense.js";
const run = await createRun({
  files,                                  // Map: path in the inputs directory -> Uint8Array
  options: { settings: { grid: 320 } },   // fields of RunOptions; output and inputs are filled in
  onEvent: (event) => { /* the objects of events.jsonl */ },
});
const report = await run.finished;        // pipeline.json; rejects on failure or cancel
const stl = run.file("mesh/mesh.stl");    // any file of the run directory, e.g. an artifact's path
run.cancel(); run.dispose();
```

The bindings underneath (`web/src/lib.rs`): `Run` with `start(options, onEvent)`
and `cancel()`, `putFile`, `getFile`, `listFiles`, `removeTree`,
`setFileSource`, `settingsSchema`, `defaultSettings`, `version`.

Measured in headless Chromium 153 with WebGPU on an Apple M1. The adapter
reports vendor `apple`, architecture `metal-3`, not a fallback adapter. The
device is created with the default limits (128 MiB per storage binding, 256 MiB
per buffer, 8 storage buffers per stage); the largest binding was 110.6 MiB and
no limit had to be raised. Native numbers are from the same machine.

| | Synthetic sphere (small settings) | Bunny, 73 photos, default settings, no live previews |
| --- | --- | --- |
| Completes | yes | yes |
| Stage sequence, artifact kinds and counts, metric names against native | same | same |
| Triangles, native / browser | 33 792 / 33 792 | 1 047 200 / 1 047 356 |
| Closed, genus | yes, 0 | yes, 11 (native 11) |
| Consistent coverage per pass, largest difference | 0 | 0.0005 |
| Depth valid in both, of valid in either, per view: median / minimum | 100 % / 99.8 % | 99.6 % / 99.0 % |
| Relative depth difference where both valid: median / p95 / p99 | 0 / 4.7e-7 / 1.8e-6 | 2.9e-7 / 1.8e-5 / 2.6e-4 |
| Silhouette IoU median, native / browser | 0.92617 / 0.92617 | 0.9667 / 0.9668 |
| Run time, native / browser | 1.7 s / 2.5 s | 52 s / 143 s |
| Peak WebAssembly memory | 47 MiB | 1921 MiB |
| Peak resident memory: renderer / GPU process | 277 / 198 MiB | 2649 / 841 MiB |

Scanner F1 of the browser's Bunny mesh (Python evaluator, evaluation only):
`above_margin` 0.963 / 0.995 / 1.000, the same as native; `all` 0.905 / 0.941 /
0.956. With live previews the Bunny also completes (186 s) but peaks at 2539 MiB
of WebAssembly memory, because each preview volume is written and meshed while
the matching data is held; for megapixel sets in a browser pass
`live_previews: false`. Where the browser's time goes on the Bunny (seconds,
native / browser): hull 1.1 / 3.8, repair with its hulls 6.0 / 13.1, the four
matching passes 4.5, 10.0, 3.1, 10.3 / 7.4, 23.0, 8.6, 36.5 (of the last, 23.5
are the single-threaded initial surfaces), writing the compressed files 7.3 /
19.8, surface 4.8 / 12.3, check 2.4 / 7.7. The synthetic scene also completes on
Chromium's software adapter (vendor `google`, architecture `swiftshader`,
fallback) in 7.9 s with the same mesh.

What a production browser build still lacks:

- **Threads.** Everything on the CPU is single-threaded (initial surfaces,
  agreement, deflate, the surface stage); a thread pool on shared memory needs
  cross-origin isolation and was left out of the default build.
- **Memory.** WebAssembly memory is limited to 4 GiB and never shrinks. The
  Bunny (73 photos of 2 megapixels, hull grid 400) peaks at 1.9 GiB without
  live previews and 2.5 GiB with them; before the inputs were left with the
  host, the grey images cropped to the canvas after mask repair and released
  before fusion, it was 2.9 and 3.2 GiB. Larger sets need streamed fusion
  output and views decoded on demand. A level whose cost volume exceeds 112 MiB
  is refused (see above), as natively.
- **Persistence.** Outputs live in memory until the host takes them; nothing is
  written to the origin-private file system, and a run cannot be resumed.
- **Photo front end.** Masks and cameras from plain photos need external
  programs, which a browser build leaves out (see `src/photos`).
- **Coverage.** One browser engine (Chromium), one GPU (Apple M1 through
  Metal) and SwiftShader; no Firefox or Safari, no Windows or Linux GPU.

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
| fs4 | MIT OR Apache-2.0 | free disk space before a run (not on wasm32) |
| web-time | MIT OR Apache-2.0 | clock that also works in a browser |
| libc (Unix only) | MIT OR Apache-2.0 | signalling the process group of an external tool (`src/photos/process.rs`) |

Browser package only (`web/`): wasm-bindgen, wasm-bindgen-futures and js-sys
(MIT OR Apache-2.0) for the bindings, console_error_panic_hook (Apache-2.0 OR
MIT) to show panics in the console; tools wasm-bindgen-cli (MIT OR Apache-2.0)
to generate the package and, for the test page only, Playwright (Apache-2.0)
with its Chromium build.
