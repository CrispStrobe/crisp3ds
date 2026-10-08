# Status

State of Crisp3DS / Crisp 3D Studio on 2026-10-08. The plan of record is
[ARCHITECTURE.md](ARCHITECTURE.md); measurements and settings are in the
[crate README](../crates/dense/README.md).

## What works

| Area | State |
| --- | --- |
| **Photos to STL, one command** | `crisp3ds-dense run --photos DIR --calibration JSON --output RUN`: threshold masks, our own turntable camera solver, GPU dense stages, closed STL. No Python, no external program. About 50–100 s per 73-photo object on an Apple M1 |
| **Native crate** (`crates/dense`, Rust + wgpu) | The primary implementation. Metal, Vulkan, DirectX 12 and WebGPU; library API, C interface, event log contract; Python package kept as the reference it was ported from |
| **Browser** | The same pipeline as WebAssembly + WebGPU, from photos or from prepared inputs, single-threaded or threaded (cross-origin isolated pages). Default Bunny from 73 photos in Chrome: measured 130–264 s with 4 threads, peak WebAssembly memory about 1.6–2.1 GiB. Live at <https://crispstrobe.github.io/crisp3ds/> |
| **Crisp 3D Studio** (app) | Desktop (macOS, Windows, Linux), iOS and Android from one code base (Tauri 2); the engine runs in-process. Start from photos with provider choice, live progress, preview surfaces, diagnostic sheets, per-step inspection sheets, example objects downloaded on request. Release [v0.2.0](https://github.com/CrispStrobe/crisp3ds/releases/tag/v0.2.0) (prerelease, unsigned desktop builds); internal TestFlight builds for iOS and macOS |
| **Masks** | `threshold` (default; contact shadow removed), `import`, `sam` (SAM 2.1 natively through ONNX Runtime or CrispEmbed's ggml engine, optional build feature), `external-sam` (PyTorch) |
| **Cameras** | `turntable` (default; our own solver, pure Rust, repeatable), `markers` (printed mat, gives scale and handedness; validated on renders only), `colmap`, `alicevision`, `import`; `--capture orbit` for camera paths that are not one ring |
| **Evaluation** | Scanner F1 against independent scans (3DLF), DTU protocol, Google Scanned Objects rendered by our own renderer (`crisp3ds-dense render`), per-step inspection sheets in every run |

## Measured results (default command, from photos)

F1 at 0.5 % of the scan diagonal, whole surface / above the support; scans used
for scoring only. Exact numbers per threshold and per commit are in the crate
README.

| Object (3DLF, 73 photos) | F1 at 0.5 % |
| --- | --- |
| Bunny | about 0.90 / 0.96 |
| Armadillo | about 0.92 / 0.95 |
| Dragon | about 0.80 / 0.85 |
| Lucy | about 0.85 / 0.88 |
| Thai statue | about 0.88 / 0.95 |
| Happy Buddha | about 0.79 / 0.83 |

| Other sets | Result |
| --- | --- |
| Google Scanned Objects renders (72 photos) | rhino 0.867, cereal box 0.833 at 0.5 % with threshold masks; light-coloured objects need better masks (Mario 0.71 → 0.83 with exact masks) |
| DTU historical dense-stage diagnostics (49 views, supplied masks; supplied or COLMAP poses) | 1.03–1.31 mm mean of accuracy and completeness on scans 63 and 65; these are not photo-only pipeline results |
| YCB turntable | cracker box 0.45 at 0.5 %; smooth objects (mustard bottle) refused by the camera gates |

## Findings worth knowing

- The 3DLF scans are mirror images of the photographed objects; our
  reconstructions have correct handedness (shown with an asymmetric synthetic
  object with known truth).
- PyTorch 2.7 on Apple MPS computes SAM 2.1's Hiera encoder wrongly (a strided
  `max_pool2d`); masks made that way before the fix were partly wrong.
- With correct arithmetic, SAM masks do not beat the threshold masks on this
  capture style; threshold is the default.
- The 3DLF light-field photos are soft (most image energy below an eighth of
  the sampling frequency). Fine relief is nevertheless visible in them and is
  poorly recovered. Per-pixel slanted-plane matching (PatchMatch-style,
  now excluded from ordinary builds pending patent review) reduces noise and improves some folds, but
  close-ups still show weak Bunny eyes, nose and head relief. Whole-object F1
  gains do not establish recovery of those features. Follow-up matching,
  fusion and smoothing experiments are recorded in
  [bunny-head-detail-review.json](../tests/evidence/bunny-head-detail-review.json);
  none was adopted as a default.
- At 10° steps (36 photos) results are almost as good as at 5°; at 15° steps
  low-texture objects fail camera recovery.

## In progress

- **Photo-textured GLB**: opt-in `run --texture` and standalone `texture`
  command, also offered in Studio for built-in engines. An embedded UV atlas
  uses the pipeline's own recovered cameras, photos and masks; unseen regions
  are grey. Geometry is preserved. Initial view-coherence/visibility checks
  reduce patchwork, but seams and captured-lighting differences remain.
  Browser package builds; real browser execution/memory and device sharing
  still need validation. Fresh Rhino: 72/72 photos, 96 s including initial
  texturing; mustard stops before stereo because threshold masks select the
  undistortion border. [Usage and limits](TEXTURED-MESH.md),
  [photo-only evidence](../tests/evidence/photo-texture-review.json).

- **Facial detail and source review**: optional neighboring-plane PatchMatch
  is excluded from ordinary engine builds and settings forms. The explicit
  `research-patchmatch` Cargo feature retains it for research, without granting
  patent rights. A targeted [source and patent screen](../tests/evidence/mvs-source-review.json)
  records the relevant claims and the limits of the review; it is not legal
  clearance. The default band pipeline remains unchanged.
  Independent per-pixel plane refinement is implemented as an experiment
  (`slanted_refine=false` by default), with no neighboring-plane candidates or
  random perturbation. Bunny eye-region F1 at 0.1% of the object diagonal
  rises from 0.579 to 0.593, below the earlier research PatchMatch result of
  0.609. Close-ups still show weak eyes and nose. Supplied scanner geometry is
  used only for posthoc evaluation, never reconstruction.
  Optional subpixel fusion (`fusion_interpolate=false`) raises the combined
  local eye-region F1 to 0.601. Neither experiment has passed the six-object
  adoption gate; both stay off. [Evidence and limits](../tests/evidence/bunny-independent-detail-review.json).
- **Example objects in the app** from several sources (CC BY renders first).
  The rhino passed the sandboxed Mac app check (72 registered photos).
  Screenshot capture now selects the viewer's upright direction for this
  object; the corrected iOS/Mac screenshot dry run and Pages check remain
  pending CI. No screenshots have been uploaded to App Store Connect.

## Known limits

- Masks assume a dark object on a light backdrop; light-coloured objects and
  dark backdrops need SAM or imported masks. Threshold masks covering over
  90% of every photo are refused before camera recovery, with a mask sheet
  kept for inspection.
- Interiors a single ring of cameras never sees (mugs, shoes) are capped.
- Thin, low-texture parts and very smooth objects remain hard.
- The marker mat has not been tested with a real print.
- Builds have been run on Apple hardware only (desktop app, iOS simulator,
  browser); Windows, Linux and Android builds are tested in CI only, and no
  real iPhone or iPad has run the app yet.

## Data published for this project

| Repository | Content | License |
| --- | --- | --- |
| [cstr/3dlf-scan-photos](https://huggingface.co/datasets/cstr/3dlf-scan-photos) | 3DLF-Scan turntable photos of seven objects (the app's example objects) | CC BY 4.0 |
| [cstr/3dlf-scan](https://huggingface.co/datasets/cstr/3dlf-scan) | the full 3DLF-Scan release, extracted (without the canonical Stanford model files) | CC BY 4.0 |
| [cstr/gso-turntable-photos](https://huggingface.co/datasets/cstr/gso-turntable-photos) | rendered turntable photos of Google Scanned Objects | CC BY 4.0 |
| [cstr/sam2.1-hiera-tiny-ONNX](https://huggingface.co/cstr/sam2.1-hiera-tiny-ONNX) | SAM 2.1 Hiera-tiny, ONNX export | Apache-2.0 |
| [cstr/sam2.1-hiera-tiny-GGUF](https://huggingface.co/cstr/sam2.1-hiera-tiny-GGUF) | SAM 2.1 Hiera-tiny for CrispEmbed's ggml engine (F16, F32) | Apache-2.0 |
