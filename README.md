# Crisp3DS

Photos of an object in, a closed printable mesh out. Crisp3DS is a rigid-object
photo scanning project: a dense reconstruction pipeline that runs on an Apple
GPU, an NVIDIA GPU or CPU, an engine that reports its progress live, a
cross-platform front end, and a portable C++20 core for projects, calibration
and marker geometry.

[![dense pipeline](https://github.com/CrispStrobe/crisp3ds/actions/workflows/dense-pipeline.yml/badge.svg)](https://github.com/CrispStrobe/crisp3ds/actions/workflows/dense-pipeline.yml)
[![studio](https://github.com/CrispStrobe/crisp3ds/actions/workflows/studio.yml/badge.svg)](https://github.com/CrispStrobe/crisp3ds/actions/workflows/studio.yml)
[![foundation](https://github.com/CrispStrobe/crisp3ds/actions/workflows/foundation.yml/badge.svg)](https://github.com/CrispStrobe/crisp3ds/actions/workflows/foundation.yml)

## What is here

| Part | Where | State |
| --- | --- | --- |
| **Dense pipeline**: calibrated turntable photos to a closed STL | [`scripts/turntable_mesh/`](scripts/turntable_mesh/README.md) | Works from the command line; scored against independent scans on two objects |
| **Native dense pipeline**: the same stages in Rust and WebGPU, without Python | [`crates/dense/`](crates/dense/README.md) | Stereo, surface, check and a single `run` command; scanner scores within 0.002 of the Python reference on four objects, about seven times faster |
| **Engine**: progress events, live preview meshes, HTTP API, replay bundles | [`docs/ENGINE-CONTRACT.md`](docs/ENGINE-CONTRACT.md) | Works; development-grade server (no TLS, no accounts) |
| **Studio**: the front end, one code base for desktop, phone and web | [`apps/studio/`](apps/studio/README.md) | Web app and desktop app work against an engine and in replay; unsigned builds for macOS, Windows, Linux, Android and the iOS simulator come from CI ([releasing](docs/RELEASING.md)) |
| **Core**: C++20 library and CLI for projects, calibration, marker boards | [`core/`](core/README.md) | Builds and tests on macOS, Linux, Windows |
| **Project workspace**: the earlier Tauri desktop app around the core | [`apps/desktop/`](apps/desktop/README.md) | Project and calibration tools; not yet connected to the dense pipeline |

The pipeline is modular: masks, cameras and undistortion are provider-based
stages in front of one native dense reconstruction, so that each platform uses
the providers it can run. The plan and its current state are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Today only desktop-class
machines run the reconstruction; phones and browsers are clients of a machine
that does, or play back recorded runs.

## From photos to a mesh

The native program (`crates/dense`, Rust and WebGPU) goes from a folder of
turntable photos and a lens calibration to a closed STL in one command, with
no Python and no other program:

```sh
cargo build --release -p crisp3ds-dense
crisp3ds-dense run --photos data/bunny/rgb \
  --calibration scripts/turntable_mesh/calibrations/3dlf-pro.json \
  --masks threshold --output runs/bunny
```

Cameras come from our own turntable solver (the default of `--cameras`).
`--masks threshold` has to be given: the default mask provider is still
`external-sam`, which starts SAM 2.1 in Python and is better at the object's
base. On the 3DLF Bunny (73 photos, Apple M1 with 16 GB) the command takes
93 seconds and the STL scores 0.967 / 0.997 / 1.000 above the support against
an independent scan; the Armadillo, Dragon and Lucy take 83, 78 and 50
seconds. Numbers per stage are in the [crate README](crates/dense/README.md)
and the solver in [`docs/TURNTABLE-SOLVER.md`](docs/TURNTABLE-SOLVER.md).

The Python package is the reference the native program was ported from:

```sh
export PYTHONPATH=$PWD
pip install -r scripts/turntable_mesh/requirements-dense.txt

# no dataset needed: an analytic test scene, about a minute on CPU
python -m scripts.turntable_mesh.synthetic_scene --output /tmp/sphere
python -m scripts.turntable_mesh.dense_pipeline --inputs /tmp/sphere --output /tmp/sphere-run --device cpu \
  --set sizes=64,128 --set grid=96 --set planes=48 --set neighbours=4 --set best_of=2 \
  --set vote_neighbours=4 --set min_votes=2,2 --set crop_padding=6 --set hull_dilate=1 \
  --set windows=5,7 --set aggregates=1,1

# a real photo set with recovered cameras and masks
python -m scripts.turntable_mesh.dense_pipeline --scene final.sfm --prepared undistorted/ --raw-masks masks/ \
  --output runs/my-object --device mps        # or cuda, or cpu
```

The pipeline repairs masks from multi-view consensus, builds a silhouette hull,
matches all views coarse to fine, fuses depth into a signed distance volume
inside the hull and extracts a closed surface. Inputs, every setting, the
design reasons and all measurements are in the
[pipeline README](scripts/turntable_mesh/README.md).

With AliceVision and SAM 2.1 installed, the same command starts from a plain
folder of turntable photos and a lens calibration (`--photos`, `--calibration`)
and recovers masks and cameras first; see
[`docs/PHOTOS-TO-INPUTS.md`](docs/PHOTOS-TO-INPUTS.md).

## Watching a run, and the front end

Every run writes an event log and, while it is still matching, coarse preview
meshes and diagnostic sheets (mask repair, hull against masks, depth per
level, mesh outline against the photos). A front end can follow that locally,
over HTTP, or from a recorded bundle:

```sh
python -m scripts.turntable_mesh.engine_server --runs runs/ --data data/ --static apps/studio/dist
```

```sh
cd apps/studio && npm ci && npm run dev     # opens with a recorded demo run, no engine needed
```

Studio shows the stage timeline, the surface improving step by step in a 3D
viewer, the diagnostics gallery, numbers and reports, and can start and cancel
runs on an engine. See [`apps/studio/README.md`](apps/studio/README.md) and the
[engine contract](docs/ENGINE-CONTRACT.md).

## Results

3DLF turntable sets: 73 photos at one elevation, Apple M1 with 16 GB, PyTorch
MPS. Scores are held-out F1 against independent structured-light scans at
0.5%, 1% and 2% of the scan's diagonal; the scans are used for scoring only.

| Object | Matching time | F1, whole surface | F1, above the support |
| --- | --- | --- | --- |
| Dragon, earlier 24-view route | about 2 min | 0.567 / 0.780 / 0.934 | 0.624 / 0.845 / 0.972 |
| Dragon, this pipeline | about 7 min | 0.756 / 0.910 / 0.975 | 0.822 / 0.960 / 0.994 |
| Armadillo, no retuning | about 7 min | 0.922 / 0.971 / 0.987 | 0.952 / 0.997 / 1.000 |
| Bunny, no retuning | about 7 min | 0.898 / 0.933 / 0.953 | 0.964 / 0.995 / 1.000 |
| Lucy, from plain photos in one command | about 15 min in all | 0.798 / 0.908 / 0.940 | 0.836 / 0.951 / 0.973 |

Known limits, in short (details in the pipeline README):

- The handedness of the results against the dataset's scans is unresolved:
  they match only as mirror images, on both objects.
- Thin parts such as horn tips and wings come out short or are lost, and
  surfaces no photo sees are silhouette bounds.
- Results use all 73 photos; reconstruction from a dozen photos with unknown
  poses is not solved.
- The NVIDIA path was checked on a synthetic scene only (Tesla T4).
- The meshes have no physical scale.

## Continuous integration

GitHub Actions build and test on macOS, Linux and Windows: the Python pipeline
(lint, unit tests, an end-to-end run through the CLI and through the HTTP
engine), the C++ core, Studio (type check, tests, build) and the Tauri
project workspace. There is no GPU in CI. See [`docs/CI.md`](docs/CI.md).

## Earlier engine comparisons

The [comparative benchmark](docs/BENCHMARK-RESULTS.md) now evaluates MVE and
[COLMAP/OpenMVS](docs/CLASSICAL-BACKEND.md) on real object photographs and a
separate upstream software control. It reports surface accuracy/completeness,
normal agreement, topology, camera diagnostics, resource limits and failed runs;
software-oracle agreement is not physical ground truth.
The latest [fresh 60-photo M1 oracle](docs/TURNTABLE-COMPLETE-PLAN.md#fresh-complete-openmvs-oracle-and-shape-result)
does finish from JPEGs through a textured OpenMVS mesh, but its shape is
**rejected** (42.23% reference-fitted F1@1% scanner diagonal; 30.92% in a
camera-transported common gauge). OpenMVS is AGPL evaluation software here,
not an app-integrated or approved commercial/App Store backend. A separate
[AliceVision-Mac audit](docs/ALICEVISION-MAC-ORACLE.md) ran an isolated Metal
depth test, not a complete AliceVision reconstruction.

## Project workspace (`apps/desktop`)

Requires Node 22.18+ (Node 24 recommended) and npm.

```sh
cd apps/desktop
npm ci
npm run dev
```

Open the local URL printed by Vite. Create a project or import a manifest from `tests/contracts/`. The example calibration is synthetic, and no photographs accompany those fixtures. Image entries are relative paths, not uploaded image data. Download the project JSON to keep a portable copy.

For the native desktop shell, install the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/), then run `npm run tauri -- dev` in `apps/desktop`. The shell currently exposes capability status only; desktop worker integration is a later milestone. Installer bundling is disabled until packaging and dependency review are complete.

During initial development a Rust toolchain was installed locally in ignored `.local-tools/` without modifying the shell profile. If using that installation, run from the repository root before native commands:

```sh
export CARGO_HOME="$PWD/.local-tools/cargo"
export RUSTUP_HOME="$PWD/.local-tools/rustup"
export CARGO_TARGET_DIR="$PWD/apps/desktop/src-tauri/target"
export PATH="$CARGO_HOME/bin:$PATH"
```

## Build the CLI

Requires CMake 3.20+ and a C++20 compiler. No external native libraries are linked into the foundation.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
./build/crisp3ds capabilities
./build/crisp3ds validate tests/contracts/calibrated-project.json
./build/crisp3ds diagnose-geometry
```

On Visual Studio generators the executable is `build/Release/crisp3ds.exe`. The [core README](core/README.md) describes commands, return codes and the C ABI.

To enable actual image-to-pose estimation, build the pinned OpenCV configuration:

```sh
cmake -S . -B build-opencv -DCMAKE_BUILD_TYPE=Release -DCRISP3DS_WITH_OPENCV=ON
cmake --build build-opencv --config Release --parallel 4
ctest --test-dir build-opencv -C Release --output-on-failure
./build-opencv/bin/crisp3ds estimate-poses build-opencv/synthetic-pose-fixture/project.json
./build-opencv/bin/crisp3ds reconstruct-sparse build-opencv/synthetic-sparse-fixture/project.json
```

This test-generated dataset is explicitly synthetic. For your photographs, supply a project with matching camera calibration and measured rotating-board geometry; see [pose contracts and gates](docs/POSE-MILESTONE.md). The CLI emits a JSON report on stdout. Save it to a separate file and use **Import pose report** in the workspace to inspect translations, rotations and reprojection errors. Importing a report does not mark pipeline stages complete. The first enabled build downloads and compiles [pinned OpenCV sources](docs/OPENCV.md).

Sparse reconstruction additionally requires a matching binary grayscale PNG object mask for every image. It re-estimates poses, matches nearby views in acquisition order, and outputs millimetre-scale points with observations and residuals. Import the matching project and use **Import sparse report** to inspect the saved JSON and orbit/zoom the point cloud. The current baseline is limited to 64 views and 256 million aggregate pixels; see [sparse contracts and verification](docs/SPARSE-MILESTONE.md).

## Verify changes

```sh
npm --prefix apps/desktop ci
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
node tests/contract-parity.mjs
python3 scripts/check_dependencies.py
```

The parity test needs the built CLI; pass a different executable path as its first argument when needed. The dependency check additionally requires Python 3.11+, Cargo and the committed lockfiles. CI runs native and web checks; a configured workflow is not evidence that every target has already passed.

Testing now includes a separate dense-stereo evaluator and a checksummed real Middlebury scene with measured reference disparities. See [test strategy](docs/TESTING.md), [dataset provenance](docs/TEST-DATA.md), and [measured results and reproduction](docs/DENSE-EVALUATION.md). This evaluator is not a production dense-reconstruction stage.

For the live Chromium test, use the development-only browser tooling (not shipped with the app):

```sh
npm install --prefix .local-tools/browser --save-exact playwright@1.63.0
PLAYWRIGHT_BROWSERS_PATH="$PWD/.local-tools/browsers" .local-tools/browser/node_modules/.bin/playwright install chromium
node scripts/live_browser_test.mjs
```

Run the OpenCV CTests first to generate the fixture. The browser script runs the actual sparse CLI, starts and stops its own Vite server on port 1430, and checks report import, orbit/zoom, rejection, state invalidation and narrow layout. It does not test the native webview or physical scanner.

## Architecture and next gate

The core owns computation and stable file contracts. The desktop host will manage isolated jobs; mobile bindings will call the library directly. Browser inspection does not imply browser-local reconstruction. Hardware capture and motor transport stay outside the core.

The next work is to improve object isolation and dense geometry against the
measured reference, then prove a licensable cross-platform backend. MVE remains
a control, not a required engine. Quality acceptance precedes worker/UI
integration; see the [active roadmap](docs/SOTA-ROADMAP.md) and [actual status](docs/STATUS.md).

- [Task plan and acceptance gates](docs/PLAN.md)
- [Project schema](docs/project.schema.json)
- [First measured dataset requirements](docs/CAPTURE-DATASET.md)
- [Dependency policy and audit limits](docs/DEPENDENCIES.md)

## License

Original Crisp3DS project code is licensed under [GNU AGPL v3 only](LICENSE) (`AGPL-3.0-only`). Third-party code, assets, and datasets retain their own licenses and attribution requirements; the project license does not relicense them. The dependency-selection policy and exact shipped dependency/source-bundle audit remain separate release gates. Distribution through the Apple App Store or another store still requires a separate compatibility and compliance review. `apps/desktop/package.json` remains `private: true` because npm publication is unrelated to GitHub repository visibility.

Every library, engine and dataset the pipeline and the apps depend on is listed
with its license in the [pipeline README](scripts/turntable_mesh/README.md#dependencies-and-licenses)
and the [Studio README](apps/studio/README.md). A successful local build is
not a statement that a combination may be redistributed.
