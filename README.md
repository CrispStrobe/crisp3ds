# Crisp3DS

A rigid-object photo scanning application, built around a portable C++20 core and a Tauri 2 / TypeScript desktop workspace. Ordinary overlapping photographs with unknown poses are the target; calibrated turntable capture is an optional mode. The active [quality-first roadmap](docs/SOTA-ROADMAP.md) defines scoped tasks and acceptance gates.

The workspace supports project creation, JSON import/export, calibration and marker-board geometry, pose-report inspection, and an interactive sparse point-cloud viewer. An optional pinned OpenCV build detects ArUco markers in PNG/JPEG images, estimates metric board-to-camera poses, and reconstructs masked object feature tracks. A separate [experimental CPU photo-to-mesh runner](docs/MVE-FULL.md) has run on real photographs on M1, but measured object quality is not accepted. **App-integrated photo-to-mesh reconstruction is not implemented.** The `reconstruct` command returns unavailable; imported stage states are unverified metadata.

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

## Run the workspace

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
