# SeedeXR AliceVision for Mac: local M1 oracle audit

Status, 2026-09-27: **Metal depth-kernel test ran; the released AliceVision photo-to-mesh commands did not start.** This is a bounded local audit, not a reconstructed YCB object or a quality comparison.

## Provenance and license

The [SeedeXR repository](https://github.com/SeedeXR/alicevision-for-mac) was cloned under `/Volumes/backups/code/alicevision-for-mac-audit` at commit `caf976050d737ffd6ec3095dc8bf9d56884400e7`. Its `v0.1.0` tag resolves locally to the same commit. The [release](https://github.com/SeedeXR/alicevision-for-mac/releases/tag/v0.1.0) asset `alicevision-for-mac-0.1.0-arm64.tar.gz` is 15,434,596 bytes, locally SHA-256 `fd48c73e3def8890314ff8e25697e568fe2581394a1519c148456d8111afabfb`. No publisher checksum or source-to-binary attestation was found; the local hash identifies the bytes inspected here.

The repository [LICENSE](https://github.com/SeedeXR/alicevision-for-mac/blob/caf976050d737ffd6ec3095dc8bf9d56884400e7/LICENSE) applies MIT to its overlay only; AliceVision and Meshroom retain MPL-2.0, vendored LEMON retains Boost 1.0, and Apple metal-cpp is Apache-2.0. This audit does not approve redistribution of the combined artifact. The release tarball contains `share/aliceVision/LICENSE`; it does not itself establish the complete dependency license inventory.

The current [README](https://github.com/SeedeXR/alicevision-for-mac/blob/caf976050d737ffd6ec3095dc8bf9d56884400e7/README.md) claims 60 binaries and four completed Meshroom template runs. The actual `v0.1.0` tarball has 12 native arm64 `aliceVision_*` commands: `cameraInit`, `featureExtraction`, `imageMatching`, `featureMatching`, `incrementalSfM`, `prepareDenseScene`, `depthMapEstimation`, `depthMapFiltering`, `meshing`, `meshFiltering`, `texturing`, and `importMiddlebury`; it also has `default.metallib`. It contains no demo photographs or AI weights. The checked-in [BUILD.md](https://github.com/SeedeXR/alicevision-for-mac/blob/caf976050d737ffd6ec3095dc8bf9d56884400e7/BUILD.md) and [INSTALL_macOS.md](https://github.com/SeedeXR/alicevision-for-mac/blob/caf976050d737ffd6ec3095dc8bf9d56884400e7/INSTALL_macOS.md) retain older phase descriptions that conflict with the README. These statements are claims and documentation history; no complete Meshroom run was reproduced here.

## Actual host checks

The clean clone is an overlay. `upstream/` is a gitignored external symlink expected by the full build, and the clone lacks `third_party/metal-cpp/`. No immutable upstream AliceVision source revision is supplied by the README quick start. A first overlay-only CMake configure failed at `cmake/Metal.cmake:31` for missing `Metal/Metal.hpp`. The official [Apple metal-cpp archive](https://developer.apple.com/metal/cpp/) was fetched into the external checkout: `metal-cpp_macOS15_iOS18.zip`, 195,223 bytes, SHA-256 `0433df1e0ab13c2b0becbd78665071e3fa28381e9714a3fce28a497892b8a184` (1.3 MiB extracted). The next configure failed for Eigen. Homebrew estimated the Eigen 5.0.1 bottle at 1.7 MB and installation at 10.5 MB; `brew install eigen` installed 10.2 MB. No other package was installed.

This command then configured the **overlay only**, excluding all upstream AliceVision CLI targets:

```sh
TMPDIR=/Volumes/backups/code/alicevision-for-mac-audit/build-audit/temp \
cmake -S /Volumes/backups/code/alicevision-for-mac-audit \
  -B /Volumes/backups/code/alicevision-for-mac-audit/build-audit \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAV_BUILD_UPSTREAM=OFF -DAV_BUILD_UPSTREAM_DEPTHMAP=OFF \
  -DAV_BUILD_PYALICEVISION=OFF
cmake --build /Volumes/backups/code/alicevision-for-mac-audit/build-audit \
  --target test_depth_pipeline --parallel 2
ctest --test-dir /Volumes/backups/code/alicevision-for-mac-audit/build-audit \
  -R '^test_depth_pipeline$' --output-on-failure -V
```

Build completed in 13.4 seconds and compiled 17 `.metal` shaders with Xcode's Metal tools. The single test passed in 0.79 seconds on the Apple M1. It ran synthetic 128×96 SGM → refine → optimize: 696 of 768 output samples were valid; the test printed median absolute truth difference 0.1022, p90 0.1399, and 652/696 valid samples within 0.15 test units. The built test executable SHA-256 is `23102f672ffd98d965cad2c8866c6d56607ad560a58e63b0835bd11c06915eb7`. This verifies that the fork's isolated Metal depth path executes locally. It does not verify the released CLIs, SfM integration, photo-derived depth, meshing, or texture quality.

The release executables are present but dynamically link Homebrew paths absent on this host. Across all 12, `otool -L` found 31 direct Homebrew library paths, 10 missing: Assimp 6, Ceres 4, Expat 1, FLANN 1.9, Geogram 1.9.9, gflags 2.3, glog 2, OpenMeshCore 11.0, and two OpenImageIO 3.1 libraries. Direct `--help` for `cameraInit`, `depthMapEstimation`, `depthMapFiltering`, `meshing`, and `texturing` each terminated at dyld with exit `-6`, before argument parsing. First missing libraries were respectively OpenImageIO, Assimp, OpenMeshCore, Ceres, and Ceres. The current Homebrew Geogram formula is 1.10.1, while the release requests `libgeogram.1.9.9.dylib`; installing today's formula cannot be presumed to satisfy that absolute path. No released command produced a depth map or mesh.

## Resource bound and next gate

After the test, `/Volumes/backups` had 18,439,256 KiB free (17.59 GiB), allowing at most 7.59 GiB of additional use above the 10 GiB floor; the internal filesystem had 23,296,100 KiB free (22.22 GiB). External artifacts were 55 MB for the clone including its 3.9 MB build directory, 46 MB for the extracted release, and 15 MB for the archive. AI models were not downloaded (`ai-models/` contains only documentation). The fork's `scripts/download_ai_models.sh` defaults to `mktemp` under the system temporary directory, so a future model fetch must explicitly stage temporary files on the external volume and verify the destination.

`brew install --dry-run` for the nine absent direct Homebrew formulae reported additional installs and upgrades, including GCC, SuiteSparse, HDF5, FFmpeg, and 28 dependencies of OpenImageIO. The nine direct formulae advertise about 22.3 MB of bottles and 82.4 MB installed; these figures **exclude** transitive packages, cache copies, and upgrades, so they are not a safe estimate for the requested full install. Full source build size is likewise unmeasured and needs the external upstream clone, an immutable upstream revision, missing dependency closure, and a separate disk budget. Do not interpret the working Metal test or the 12-file release as evidence of a complete local AliceVision turntable oracle.

The next feasible local route is a source rebuild against one pinned upstream AliceVision revision and an audited, version-consistent dependency set on a roomier Apple Silicon volume. A second candidate is the release's 1,247,000,900-byte Meshroom DMG, whose README says it bundles runtime libraries; that claim and its actual executable behavior remain unverified here. An official Linux CUDA release on a GPU host is a separate oracle route. None of these routes has been run, and the present release tarball cannot establish a complete turntable pipeline on this Mac.

## Subsequent bounded DMG audit, 2026-09-27

The [official `v0.1.0` release](https://github.com/SeedeXR/alicevision-for-mac/releases/tag/v0.1.0) and its [GitHub release API record](https://api.github.com/repos/SeedeXR/alicevision-for-mac/releases/tags/v0.1.0) identify asset `Meshroom-0.1.0-arm64.dmg` (asset ID `428721365`), exactly **1,247,000,900 bytes**, with publisher SHA-256 `41f397e834ab862d9792cd2220f104a6340efb9194eb6c0b951f405ebe0ab5b2`. It was downloaded to the fresh external path `/Volumes/backups/code/alicevision-for-mac-audit/dmg-oracle-audit-001/Meshroom-0.1.0-arm64.dmg`. Local size and SHA-256 matched the API record, and `hdiutil verify` reported a valid disk-image checksum. No installer or Python package was installed.

The DMG mounted at `/Volumes/Meshroom 0.1.0` as a read-only APFS volume. Its `Meshroom.app` reported about 3.0 GiB of mounted content. The app contains 60 arm64 `aliceVision_*` executables, 283 regular `.dylib` files in `Contents/Resources/lib`, two BiRefNet `.mlpackage` directories, a bundled Meshroom batch script, and the `photogrammetryObjectTurntable.mg` template. That 13-node template includes BiRefNet-lite foreground segmentation, feature matching with a 3-pixel minimum 2D motion setting, SfM, depth, meshing, and texturing. This is an inventory of packaged files and descriptors, not a successful run.

Static `otool -L` inspection found that the 60 executables' direct non-system library references point into the bundle and that those direct library filenames exist there. However, at least 20 `@rpath`/`@loader_path` filenames referenced by bundled libraries are absent from `Contents/Resources/lib`, including `libgcc_s.1.1.dylib`, `libjxl_cms.0.11.dylib`, and VTK libraries. They may be reachable from another host path, but this inventory does not establish a self-contained dependency closure. More decisively, `Contents/Resources/meshroom-venv/bin/python` is a broken link through `python3.13` to `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`, which does not exist on this host. `codesign --verify --deep --strict` for the whole app reported a missing file; the individual `aliceVision_cameraInit` executable's ad-hoc signature verified.

The app launcher `--help` and direct `aliceVision_cameraInit --help` probes each produced no useful output and were terminated after roughly 50 seconds. Their stall cause was not established, so no CLI runtime success is claimed. **No photographs, Meshroom graph, depth map, or mesh were processed. The DMG is no-go for a local photo oracle as packaged.** A supported next attempt needs a repaired or rebuilt bundle on a roomier disk with a relocatable Python 3.13 runtime, complete transitive library closure, signing, and bounded CLI-help checks before a three-photo smoke; only after that smoke could a 48-photo run be considered.

The new download/inventory directory used **1,228,908 KiB** (about 1.17 GiB), within the 1.5 GiB cap. At the final check, external free space was **15,086,732 KiB** and internal free space **23,415,252 KiB**, leaving both above the 10 GiB floors and preserving a separate 1 GiB external reserve for the LightGlue work. A read-only mount consumed no 3.0 GiB app copy. Ordinary detach returned `resource busy`; after the help probes ended and `fuser` reported no holder, a forced detach succeeded. The mount is gone, no probe processes remain, and the hash-verified DMG is retained externally for future inspection. No prior artifact was changed.
