# COLMAP → OpenMVS CPU experiment

This is an image-only research CLI for unknown camera poses. It estimates sparse cameras and points with PyCOLMAP 3.11.1, exports undistorted COLMAP images, then runs OpenMVS import, densification, mesh reconstruction, refinement and texturing. No supplied scan, depth, reference pose or fitted scale enters reconstruction. The output gauge is arbitrary. This experiment does not replace the app's `reconstruct` command.

## Pinned local toolchain

The existing `.local-tools/colmap-sparse/venv` supplies PyCOLMAP 3.11.1. The OpenMVS tools come from the upstream [v2.4.0 macOS arm64 release](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0), asset `OpenMVS_macOS_arm64.zip`, SHA-256 `3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b`. The archive is 68,347,008 bytes; five selected CLI executables are extracted to `.local-tools/classical-backend/bin`. The [fetch helper](../scripts/classical_backend/fetch_openmvs.py) verifies the exact archive size/hash and extracts only those five members, keeping the 10 GiB free-space reserve. Each executable's SHA-256 is recorded in `result.json`. The native `InterfaceCOLMAP --help` starts on this M1 and `otool -L` shows macOS system libraries only. No global packages were installed. A local source build was not attempted because OpenCV, Eigen, CGAL and Ceres development packages are absent; the release binary is the bounded experiment toolchain.

OpenMVS identifies its project as [AGPL-3.0](https://github.com/cdcseacave/openMVS/blob/v2.4.0/LICENSE). It is a commercial desktop/server integration candidate under the approved AGPL roadmap, but this experiment is not release approval. App Store distribution is a separate question. `otool -L` shows dynamic links only; statically linked components and exact packaged dependencies still need review.

## Run

From the repository root, after placing the pinned binaries as above:

```sh
TMPDIR="$PWD/.local-tools/tmp" .local-tools/colmap-sparse/venv/bin/python -m scripts.classical_backend.fetch_openmvs
TMPDIR="$PWD/.local-tools/tmp" .local-tools/colmap-sparse/venv/bin/python -m unittest scripts.classical_backend.test_run -v
TMPDIR="$PWD/.local-tools/tmp" .local-tools/colmap-sparse/venv/bin/python -m scripts.classical_backend.run \
  --images .local-tools/test-data/ycb-cracker-box/photos \
  --output build-opencv/classical-ycb-001 \
  --max-views 60 --max-image-size 1600 --max-threads 2 \
  --max-gib 2 --timeout-minutes 30
```

The output must be fresh. The runner copies and hashes selected photos, caps total run time, logs, output bytes and sampled child RSS, and stops a stage when free disk approaches 10 GiB. `--image-list` accepts one basename per line. Every stage gets a log and a structured result; a nonzero exit or missing geometry marks the run failed. It requires at least three cameras and points from SfM, at least 70% photo registration by default, nonempty dense points and mesh faces validated through the full PLY payload, and a textured OBJ with valid face indices and nonempty texture assets. These checks establish artifact presence, not shape accuracy. The final `result.json` always marks `shipping_approved`, `metric_scale_verified` and `quality_accepted` false.

For a controlled shared-SfM experiment, `--sparse-model PATH` takes a binary COLMAP model from an independently recorded run. Add `--sparse-provenance`, `--sparse-manifest` and `--sparse-result` to verify the object-motion producer's photo and photo-derived mask hashes, exact options, and sealed binary model hashes. The runner checks registered photo names, copies and hashes the original photos, and records producer file hashes. Without those producer files it labels the model `external_unverified_model`, so no image-only provenance claim follows. It distinguishes raw from photo-mask SfM. This skips internal feature extraction, matching and mapping and does not demonstrate a single-invocation photo-to-mesh run.

The selected OpenMVS CLI sequence follows its [COLMAP import instructions](https://github.com/cdcseacave/openMVS/blob/v2.4.0/docs/wiki/Usage.md): `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `TextureMesh`. CPU execution is restricted to two native threads; PyCOLMAP feature extraction and matching also use CPU. Its internal SfM uses one unknown shared `SIMPLE_RADIAL` camera, exhaustive matching and seed 0. `--max-image-size` limits undistorted images and densification uses resolution level 2 to keep the local trial bounded. The shared YCB model's producer uses different documented SfM options, so the two modes must be reported separately.

## Platform and result limits

Only the M1 release executables have been inspected. Linux and Windows have not been executed. A successful run would demonstrate a complete local CLI path, not textured object quality or metric scale. Reference surface scoring, if done, must use independent scan geometry only after reconstruction and must label any fitted similarity transform.
