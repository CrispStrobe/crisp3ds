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

The selected OpenMVS CLI sequence follows its [COLMAP import instructions](https://github.com/cdcseacave/openMVS/blob/v2.4.0/docs/wiki/Usage.md): `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `TextureMesh`. CPU execution is restricted to two native threads; PyCOLMAP feature extraction and matching also use CPU. By default, internal SfM uses one unknown shared `SIMPLE_RADIAL` camera, exhaustive matching and seed 0. `--max-image-size` limits undistorted images and densification uses resolution level 2 to keep the local trial bounded. The shared YCB model's producer used different documented SfM options, so the two modes must be reported separately.

The current standalone CLI can also reproduce the *settings* of the foreground producer without hardcoding the mask detector: `--pose-mask-dir` copies and hashes one `IMAGE_NAME.png` file per photo; `--matching sequential --sequential-overlap 8 --sift-max-features 1800 --sift-max-image-size 1200 --seed 20260927` select the bounded SfM configuration. Masks must be prepared independently from photos. This option does not accept scan geometry or supplied camera poses. Native handoffs explicitly pass `dense.ply` to `ReconstructMesh`, `mesh.ply` to `RefineMesh`, and `refined.ply` to `TextureMesh`; the `.mvs` interface scene alone retains the original sparse points.

## M1 YCB observations (2026-09-27)

The independent [object-motion raw and foreground SfM reports](OBJECT-MOTION.md) used 60 real YCB cracker-box photos. The raw arm registered only 2 cameras. A photo-derived coarse pose-mask arm registered all 60 and produced 3,939 sparse points; the separately sealed producer report binds photo, mask and binary-model hashes. The shared-model OpenMVS run and its continuations are recorded as separate immutable results:

| Result | Actual stage outcome |
| --- | --- |
| `build-opencv/classical-ycb-foreground-001/result.json` | Adapter provenance variable-shadowing failure before native stages. |
| `build-opencv/classical-ycb-foreground-002/result.json` | 60 images undistorted and imported; native depth estimation completed 60 base maps and 45/60 maps in its second geometric pass, then a transient-file race in the runner's disk monitor stopped it. |
| `build-opencv/classical-ycb-foreground-003-continuation/result.json` | Copied the complete base depth-map cache; OpenMVS fused 60 maps in 7.9 s with `--geometric-iters 0` to a 56,684-point dense cloud. Native densification succeeded; the then-current validator rejected OpenMVS list properties in its PLY. |
| `build-opencv/classical-ycb-foreground-004-finish/result.json` | Reimported the same 60 cameras and explicitly passed that dense PLY to the native mesh tools. Rough mesh: 22,320 vertices / 44,561 faces. Refined and textured OBJ: 3,359 vertices / 6,612 faces, one 1024-pixel JPG atlas. Result `complete`; output 61.2 MB; free disk 20.58 GB. |
| `build-opencv/classical-ycb-foreground-005-e2e/result.json` | Fresh one-command masked run copied all 60 photos/masks but registered only 2 cameras / 357 points and stopped at the 3-camera gate before dense processing. Its feature count matched the producer, but sequential verified match totals differed slightly; that result does not reproduce the shared SfM success. |
| `build-opencv/classical-ycb-foreground-006-filtered/result.json` | Photo-mask point-support filter retained 34,172 of the same 56,684 native dense points at a frozen 48/60 camera threshold. With the same cameras and native mesh/refine/texture settings, the filtered run completed a 2,225-vertex / 4,404-face textured OBJ. It is a separate composed continuation. |

The successful textured artifact is a **composed continuation**, not a fresh one-command photo-to-mesh success or a single elapsed-time measurement. `004` took 0.51 s for reimport, 3.57 s for meshing, 92.05 s for refinement and 6.18 s for texturing; its largest sampled single-child RSS was 780 MB. The dense run spent roughly 10 minutes estimating depths before the runner race; `003` then fused cached maps in 7.9 s. The final geometry has no exact-zero-area triangles under full payload validation. Independent whole-mesh reference-fitted shape F1 at 1% of scan diagonal was **12.77%**; the visible mesh contains substantial board/background. This rejects object-quality acceptance. The fitted alignment cannot establish physical scale.

`003` and `004` are kept separate from the failed sources; the continuation runners log source hashes, stage commands, limits and native artifacts. OpenMVS's [usage guide](https://github.com/cdcseacave/openMVS/blob/v2.4.0/docs/wiki/Usage.md) documents reusing available depth maps for fusion. The completed 60 base maps were copied while partial geometric maps were omitted. `003` was thus a cached-depth diagnostic with geometric consistency disabled, and `004` used only its verified dense PLY plus freshly imported cameras. These choices are part of the result and cannot be silently presented as the original default dense run.

The [point-support filter](../scripts/object_motion/filter_cloud.py) is predeclared in the object-motion experiment and uses only the foreground SfM cameras and photo-derived coarse masks. Its output preserves every retained OpenMVS point record, including view lists and weights, byte-for-byte. `006` verifies the filter report's source dense-cloud, model, photo, mask and output hashes before calling `ReconstructMesh -p` on the filtered cloud. A repeated filter run produced the identical native PLY hash. No independent reference geometry entered this filter or either mesh run. Independent reference-fitted scoring improved F1 at 1% of scan diagonal from **12.77%** for `004` to **38.15%** for `006` (precision 42.63%, recall 34.52%). At 0.5% and 2%, filtered F1 was 19.48% and 63.54%. This is a substantial cleanup of background geometry but still rejects object-quality acceptance: important object surfaces are missing or distorted, and fitted alignment does not establish metric scale.

## Platform and result limits

Only the M1 release executables have been inspected. Linux and Windows have not been executed. A successful run would demonstrate a complete local CLI path, not textured object quality or metric scale. Reference surface scoring, if done, must use independent scan geometry only after reconstruction and must label any fitted similarity transform.
