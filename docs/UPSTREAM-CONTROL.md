# Pinned upstream OpenMVS control

This control separates camera recovery from OpenMVS dense/mesh stages. It pins the [OpenMVS sample v2.3.0 commit](https://github.com/cdcseacave/openMVS_sample/tree/8855e0405b2c80c8286338ff21f3c09f22e9be16), before the sample's May 2026 switch to native `CreateStructure`. Its 11 Sceaux Castle JPEGs, `scene.mvs`, and selected dense/mesh outputs are downloaded into an ignored local research folder. The pinned `scene.mvs` contains upstream image-derived cameras and sparse points. Using it is a **stage-oracle lane**: it can check our OpenMVS 2.4.0 executable chain on a known upstream scene, but it cannot measure this project's image-only SfM accuracy or object geometry accuracy. The upstream dense/mesh artifacts are regression examples, not measured ground truth.

The upstream v2.3 [README](https://raw.githubusercontent.com/cdcseacave/openMVS_sample/8855e0405b2c80c8286338ff21f3c09f22e9be16/README.md) says the cameras and sparse cloud came from OpenMVG, with outputs automatically produced by OpenMVS. The current [sample README](https://github.com/cdcseacave/openMVS_sample) instead documents `CreateStructure`; the locally pinned OpenMVS 2.4.0 arm64 ZIP has no such executable. Mixing the current sample's `scene.mvs` with the older binary is therefore avoided. The v2.3 sample lists an AGPL-3.0 repository license, while the [original Sceaux photo repository](https://github.com/openMVG/ImageDataset_SceauxCastle) credits Copyright 2012 Pierre Moulon without a separate photo redistribution license. These photos are retained only for local evaluation; no shipping rights are inferred.

## Frozen protocol

Fetch only 11 JPEGs, `scene.mvs`, `scene_dense.mvs`, reference `scene_dense.ply` and `scene_dense_mesh.ply`, two logs, README, and LICENSE. The pinned commit's Git tree gives an exact path, size and blob SHA-1 for every selected file; the fetcher verifies all three plus a measured local SHA-256. The selected payload is 89,456,041 bytes across 19 files, far below the 150 MiB fetch cap. The source folder is `.local-tools/upstream-control/openmvs-sceaux-v23/`, with 11 originals in `images/` and file identities in `manifest.json`. Neither reference PLY is fed into reconstruction. The source folder is copied into a fresh run folder so OpenMVS cannot mutate the verified input set.

Run from the repository root:

```sh
python3 scripts/upstream_control/fetch.py
python3 -m unittest discover scripts/upstream_control -v
python3 scripts/upstream_control/run.py
```

The frozen OpenMVS 2.4.0 CPU profile uses the local pinned arm64 binaries, two threads, and `DensifyPointCloud` with maximum and minimum resolution 640 and resolution level 2. `ReconstructMesh`, `RefineMesh`, and `TextureMesh` consume the preceding stage's `.mvs` plus explicit point-cloud/mesh PLY paths. Refine and texture use resolution level 2. A 15-minute total deadline, 1.5 GiB combined selected-download-plus-run-output cap, bounded logs, and 10 GiB free-space floor apply. This 640px CPU profile differs from the upstream v2.3 GPU/default-resolution run; counts and elapsed time are descriptive regression context, not bitwise acceptance targets. It is also separate from the 60-photo YCB object experiment and must not be treated as evidence of YCB shape quality.

The fetched upstream dense PLY validates at 443,170 points, and its mesh PLY at 363,631 vertices and 726,814 faces. These are reference intermediates. The local CPU run will record each stage's command, status, output hash, counts, and bounded resource status in `build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001/report.json`. A failure remains a measured result. No reference intermediate is allowed as an input to a local stage.

## Independent image-only route

The same 11 verified original JPEGs are suitable for the already pinned local MVE `sfmrecon` pipeline. That run must start from photos only and report its own camera coverage, cloud, and mesh. It cannot consume `scene.mvs` or upstream PLYs. MVE is the practical independent local SfM candidate; OpenMVG executables are not installed, and rebuilding it for this control would add a distinct toolchain and schedule. The prior YCB raw MVE attempt failed SfM, which remains part of the same-input pipeline comparison. Supplied-camera Sceaux and image-only Sceaux results must remain separate rows in the benchmark.

## Measured outcome

The first bounded CPU replay reached dense and rough-mesh output, then failed at refinement because its command handed `RefineMesh` a nonexistent `scene_dense_mesh.mvs`. That failure is retained in [the original report](../build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001/report.json); it is not silently relabeled a complete run. Densification took 128.34 s and yielded 98,954 points (SHA-256 `f11bc0fd2edba24616fc965d220cf7da79035df13dd589db2c55122f9a3da4c9`). Meshing took 10.02 s and yielded 93,958 vertices and 187,793 faces, with zero exact-area faces (SHA-256 `9114dbf32b51ffbf820d6812053daf3b90ea50ed06f3cee5fcc6b075df9ad574`). These counts are much smaller than the upstream GPU/default-resolution intermediates, as expected for the declared 640 px CPU profile; they are not an accuracy score.

The handoff was corrected for future fresh runs. A hash-checked continuation copied the first run's `scene_dense.mvs`, rough mesh, and 11 photos to a **fresh** folder and supplied that scene plus an explicit mesh PLY to each subsequent stage:

```sh
python3 scripts/upstream_control/continue_run.py
```

[The continuation report](../build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001-continuation/report.json) binds the failed report's hash, the verified sample manifest, and every copied input hash. Refinement succeeded in 102.36 s with 30,491 vertices and 60,727 faces, zero exact-area faces; [the refined PLY](../build-opencv/upstream-control/sceaux-v23-scene-v24-cpu-001-continuation/scene_dense_mesh_refine.ply) has SHA-256 `60d5487c49a0147344739db35b4388c5fbcffa3367f772b2d1d333d15bb8e4ed`. Texturing succeeded in 22.08 s with a 30,491-vertex, 60,727-face OBJ and one atlas; its OBJ SHA-256 is `9bfc7d291957ab69792b2397b77ba3530242c2ef19ca3c6383dfe37fc922ea00`. This establishes that the pinned local OpenMVS dense-to-texture executable chain can complete with the upstream supplied-camera scene. It does not establish camera recovery or surface quality on photos alone.

The independent pinned MVE run on the same 11 JPEGs started from images only, registered 11/11 cameras and produced a 41,339-face mesh in `build-opencv/mve-upstream-sceaux-001`. A surface comparison against the pinned upstream mesh is a descriptive regression diagnostic, not ground-truth accuracy: the upstream mesh used different cameras, resolution, and processing settings. These two runs must remain separate benchmark rows.

For a separate post hoc camera-center diagnostic, the original supplied `scene.mvs` was reverse-exported to COLMAP text with OpenMVS 2.4.0:

```sh
.local-tools/classical-backend/bin/InterfaceCOLMAP \
  -i "$PWD/.local-tools/upstream-control/openmvs-sceaux-v23/scene.mvs" \
  -o "$PWD/build-opencv/upstream-control/sceaux-v23-camera-export-001/export" \
  --working-folder "$PWD/build-opencv/upstream-control/sceaux-v23-camera-export-001" \
  --max-threads 2 --binary 0 --no-points 1
```

Its [images.txt](../build-opencv/upstream-control/sceaux-v23-camera-export-001/export/sparse/images.txt) contains 11 standard COLMAP world-to-camera quaternions/translations; camera centers are `-R(q)^T t` in the supplied scene's arbitrary frame. The source `scene.mvs` SHA-256 is `b69f87384284d7716d700ce61296e9f2a4cd8973cb4527d549126b0968edafde`; exported `images.txt` and `cameras.txt` hashes are `e210acf8eb4c99107b8d8bec0f4a29c35703c82062a461f665dbcc0d01c77b06` and `85a5ae76af0712f50e34b845be1c50fc0510582485f46d15a3f27c52dc8507ab`. Aligning these centers to MVE's recovered cameras is an *evaluation-only* Sim(3) diagnostic, not an input to image-only reconstruction.
