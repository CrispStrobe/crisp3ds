# Dense turntable pipeline: calibrated photos to a closed STL

This directory holds the photo-to-mesh workers of Crisp3DS. This README covers
the **all-view dense pipeline**, the route that currently gives the best closed
models from a turntable photo set on an Apple M1 (GPU through PyTorch MPS, or
CPU). The development tree holds further experimental workers (the earlier
plane sweep, the Apple Object Capture and board/SAM session routes). They are
not all published yet and are not described here.

It starts from photos whose cameras are already recovered and which already have
one object mask per photo. Camera recovery and segmentation are earlier,
separate steps (see [Inputs](#inputs)).

## What it does

```
inputs  ->  mask repair  ->  silhouette hull  ->  multiscale stereo  ->  TSDF  ->  surface  ->  check
```

| Stage | Module | What happens |
| --- | --- | --- |
| inputs | `dense_all_views_inputs.py` | One camera table for every registered view of an AliceVision scene; raw masks are remapped through the lens undistortion |
| mask repair | `multiscale_stereo.py` | Pixels a single-view segmentation dropped are added back when most views agree in 3D **and** the pixel is object-dark in that photo |
| silhouette hull | `multiscale_stereo.py` | Voxels that project inside (almost) every mask; it bounds all later steps |
| multiscale stereo | `multiscale_stereo.py` | Masked NCC with contrast normalised inside the mask; full depth sweep at the coarsest level, then a narrow band around the smoothed coarser depth; cost aggregation along the surface; a hull-front candidate for thin parts; cross-view depth agreement at every level |
| TSDF | `multiscale_stereo.py` | Depth is averaged into a truncated signed distance volume inside the hull. The silhouette rim band is left to the hull |
| surface | `tsdf_hull_mesh.py` | Unobserved hull voxels fall back to the hull's own signed distance; marching cubes on the float field; largest component; light Taubin smoothing; binary STL |
| check | `mesh_photo_check.py` | Silhouette IoU of the mesh against every mask and a preview sheet |

`scan_evaluate.py` scores a finished STL against an independent scan.
`dense_pipeline.py` runs all of this as one command. `dense_config.py` holds
every tunable. `stl_compare_render.py` renders several STLs through the same
cameras. `synthetic_scene.py` writes a small analytic test scene.

Why it is built this way, with the measurements on the 3DLF Dragon behind each
choice, is in [Design notes](#design-notes).

## Quick start

No dataset needed; this runs the whole CLI on an analytic sphere in about 15 s
on CPU:

```sh
cd /path/to/crisp3ds
export PYTHONPATH=$PWD
python -m scripts.turntable_mesh.synthetic_scene --output /tmp/sphere
python -m scripts.turntable_mesh.dense_pipeline --inputs /tmp/sphere --output /tmp/sphere-run \
  --device cpu --set sizes=64,128 --set grid=96 --set planes=48 --set neighbours=4 --set best_of=2 \
  --set vote_neighbours=4 --set min_votes=2,2 --set crop_padding=6 --set hull_dilate=1 \
  --set windows=5,7 --set aggregates=1,1
```

A real photo set (defaults are tuned for megapixel photos):

```sh
python -m scripts.turntable_mesh.dense_pipeline \
  --scene final.sfm --prepared undistorted-pngs/ --raw-masks masks/ \
  --output runs/my-object --device mps
```

Result: `runs/my-object/mesh/mesh.stl`, `check/preview.png`, `check/result.json`
and `pipeline.json` (stage timings and status). The output directory must not
exist; nothing is ever overwritten.

### Interpreters

The stages need different libraries, and the local development setup keeps them
in two virtual environments. The driver starts each stage as a separate process:

| Stages | Needs | Selected by |
| --- | --- | --- |
| inputs, surface, check | NumPy, SciPy, scikit-image, OpenCV, Pillow | `--python` or `CRISP3DS_PYTHON` |
| stereo | PyTorch, NumPy, Pillow | `--torch-python` or `CRISP3DS_TORCH_PYTHON` |

Both default to the interpreter running the driver, so one environment with all
of the packages works without any flag. On the development machine:

```sh
export CRISP3DS_PYTHON=$PWD/.local-tools/colmap-sparse/venv/bin/python
export CRISP3DS_TORCH_PYTHON=$PWD/.local-tools/sam21-m1-local-cache-001/venv/bin/python
```

`--device mps` uses the Apple GPU and refuses to run if MPS is unavailable
(`PYTORCH_ENABLE_MPS_FALLBACK=0`); `--device cpu` runs the same code on CPU.

### Running stages separately

```sh
python -m scripts.turntable_mesh.dense_all_views_inputs --scene final.sfm --prepared pngs/ --raw-masks masks/ --output run/inputs
python -m scripts.turntable_mesh.multiscale_stereo --inputs run/inputs --output run/stereo --device mps
python -m scripts.turntable_mesh.tsdf_hull_mesh --volume run/stereo/volume.npz --output run/mesh
python -m scripts.turntable_mesh.mesh_photo_check --inputs run/inputs --mesh run/mesh/mesh.stl --output run/check \
  --repaired-masks run/stereo/masks-repaired
```

`multiscale_stereo --reuse-depths run/stereo/depths.npz` skips matching and only
re-fuses, which is the quick way to try fusion settings. Re-meshing an existing
`volume.npz` with other `mesh_*` settings takes about a minute. The driver
deletes `volume.npz` unless `--keep-volume` is given.

## Inputs

Either an existing inputs directory, or the three things it is built from.

**Inputs directory** (`--inputs`):

- `cameras.json`: `{"views": [{name, image, mask, width, height, k, rotation, translation}, …]}`.
  `rotation`/`translation` are world-to-camera. `k = [fx, fy, cx, cy]` in the
  **half-pixel-centre** convention: the ray through pixel `(x, y)` is
  `((x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, 1)`. Images must be undistorted
  pinhole images. `image`/`mask` may be absolute or relative to the directory.
- `sparse_points.npy`: N×3 photo-reconstructed points; only used to find the
  object's rough location and the scene centre for neighbour selection.
- Masks are 8-bit, object > 127.

**From AliceVision** (`--scene`, `--prepared`, `--raw-masks`): an SfM file with
poses and one shared `radialk3` intrinsic, the PNGs written by AliceVision
`prepareDenseScene` (named `<viewId>.png`), and one 0/255 mask per source photo
with the source photo's file name, in the original distorted frame.

Requirements and assumptions:

- At least `neighbours + 1` views, each with two or more other views between
  `minimum_angle` and `maximum_angle` (3°–40° by default) as seen from the object.
- The object is darker than its backdrop for mask repair to add anything
  (`--set repair_masks=false` turns it off).
- A mask that does not touch the image border is taken to show the whole object.
- World scale is whatever the SfM produced. **The STL has no physical scale.**

## Settings

Every setting can be given three ways, later ones winning: defaults in
`DenseConfig`, a JSON file (`--config`; any earlier `config.json` or
`result.json` works), and `--set key=value` (repeatable; tuples as
`a,b,c`). `python -m scripts.turntable_mesh.dense_pipeline --list-settings`
prints all of them. Lengths are relative (voxels, level pixels, fractions of
depth), so they do not depend on SfM scale. Per-level tuples have one entry per
pyramid level; a shorter tuple repeats its last entry.

| Setting | Default | Meaning |
| --- | --- | --- |
| `sizes` | 256, 512, 1024 | Longest canvas side per pyramid level, capped at native resolution |
| `crop_padding` | 24 | Native pixels kept around each mask box |
| `stretch_percentiles` | 1, 99 | Grey range inside the mask mapped to 0..1 before matching |
| `neighbours`, `best_of` | 6, 3 | Matching neighbours per view; mean of the best N scores is used |
| `minimum_angle`, `maximum_angle` | 3, 40 | Allowed angle between a view and its neighbours, degrees |
| `planes` | 128 | Inverse-depth planes of the full sweep at the coarsest level |
| `windows` | 7, 9, 11 | NCC window per level, pixels |
| `aggregates` | 1, 1.5, 2 | Cost blur sigma per level, pixels; 0 disables |
| `passes` | 1, 2, 2 | Refinement passes per level |
| `band_steps` | 8, 5 | Search half-width in depth steps for the first and later passes |
| `min_score`, `min_variance`, `window_fill` | 0.55, 1e-4, 0.6 | NCC acceptance, texture gate on normalised grey, minimum valid window fraction |
| `tolerances`, `min_votes`, `vote_neighbours` | 0.006/0.003/0.002, 2/3/3, 10 | Cross-view depth agreement: relative tolerance, required agreeing views, views asked |
| `grid` | 400 | Voxels along the longest side of the object |
| `hull_dilate`, `hull_allowed` | 2, 2 | Mask tolerance in native pixels; views allowed to disagree |
| `repair_masks`, `repair_loose`, `repair_base_margin` | true, 8, 8 | Mask repair on/off; consensus tolerance; voxels above the support kept out of repair |
| `hull_front`, `hull_front_level`, `hull_front_min_score`, `hull_front_margin` | true, 1, 0.6, 0.004 | Hull-front depth candidate for thin parts |
| `fallback_level` | true | Where the finest level fails its checks, use the previous level's depth |
| `rim_fraction` | 0.55 | Silhouette band excluded from fusion, in matching windows |
| `truncation_voxels`, `behind_voxels`, `behind_weight` | 3, 12, 0.25 | TSDF truncation; depth and weight of the weak inside vote behind a surface |
| `mesh_smooth`, `mesh_fill_sigmas`, `mesh_final_smooth` | 1.0, 2/4, 0.6 | Field smoothing, extrapolation reach into unobserved hull, final blur; voxels |
| `mesh_minimum_weight`, `mesh_confidence_cap` | 0.5, 4 | Evidence needed to count as observed; views at which confidence saturates |
| `mesh_taubin_cycles` | 5 | Taubin smoothing cycles on the mesh |

Faster, coarser runs: `--set sizes=256,512 --set grid=256`.

## Outputs

| File | Content |
| --- | --- |
| `mesh/mesh.stl` | Binary STL, largest component, outward normals |
| `mesh/result.json` | Triangles, closedness, boundary and non-manifold edge counts, genus, observed and extrapolated hull fractions |
| `stereo/result.json` | Configuration, hull size, mask repair statistics, per-level coverage and timings |
| `stereo/depths.npz` | Final depth per view on the common canvas (`depth_000`, …) |
| `stereo/depth-level-N.png`, `depth-merged.png` | Photo, depth and depth shading for three views |
| `stereo/masks-repaired/` | Masks after multi-view repair |
| `check/result.json`, `check/preview.png` | Silhouette IoU against input and repaired masks; preview sheet |
| `pipeline.json`, `*.log`, `config.json` | Stage commands, exit codes, timings, logs, the resolved configuration |

What the numbers do and do not mean: silhouette IoU is agreement with the
photos' outlines. It cannot see errors along the viewing direction and is not
scanner accuracy. Closedness and genus describe the exported topology, not the
true object. Surface inside the hull that no photo measured (for example the
underside on a turntable) is the silhouette bound, not a measurement.

## Results so far

3DLF Dragon, 73 turntable photos at one elevation, 71 registered by AliceVision
global SfM with a fixed lens, experimental SAM masks, Apple M1 16 GB.

| | Earlier 24-view plane sweep + Poisson (control 204) | This pipeline (control 214, `mesh-009`) |
| --- | --- | --- |
| Views used | 24 | 71 |
| Median foreground depth coverage | about 38% | 65% native, 82% with level fallback |
| Silhouette IoU vs original masks, 71 views, median | 0.897 | 0.946 |
| Topology | closed, genus 78 | closed, genus 23 |
| Dense time | about 2 min | about 12.5 min stereo + 1 min surface (MPS) |

Independent scanner check (`scan_evaluate.py`, evaluation only, never fed back
into reconstruction). The platform is removed from the scan, the mesh is fitted
by a similarity transform, and distances are reported on held-out surface
samples as a percentage of the scan's bounding-box diagonal. "Above support"
excludes the invented underside and the contact band.

| Mesh | Region | Accuracy median / p90 | Completeness median / p90 | F1 @0.5% | F1 @1% | F1 @2% |
| --- | --- | --- | --- | --- | --- | --- |
| This pipeline (`mesh-009`) | all | 0.30 / 1.33 | 0.24 / 0.64 | 0.755 | 0.911 | 0.973 |
| This pipeline (`mesh-009`) | above support | 0.25 / 0.76 | 0.22 / 0.59 | 0.819 | 0.961 | 0.994 |
| Earlier route (control 204) | all | 0.47 / 2.00 | 0.36 / 1.29 | 0.567 | 0.780 | 0.934 |
| Earlier route (control 204) | above support | 0.39 / 1.40 | 0.34 / 1.11 | 0.624 | 0.845 | 0.972 |

The largest remaining errors are at the top of the head, the horn tips and the
concavities between the tail loops and legs. The fit is shape-only, so absolute
scale is not evaluated.

**Handedness is unresolved.** Both meshes fit the scan only as mirror images
(trimmed residual 0.65 mirrored against 2.05 unmirrored for `mesh-009`), and the
photo-derived camera orbit turns in the opposite sense to the dataset's
depth-derived poses. A perspective SfM solution with a fixed lens and
sub-pixel residuals cannot itself be mirrored, so either the dataset's photos
are flipped relative to its depth and scan data, or those use a left-handed
frame. Until that is settled, treat the STL's chirality as unverified; the
evaluator reports which handedness it scored.

```sh
python -m scripts.turntable_mesh.scan_evaluate --mesh run/mesh/mesh.stl --reference scan.ply --output run/scan-evaluation
```

Status and limits, stated plainly:

- The Dragon result above was produced by the code before it was reorganised
  around `DenseConfig` and `dense_pipeline.py`. The reorganised code passes the
  synthetic unit tests and the synthetic end-to-end CLI run on CPU; the full
  Dragon run has **not yet been repeated** with it.
- Settings were chosen on one object. A second object has not been run.
- Horn tips are still shorter and thinner than in the photos, and the body has
  residual dimples. Mask repair can add contact shadow near the base.
- The cameras come from all 73 photos. This is not a dozen-photo result.

## Tests

```sh
export PYTHONPATH=$PWD
$CRISP3DS_TORCH_PYTHON -m unittest scripts.turntable_mesh.test_multiscale_stereo   # config + stereo on an analytic sphere
$CRISP3DS_PYTHON       -m unittest scripts.turntable_mesh.test_tsdf_hull_mesh      # surface extraction on analytic volumes
$CRISP3DS_PYTHON       -m unittest scripts.turntable_mesh.test_scan_evaluate       # similarity fit, platform removal, mirror detection
```

Each test file skips the cases whose libraries are missing in that interpreter.

## Design notes

Measured on the Dragon before this pipeline existed:

- **Texture gate.** The old gate was an absolute variance on raw 8-bit grey with
  a fixed 5×5 window. On the dark figure only 50–59% of foreground patches
  passed at 256 px, 26–30% at 512 px and 15–19% at 1024 px, so more resolution
  made depth maps emptier. With grey normalised inside the mask and an 11×11
  window, 98%, 84–92% and 71–84% pass.
- **Neighbours.** 24 of 73 photos gave 15° and 30° neighbours. On a turntable
  the lights are fixed and the object turns, so shading moves across the surface
  between views. All registered views give 5° and 10° neighbours.
- **Masks.** The segmentation dropped whole horns in runs of consecutive views.
  Any strict silhouette intersection then deletes them even though the cameras
  are right. Loosening the hull globally fills real gaps instead, so the repair
  is per pixel and needs both 3D consensus and photo evidence.
- **Thin parts.** A matching window that straddles the silhouette matches the
  apparent contour, which triangulates behind the true surface and carves thin
  parts from both sides. The hull is exact at the rim, so the rim is left to it,
  and thin parts get a depth candidate from the hull's front surface.
- **Surface.** Poisson on oriented points needs normals from depth maps full of
  holes, and re-voxelising its output to a binary grid produced terraces. A
  signed distance volume needs no normals and is meshed from float values.
- **MPS speed.** Every view shares one canvas size per level, because MPS
  recompiles its kernels for each new tensor shape (5–8× slower otherwise), and
  best-of-N neighbour scores use elementwise min/max insertion, because indexed
  `max` and `scatter_` are very slow on MPS.

## Dependencies and licenses

Crisp3DS itself is **AGPL-3.0-only** (`LICENSE` in the repository root). Nothing
below is vendored into this repository; the table records what the code imports,
launches or was tested with, so that anyone packaging it can check obligations.
Versions are the ones installed on the development machine on 2026-10-05.
Licenses marked † were not re-read from the pinned source for this list and
should be confirmed before redistribution.

### Imported by the all-view dense pipeline

| Library | Version tested | License | Used for |
| --- | --- | --- | --- |
| Python | 3.11.1 | PSF-2.0 | Runtime |
| NumPy | 1.26.4 | BSD-3-Clause | Arrays everywhere |
| PyTorch | 2.7.0 | BSD-3-Clause | Stereo, hull and TSDF on MPS or CPU |
| Pillow | 11.2.1 / 12.0.0 | MIT-CMU (HPND) | Image reading, resizing, previews |
| SciPy | 1.17.1 | BSD-3-Clause | Distance transform, filters, sparse graphs in surface extraction |
| scikit-image | 0.25.2 | BSD-3-Clause | Marching cubes |
| OpenCV (`opencv-python-headless`) | 4.10.0.84 | Apache-2.0 | Mask undistortion, rasterising, previews |

Installed alongside scikit-image and Torch as their own dependencies (not
imported directly): networkx (BSD-3-Clause), imageio (BSD-2-Clause), tifffile
(BSD-3-Clause), lazy_loader (BSD-3-Clause), packaging (Apache-2.0 OR
BSD-2-Clause), and Torch's bundled components under their own notices.

### Upstream steps this pipeline consumes

| Software | License | Role | How it is used |
| --- | --- | --- | --- |
| AliceVision (fork `CrispStrobe/AliceVision`, branch `crisp3ds/m1-sycl`) | MPL-2.0; parts MIT (libmv) | Camera recovery (global SfM), `prepareDenseScene` undistortion; optional native depth/meshing engine | External executables; this pipeline reads its `.sfm` JSON and PNGs |
| AdaptiveCpp (fork `CrispStrobe/AdaptiveCpp`) | BSD-2-Clause | SYCL runtime for AliceVision's Metal depth stages | Linked by the AliceVision build only |
| Apple metal-cpp | Apache-2.0 | Headers for the AdaptiveCpp Metal backend | Build-time only |
| SAM 2.1 (Meta) | Apache-2.0 † (code and checkpoints) | Object masks | Separate segmentation step; masks are files |
| hydra-core, omegaconf, iopath, tqdm | MIT; BSD-3-Clause †; MIT †; MPL-2.0 AND MIT | SAM 2 runtime dependencies | Segmentation step only |
| Eigen, Ceres Solver, Boost, OpenImageIO, nanoflann, LLVM/Clang | MPL-2.0 †; BSD-3-Clause †; BSL-1.0 †; Apache-2.0 †; BSD-2-Clause †; Apache-2.0 with LLVM exception † | AliceVision build dependencies | Build-time and linked by AliceVision only |

### Other engines and tools in this repository's workflows

Not used by the all-view dense pipeline; listed because other workers and
comparisons in the development tree launch them. Some of those workers are not
published in this repository yet.

| Software | License | Role |
| --- | --- | --- |
| MicMac (fork `CrispStrobe/micmac`) | CeCILL-B | Optional comparison engine, CPU only |
| PoissonRecon (binary shipped inside MicMac's `binaire-aux`) | MIT † | Surface from oriented points in the older plane-sweep route |
| COLMAP | BSD-3-Clause † | Optional sparse/dense comparison |
| OpenMVS | AGPL-3.0 † | Optional evaluation backend only; not an approved redistribution dependency |
| OpenMVG | MPL-2.0 † | Optional sparse comparison (see `docs/OPENMVG-LICENSE-CLOSURE.md`) |
| MVE | BSD-3-Clause | Optional comparison engine |
| Apple Object Capture (RealityKit `PhotogrammetrySession`) | Proprietary, part of macOS | Separate Apple reconstruction route |
| LoFTR (via learned matching experiments) | Apache-2.0 † | Camera-recovery experiments |

### Application shell

| Software | Version pinned | License |
| --- | --- | --- |
| Tauri (`tauri`, `@tauri-apps/api`, `@tauri-apps/cli`) | 2.11.x | MIT OR Apache-2.0 † |
| serde, serde_json, libc (Rust) | 1.0.229, 1.0.151, 0.2.189 | MIT OR Apache-2.0 † |
| TypeScript | 7.0.2 | Apache-2.0 † |
| Vite, Vitest | 8.3.1, 5.0.2 | MIT † |

Transitive Rust and npm dependencies are not enumerated here; use
`cargo license` / `npx license-checker` against the lock files before packaging.

### Test data

| Data | License | Note |
| --- | --- | --- |
| 3DLF scan dataset (Bunny, Dragon, Armadillo, …), Mendeley Data `ngvgpsvd8b` v1 | CC BY 4.0 | Photos and independent structured-light scans; not redistributed here |
| YCB object set | See YCB terms † | Not redistributed here |
| ETH3D | CC BY-NC-SA 4.0 | Non-commercial; separately gated; not redistributed here |
| Middlebury stereo | See Middlebury terms † | Not redistributed here |

Engine and data licenses are independent of this repository's AGPL license. A
successful local build or run is not a statement that a combination may be
redistributed.
