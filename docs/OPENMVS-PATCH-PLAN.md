# OpenMVS depth/fusion coverage investigation

Root additionally verified that the cache-only filter-2 control's `dense.ply`
is byte-identical to the original 008 `dense.ply`: both SHA-256 values are
`3c200ef4e7bfb2312af85cc2ebeb54aa9abb905a4895afff0d3291239d3bb271`.

Status: pinned-source audit and one approved **cached-depth fusion-only** paired
run completed; no fork, depth estimation, meshing, or quality improvement claim.

## Controllable loss point

The native masked 008 run used 60 image-estimated cameras, 60 photo-derived
coarse masks, 641×512 final depth maps, and OpenMVS v2.4.0's default
`--fusion-filter 2` (dense-fuse). It produced 40,894 dense points. A later
recovered-camera 015 run produced 40,450 points, but its cameras and depth-map
size differ; that comparison cannot isolate fusion. The paired original-photo
pixel [depth-support audit](DEPTH-SUPPORT-AUDIT.md) located some 015 loss before
meshing, without establishing why.

Pinned [SceneDensify.cpp](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/SceneDensify.cpp)
has an explicit dense-fuse acceptance gate: a candidate is emitted only with
at least `nMinPixelsFuse` agreeing pixels **and** `nMinViewsFuse` distinct views.
The defaults are 5 and 2, respectively, in pinned
[DepthMap.cpp](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/DepthMap.cpp).
The [DensifyPointCloud CLI](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/DensifyPointCloud/DensifyPointCloud.cpp)
exposes `--fusion-filter 2` for this path and `--fusion-filter 1` for the
simpler view-consistency fusion path. Filter 1 still enforces a minimum number
of agreeing views, but follows a different algorithm and may admit noise. The
gate is a plausible *controllable* coverage loss, not a proven explanation of
the 008/015 difference.

The source checks for an existing `depthNNNN.dmap` before estimation. With
`--geometric-iters 0` and default `--optimize 0`, it can fuse a completed cache
without rerunning depth estimation or geometric passes. `--fusion-mode 1` is
**not** fusion-only; it exports depth maps and returns before fusion. Neither
arm will use it.

## Frozen one-variable experiment — approved and executed

Use **the same final 008 geometric depth maps** in two fresh, independent
working directories. Copy the exact 60 maps, `scene.mvs`, 60 undistorted images,
and 60 warped masks into each. Both arms use the existing, hash-checked
OpenMVS binary, 2 threads, resolution level 2, max resolution 1600, min
resolution 640, tower mode 4, ignore-mask-label 0, number-views-fuse 2,
geometric-iters 0, optimize 0, and fusion-mode 0. The only changing native
option is `--fusion-filter`: 2 for cached control, 1 for ablation. The control
is a *new cache-only* control, not a direct rerun of the original end-to-end
008 depth stage. No camera, mask, depth, or image changes are permitted.

`python3 -m scripts.classical_backend.openmvs_fusion_plan --control build-opencv/classical-ycb-fusion-control-016 --ablation build-opencv/classical-ycb-fusion-simple-017`

This command only reads and hashes inputs and prints a plan; it cannot execute
OpenMVS or create the output directories. On the current workspace, it
verified the pinned 008 report SHA-256
`8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9`,
the native binary hash, all 60 images/masks/maps, and **447,520,816 bytes**
of source files to copy per arm. Both planned outputs are fresh. Each arm
would have a 700 MiB hard output cap; combined new output ≤1,400 MiB plus a
256 MiB preflight buffer. The preflight requires free space ≥10 GiB reserve +
both caps + buffer. At the audit it observed 12,686,807,040 free bytes via
`shutil.disk_usage`, so the preflight passed narrowly. Recheck immediately
before any actual run; no workspace cleanup is authorized merely to fit it.
An approved executor must impose a wall-clock and RSS cap, inspect the native
stage logs, validate dense PLY payloads, and rehash every source/map before
and after copying and execution. The planner is not that executor.

The bounded companion is
`python3 -m scripts.classical_backend.openmvs_fusion_execute --control build-opencv/classical-ycb-fusion-control-016 --ablation build-opencv/classical-ycb-fusion-simple-017`.
It copies files independently, preserves a full JSON result in the control
arm, and limits both arms to 600 seconds total. At the later execution gate,
free space had fallen to **12,369,154,048 bytes**, below the required
**12,473,860,096 bytes** by 104,706,048 bytes. The planner rejected the run;
there are **no 016/017 native artifacts**. No caps were weakened and no data
was deleted to make it fit.

After the separately authorized storage relocation freed internal space, the
same planner rehashed the sealed 008 source and all **181 consumed files**:
`scene.mvs`, 60 undistorted images, 60 masks, and 60 final geometric depth maps.
The source result SHA remained `8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9`;
the OpenMVS binary SHA was `8ea970b0349270754d23f31a11dea87417c06d1001fa1ef437ee69b6146a1ce0`.
Free space was 16,427,102,208 bytes, above the unchanged 12,473,860,096-byte
preflight requirement. The 016/017 outputs were fresh. Twelve planner,
executor, and independent isolation tests passed before native launch.

The approved paired executor completed in **9.514 s** overall (including
copy/hash checks); native fusion took 1.544 s and 1.539 s respectively, with
2 threads, 4 GiB sampled-child-RSS cap, 700 MiB/arm output cap, 600 s global
deadline, and 10 GiB disk reserve. Both logs explicitly show 60 **cached
depth maps fused** and no new depth estimation. All source and arm-local copied
input hashes remained unchanged. The complete
[pair report](../build-opencv/classical-ycb-fusion-control-016/fusion-pair.json)
has SHA-256 `7bca23a2f64e3f3ee3af53228a38bac6f9bdb5d82e83806233c064444e052a2d`.

| Cache-only arm | Fused depths in native log | Points after fixed ROI trim | Dense PLY SHA-256 |
| --- | ---: | ---: | --- |
| `016`, filter 2 | 396,568 | 40,894 | `3c200ef4e7bfb2312af85cc2ebeb54aa9abb905a4895afff0d3291239d3bb271` |
| `017`, filter 1 | 674,518 | 120,147 | `7dc0f519bb002b28575c767ee4944392b60e345a65253878b67bc3c89a854288` |

### Fixed-gauge point-cloud diagnostic, not mesh or metrology

Before scoring, the diagnostic fixed the already selected `sensor-depth-003`
2,048 observed rays in each of three Berkeley frames (0°, 120°, 240°), its
coarse/interior support, the original 008 **named-camera** world-to-table
Sim(3), and the same Berkeley calibration/poses. Each arm's points are projected
to the depth camera, nearest-integer-pixel z-buffered, then queried within a
fixed **±2-pixel Chebyshev square**. A hit is a nearby cloud point, **not** a
first-hit triangle or a continuous surface. The larger cloud has an inherent
opportunity to hit more squares; this proxy is not directly comparable to the
rough-mesh sensor-depth percentages. Missing predictions remain missing.
Separately, 4,096 uniformly sampled cloud points (seed 2027) were measured to
exact reference triangles with the unchanged **008 reference-fitted** Sim(3).
That scanner-relative point metric is a development diagnostic with an
unreliable fit and unverified reference units, not independent physical
accuracy. No transform, support, radius, threshold, or sample seed was tuned
after results.

The first [diagnostic 001](../build-opencv/classical-ycb-fusion-point-diagnostic-001/report.json)
(SHA-256 `125b39404adec41332e918456266f98f3d8f52a970da86abceb1718edfb6f7f4`)
was run before full alignment/source rehash and RSS enforcement were reviewed.
It is retained as **provisional, not accepted evidence**. The repaired code
bound the complete fit, camera, projection, calibration, reference, pair,
sensor, and both cloud files before and after work. Fresh
[supervised diagnostic 002](../build-opencv/classical-ycb-fusion-point-diagnostic-supervised-002/diagnostic/report.json)
completed in 14.51 s under a 120 s, 2 GiB sampled-RSS, 20 MiB output limit;
peak sampled child RSS was 224,624,640 bytes. Its report SHA-256 is
`d0fed1078700a3e566e56426cbef4651777d40cdfeefad8bbd689c23424e4e23`.
Three new analytic point-projection/transition/PLY tests and all twelve prior
fusion tests passed.

| Fixed sensor panel | Supported rays | Filter 2 hits | Filter 1 hits | Shared hits | Filter 1 gained / lost | Shared-hit MAE, filter 2 / 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse | 6,144 | 5,512 | 5,525 | 5,511 | 14 / 1 | 9.060 / 10.913 mm |
| 16-pixel interior | 3,787 | 3,566 | 3,571 | 3,565 | 6 / 1 | 6.916 / 8.050 mm |

The 14 newly supported coarse rays have **76.745 mm** mean absolute
disagreement; the six newly supported interior rays have **14.724 mm**.
Under the frozen 008 scanner fit, sampled point-to-reference mean distance is
0.00958 versus 0.00988 reference units (filter 2 versus 1), with p95 0.03429
versus 0.03515. Filter 1 thus emits about 2.94× as many points but yields
almost no new fixed-ray support and larger residuals on the shared hits in
this *point-cloud proxy*. It does not establish a better reconstruction;
nor can this test alone attribute every bad point to the fusion filter rather
than upstream depth/map/calibration uncertainty. No meshing was run.

Success would mean valid output clouds under the same source/limits; the
comparison should report point count, per-view support, and independent
geometry/sensor diagnostics, with any extra coverage balanced against outliers.
Do not claim filter 1 is a better reconstruction from point count alone. If
both fusion paths lose the same pixels, the likely loss point is earlier
(depth estimation/geometric consistency/masking), and this hypothesis is not
supported. The existing 008 baseline and all 015 evidence remain untouched.

## Decision and bounded next step

**Keep filter 2 as the default.** Filter 1's much larger point cloud did not
produce meaningful fixed-ray support gain and its shared-hit residuals were
worse under this frozen diagnostic. No OpenMVS source patch, threshold search,
or further reconstruction arm is justified by these results. A future quality
investigation should first examine upstream depth-map/object-mask support and
camera/calibration uncertainty with independent fixed observations, rather
than counting additional fused points as recovered object surface.
