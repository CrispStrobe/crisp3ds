# OpenMVS depth/fusion coverage investigation

Status: read-only source audit and preflight only. No fork was built, no native
reconstruction was launched, and no quality improvement is claimed.

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

## Proposed one-variable experiment — requires separate approval

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

Success would mean valid output clouds under the same source/limits; the
comparison should report point count, per-view support, and independent
geometry/sensor diagnostics, with any extra coverage balanced against outliers.
Do not claim filter 1 is a better reconstruction from point count alone. If
both fusion paths lose the same pixels, the likely loss point is earlier
(depth estimation/geometric consistency/masking), and this hypothesis is not
supported. The existing 008 baseline and all 015 evidence remain untouched.

## Fork/patch mechanism, if the paired test warrants it

No source rebuild is presently justified or feasible under the tight Mac disk
reserve; the available tool is the pinned v2.4.0 binary. If the controlled
comparison shows a material fusion-only difference, a small future fork could
add **counters only** at the dense-fuse acceptance branch in
`libs/MVS/SceneDensify.cpp`: candidates with no valid depth, below 2 views,
below 5 agreeing pixels, and emitted points, per input view. Counters would
be thread-local and reduced at the end; no candidate decisions or floating
point operations would be altered. A parity run against the pinned binary
must confirm identical depth-map and cloud hashes before interpreting those
diagnostics. Threshold or algorithm changes would be a separate predeclared
experiment, never silently folded into this instrumentation.
