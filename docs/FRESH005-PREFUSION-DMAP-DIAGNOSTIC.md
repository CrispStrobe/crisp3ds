# Fresh 005 pre-fusion DMAP inventory

The sealed fresh 005 run has 60 final geometric OpenMVS depth maps. A read-only
inventory used the pinned `result.json` SHA-256
`dafc6ccb2a90f0c5a45748291571455237acd8ae08da01dd80adc10222a3625e`,
the run's own per-map hashes, the camera-check report, and the mask report.
It parsed the depth and confidence channels documented in OpenMVS v2.4.0's
[`HeaderDepthDataRaw`](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/Interface.h),
with exact file-length and image-order checks. The diagnostic does not launch
OpenMVS or write reconstruction data.

`python -m scripts.classical_backend.fresh005_prefusion_diagnostic`

Across 80,310,240 DMAP pixels, 2,607,531 have positive depth (3.25%). The
per-view positive-depth fraction ranges from 2.07% to 4.36%, with a median of
3.22%. No positive-depth pixel has zero confidence. The per-view median raw
confidence ranges from 0.857 to 0.945; the per-view median camera depth ranges
from 5.344 to 5.830 in the run's unverified world units. These distributions
are descriptive. They do not define a confidence cutoff or metric box size.

The existing camera check verifies all 60 DMAP cameras against the undistorted
COLMAP model and reports zero positive-depth pixels outside the warped masks.
The inventory checks that every DMAP hash and embedded image name matches the
ordered camera-check rows. The original dense PLY is also checked against the
sealed run result; it remains the unmodified control.

These data cannot identify which DMAP pixels produced a particular fused point.
The PLY and retained DMAP reports do not carry per-point source view/pixel IDs,
agreement counts, or the individual depth/confidence values used by fusion.
Thus the inventory can confirm that positive masked depth exists before fusion,
but cannot attribute the perceived wide box to dense depth versus fusion. A
future causal test needs a read-only fusion trace or instrumented evaluation
build that records those inputs and rejection/aggregation reasons for each
candidate, bound to the same sealed maps and cameras. The earlier 008
filter-1/filter-2 paired experiment remains the available filter ablation; its
larger filter-1 cloud did not meaningfully increase fixed sensor-ray support.
