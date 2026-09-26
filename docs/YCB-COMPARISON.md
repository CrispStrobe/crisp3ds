# YCB cracker-box development comparison

Protocol started 2026-09-27 before inspecting any reconstructed YCB surface.
This is a development case, not held-out acceptance. Source/photo/reference hashes
and attribution: [dataset manifest](../tests/datasets/ycb_cracker_box.json).

## Inputs and lanes

All 60 NP3 real JPEGs, 1280x1024, original bytes, full turntable revolution.
No scan geometry, depth or supplied poses may initialize reconstruction. Acquisition
order is permitted; the recorded nominal angles are not fixed camera poses.
Separate Google scanner geometry is an evaluation oracle with unverified physical
units/cross-sensor alignment. Whole-reference comparisons include the recorded
reference population; do not silently crop it to match an incomplete output.

1. **MVE raw control:** selected CMake tools, 60 views, scale 2, standard runner
   settings, 1 GiB output cap and 5-minute global timeout. Unknown cameras, original
   photos, no masks. Runtime is a bounded trial, not best attainable MVE quality.
2. **Classical raw/shared camera:** PyCOLMAP 3.11.1, SIMPLE_RADIAL, SINGLE shared
   camera, seed 20260927, SIFT max image size 1200 and 1800 features, two CPU
   threads, sequential overlap 8 without quadratic overlap or loop detection,
   one incremental model. Reuse this exact sparse output for OpenMVS conversion
   and record its provenance rather than estimate cameras a second time. This
   bounded matching configuration is not a claim of the best possible baseline.
3. **Object-support ablation:** same camera and matching settings with an
   image-only support mask. A red-packaging heuristic is object-specific and only
   a pose-support experiment, not a general segmentation algorithm or exact mesh
   silhouette. Do not attribute combined camera/backend/mask changes to one factor.

The raw cross-engine comparison measures complete configured pipelines, not an
isolated dense algorithm. Dense-only conclusions require identical frozen cameras.
Run heavy experiments sequentially on the M1 where practical; retain resource load
and contention caveats. A timeout is reported as a timeout, never a quality score.

## Scoring and review

Use the existing frozen proper-Sim3 alignment procedure (1024 alignment samples,
seed 2026, 24 proper PCA seeds, trimmed fitting, 60-second limit). Retain the
unaligned output and transform; this is shape normalization, not recovered scale.
For outputs that exceed evaluator geometry limits, report the blocker rather than
silently simplifying/cropping them. Any simplification must be a separate variant.

Score independent area-weighted samples against target triangles: 2048 queries in
each direction, seed 2027/2028, at 0.5%, 1%, 2% of the reference bounding-box
diagonal. Run a reference self-control. Report direction-specific median/p90,
precision, recall and F-score. Sampling approximates a surface integral; exact
target-triangle distances do not remove sampling or reference uncertainty.

Root inspects bare mesh projections before accepting a result. Check recognizable
box shape, planar faces, edges, background/turntable contamination and missing
surfaces. Camera count and triangle count are structural diagnostics only.

## Results

### MVE raw control: failed camera initialization

Root ran the frozen 60-image trial in `build-opencv/mve-ycb-raw-001`.
Import succeeded in 1.57 s. SfM failed with exit 1 after 265.20 s (267.14 s
total), before the five-minute limit. After all-pairs matching it selected views
2 and 7, triangulated zero tracks, rejected 33 for large error and 28 for unstable
angle, then reported `No Jacobian given` during bundle adjustment. No dense cloud,
mesh or surface score exists for this run. Peak single-child RSS was 172.56 MiB;
this is not aggregate process-tree memory. Failure logs and result JSON are retained.

This is evidence against accepting this raw configuration, not proof that every
MVE configuration fails or that the competing pipeline succeeds.

### COLMAP raw/shared camera: insufficient registration

The frozen sequential/shared-camera arm wrote a model with only 2/60 views
(162 and 186 degrees), 297 points and 594 observations. Root independently
loaded the binary model. Focal length was 4584.95 pixels (3.58 image widths)
and SIMPLE_RADIAL distortion -5.37008. Of the tracks, 54 were wholly inside
the photo-derived support region, 6 mixed and 237 wholly outside. This is not
an accepted object-frame camera solution and cannot pass the dense adapter's
three-camera/70%-registration gate.

The wrapper failed at 178.44 s because diagnostics called a PyCOLMAP integer
property as a method. The source bug was fixed; a diagnostic-only continuation
read and hashed the existing model, retaining the original failure status and
without rerunning or repairing SfM. This distinguishes a reporting bug from the
underlying inadequate two-camera result.

### Foreground/shared camera: complete sparse coverage, not yet mesh acceptance

Changing only feature eligibility to the photo-derived support masks registered
60/60 views, with 3,939 points and 24,396 observations. The worker reported
93.80 s and wrapper 96.19 s. Root independently loaded the binary model: mean
reprojection error 0.50923 pixels, mean track length 6.19345, focal 1077.58 pixels
(0.842 image widths), radial parameter -0.00723354, finite points. The camera
center covariance has two large eigenvalues (6.907, 7.170) and a small third
(0.000317), consistent with an approximately planar orbit but not independent
pose truth.

3,909 tracks are wholly inside the support masks, 29 mixed and 1 wholly outside;
24,362 observations inside versus 34 outside. Masks are checked at the image
keypoint coordinate, while feature support/rounding can cross the boundary. The
coarse mask can also include non-object pixels. This is strong evidence that
foreground selection fixes this camera-estimation failure, not yet proof of
correct surfaces or a general segmentation solution. Root also inspected the
180-degree photo and mask: most carton pixels are retained, with small edge
exclusions and possible checkerboard leakage.

The sealed binary model and producer metadata are under
`build-opencv/object-motion/foreground`. The complete dense-backend run will
reuse those exact cameras and photo hashes. Surface acceptance remains pending.
