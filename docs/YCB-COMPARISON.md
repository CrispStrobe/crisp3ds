# YCB cracker-box development comparison

Protocol started 2026-09-27 before inspecting any reconstructed YCB surface.
This is a development case, not held-out acceptance. Source/photo/reference hashes
and attribution: [dataset manifest](../tests/datasets/ycb_cracker_box.json).

The latest [fresh 60-photo camera-gated OpenMVS run](TURNTABLE-COMPLETE-PLAN.md#fresh-complete-openmvs-oracle-and-shape-result)
completed full-resolution dense reconstruction through texture on this M1 and
passed the 60-view mask audit, but its bare mesh remains rejected: F1@1% is
42.23% after its own reference fit and 30.92% in the camera-transported older
`006` gauge. This is not a KIRI comparison or independently calibrated metric
accuracy. The [AliceVision M1 audit](ALICEVISION-MAC-ORACLE.md) ran an isolated
Metal depth test, not an AliceVision photo-to-mesh pipeline.

## Latest fixed-camera ablations

The [consolidated comparison](BENCHMARK-RESULTS.md) and
[shared-alignment diagnostic](SHARED-GAUGE.md) add native OpenMVS depth masks
and a full-resolution run. Native masks were verified in all 60 depth maps:
positive depths outside the masks fell from 16,261,310 to zero. These masks
are still photo-derived support heuristics, not ground-truth silhouettes.

At 641 × 512, independent-fit F at 1% diagonal is 42.04%, versus 38.15% for
the earlier cloud filter. Under the unchanged cloud-filter alignment, however,
it is 35.49%. Normalized symmetric mean distance also worsens from 2.525% to
3.053%. Do not present this as a clear quality improvement.

Full-resolution 1282 × 1024 depth produces 151,387 fused points and 10,810
refined faces, but almost unchanged scores: 42.06% independently fitted F,
35.52% under the shared alignment, and 3.128% normalized mean distance.
The first attempt (`009`) exceeded a declared 2 GiB output budget. A distinct
hash-verified continuation (`011`) reused all 60 complete initial depth maps,
excluded partial geometric maps, then reran both geometric passes, meshing,
refinement and texturing under a 3 GiB cap. Its timing is composed, not fresh e2e.

Both native-mask meshes are closed single components without counted
nonmanifold edges. That topology does not establish correct shape; quality
remains rejected. A fresh explicitly seeded image-to-SfM replay (`010`) also
registered 60/60 views, but automatic initialization remains unresolved and
that replay did not itself run dense reconstruction. See
[native commands, failures and provenance](CLASSICAL-BACKEND.md).

An additional [Berkeley camera-reference diagnostic](YCB-REFERENCE-FRAMES.md)
checks trajectory and intrinsics independently of the Google mesh. It does not
provide a known transform between that mesh and the photo reconstruction.

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
`build-opencv/object-motion/foreground`. The complete dense-backend run reuses
those exact cameras and photo hashes.

### First complete classical surface: rejected, contaminated geometry

`classical-ycb-foreground-002` was interrupted at 603.005 s by the wrapper's
resource scanner racing a native temporary-file rename. Native base depth maps
were retained. A fresh `003-continuation` copied only completed base maps, omitted
partial geometric-pass maps, and fused them with further geometric iterations
disabled. The reimported scene is byte-identical to the original scene. Native
fusion succeeded with 56,684 points, but the old validator rejected OpenMVS PLY
type aliases/visibility lists. Both failures remain recorded, not overwritten.

After extending strict PLY validation, `004-finish` explicitly passed the dense
PLY to meshing and the resulting mesh to refinement/texturing. This matters
because the interface scene alone still contains the original sparse points.
Meshing produced 22,320 vertices/44,561 faces; one-scale refinement and texturing
produced 3,359 vertices/6,612 faces and one 1024-pixel JPEG texture. Stage times
were reimport 0.51 s, mesh 3.57 s, refine 92.05 s, texture 6.18 s. These are
composed-stage recovery times, not a fresh one-command performance benchmark.

The final native geometry was normalized without changing any coordinate or
triangle. Native SHA-256: `2b6023960367f0ded70f8840638f9cd7d0a1ea62ec6e4a157dab3d31b91adfc9`;
normalized SHA-256: `f571ba83f2fc7c7284b1a8922af0369b3cdf041187c08e67b03e2edfcf88cfd9`.
The frozen 1024-sample reference fit found scale 0.0827699088. At the declared
2048-query scoring settings, whole-mesh F is 6.57%, 12.77%, 23.98% at 0.5%, 1%,
2% diagonal. At 1%: precision 14.79%, recall 11.23%. Root's bare-shape preview
shows substantial non-object/background geometry. This is **rejected quality**,
not competitive object reconstruction. Alignment contamination/uncertainty also
limits the score; do not reinterpret it as a percentage of KIRI performance.

Artifacts: `build-opencv/classical-ycb-foreground-004-finish/reference-fit-v1.json`,
`surface-metrics-v1.json`, `textured.obj`, and local
`.local-tools/classical-ycb-shape-preview.png`.

### Next frozen ablation: multi-view photo support before meshing

Before inspecting any filtered result, fix the rule: project each original dense
point into all 60 original distorted cameras and retain it only if at least
48/60 projections fall within the frozen photo-derived pose-support masks.
Behind-camera and outside-image projections count as unsupported. No reference
scan, fitted scale, supplied geometry or GT-derived bounds enter this filter.
Keep all retained original PLY record bytes (including colors/normals/visibility)
and alter only the count and selection. Keep the same native mesh/refine/texture
settings as `004` and score by the same frozen protocol.

This is a visual-support constraint using coarse masks, not an exact silhouette
or visibility oracle. It can remove real geometry near imperfect masks; report
retained/rejected counts and completeness, not only improved precision. Preserve
the unfiltered mesh and do not tune the 48-view threshold against reference scores.

### Frozen filter result: improvement, still rejected

The rule retained 34,172/56,684 points. Retained native vertex records are
byte-identical; an independent repeat produced the same filtered SHA-256.
`006-filtered` completed with 2,225 vertices/4,404 faces and one texture.
Reimport/mesh/refine/texture took 0.51/1.53/66.45/4.62 seconds. These exclude
camera estimation, depth reconstruction and filtering; they are not e2e times.

| Whole-mesh threshold | Unfiltered F | Filtered F | Filtered precision | Filtered recall |
| --- | ---: | ---: | ---: | ---: |
| 0.5% reference diagonal | 6.57% | 19.48% | 22.07% | 17.43% |
| 1% reference diagonal | 12.77% | 38.15% | 42.63% | 34.52% |
| 2% reference diagonal | 23.98% | 63.54% | 69.68% | 58.40% |

Root independently normalized, aligned, scored and inspected the bare surface.
The box shape is much clearer, but substantial missing/distorted geometry remains:
**quality is still rejected**. Both geometry and reference-fitted alignment change;
this is not a physical accuracy claim or a percentage of KIRI quality. The frozen
fit scale is 0.1638040991. Normalized geometry SHA-256:
`b2b410156b4b2386962c9c9e88f59a5f19eef102874e6d3ae6321c3c853dbaff`.
Reports reside in `build-opencv/classical-ycb-foreground-006-filtered/`;
preview: `.local-tools/classical-ycb-filtered-preview.png`.

### Fresh replay reliability remains unresolved

`005-e2e` attempted the standalone masked pipeline from photographs and stopped
at its registration gate with only 2/60 cameras and 357 points. The earlier
successful camera trial and this replay have matching feature counts but slightly
different match totals; separate-process execution/random state are hypotheses,
not established causes. Preserve this failed replay in reliability reporting.
Successful composed recovery through texture does not establish reliable one-command
e2e operation. Next priority: reproduce and stabilize image-only alignment, improve
object masks/depth support without reference leakage, then test held-out objects.

### Additional stage diagnostic: rough surface versus refinement

Before inspecting new scores, freeze an additive stage comparison: normalize
the native `mesh.ply` outputs of `004` and `006` without changing vertices or
triangles, fit each with the existing 1024-sample alignment procedure, and score
with 2048 samples, seeds 2027/2028 and unchanged 0.5%/1%/2% thresholds. The goal
is to determine whether the one-scale refinement is losing object surface;
this is not another reconstruction engine. Retain rough and refined outputs,
including their separate fitted alignments. No reference information feeds back
into native reconstruction or determines which triangles to keep.

Root's stage scores (each with its own frozen-algorithm reference fit):

| Variant | Rough F at 1% | Refined F at 1% | Rough normalized mean distance | Refined normalized mean distance |
| --- | ---: | ---: | ---: | ---: |
| 004 unfiltered | 14.95% | 12.77% | 5.331% | 5.704% |
| 006 photo-support filtered | 33.87% | 38.15% | 2.496% | 2.525% |

Mean distance is half the two directed means, divided by reference diagonal.
Refinement improves filtered F but slightly worsens its mean distance; the poor
surface is already present before refinement. Thus the large face-count decrease
is not evidence that refinement alone caused the failure. These are not identical
alignment transforms, and small metric changes include sampling/alignment effects.
Both rough outputs remain rejected. Reports: `rough-reference-fit-v1.json` and
`rough-metrics-v1.json` in their respective original run folders.
