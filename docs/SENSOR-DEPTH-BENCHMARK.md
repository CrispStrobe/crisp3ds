# Frozen three-view sensor-depth benchmark protocol

## Next candidate: recovered image-only cameras (015)

Pre-score plan: use the sealed `initialization-recovery-005-delayed-refinement`
camera model for one new native rough-mesh arm, `classical-ycb-recovered-015`.
Keep the existing 60 photo-derived masks and the 013 level-2/min600/max1600,
two-geometric-iteration profile; warp masks anew for these estimated intrinsics.
The bounded native run may fail and must retain that status. It is a cached
sparse-model continuation, not fresh end-to-end reconstruction.

If the rough mesh and all depth-camera/mask checks pass, compare it against
rough 008 and rough 014 using exactly the same frozen rays, support panels,
thresholds and depth assumptions below. First bind a new camera-only Sim(3)
to the exact 005 model and existing Berkeley metadata. Never transfer a prior
transform into this new gauge or fit a mesh to improve the score. Reference
metadata is permitted only in this post-reconstruction evaluator. Report camera
fit diagnostics alongside geometry residuals; no parameter retries are selected
from these scores. This is a repeatedly inspected development object, not an
unseen acceptance test.

Outcome: the three-candidate `sensor-depth-004` run completed. Its rays/support
are identical to 003, whose two baseline pooled results were reproduced exactly.
Recovered 015 did not beat 008: coverage 80.58% versus 84.08%, hit-only mean
absolute depth disagreement 2.922 versus 2.881 mm. See the full
[rough-mesh comparison and limitations](RECOVERED-DENSE.md#subsequent-fixed-ray-sensor-comparison-not-a-quality-win).

This protocol is fixed before any reconstructed-mesh sensor-depth score. Use
only the already extracted Berkeley NP3 depth frames at 0°, 120° and 240°,
the previously prepared **photo-derived** NP3 RGB pose-support masks, the
Berkeley calibration/NP5 per-angle poses, and a separately named-camera-fitted
proper Sim(3) for each candidate. No Google mesh, supplied depth, Berkeley
pose, or Berkeley mask was used to reconstruct the candidate. The evaluator
does **not** fit or adjust a mesh against these depths. Initial candidate
comparisons are the prior 008 native-masked mesh and, if complete and sealed,
the calibrated 014 mesh. Each candidate must match the sparse-model hashes
bound to its own named-camera transform report; no transform is transferred
between different camera gauges without exact sparse-model identity.

The source `/depth` HDF5 arrays are 640×480 uint16. Zero means missing, not
zero distance. The actual NP3 IR depth scale is 1.0016965552242483 and IR
depth bias is exactly zero. Convert this pinned dataset as
`z = raw × 0.0001 × NP3_ir_depth_scale` metres, using the published 100 µm
unit. Applying the local IR-scale field this way is a frozen,
assumption-qualified interpretation; the original processing script was not
reachable to prove application order. The 0.17% scale correction is about
1 mm at the observed distances. The distinct generic `NP3_depth_scale` is 1
and is not additionally applied; nonzero bias units are unverified and
rejected. Depth rays are
`((u-cx)/fx, (v-cy)/fy, 1)` at integer depth pixel addresses and use
`NP3_depth_K`; do not apply IR lens distortion again to the depth map. Map
table points to the depth camera with
`H_NP3_ir_from_NP5 × inverse(H_table_from_reference_camera)` and the mesh
into the table frame with the candidate's named-camera Sim(3). The RGB
support mask is addressed by projecting each **measured** valid depth point
through `H_NP3_from_NP5 × inverse(H_NP3_ir_from_NP5)` with the supplied
`NP3_rgb_K` and Brown–Conrady `NP3_rgb_d`, then nearest integer RGB pixel.
The independent RGB–depth projection check supplies analytic checks and
plausible overlays under the explicit integer-address/rectified-depth
assumptions, hash-bound to these exact inputs. This does not conclusively
verify a pixel-center convention or certify calibration; RGB/depth
disocclusion and calibration error remain uncertainties.
The supplied `NP3_ir_d` is nonzero: the thesis discussion of the convolution
offset does **not** establish that stored depth pixels are distortion-free.
Treating their grid as rectified pinhole coordinates without another IR
undistortion is a frozen unresolved assumption, not a verified correction.

Support is independent of candidate geometry: (A) the existing coarse
photo-only reddish-package scanline masks and (B) a separate fixed 16-RGB-pixel
binary erosion of those masks. Neither uses Berkeley's model-generated RGB
masks. This is a **support-conditioned object-interior proxy**, not a proven
object-only segmentation: table/background depth may reproject into the mask
at occlusion boundaries, and the erosion can omit legitimate object surface.
Panel B is an interior sensitivity analysis, not whole-object completeness.
No mask, crop, or erosion radius is changed after viewing candidate residuals.

For each frame, enumerate **all** depth pixels in row-major order. Eligible
pixels have nonzero observed depth between 0.2 and 1.5 m,
valid RGB reprojection, and panel-A mask support. Draw at most 2,048 eligible
pixels without replacement using NumPy PCG64 seed `20260927 + angle`, then
sort linear pixel indices; every candidate uses these exact IDs. Panel B
uses only selected A rays that also land in the eroded mask, never a second
candidate-specific draw. Report complete frame/valid/eligible denominators,
selected IDs and their SHA-256, sampled fraction, and per-panel selected counts.
An insufficient/empty panel remains unavailable; it is not filled by changing
support or sampling.

Pre-score input-only preparation through the sealed projection-002 gate yielded
eligible counts 17,595 / 8,958 / 10,572 and 2,048 selected rays per 0° / 120° /
240° frame, respectively. SHA-256 of sorted little-endian uint32 linear IDs:
`3944d636194653b6d6c832dfd91262a6231ece210050583fec2d1641c2b45f68`,
`6cdc1c81278dcf8302be4bcb3cbd9f59d1121ecdd1a9f4db357542dc117d5c7d`,
`0cf8f4aaacda2785d4aa5b78f95313e94dcb0d3e5623c8d547e64873319fabaf`.
The fixed interior subsets contain 1,622 / 1,107 / 1,058 rays. These counts
and hashes were recorded before any candidate residual was computed.

For each selected ray, render the **first positive double-sided triangle
intersection** from the transformed output mesh in the supplied IR/depth
frame, within 0.2–1.5 m. Use a bounded triangle AABB tree, no backface
culling, and exclude/count exactly zero-area triangles. Rays with no hit are
missing predictions, not artificial zero-depth residuals. For each frame and
panel report hit/no-hit counts and hit fraction of observed-supported rays;
on rays with both observed and predicted depth report signed residual
`predicted − observed` (positive means farther from the camera), absolute
mean/median/p90/p95, signed mean/p5/median/p95, and fractions within
5/10/20 mm both among hits and among all selected supported rays. Include
pooled summaries without hiding per-view failures. These are conditional
sensor-depth agreement/coverage diagnostics, not certified metric accuracy,
whole-object completeness, or a product-quality acceptance gate.

Resource bounds: at most three pinned frames, 2,048 rays per frame, at most
the existing 2-million-face/1-million-vertex evaluator geometry cap,
300 seconds total for a real comparison, ≤20 MB result JSON, no new Python dependency or
download, and at least 10 GiB free disk. Unit tests include known planes,
first-hit occlusion, parallel/miss rays, proper transforms and mask/sample
determinism. Real scoring remains disabled until coordinate and provenance
gates pass and the root explicitly approves it. The 003 named-camera
transform is bound only to the original 008 sparse source; calibrated 014
requires its own exact-model-hash-bound transform against the same pinned
reference poses. A fitted global camera Sim(3) plus supplied per-view table
poses conflates geometry, reconstruction-camera, and reference-pose errors in
the residuals; it is not pure physical shape accuracy. No pose is adjusted
from candidate mesh residuals. The initial live arm is frozen to stage-matched
native **rough** 008 versus native **rough** calibrated 014; the 014 refined
stage timed out, so no refined-to-rough quality ranking is attempted. Native
meshes are losslessly exported to bounded evaluator-compatible triangle PLYs,
with source and output hashes. The runner also binds producer-result hashes,
checks exact named dense-camera poses against each fitted source model,
rechecks critical inputs after scoring, and checkpoints completed candidates
or failure status under the 300-second total cap.

Independent numerical checks before live scoring: 2,000 random ray/triangle
cases agreed with brute-force barycentric intersection to 2.22×10⁻¹⁶ m,
including hit/miss flags. One thousand random RGB projections agreed with
OpenCV Brown–Conrady `projectPoints` to 1.14×10⁻¹³ pixel. Berkeley documents
turntable-pose/depth/normal ambiguities; 5-mm sensor rates must not be read
as a global mesh-quality percentage.

## Results

The final, per-ray-auditable [report](../build-opencv/sensor-depth-003/report.json)
has SHA-256 `40b1d821b9e7c8a9eaddf6946815922af93c3d72bacee25cfff0006dd3b0e8a7`
and status `complete`. It stores each selected pixel ID, observed depth,
predicted depth (`null` for no hit), and interior membership. Both native rough
meshes were normalized losslessly in the ignored output directory, leaving
their sources unchanged. The full comparison took about seven seconds, below
the one-batch 300-second cap. The earlier
[001](../build-opencv/sensor-depth-001/report.json) and
[002](../build-opencv/sensor-depth-002/report.json) reports remain preserved:
001 lacked pooled/per-ray fields; 002 added pooled fields; 003 added per-ray
serialization. All six per-view score summaries in 003 exactly equal 001,
and both pooled summaries exactly equal 002. The selected ray hashes never
changed.

| Fixed support, pooled over 0°/120°/240° | rough 008 | rough calibrated 014 |
| --- | ---: | ---: |
| Coarse mask: first-hit coverage | 5,166/6,144 (84.08%) | 5,140/6,144 (83.66%) |
| Coarse mask: hit-only absolute mean | 2.88 mm | 5.96 mm |
| Coarse mask: ≤5 mm among *all supported rays* | 70.72% | 38.56% |
| Eroded interior: first-hit coverage | 3,506/3,787 (92.58%) | 3,568/3,787 (94.22%) |
| Eroded interior: hit-only absolute mean | 2.34 mm | 5.64 mm |

Per-view coarse hit-only absolute means were 1.93 / 3.32 / 3.32 mm for 008
versus 4.10 / 7.71 / 5.91 mm for 014; thus the pooled difference is not
confined to one view. Pooled signed means were −0.42 mm and +2.72 mm,
respectively (predicted minus observed). Pooled means weight individual hit
rays, not view-level medians or percentiles. The 014 interior has slightly
more first hits but larger depth errors; coverage and residual accuracy are
different quantities.

On these fixed three views, rough 008 agrees more closely with Berkeley
sensor depth than rough 014 **under this camera/pose/calibration chain**.
This does not prove that 008 has better physical shape or generalize to
unobserved surfaces: the support mask is only photo-derived, 16-pixel erosion
is an interior proxy, missing predictions differ, depth-to-RGB projection
and IR-scale application are assumption-qualified, and the global camera
Sim(3) plus supplied table poses can contribute millimetre-scale error.
Neither supplied depths nor poses entered either reconstruction; 014 did
use supplied camera intrinsics and is not an image-only result. Depths and
poses are post hoc diagnostic references, not certified ground truth.
