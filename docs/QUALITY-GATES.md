# Paired depth quality gate

This gate compares two MVE radial depth maps from the **same rendered scene, reference view, resolution, and estimated camera**. It is a provisional development experiment, not a physical product tolerance or approval for production use. Ground-truth planes enter only this scorer. They must never tune the matcher or choose a candidate after inspecting its score.

Run from the repository root, with `TMPDIR=.local-tools/tmp` if scratch space is needed:

```sh
python3 -m unittest scripts.quality_gate.test_compare_depth -v
python3 scripts/quality_gate/compare_depth.py \
  --baseline-depth build-opencv/mve-spike/scene-supervised/views/view_0001.mve/depth-L2.mvei \
  --baseline-scene build-opencv/mve-spike/scene-supervised \
  --candidate-depth build-opencv/mve-masked/scene/views/view_0001.mve/depth-L2.mvei \
  --candidate-scene build-opencv/mve-masked/scene \
  --fixture build-opencv/synthetic-sparse-fixture --view 1 \
  --baseline-binary .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --candidate-binary .local-tools/mve-masked/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --output build-opencv/mve-masked/paired-quality-v2.json
```

The scorer refuses a changed output size or any changed MVE camera value. It also checks that the camera agrees with the fixture's *estimated sparse pose* and calibration, and refuses nonzero lens distortion. A pose-refined or differently scaled run is **unpaired** for this gate. It may be described separately, with its own fixed truth population and geometry accounting, but its scores must not be passed off as a pixelwise paired improvement.

For each pixel, a ray from the estimated MVE camera intersects the visible synthetic object rectangles. The nearest positive hit is the truth depth. MVE stores Euclidean camera-to-point range along the normalized ray; the scorer converts the plane intersection to that radial convention. All lengths are millimetres. The fixed truth pixels define all object pixels, separate two-pixel inside-object silhouette and surface-step bands, their union as the boundary band, and the remaining interior. Adjacent pixels hitting different planes mark a surface step. This includes object pixels on which either algorithm emits no depth. The scorer records valid depths outside the object as an object-isolation diagnostic; the synthetic planes do not provide a complete background truth, so those depths are not classified as matching errors.

Each region reports coverage, missing count, matched-pixel MAE and p95, and missing-inclusive bad-1/bad-2/bad-5 fractions. A missing, nonpositive or nonfinite depth counts as bad at every threshold. A valid depth is bad at a threshold only when its absolute error is **greater than** that threshold. The denominator is always the fixed ground-truth pixel count. Shared-support MAE is also reported, with its fraction of truth pixels, and is explicitly labeled selection biased: deleting difficult predictions can improve it without improving reconstruction quality.

The predeclared provisional pass condition requires all three checks:

1. At least 25% relative reduction in full-object missing-inclusive bad-2, compared with the paired baseline.
2. Full-object coverage loss no larger than two percentage points.
3. No increase in boundary-band missing-inclusive bad-2.

The pass decision only selects a candidate for further development. It cannot establish real-world accuracy or completeness: this fixture uses two simple rendered planes and one view, and its estimated camera has pose error. A measured multi-view reference and independent held-out captures are required before any physical tolerance or backend selection claim.

## Paired mask experiment, 2026-09-26

The independent paired score is [`paired-quality-v2.json`](../build-opencv/mve-masked/paired-quality-v2.json). It records hashes for depths, metadata, fixture inputs, conversions, and both dense binaries. The camera metadata files have the same SHA-256; both depth maps are 400×300. This fixture's two visible rectangles are separated in the reference projection, so the surface-step band is empty; the boundary band here consists of the silhouette band.

| Fixed full-object population (7,670 pixels) | Baseline | Masked MVE |
| --- | ---: | ---: |
| Coverage | 94.20% | 76.53% |
| Missing-inclusive bad-2 | 66.77% | 73.44% |
| Matched MAE | 3.098 mm | 2.624 mm |
| Matched p95 | 5.562 mm | 5.139 mm |
| Boundary bad-2 (1,022 pixels) | 78.18% | 100.00% |
| Valid depths outside the object-plane truth | 36,927 | 0 |

All three provisional checks fail. The mask isolates the object-plane output but removes every boundary-band prediction. On the 5,870 pixels shared by both outputs, MAE changes only from 2.629 to 2.624 mm; the larger matched-only MAE improvement mostly reflects deletion of difficult predictions. This result is a diagnostic for the tested mask implementation, not a verdict on all possible mask handling.

The independent [L1 paired score](../build-opencv/mve-masked/paired-quality-L1.json) repeats the conclusion at 800×600: fixed full-object bad-2 rises from 52.15% to 56.51%, coverage falls from 98.46% to 89.46%, and boundary bad-2 rises from 63.39% to 100%. Matched MAE moves from 2.133 to 2.104 mm, while on the 27,309 shared pixels it changes only from 2.10424 to 2.10422 mm. Thus the matched-only improvement is again almost entirely due to removing predictions.

## Independent ArUco corner-refinement fixture

The existing R02 rendered fixture has a possible corner convention bias: its marker texture pixel centres were mapped to physical marker corners. To test the subpixel corner-refinement hypothesis without that bias, [`pose_fixture.cpp`](../scripts/quality_gate/pose_fixture.cpp) renders marker *outer boundaries* at the known 40 mm board coordinates. Each output pixel integrates a 4×4 supersampled image with coordinates spanning the correct outer-pixel interval, from -0.5 to width or height minus 0.5. Eight fixed fractional camera translations and varied rotations, light gradients, and Gaussian blur values from 0 to 1.1 px were specified before the first valid score. Four DICT_4X4_50 markers are rendered in each 800×600 image; all eight images, the [numeric report](../build-opencv/quality-ablation/independent-pose-fixture-v6.json), and its [provenance sidecar](../build-opencv/quality-ablation/independent-pose-fixture-v6.json.provenance.json) are retained. Rebuild with `python3 scripts/quality_gate/run_pose_fixture.py --output NEW_REPORT.json` using the existing local OpenCV build.

The initial [v1](../build-opencv/quality-ablation/independent-pose-fixture.json) and [v2](../build-opencv/quality-ablation/independent-pose-fixture-v2.json) runs detected no markers because the rendering homography was assembled incorrectly; their nearly black output is retained. V3 corrected the matrix to `K * [r1 r2 t]`, without changing camera, blur, lighting or detector settings. V4 saved all eight images instead of only the first. These are development diagnostics, not held-out validation cases or a tuned benchmark. The hardened runner now checks free disk space, refuses existing output/image targets, verifies all fits, and writes source, binary and artifact hashes in a sidecar.

Both detectors found all four markers in every view. Against the known projected *outer* marker corners, default detection averaged 0.685 px corner error and `CORNER_REFINE_SUBPIX` averaged 0.277 px. The same IPPE plus Levenberg–Marquardt pose solver averaged 2.248 mm versus 0.504 mm camera-center error, and 0.184° versus 0.039° rotation error. Subpixel refinement improved corner and pose errors in each of the eight views. These are rendered ideal board images and do not prove improvement on real photographs; they show that the adverse R02 result is fixture-dependent and that changing the production detector should await a measured capture test.

An independent [supervisor replay](../build-opencv/quality-ablation/independent-pose-supervisor.json), with its own [provenance sidecar](../build-opencv/quality-ablation/independent-pose-supervisor.json.provenance.json), reproduced successful detection and pose fitting on all eight cases. The replay used the hardened runner, including the local temporary directory and free-space preflight.

## Changed-pose world-frame comparison

When estimated cameras change, radial depths cannot be compared at the same pixel as if their rays were identical. [`compare_world.py`](../scripts/quality_gate/compare_world.py) instead fixes eligibility using the synthetic truth camera and visible rectangle at each pixel. It backprojects each valid radial depth with its own MVE camera and scores 3D Euclidean distance to the *same finite visible truth rectangle* in the object frame; plane-Z error is a separate diagnostic. The two outputs must still have the same view, 800×600 L1 resolution, fixture intrinsics and millimetre units. The baseline camera must match the original sparse report. Empty truth, boundary or interior populations fail closed. All-valid bad-1/2/5, coverage and matched MAE/p95 are reported on fixed full-object, silhouette/surface-step boundary and interior populations, with shared-support selection bias labeled. The same provisional relative thresholds apply, but a pass remains a development signal only.

The scorer's [estimated versus known-pose L1 integration check](../build-opencv/quality-ablation/world-known-pose-vs-estimated-L1-v2.json) uses 30,514 fixed truth pixels, matching the existing ablation population. It verifies that a large camera improvement is visible in the world metric: full-object missing-inclusive rectangle bad-2 falls from 51.74% to 5.54% at 98.51% versus 98.60% coverage; boundary bad-2 falls from 62.67% to 37.52%. Known-pose reconstruction is an oracle diagnostic and cannot be selected as a feasible camera pipeline. The independent [world scorer fixtures](../scripts/quality_gate/test_compare_world.py) cover radial backprojection, missing versus wrong, dropping difficult predictions, changed poses, empty truth and malformed cameras.

An optimized pose run that fails its independent frozen marker reprojection safeguard remains rejected even if a subsequent dense-depth diagnostic scores well. The surface quality gate and the upstream camera acceptance check answer different questions; passing one does not override failure of the other.

The rejected bundle-adjustment diagnostic was independently scored at the same 800×600 L1 grid in [`world-fixed-truth-L1.json`](../build-opencv/bundle-quality/world-fixed-truth-L1.json). On 30,514 fixed object-truth pixels, finite-rectangle missing-inclusive bad-2 fell from 51.74% to 27.23% (47.4% relative reduction), coverage rose from 98.51% to 98.62%, and the fixed two-pixel boundary bad-2 fell from 62.67% to 50.00%. Matched finite-rectangle MAE fell from 2.130 to 1.580 mm. All three provisional *surface* checks pass. The [optimization diagnostic](../build-opencv/bundle-quality/rejected-diagnostic.json) records that frozen marker RMS increased from 1.10791 to 1.11717 px, so this pose solution remains **rejected upstream**. These development-scene truth scores cannot be used to waive that independent marker safeguard or to claim real-scan accuracy.
