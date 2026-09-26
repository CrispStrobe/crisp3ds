# Fixed tree epipolar diagnostic

This experiment compares supplied camera epipolar geometry with geometry fitted
from image matches before any supplied-pose filtering. It is a full-frame test:
features may lie on the tree, buildings, cars, ground, or other background. It
does not establish an object-specific pose, depth truth, or metric scale.

## Protocol frozen before the live run

Input is the ten corrected 768×512 images and `views.tsv` in
`build-opencv/tree-refine/baseline/scene`. No image, intrinsic, or pose is
changed. Each image gets OpenCV SIFT with `nfeatures=3000`. OpenCV can retain
ties at the response cutoff, so keypoints and corresponding descriptors are
truncated in returned order to exactly 3,000. The exact counts are recorded.
For every one
of 45 unordered view pairs, L2 two-nearest-neighbor matches must pass ratio
0.8 in both directions and agree mutually. No camera geometry filters them.
Matches are sorted by source descriptor index, with target index as tie break.
Zero-based positions congruent to 4 modulo 5 form the held-out set; all other
positions train. This partition is independent of image geometry and residuals.

For each pair with at least 16 training and 8 held-out matches, fit one
fundamental matrix to training points only using OpenCV `FM_RANSAC`, a 1 px
threshold, 0.999 confidence, at most 10,000 iterations, and a fixed per-pair
OpenCV RNG seed. There is no held-out model selection or refit. The supplied
fundamental matrix is
`Kb^-T [tb - Rb Ra^T ta]x Rb Ra^T Ka^-1` for world-to-camera
poses. Normalize both F matrices to unit Frobenius norm; their sign is
immaterial. Report held-out square-root Sampson distance in pixels for both,
using the *same* held-out matches. A zero or nonfinite denominator is invalid.
Report total, finite, and invalid counts, median and nearest-rank p90 of finite
distances, and counts and fractions within 1, 2, and 4 px. Fractions use all
held-out matches as denominator, so invalid results count as failures. An
unavailable pair has an explicit reason and retains its raw matches.

Also fit one homography with `RANSAC` on training points only, 3 px reprojection
threshold, 0.999 confidence, at most 10,000 iterations, with fixed per-pair RNG
seed. Record its training inlier count and finite held-out symmetric transfer
distance in pixels as *degeneracy context*. Homography transfer error and
Sampson distance are different metrics and must not be ranked numerically.
A planar or low-parallax scene can fit a homography without validating a camera.
The initial 45-pair phase includes no essential-matrix pose recovery; the
single-pair calibrated follow-up is declared separately below.

The output stores image paths and source/binary hashes, cameras, every match's
keypoint IDs and coordinates, its descriptor-order position and partition,
model matrices, and metrics. Input and code/binary hashes are checked before
and after. The runner requires a fresh output directory and keeps its output
below 100 MiB. Use at least 10 GiB free space; no dependency downloads. The
focused pair is original views 1 and 4, but all 45 pairs are processed.

## Run

The local runner uses existing `/usr/local/bin/python3` with OpenCV Python
4.10.0 and NumPy 1.26.4. This is a separate OpenCV build from the native 4.12
tree tools. No new package is installed. The run has a 600-second process limit.

```sh
TMPDIR="$PWD/.local-tools/tmp" /usr/local/bin/python3 scripts/tree_epipolar/run.py \
  --scene build-opencv/tree-refine/baseline/scene \
  --output build-opencv/tree-epipolar-capped
```

Synthetic tests run with `/usr/local/bin/python3 scripts/tree_epipolar/test_run.py`.
The independent verifier can recompute each residual from `report.json` and
`matches/AA_BB.json` without feature detection or fitting.

## Predeclared calibrated pair 1/4 follow-up

After the fixed F result, one bounded calibrated pose check uses **only** the
saved pair 1/4 matches in the capped report. The same descriptor-order train
and held-out split remains frozen. Normalize training image coordinates with
each view's own K. Fit one essential matrix with OpenCV `findEssentialMat`
RANSAC, confidence 0.999, normalized threshold `1 / mean(fx_a, fx_b)`, at most
10,000 iterations, and fixed RNG seed 3414. `recoverPose` receives only the
essential RANSAC training inliers, with an explicit distance threshold of 50
unit baselines, and does no independent feature matching or held-out fitting.
Its cheirality-positive count, which also reflects that distance gate, is
recorded separately from RANSAC inliers. The recovered
unit translation and rotation define a calibrated F via the original two K
matrices; score that F on **the same untouched held-out matches** with the
same square-root Sampson metric and thresholds as above.
For bookkeeping, triangulate all essential RANSAC training inliers once under
the recovered pose, without refitting anything. Report finite, positive in
both views, nonpositive in either view, positive with either depth over 50,
positive with either Euclidean camera distance over 50, and minimum/median
ray angle over all raw positive points. The recoverPose count alone must not
be interpreted as saying every rejected point is behind a camera.

For context, report the angular difference between recovered and supplied
relative rotations and between their translation directions, plus the median
and p90 triangulation-ray angle on cheirality-positive training points. The
translation magnitude is arbitrary, and the E decomposition has sign and
pose hypotheses resolved only under its cheirality assumptions. Neither
metric validates absolute scale or tree-only geometry. This follow-up fits no
pose variants, third view, or revised intrinsics, and changes no source pose.
It requires a fresh report path, checks inputs and source/binary hashes before
and after, and has a 60-second process limit.

```sh
TMPDIR="$PWD/.local-tools/tmp" /usr/local/bin/python3 scripts/tree_epipolar/essential.py \
  --report build-opencv/tree-epipolar-capped/report.json \
  --output build-opencv/tree-epipolar-capped/essential-report-v2.json
```

## Frozen run result

The final capped run at
[`build-opencv/tree-epipolar-capped/report.json`](../build-opencv/tree-epipolar-capped/report.json)
completed in 3.5 seconds and wrote 1,579,037 bytes. All source image,
`views.tsv`, script, and OpenCV binary hashes matched before and after. OpenCV
returned 3,000 retained features for every view. Across 45 pairs there were 8,755 mutual matches. Thirty
pairs met both sample count gates; 15 are explicitly unavailable. All 30
available pairs had finite Sampson distances for every held-out match.

| Pair set / metric | Supplied F | Image-fitted F |
| --- | ---: | ---: |
| Pair 1/4 held-out matches | 83 | 83 |
| Pair 1/4 median √Sampson distance | 1.831 px | 0.199 px |
| Pair 1/4 p90 √Sampson distance | 5.323 px | 1.783 px |
| Pair 1/4 within 4 px | 64/83 | 83/83 |
| Median of 30 pair medians (unweighted) | 7.305 px | 0.282 px |

The image-fitted F has a smaller held-out median in all 30 available pairs.
For pair 1/4, the F fit used 333 training matches and found 271 RANSAC
inliers. The homography had 139/333 training inliers and a held-out symmetric
transfer median/p90 of 3.065/48.111 px; those transfer values are a separate
metric and only provide degeneracy context. The 15 unavailable pairs have
between 3 and 35 mutual matches, below the predeclared count gate.

These results show that the supplied cameras predict these full-frame SIFT
correspondences much less closely than a train-only image-derived F. The
result does not identify which camera parameter or pose is wrong, and it does
not by itself establish a better 3D tree reconstruction. Repeated textures,
spatially correlated matches, and background features limit what this
descriptor-order holdout can independently validate.

An initial run at
[`build-opencv/tree-epipolar-fixed/report.json`](../build-opencv/tree-epipolar-fixed/report.json)
predated the deterministic 3,000-feature truncation: OpenCV returned 3,001
for views 1 and 8 because of tied responses. It is preserved for audit, but
its source hash is historical; the capped run is the final reference. The
truncation did not change the pair matches or reported scores.

The calibrated pair 1/4 follow-up is at
[`build-opencv/tree-epipolar-capped/essential-report-v2.json`](../build-opencv/tree-epipolar-capped/essential-report-v2.json).
It used the same 333 training and 83 held-out coordinates. Essential RANSAC
retained 309 training matches; recoverPose returned 140 positive-depth
matches within the declared 50 unit-baseline distance gate. Their median
ray angle was 1.771°. The pose-derived calibrated F reached held-out median
and p90 √Sampson distances of 0.252 and 0.778 px; 82/83 were within 1 px
and 83/83 within 2 px. Relative to the supplied pair, recovered rotation
differs by 5.692° and translation direction by 12.792°. That unit
translation has no metric scale, and its direction depends on cheirality
assumptions. The result supports a calibrated two-view pose that fits these
full-frame correspondences, without establishing an absolute or tree-only
camera solution.

Raw triangulation of all 309 essential RANSAC training inliers under that
recovered pose gave 309 finite points, 146 positive depth in both cameras,
and 163 nonpositive depth in at least one. Six of the 146 positive points
were over 50 unit baselines deep or distant in at least one camera; this
accounts for the 140 recoverPose retained points. Across all 146 raw positive
points, the ray-angle minimum/median was 0.080°/1.756°. The earlier
`essential-report.json` is retained as a historical first pass; its script
hash predates this triangulation bookkeeping.
