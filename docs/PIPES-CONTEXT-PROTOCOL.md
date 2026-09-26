# Pipes larger-context correspondence check v1

Frozen before executing the experiment, 2026-09-26. Development diagnostic,
not unseen acceptance: the pipes outlier images and laser errors have already
been inspected. No threshold sweep or selective removal of named outliers.

## Question and fixed inputs

Can a larger image neighborhood flag repeated-hardware correspondences missed
by the existing SIFT checks, without discarding excessive valid support?
This experiment only classifies the 258 existing points. It cannot repair
coordinates, recover missing geometry, or improve fixed-population coverage.

Use the four images, supplied cameras, 1024-pixel resize, SIFT feature limit,
and exact feature IDs from `pipes-sparse/run-003`. Reproduce extraction using
the original implementation and validate saved feature centers within 1e-6
pixels and baseline pair-match counts before proceeding. Original points SHA:
`49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0`.
Original report SHA:
`00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e`.

## Single candidate policy

Compute SIFT descriptors at the same detected centers and orientations, with
keypoint size multiplied by exactly **2**. Keep other keypoint metadata and
the same downsampled images. This enlarges context, not image resolution.
Preserve feature identity through descriptor computation; dropped descriptors
are explicit missing evidence, never silently renumbered. Leave boundary
handling to the existing SIFT implementation, with no additional mask or crop.

Use every original detected feature as a competitor, not only features in
accepted tracks. For each image pair, compute exact L2 nearest-two descriptor
matches in both directions. Require strict ratio **0.8** and mutual agreement.
A frozen point is retained only if **every pair** of its recorded observations
passes this context check. Missing evidence rejects with an explicit reason.
Also report an unchanged-size **all-pair control** using the original
descriptors: original tracks need only be connected, not complete pairwise
cliques. Compare size 2 against size 1 under the same all-pair rule to separate
the effect of larger context from this stricter graph policy.
Do not move observations, refit points/cameras, use laser geometry or scores,
alter parallax/reprojection thresholds, or special-case inspected point IDs.

## Evaluation after verdict sealing

Read the existing exhaustive `pipes-score/run-002/score.json` only in the
separate evaluator after sealing verdicts. Verify its input hashes correspond
to the unchanged point/report files. Reuse those exact per-point distances:
there is no new geometry requiring another scan nearest-neighbor search.

Report retained/rejected points, observations, image support, retained
median/p90/p95/max proximity, and fractions within 10/20/50/100 mm. At each
threshold also report retained-within-threshold over the original **258**
points and counts of near-reference and far-reference points discarded.
These are laser proximity diagnostics, not correct-match labels, surface
accuracy, or completeness. Missing-inclusive error cannot improve through
pure rejection; any cleaner retained subset must be presented with that cost.
Report the already-inspected three-view subset as a descriptive comparator,
not an independent experimental discovery. An empty result has null distance
quantiles and zero retained coverage, never perfect accuracy.

## Bounds and promotion

Use one OpenCV worker, 300-second guard, at most 128 MiB generated output,
and at least 10 GiB free disk plus output allowance. Hash images, input files,
protocol, implementation and runtime dependencies before and after. Preserve
fresh failed/successful run directories; do not overwrite baseline artifacts.
No installs, new datasets, GPU jobs, or remote compute are required.

There is no automatic production promotion. This tests a potential confidence
signal only. A later recovery method must retain or add reliable support and
be evaluated on a separate measured acceptance dataset.
