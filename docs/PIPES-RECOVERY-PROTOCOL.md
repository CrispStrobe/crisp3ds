# Pipes additional-view anchor recovery v1

Frozen before execution, 2026-09-26. This is a development experiment following
the context rejection result, not unseen acceptance or a production algorithm.
No per-point laser errors may enter matching, fitting, or replacement decisions.

## Fixed question and population

Can additional photographs re-estimate the surface location of an existing
image feature without trusting its original cross-image correspondences?
Use all 258 original point IDs from `pipes-sparse/run-003`; do not select only
inspected outliers or points rejected by the context experiment. Each point's
anchor is its lexicographically first original observation (image name then
feature ID). Anchor pixels and feature IDs never move.

Original points SHA-256:
`49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0`.
Original report SHA-256:
`00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e`.
Frozen original distance score SHA-256:
`9df4234128ca06e1ee10842e34d93d9fc66336ca37f81c598e28af6921523796`.

## Image-only reconstruction policy

1. Replay the original four views' extraction, original pair counts, saved
   feature coordinates and runtime versions/binaries. Add **all ten** other
   prepared images, DSC_0638.JPG through DSC_0647.JPG. These ten are now
   reconstruction/training views, not held-out validation images. Use the
   existing supplied intrinsics and poses, longest image edge 1024, actual
   per-axis intrinsic scaling, pixel-edge coordinates, and SIFT limit 4096.
2. Compute exact global strict mutual nearest-two L2 matches with ratio 0.8
   at original SIFT size. Use all extracted features as competitors. Keep
   matches only at fixed-pose square-root Sampson error <=2 pixels. Match
   original anchor images to new views and every new-view pair. Do not search
   around projections of the old XYZ or restrict candidates using reference
   geometry, laser errors, or the context verdicts.
3. For each anchor, collect every directly matched new-view feature. Require
   at least **two distinct new views**. Require every pair of the collected
   observations to pass the same mutual matching and epipolar checks. Reject
   inconsistent sets wholesale: no subset search, best-pair selection, or
   parameter sweep. Triangulate only the original anchor plus these new-view
   observations, excluding every other original-four-view observation.
4. Use the unchanged fixed-ray least-squares reconstruction and gates:
   all depths positive, acute parallax >=1 degree, maximum reprojection
   <=4 pixels, normal-matrix condition <=1e8. Reject all otherwise admissible
   candidates that share any new-view feature with another original point ID;
   do not count duplicated supporting evidence as independent recovery.
5. If accepted, propose the independently re-estimated XYZ for that anchor ID.
   If unresolved/rejected, retain the exact original XYZ as an explicit
   **fallback**, not as a confirmed point. Save all decisions and reasons.
   The candidate comparison therefore has the same 258 anchor IDs as baseline.
   No observation, camera, scale, or registration is adjusted against the scan.

This measures both an additional-data and a matching-policy change; it is not
a same-four-image algorithm comparison. Supplied cameras are an oracle input
with scan-informed calibration, not independent pose ground truth. Repeated
features can still fool multi-view matching. No metric implies correctness
of the physical surface association just because it is near the laser cloud.

## Sealing, scoring and bounds

Seal candidate coordinates and decision hashes before opening reference
distances. Validate input/protocol/runtime hashes before/after reconstruction.
The separate evaluator checks exact IDs, anchors and original fallbacks,
positive depths and reprojection of accepted new observations. Re-estimated
points are compared exhaustively in float64 against every measured scan
vertex with the published scan transform applied once, no fitted alignment.
Unchanged fallback distances reuse the exact frozen original score.

Report all-258 median/p90/p95/max and counts within 10/20/50/100 mm. At each
threshold report paired changes from near to far and far to near. Also report
the re-estimated subset, unresolved count/reasons, observations and per-view
support, and runtime/RSS. Report adverse movements, not only improvements.
No recovered points means unchanged baseline, not a successful recovery.
Sparse proximity is not surface completeness, dimensional certification or
official ETH3D accuracy. A separate measured scene is required for acceptance.

Each reconstruction/scoring job is bounded to 300 seconds, 256 MiB generated
output, and a 10 GiB free-disk reserve plus output allowance. Use one OpenCV
worker. Preserve fresh run directories and the frozen baseline. Existing
local images only; no downloads, packages, GPUs, or remote execution.
