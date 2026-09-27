# Mustard photo-component two-view pose experiment — sealed preflight

Status, 2026-09-27: **plan only; no new fit or native reconstruction has run.**
This is a TRAIN-photo test of whether stationary-scene correspondences dominate
the mustard camera failures, or whether correspondences on the rotating bottle
remain intrinsically ambiguous. It uses the existing six-pair panel and one
existing fixed-intrinsic estimator. A later full 48-view SfM arm is outside
this plan.

## Why this is the next controlled question

The unmasked OpenMVG incremental run registered 46/48 TRAIN images and only
222 tracks; fixed-intrinsic and GLOBAL variants also registered 46/48. The
paired COLMAP coarse-mask arms both registered 48/48 but folded opposing
camera labels, while limiting COLMAP matching to four neighboring acquisition
positions left only 24/48 registered. Thus registration count, local graph
connectivity, and small reprojection residuals have not identified a sound
object-relative camera path.

A read-only decode of OpenMVG's E-filtered graph found 25,439 matches in 458
pairs, but only 385 matches had both endpoints inside coarse pose-support
masks. Those masks cover 1.345% of pixels and are **not object labels**;
neither-inside does not prove background. The existing six-pair independent
essential estimates are sensitive to individual verified rows and row order.
This experiment makes the missing correspondence provenance explicit from
original RGB before using it in any pose fit.

## Frozen population and inputs

Use only the `cached` arm of the already sealed six-pair TRAIN panel in
[MUSTARD-TWO-VIEW-POSE-PLAN.md](MUSTARD-TWO-VIEW-POSE-PLAN.md):

| Stratum | Exact pair | Verified rows |
| --- | --- | ---: |
| Near | `NP3_006/012` | 136 |
| Near | `NP3_012/018` | 129 |
| Middle | `NP3_012/336` | 41 |
| Middle | `NP3_036/126` | 24 |
| Opposing | `NP3_030/198` | 25 |
| Opposing | `NP3_036/222` | 25 |

All **380** original verified rows are the denominator. Preserve the sealed
SQLite row order and feature indices; no row is selected using an existing
camera, residual, scanner, reference pose, or later outcome. The cached
database SHA-256 is
`5858be16dfa3f0fc50303d0eb4a001e21ee14036f67755a9997d5dac374b3075`,
and its producer result is
`5a83c3e05e2b66887411799ecc8c5f3f41e3b8af3bbd775d308055ef7ee0041e`.
The TRAIN staging report is
`bbcba624f6a51f82e0ebdebf6e7e4cda10804bc8df34cfd0d5b3242983a225d0`;
its per-image SHA-256 values bind the source RGBs. The capture-order profile is
`bb00d17e1940f121dc12f7ddb31d4adf80b7e9a935f18e5535cfc1a2cde1998d`.
Acquisition suffixes define pair strata only, not supplied camera angles.
The unchanged estimator source is
`scripts/object_motion/mustard_two_view_pose.py`, SHA-256
`0ae09a5b5ea23b852a3166d42ab84ad0670d0255195f791e7cd2619bec135e05`.

The preflight must open SQLite `mode=ro&immutable=1`, reject nonempty WAL/SHM,
assert the 48 exact TRAIN names, the six row counts, the single fixed initial
`SIMPLE_RADIAL` camera `[1536,640,512,0]`, finite in-frame keypoints, and
unchanged input hashes before and after. The original TRAIN RGBs may be read
only to inspect feature locations. Do not open Berkeley per-angle poses,
capture-calibration H5, checkerboard pose reports, scanner/Google geometry,
depth, masks, held-out photos, or saved SfM camera models. The assumed focal
length is a diagnostic camera choice, not measured calibration.

## Photo review before fitting

For every one of the 380 indexed matches, prepare the two original-RGB
keypoint crops at fixed 64×64 and 192×192 sizes plus full-frame locator
thumbnails. Show the pair name, original feature IDs, coordinates, and row
ordinal. The display must not reveal SfM inlier status, recovered pose,
reference pose, mask overlap, or a laser score. Two independent reviewers
label **each endpoint** as `bottle_surface`, `rotating_board_or_support`,
`stationary_scene`, or `uncertain`, using visible RGB evidence across the two
views. A match is `bottle_only` only when both endpoints are bottle surface;
`stationary_only` only when both are stationary scene. Mixed, disputed, and
uncertain rows enter neither semantic arm. The unchanged complete 380-row arm
remains the control. A reviewer may mark a whole row uncertain rather than
guess at a blurred edge or repeated logo. Coarse support masks never decide a
class.

Seal a machine-readable decision for all 760 endpoints, reviewer identities,
disagreement resolution, the exact indexed-row/coordinate manifest, and all
source/crop hashes **before** any component-specific fit. Do not add or drop a
pair after seeing label counts. The preflight records per-pair counts for all
four endpoint classes, bottle-only, stationary-only, mixed, and uncertain.
Manual labels are photo evidence, not ground truth; disagreement and incomplete
coverage are explicit results.

## Exact launch gate for the one Python pose batch

The batch may launch only after the complete 380-row/760-endpoint review and
source seals pass, no reviewed class is inferred from the coarse masks, and
the immutable label manifest has been separately inspected. The fit reads
only the same six sealed indexed rows, their labels, and the unchanged fixed
initial intrinsic. At least **one pair in each of the near, middle, and
opposing strata** must have at least 20 bottle-only rows and at least as many
stationary-only rows, with at least 16 distinct feature locations per image
in each selected arm. If this prespecified support gate fails, write `insufficient_labeled_support`
with all counts and stop before fitting; no pair substitution, relaxed count,
or new threshold is permitted. The gate checks sample adequacy, not camera
quality.

For each eligible pair, run exactly three arms: all original verified rows;
all bottle-only rows; and a count-matched stationary-only control chosen by
ascending SHA-256 of `(pair names, first feature ID, second feature ID)`.
Use the first `N` stationary rows where `N` is that pair's bottle-only count.
No random resampling or best-subset selection. Preserve the selected rows'
original SQLite order for the primary fit. Repeat once in reverse row order
as a sensitivity diagnostic; do not choose between orders. Reuse the existing
OpenCV 4.10 five-point RANSAC settings, three seeds `17,23,31`, normalized
2 px threshold, 2,000 iterations, 0.999 confidence, `recoverPose`, competing
homography, and the already frozen essential/cheirality/parallax gates.
Report every seed and both orders, input/inlier counts, estimated rotations,
abstentions, and pairwise rotation spreads. Failed gates remain unavailable.
No 48-view mapper, native OpenMVG, native COLMAP, bundle adjustment, dense
stage, or reference score is launched here.

Bound the later batch to one run with a 120 s wall limit, one OpenCV thread,
2 GiB process RSS, 20 MiB JSON output, and a 10 GiB free-space floor on
each volume. It writes only to a fresh external output path, retains failed
reports, and rehashes every input afterward. A new adapter implementing the
manifest and subset selection needs its own source review and synthetic tests
for complete-label accounting, endpoint agreement, deterministic
count-matching, order preservation/reversal, and fail-closed support gates.
Those tests are not written now because this plan does not launch or implement
the adapter.

## Predeclared interpretation and falsification

`Background-dominant` means stationary-only rows outnumber bottle-only rows
on at least four of the six fixed pairs, including at least one pair in every
stratum. If this count condition fails, the proposed dominance explanation is
falsified **for this panel** before fitting. A component-specific camera
explanation is supported only if the label gate passes and the bottle-only
fit passes the existing reliability gates in both row orders with at most 5°
rotation difference across orders and seeds, while the count-matched
stationary arm is near-identity or unavailable on a middle or opposing pair.
This is a discriminating photo-only pattern, not physical pose certification.

If bottle-only fits remain unavailable or change by more than 5° despite
adequate labeled support, simply removing confirmed stationary background is
not a sufficient explanation for this pair's camera failure; repeated bottle
texture, low parallax, planarity, and the assumed intrinsic remain live causes.
If the label support gate fails, the cause is unresolved, not evidence for
either explanation. If both semantic arms give similarly stable distinct
poses, the simple background-versus-object-motion dichotomy is falsified.
Near pairs are reported but not required to have a large physical rotation;
capture-order suffixes are not pose truth. Even a favorable paired result
does not authorize full-orbit camera acceptance, a mesh, or production use.
