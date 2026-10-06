# The `turntable` camera provider: approach and prototype

Goal: cameras for ordered turntable photos with a known lens from our own
code, so that platforms without external programs (phones, browsers) do not
need AliceVision or COLMAP.

**Status: the approach is settled by a Python prototype
(`crates/dense/tools/turntable_prototype.py`) that reaches COLMAP's scanner
scores on the four test objects. The Rust provider is not written yet;
`--cameras turntable` does not exist.** The prototype uses OpenCV's SIFT and
essential-matrix code; its initialisation and its bundle adjustment are its
own.

## Method

1. **Features.** SIFT on the contrast images, inside the object masks (eroded
   by two pixels), up to 6000 per photo at a contrast threshold of 0.01.
2. **Matches** between every photo and its next four, with the turn closed
   (the last photos are matched with the first): ratio test at 0.8, mutual.
3. **One motion.** A turntable repeats one motion, a rotation about a fixed
   axis. The essential matrix of every consecutive pair gives a rotation and a
   translation direction; the axis direction and the direction from the camera
   to the axis are the medians over all pairs. The distance from the camera to
   the axis is set to 1, which fixes the scale.
4. **The angle of every step** by a one-dimensional search on that pair's
   matches with the axis fixed. A pair without image motion gets a step of
   zero (the Bunny's photos 13 and 14). Steps far from the median are replaced
   by it.
5. **The turn closes.** The steps, the one from the last photo back to the
   first included, are scaled to add up to 360 degrees (unless `--open-turn`).
   From a single pair, the angle of a step and the distance to the axis are
   hard to tell apart, and the raw sums were 352 to 375 degrees; without this
   the matches across the seam fail their check against the poses and the
   adjustment has nothing that closes the ring.
6. **Tracks.** Matches that agree with the poses (Sampson distance below
   4 px) are joined; tracks of at least three photos are triangulated.
7. **Bundle adjustment** of all poses and points with the lens fixed:
   Levenberg-Marquardt, points eliminated by a Schur complement (the reduced
   system is 6 x photos square), Huber loss at 1 px, the first camera held.
   Every pose has six free parameters; nothing pulls them back to the
   turntable model, which only served as the starting point. After the first
   adjustment the matches are checked again against the adjusted poses, then
   observations far off are dropped twice.

The result is written as a COLMAP text model and enters the pipeline through
`--cameras import:DIR`, so gates, undistortion, scene and dense stages are the
usual ones.

## Result

Same SAM masks for all three providers, native dense stages, scanner F1 at
0.5 % of the diagonal (`above_margin`; `all` in brackets). One run each.

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| `alicevision` | 0.963 (0.905) | 0.952 (0.923) | 0.834 (0.785) | 0.826 (0.790) |
| `colmap` | 0.965 (0.905) | 0.952 (0.924) | 0.844 (0.795) | 0.865 (0.828) |
| prototype | 0.967 (0.909) | 0.953 (0.926) | 0.840 (0.792) | 0.863 (0.827) |
| prototype: registered | 73 of 73 | 73 of 73 | 73 of 73 | 73 of 73 |
| prototype: points, observations | 6455, 32 422 | 7605, 40 147 | 4952, 25 761 | 2572, 13 766 |
| prototype: reprojection median, p95 (px) | 0.35, 1.19 | 0.36, 1.17 | 0.41, 1.31 | 0.44, 1.41 |
| prototype: ring radius spread | 0.53 % | 0.38 % | 0.28 % | 0.36 % |
| prototype: seconds, total (of which features) | 85 (45) | 60 (25) | 38 (24) | 39 (26) |
| `colmap`: seconds | 64 | 105 | 182 | 133 |

The prototype is within 0.004 of COLMAP on every object, above AliceVision on
all four. Its times include Python overhead and were taken on a shared
machine.

What went wrong on the way, because the gates did not see it: the first
version (steps not closed to 360 degrees, a least-squares routine that did not
move the cameras) passed every gate on the Bunny with 0.70 px median
reprojection and a ring spread of 0.12 %, and gave F1 0.64 instead of 0.96.
Its orbit swept 363.6 degrees from the first photo to the last where COLMAP's
sweeps 355.5, with a median step of 5.28 degrees against 5.02. A ring that is a perfect circle with plausible reprojection can
still be wrong in its angles; only the reconstruction showed it.

## What the Rust provider needs

In `crates/dense/src/photos/turntable/`, pure Rust, reusing
`photos/markers/linalg.rs`:

| Part | Notes |
| --- | --- |
| Feature detector and descriptor | The one part without a counterpart in the prototype's own code. SIFT (the patent expired in 2020) written here, or an existing permissive crate, to be judged by the same scores. The Dragon and Lucy have the fewest features and will show a weaker detector first |
| Matching | brute force over 128-d descriptors, ratio and mutual tests; a few thousand features per photo |
| Essential matrix per pair | eight-point with RANSAC on normalised coordinates and decomposition with the cheirality test; only its rotation and translation direction are used |
| Axis, steps, closure | as above; medians and a one-dimensional search |
| Tracks, triangulation | union-find; linear triangulation |
| Bundle adjustment | as in the prototype: analytic Jacobians, Schur complement, dense solve of a 438 x 438 system for 73 photos |

Open points: photos that do not close a turn (the prototype has `--open-turn`
but it was not tested); a camera walking around an object at rest does not
repeat one motion and needs incremental registration instead of step 3; the
repeatability of the result was not measured (the prototype has no random
choice of its own, but OpenCV's RANSAC has).
