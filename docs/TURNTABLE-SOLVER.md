# The `turntable` camera provider

`crisp3ds-dense photos --cameras turntable`: cameras for one turn of ordered
turntable photos with a known lens, from this crate's own code
(`crates/dense/src/photos/turntable/`). No external program, no random choice,
plain Rust that builds for WebAssembly, so it is the camera provider for
phones and browsers as well as desktops.

It assumes what its name says: the photos are in capture order and each is the
previous one turned about one fixed axis (the object on a turntable under a
fixed camera). The axis should be roughly upright in the photos (within about
50 degrees) and the object roughly in front of the camera. A full turn is the
default; `--open-turn` is for photos that do not close one.

## Method

1. **Features** (`features.rs`): SIFT after Lowe, written here (the patent
   expired in 2020): Gaussian scale space of the doubled image, extrema of the
   differences of Gaussians located to sub-pixel accuracy, low contrast and
   edges rejected, orientations from gradient histograms, 4 x 4 x 8
   descriptors. On the contrast images, inside the object masks shrunk by two
   pixels, at most `--turntable-features` (6000) per photo.
2. **Matches** (`matching.rs`) between every photo and its next
   `--turntable-span` (4) photos, around the closed turn: nearest neighbour by
   descriptor distance in both directions, ratio test at 0.8, both agreeing.
   **Order check:** neighbours in name order must share at least 1.5 times as
   many matches (median over the turn) as the farthest pairs matched;
   otherwise the run is refused with a message to name the photos in capture
   order or to choose `--cameras colmap`. Measured: 2.9 to 14 for ordered sets
   (73, 37 and 25 photos), 0.93 to 1.08 for shuffled names.
3. **One motion** (`solver.rs`). The axis direction, the direction from the
   camera to the axis (its distance is set to 1, which fixes the scale) and
   the angle of every step are fitted to the matches of all consecutive pairs
   together, by their Sampson distances cut off at 3 px. The fit starts from
   the best of eleven axis tilts and both turning directions and refines the
   three shared numbers over shrinking ranges, re-fitting the step angles in
   every evaluation. A pair without image motion gets a step of zero (the
   Bunny's photos 13 and 14); a step far from the median is replaced by it.
4. **The turn closes.** Unless `--open-turn`, the steps, the one from the last
   photo back to the first included, are scaled to add up to 360 degrees.
   Steps that add up to less than 0.75 or more than 1.15 turns stop the run
   with that reason instead: the motion was not found. Correct solutions
   measured 0.78 (the Happy Buddha, whose rotation is poorly separated from
   translation; the adjustment then corrects the poses) to 1.07 (25 photos at
   15 degrees); a wrong axis gave 1.19 (a smooth bottle seen from a steep
   camera, `docs/OTHER-IMAGE-SETS.md`), and scaling it to one turn let it pass
   every later gate. The window was 0.6 to 1.4 for a while; all test sets
   have the same cameras under both.
5. **Tracks.** Matches that agree with these poses are joined; tracks of at
   least three photos are triangulated.
6. **Bundle adjustment** (`adjust.rs`) of all poses and points with the lens
   fixed: Levenberg-Marquardt, the points eliminated by a Schur complement,
   Huber loss at 1 px, the first camera held. Every pose has six free
   parameters; the turntable model was only the starting point. After the
   first adjustment the matches are checked again against the adjusted poses;
   then observations far off are dropped twice.

The Python prototype this was ported from
(`crates/dense/tools/turntable_prototype.py`) took step 3 from the essential
matrices of single pairs (OpenCV's five-point RANSAC) and medians over them.
An eight-point RANSAC written here could not do that: a small object in a
narrow field barely tells a 5 degree rotation from a translation, and its
axes scattered by 48 degrees. The joint fit needs no essential matrix and no
random sampling, and its raw step sums are closer to a full turn than the
prototype's (353 to 364 degrees against 352 to 375). The eight-point code is
still in `geometry.rs`, unused, for captures that are not turntables.

## Result

Same SAM masks for every provider, native dense stages, scanner F1 at 0.5 % of
the diagonal, `above_margin` (`all` in brackets). One run each for
`alicevision` and `colmap`, which are not repeatable.

| | Bunny | Armadillo | Dragon | Lucy |
| --- | --- | --- | --- | --- |
| `alicevision` | 0.963 (0.905) | 0.952 (0.923) | 0.834 (0.785) | 0.826 (0.790) |
| `colmap` | 0.965 (0.905) | 0.952 (0.924) | 0.844 (0.795) | 0.865 (0.828) |
| prototype (OpenCV SIFT, Python) | 0.967 (0.909) | 0.953 (0.926) | 0.840 (0.792) | 0.863 (0.827) |
| Rust solver on the prototype's matches | 0.968 (0.909) | 0.953 (0.925) | 0.837 (0.789) | 0.864 (0.827) |
| **`turntable`** (Rust features and solver) | **0.966 (0.908)** | **0.953 (0.926)** | **0.837 (0.791)** | **0.860 (0.824)** |
| `turntable` minus `colmap` | +0.002 | +0.001 | -0.007 | -0.005 |
| Registered | 73 of 73 | 73 of 73 | 73 of 73 | 73 of 73 |
| Features per photo, median: OpenCV / here | 1137 / 1107 | not recorded / 1266 | not recorded / 919 | not recorded / 378 |
| Matches over 292 pairs: OpenCV / here | 59 513 / 59 250 | not recorded / 73 072 | not recorded / 51 725 | not recorded / 30 521 |
| Points, observations: prototype | 6455, 32 422 | 7605, 40 147 | 4952, 25 761 | 2572, 13 766 |
| Points, observations: `turntable` | 6462, 32 531 | 7549, 39 973 | 4932, 25 914 | 2585, 13 690 |
| Reprojection median, p95 (px): prototype | 0.35, 1.19 | 0.36, 1.17 | 0.41, 1.31 | 0.44, 1.41 |
| Reprojection median, p95 (px): `turntable` | 0.35, 1.19 | 0.36, 1.16 | 0.41, 1.29 | 0.43, 1.42 |
| Camera recovery, seconds: `colmap` / prototype / `turntable` | 64 / 85 / 18 | 105 / 60 / 24 | 182 / 38 / 18 | 133 / 39 / 13 |

`turntable` times are features, matching and solving on an M1 with four
threads while another job used the machine (Bunny: 7.8 + 7.3 + 3.2 s).

- **The port.** With the prototype's own matches the Rust solver gives the
  prototype's cameras: rotations within 0.006 degrees and centres within
  0.009 % of the ring radius in the median on the Bunny, the Armadillo and the
  Dragon (largest 0.023 degrees), 0.10 degrees and 0.12 % on Lucy, where the
  different step fit shows most. Scanner F1 is within 0.001 on three objects
  and 0.003 lower on the Dragon.
- **The features.** With its own SIFT the provider is within 0.01 of COLMAP
  on all four objects, the criterion set for it. It is 0.007 and 0.005 below
  COLMAP on the Dragon and Lucy, the two objects with the fewest features,
  and, against its own solver fed with OpenCV's features, equal on the Dragon
  and 0.004 lower on Lucy.
- **Repeatability.** Two runs on the Bunny and on the Dragon, with two and
  with four threads, wrote identical `cameras.json` files, byte for byte.
  Nothing in the provider is random or depends on the order in which threads
  finish. (AliceVision's cameras moved by 0.7 degrees between runs.)
- **An open turn.** The first 50 Bunny photos (247 degrees) with
  `--open-turn --maximum-angular-gap-deg 360`: all registered, 0.33 px median.
  Against the full turn's cameras they differ by 0.29 degrees and 0.33 % of
  the radius in the median (1.07 degrees and 1.5 % at most): without the
  closure the ring is held less firmly. Without `--open-turn` the same photos
  are rejected by the closure gate.

## The closure gate

Step angles that are all a few percent too large or too small leave
reprojection and ring shape plausible. The first prototype passed every gate
on the Bunny with a 0.70 px median and scored 0.64 instead of 0.96; its orbit
swept 363.6 degrees from the first photo to the last. For a capture declared
a full turn (the default; `--open-turn` says otherwise), every camera provider
now has to close: what is left from the last photo to the first, per capture
step, must not be below minus a quarter of the median step and not above
`--maximum-closing-step-ratio` (2.5) times it. A unit test rebuilds that
363.6 degree ring and is rejected ("the cameras turn 363.6 degrees from the
first photo to the last, past a full turn"); the original files of that run
were overwritten, so the gate was not run on them.

## From photos to a scored STL in one command

`crisp3ds-dense run --photos DIR --calibration scripts/turntable_mesh/calibrations/3dlf-pro.json
--output RUN`, masks and cameras left at their defaults: threshold masks, this
solver, the dense stages, nothing else started. 73 photos each, Apple M1
with 16 GB, one run per object; the scans are used for scoring only (F1 at
0.5 %, 1 % and 2 % of the scan's diagonal).

| Object | Whole run | Photos stage (cameras in it) | Stereo | Mesh | Check | F1, whole surface | F1, above the support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bunny | 93 s | 41 s (31 s) | 45 s | 5 s | 2 s | 0.904 / 0.939 / 0.956 | 0.964 / 0.995 / 1.000 |
| Armadillo | 83 s | 42 s (32 s) | 35 s | 4 s | 2 s | 0.924 / 0.972 / 0.987 | 0.952 / 0.997 / 1.000 |
| Dragon | 78 s | 33 s (23 s) | 39 s | 4 s | 2 s | 0.802 / 0.933 / 0.982 | 0.847 / 0.965 / 0.995 |
| Lucy | 50 s | 23 s (13 s) | 25 s | 1 s | 1 s | 0.822 / 0.940 / 0.965 | 0.861 / 0.985 / 0.999 |

Scores are from the default command at `ac36992` (threshold masks with the
contact shadow taken out, `--threshold-shadow 0.25`). Times are from the same
command with `--masks threshold` at `545356e`, before the shadow step, on an
otherwise idle machine; the runs at `ac36992` shared the machine with other
work and their times say nothing.

All 73 photos registered every time and every gate passed. Above the support
the scores equal those with SAM masks and the same cameras (0.966, 0.953,
0.837, 0.860 at 0.5 %). Sheets with the SAM-mask mesh beside
each: `checkpoint-renders/photos-to-stl-native-<object>.png`.

## Robustness

Default command (`threshold` masks, these cameras): the Bunny and Dragon
with 73 photos and the Thai statue at `ac36992`, the rest with the order check
and the closing window (which leave those three unchanged); 3DLF photos, scanner F1 at 0.5 / 1 / 2 % of the scan's
diagonal (evaluation only).

| Set | Photos (step) | Registered, gates | F1, whole surface | F1, above the support |
| --- | --- | --- | --- | --- |
| Bunny | 73 (5 deg) | 73, pass | 0.904 / 0.939 / 0.956 | 0.964 / 0.995 / 1.000 |
| Bunny, every 2nd | 37 (10 deg) | 37, pass | 0.902 / 0.938 / 0.956 | 0.961 / 0.995 / 1.000 |
| Bunny, every 3rd | 25 (15 deg) | 25, pass | 0.869 / 0.924 / 0.949 | 0.926 / 0.977 / 0.992 |
| Dragon | 73 (5 deg) | 73, pass | 0.802 / 0.933 / 0.982 | 0.847 / 0.965 / 0.995 |
| Dragon, every 2nd | 37 (10 deg) | 37, pass | 0.791 / 0.920 / 0.978 | 0.838 / 0.953 / 0.991 |
| Dragon, every 3rd | 25 (15 deg) | 25, **refused by the camera audit** (too few landmarks per view) | | |
| Thai statue | 73 (5 deg) | 73, pass | 0.872 / 0.933 / 0.953 | 0.935 / 0.995 / 1.000 |
| Happy Buddha | 73 (5 deg) | 73, pass (raw steps 0.78 turn, inside the closing window; refused by the closure gate before the window existed) | 0.779 / 0.912 / 0.967 | 0.817 / 0.944 / 0.985 |
| Asian dragon | 73 (5 deg) | 73, pass | mesh looks right; the scan evaluator cannot isolate the object from the platform in this scan | |
| Bunny, Dragon, names shuffled | 73 | refused by the order check | | |

Where it breaks: at 15 degree steps the Dragon keeps too few points per view
for the audit (the Bunny still passes, 0.03 lower above the support); the
Happy Buddha's face and the holes between its arms come out soft. Photos out
of order are refused, not sorted. YCB turntable sets are not supported: their
lens has tangential distortion, which the `radialk3` calibration cannot carry,
and their backdrop is dark.

## Limits and open points

- **Turntables only.** A camera walking around an object at rest does not
  repeat one motion; that needs registration photo by photo, which is not
  written.
- **The default camera provider, on thin evidence.** For: equal
  scores, no external program, every platform, repeatable, three to ten times
  faster than COLMAP here. Against: it has been run on seven objects from one
  camera, lens and turntable, at 5 to 15 degree steps; the axis
  search assumes an upright axis and a centred object; it does not handle
  captures that are not turntables, which `colmap` does; the Dragon and Lucy
  are a little below COLMAP.
- **In a browser it is single-threaded.** The Bunny from its 73 photos
  completes in Chrome with WebGPU (351 s in all; features 57 s and matching
  55 s single-threaded; cameras equal to the native ones to 5e-11); see
  `crates/dense/README.md`.
- **The order check looks at medians.** A turn in name order with a few
  photos swapped passes the check and goes to the solver.
- **Memory and time grow with the mask's bounding box** (the doubled image
  and six blurred copies of it per octave): about 8 s for 73 photos of
  1749 x 1155 with four threads here. Matching compares all descriptors of a
  pair: 1 to 11 s for 292 pairs.
- Not tried: steps larger than 15 degrees, a strongly tilted axis.
