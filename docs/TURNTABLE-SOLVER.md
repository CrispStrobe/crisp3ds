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

## Limits and open points

- **Turntables only.** A camera walking around an object at rest does not
  repeat one motion; that needs registration photo by photo, which is not
  written.
- **Proposed as the default camera provider, not made it.** For: equal
  scores, no external program, every platform, repeatable, three to ten times
  faster than COLMAP here. Against: it has been run on four objects from one
  camera, lens and turntable, all with 73 photos at 5 degree steps; the axis
  search assumes an upright axis and a centred object; it does not handle
  captures that are not turntables, which `colmap` does; the Dragon and Lucy
  are a little below COLMAP.
- **Not run in a browser.** The provider's own modules build for wasm32; the
  `photos` command around them (staging, files, the stage runner) is not
  built for the browser yet, so nothing calls them there.
- **Memory and time grow with the mask's bounding box** (the doubled image
  and six blurred copies of it per octave): about 8 s for 73 photos of
  1749 x 1155 with four threads here. Matching compares all descriptors of a
  pair: 1 to 11 s for 292 pairs.
- Not tried: fewer photos or larger steps than 5 degrees, a strongly tilted
  axis, photos out of order.
