# Selected MVE dense execution spike

This is a test-only execution of MVE's selected `math`, `util`, `mve`, and
`dmrecon` code on the existing rendered sparse fixture. It does not select MVE
for the application. No global dependency was installed and no SfM library,
`make all`, or GPL `sceneupgrade` target was built.

## Source and build

The source is the pinned [MVE commit
`bf2279f161ba962072ecac85224c15e82bc5f52e`](https://github.com/simonfuhrmann/mve/tree/bf2279f161ba962072ecac85224c15e82bc5f52e).
The downloaded codeload archive is 611,809 bytes with SHA-256
`0e33d40b150313617d28ab4297459939e9ad59c47f21566ed649daf3b34c47f6`.
The selected source's `LICENSE.txt` has SHA-256
`79c48dfce85b3658b579a6b6d80be1d8aa31f78c91bc45fcb1bc4ccab8d70c91`
and states BSD-3-Clause terms; its separate non-free disclaimer names only
`libs/sfm/sift.cc` and `libs/sfm/surf.cc`, which this build excludes. The
`apps/dmrecon/dmrecon.cc` and `libs/dmrecon/dmrecon.cc` SHA-256 values are
`a7d9d72e102b1886f1085be7409334f91d7168b45d62c4a2d1c876742313fe2f`
and `4115fb4e861815aff895279f0d8fc35f798a8f143ac648d29cf04cfb5c4be8b7`.
This records the selected code; it is not a redistribution or complete
transitive-license approval.

The build used Apple clang 17.0.0, `make -j2 CXX=clang++` separately in
`libs/math`, `libs/util`, `libs/mve`, `libs/dmrecon`, and `apps/dmrecon`.
Existing pkg-config packages were libjpeg 3.2.0, libpng 1.6.58, and
libtiff-4 4.7.2. The linked `apps/dmrecon/dmrecon` SHA-256 is
`28b7cc9a91a506677f27bf606fa3b5d0636e2a3bf01bd73f31bc9ec657d56266`.
Build outputs are under ignored `.local-tools/mve-spike/`.

## Scene conversion and camera check

`scripts/mve_spike/convert_scene.py` writes a fresh MVE scene from
`build-opencv/synthetic-sparse-fixture/project.json` and `sparse-report.json`.
It uses three object-to-camera poses estimated from rendered markers and 1,189
reconstructed sparse points with 2,843 rendered-image observations. The 8-bit grayscale
input PNGs are copied as MVE `undistorted` embeddings. Ground-truth planes
and truth camera poses are excluded from conversion.

For the 1600×1200 fixture, MVE normalized focal length is `1500/1600 =
0.9375`, pixel aspect is 1, and principal point is
`((800+0.5)/1600, (600+0.5)/1200)`. MVE's `fill_calibration` adds the half
pixel to K; its projection convention subtracts 0.5 to recover observed pixel
coordinates. Rotation and translation are already world/object to camera and
are copied without inversion. The converter checks every seed reprojection
before writing. `check_scene.cc` then asks the actual selected MVE library to
load the scene and print its K, rotation, translation, and seed counts;
`verify_loaded.py` checks the loaded values and reprojects all observations.
The loaded-camera RMS is 0.350443 px, maximum 1.909821 px. The result is
[camera-check.json](../build-opencv/mve-spike/camera-check.json).

The source and invocation are:

```sh
TMPDIR=.local-tools/tmp python3 scripts/mve_spike/convert_scene.py \
  --fixture build-opencv/synthetic-sparse-fixture \
  --scene build-opencv/mve-spike/scene

clang++ -std=c++17 -O2 \
  -I.local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/libs \
  scripts/mve_spike/check_scene.cc \
  .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/libs/mve/libmve.a \
  .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/libs/util/libmve_util.a \
  -o build-opencv/mve-spike/check_scene $(pkg-config --libs libjpeg libpng libtiff-4)

TMPDIR=.local-tools/tmp python3 scripts/mve_spike/verify_loaded.py \
  --inspector build-opencv/mve-spike/check_scene \
  --scene build-opencv/mve-spike/scene \
  --fixture build-opencv/synthetic-sparse-fixture \
  --output build-opencv/mve-spike/camera-check.json
```

## Dense run and result

The bounded run used one reference view, two neighbors, and scale level 2
(400×300). It completed in about 1.48 seconds of reported reconstruction time
and saved `view_0001.mve/depth-L2.mvei`:

```sh
TMPDIR=.local-tools/tmp \
  .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --master-view=1 --scale=2 --neighbors=2 --local-neighbors=2 --progress=simple \
  build-opencv/mve-spike/scene

TMPDIR=.local-tools/tmp python3 scripts/mve_spike/evaluate_depth.py \
  --depth build-opencv/mve-spike/scene/views/view_0001.mve/depth-L2.mvei \
  --scene build-opencv/mve-spike/scene \
  --fixture build-opencv/synthetic-sparse-fixture --view 1 \
  --output build-opencv/mve-spike/depth-evaluation-radial.json
```

The [corrected radial-depth evaluation](../build-opencv/mve-spike/depth-evaluation-radial.json) uses
the two rendered object planes solely as a scoring reference. For each output
pixel, it intersects the ray from the converted camera with those planes and
uses the nearest intersection inside a plane rectangle. MVE's stored depth is
Euclidean camera-to-point distance, established by its seed initialization
and normalized inverse-K rays, so truth intersections are converted to the
same distance. The score uses estimated sparse camera poses, so its error
includes pose error. It compares only pixels whose ray hits an object plane;
it is a synthetic, visible-panel diagnostic, not full-surface accuracy.

| Metric, reference view 2 | Result |
| --- | ---: |
| Object-plane truth pixels | 7,670 |
| Valid depths on object planes | 7,225 (94.20% coverage) |
| Invalid depths on object planes | 445 |
| Mean absolute error on matched object pixels | 3.10 mm |
| Median absolute error on matched object pixels | 2.46 mm |
| Matched object pixels with error >5 mm | 7.61% |
| Valid depths outside the two object planes | 36,927 |

An independent replay used a fresh
[`scene-supervised`](../build-opencv/mve-spike/scene-supervised/conversion.json)
with the same selected binary and command options. Its
[loaded-camera check](../build-opencv/mve-spike/camera-check-supervised.json)
and [radial-depth score](../build-opencv/mve-spike/depth-evaluation-supervised-radial.json)
match all counts and displayed metrics above; MVE reported 44,152 filled
pixels and 1,474 ms reconstruction time on that replay.

The outside-object count is evidence that the result is not object isolated;
it is not classified as a stereo matching error. The scene has no enforced
object mask. MVE's selected image pyramid removes alpha rather than using it
as a dense matching mask, and these grayscale embeddings contain no alpha.
The application must supply and validate actual mask enforcement before this
backend can be considered for production. This single rendered fixture does
not establish performance on measured turntable captures, physical dimensions,
complete surfaces, memory limits, or other platforms.

An initial diagnostic incorrectly compared MVE's Euclidean depth to camera-Z
truth and reported 6.30 mm MAE. That metric is invalid; its values are retained
with an explicit invalid marker in
[`initial-z-score-invalid.json`](../build-opencv/mve-spike/initial-z-score-invalid.json).
The corrected radial score above is the only depth-error result used here.

Converter convention test:

```sh
TMPDIR=.local-tools/tmp python3 -m unittest \
  scripts/mve_spike/test_convert_scene.py scripts/mve_spike/test_evaluate_depth.py -v
```
