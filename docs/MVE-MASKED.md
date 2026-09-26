# Mask-aware selected MVE dense spike

This is a test-only isolated fork of the pinned selected MVE source described in
[MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md). It uses the fixture's real `maskPath`
images and enforces them inside dense matching. It does not alter application
depth code or select MVE as the product backend.

## Reproduce

The repeatable patch is
[`scripts/mve_masked/mve-object-mask.patch`](../scripts/mve_masked/mve-object-mask.patch).
It applies to MVE commit `bf2279f161ba962072ecac85224c15e82bc5f52e`.
The local source copy and binaries are ignored under `.local-tools/mve-masked/`.
`build.sh` refuses an existing destination; pass another build root to replay it.
It copies the pinned local source, cleans the five selected components in the
new copy, then builds only `math`, `util`, `mve`, `dmrecon`, and the `dmrecon` app,
with no global installation. It requires a 10 GiB free-space reserve plus a
copy/build allowance and uses `.local-tools/tmp`. It relies on the same existing
libjpeg, libpng, and libtiff packages as the original spike. The scene converter and verification
scripts use Pillow 12.2.0 in this environment.

```sh
scripts/mve_masked/build.sh .local-tools/mve-masked-replay
TMPDIR=.local-tools/tmp python3 scripts/mve_masked/convert_scene.py \
  --fixture build-opencv/synthetic-sparse-fixture \
  --scene build-opencv/mve-masked/scene-replay
TMPDIR=.local-tools/tmp \
  .local-tools/mve-masked-replay/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --master-view=1 --scale=2 --neighbors=2 --local-neighbors=2 --progress=simple \
  build-opencv/mve-masked/scene-replay
```

The converter first runs the original sparse conversion, so camera metadata,
seed bundle, and `undistorted.png` bytes are unchanged. It then copies an
`object-mask.png` embedding per view, matching `maskPath` by the sparse view's
image ID. It rejects wrong dimensions, nonbinary pixels, empty masks, duplicate
IDs, and mask paths outside the fixture. The C++ backend also checks dimensions,
channel count, and binary values when loading a mask. An all-white mask is
allowed for parity testing. The converter requires each view's mask; the patched
MVE binary retains ordinary unmasked behavior if no mask embedding exists.

The C++ validity pyramid marks a downsampled pixel valid only if every pixel
in the Gaussian's 4×4 parent support is valid. Reference patch centers and
all 5×5 reference pixels must be valid. Every projected neighbor sample must
have all four bilinear pixels valid. The analytic color derivative reads those
same four pixels. Seeds and queue propagation are rejected outside the
reference validity mask. This deliberately erodes object boundaries at coarse
scales; excluded pixels cannot influence accepted photometric values through
the image pyramid.

## Verification and quality

The bounded command used view ID 1, two neighbors, and each listed scale. The
same radial scorer from the original spike used the two rendered object planes
only for scoring. It was not available to the matcher. Counts and errors below
are for the estimated sparse camera poses, and each pair uses the same camera
metadata and resolution.

| Scale | Path | Object coverage | Object MAE | Object error >5 mm | Depths outside planes |
| --- | --- | ---: | ---: | ---: | ---: |
| L2, 400×300 | unmasked | 7,225/7,670 (94.20%) | 3.10 mm | 7.61% | 36,927 |
| L2, 400×300 | masked | 5,870/7,670 (76.53%) | 2.62 mm | 5.60% | 0 |
| L1, 800×600 | unmasked | 30,057/30,527 (98.46%) | 2.13 mm | 1.10% | 137,040 |
| L1, 800×600 | masked | 27,309/30,527 (89.46%) | 2.10 mm | 0.75% | 0 |

The mask removes outside-object depths and reduces the matched-pixel error
tail, but reduces object coverage by 17.7 percentage points at L2 and 9.0
points at L1. MAE at L1 changes little. This is an isolation result, not an
overall dense-quality win. Boundary erosion and fixture-specific camera/plane
scoring remain limitations. The independent paired quality analysis is recorded
separately in the quality ablation artifacts.

`verify_gate.py` rejected the supplied wrong-size and nonbinary masks (the
nonbinary image contains an interior nonbinary pixel with valid 0/255 extrema),
rejected an escaping symlink, and preserved mask/view association when project
image entries were reordered. An all-white mask reproduced every one of the
120,000 unmasked L2 depth values exactly. Inverting 5,388,035 outside-mask
source pixels across the three images changed none of the 120,000 masked L2
depths; the same mutation changed none of the 480,000 masked L1 depths. The
full L2 verification record is at
`build-opencv/mve-masked/verification-v3/verification.json`. That record includes
SHA-256 hashes of the binary, fixture inputs, reference depths, mutated images,
and generated depths. Each reconstruction has a 30-second timeout.
An independent supervisor replay in
`build-opencv/mve-masked/verification-supervisor/verification.json` reproduced
the all-valid parity and 5,388,035-pixel outside-mask mutation results with
zero depth mismatches; it also rejected the malformed and escaping masks.

```sh
TMPDIR=.local-tools/tmp python3 scripts/mve_masked/verify_gate.py \
  --fixture build-opencv/synthetic-sparse-fixture \
  --output build-opencv/mve-masked/verification-replay \
  --binary .local-tools/mve-masked/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --baseline-depth build-opencv/mve-spike/scene-supervised/views/view_0001.mve/depth-L2.mvei \
  --reference-masked-depth build-opencv/mve-masked/scene/views/view_0001.mve/depth-L2.mvei
```

MVE's selected source has BSD-3-Clause notices as recorded in the original
spike. This local experiment does not resolve complete transitive licensing,
measured capture quality, full-surface accuracy, memory limits, or portability.
