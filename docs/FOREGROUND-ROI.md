# Manual foreground ROI support audit

This protocol takes ten manually inspected polygon envelopes for the target
tree in the frozen 768×512 PNGs. It does not infer a mask, segment images,
alter features, rerun matching, or reconstruct geometry. The polygons are
approximate; gaps can contain background. A SIFT feature centre inside an
envelope can still have a descriptor that sees background. No ROI category
is physical ground truth.

The versioned input schema is `foreground_roi_v1`. Every one of the ten
frozen image names must occur exactly once. `original_sha256` is the SHA-256
of the exact staged PNG used for feature extraction (as recorded in
`heldout-v2-001/holdout.json`). The provenance text is fixed. Each image has
one or more simple polygons; coordinates are in the original 768×512 image
frame. COLMAP feature centres are used directly, with the first pixel centre
at `(0.5, 0.5)` and no half-pixel shift. A centre exactly on any polygon edge
or vertex counts as inside. Vertices must be finite, within `[0,768] ×
[0,512]`, nonrepeated, and must form non-self-intersecting nonzero-area
polygons. Multiple polygons are combined by union, without hole semantics.

Template for **one** image entry (the real file must have all ten):

```json
{
  "schema": "foreground_roi_v1",
  "provenance": {
    "method": "manual",
    "description": "approximate target-tree envelope; gaps may contain background"
  },
  "images": [
    {
      "name": "The_Tree-34.png",
      "width": 768,
      "height": 512,
      "original_sha256": "<64 lowercase hex characters from frozen holdout manifest>",
      "polygons": [[[100, 80], [300, 80], [300, 400], [100, 400]]]
    }
  ]
}
```

The rectangle above only illustrates the JSON shape; it is not a proposed
tree envelope. Polygon vertices must come from manual image inspection.

The read-only audit first verifies polygon syntax and all source hashes.
It then reports, separately for each image, how many original training and
held-out feature centres are inside. For both training and raw held-out
descriptor pairs it counts both endpoints inside, one inside, or neither.
For every saved sparse 3D track, the compact mapper rows are mapped back to
original feature IDs before counting inside observation links. Points are
classified by all, any, or strict-majority inside links, with half ties
reported separately. For raw held-out pairs with two registered cameras it
recomputes the existing pose-derived square-root Sampson residual and reports
inside-pair versus other-pair summaries, including unregistered, numerical
invalid, finite above-4-px, and missing-inclusive above-4-px or unavailable
counts on the original raw denominators. It also divides the frozen 181
closed held-out cycles into all-three-inside versus other while preserving
all availability statuses and the original full-set metrics separately.

These ROI subsets are explicitly **post hoc development diagnostics** on
already-seen images and matches. They cannot replace the frozen full-frame
holdout result or establish foreground-only depth accuracy. An HTML overview
references the original PNG bytes and overlays the manual polygons with
native SVG; no raster masks or generated imagery are written.

The run requires a fresh output directory, at least 10 GiB free plus a
50 MiB cap, and checks input hashes before and after. It writes no feature
database or model files.

```sh
python3 -m unittest scripts.foreground_roi.test_run
python3 -m scripts.foreground_roi.run \
  --polygons PATH/ten-manual-polygons.json \
  --output build-opencv/foreground-roi/run-001
```

## Frozen manual input and support result

Root manually inspected all ten PNGs and supplied
[`tree-envelope-v1.json`](../tests/datasets/tree-envelope-v1.json) before
seeing ROI support metrics. Its SHA-256 is
`4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853`.
All ten polygons passed the schema, bounds, intersection, and source PNG
hash checks. Root inspected the
[`overlay-review-001` screenshot](../build-opencv/foreground-roi/overlay-review-001/overview.png)
and accepted these approximate envelopes before the support run. Seven
focused tests passed.

The read-only support [report](../build-opencv/foreground-roi/run-001/report.json)
and [overlays](../build-opencv/foreground-roi/run-001/overlays.html) used the
exact accepted JSON. All input and code hashes matched before and after.
Root's fresh independent replay at
[`supervisor-001`](../build-opencv/foreground-roi/supervisor-001/report.json)
matched every count and input hash; four residual summary values differed
only at floating-point rounding, within `1e-12`.

| Saved population | Centres/links or pairs inside the manual envelopes |
| --- | ---: |
| Training descriptor pairs, both endpoints | 1,076 / 14,651 |
| Raw held-out descriptor pairs, both endpoints | 82 / 1,121 |
| Sparse 3D points, all track links | 151 / 1,773 |
| Sparse 3D points, any track link | 256 / 1,773 |
| Sparse 3D points, strict majority of links | 181 / 1,773 |
| Sparse point-track links | 684 / 7,046 |
| Closed held-out cycles, all three centres | 2 / 181 |

All 82 raw held-out pairs with both endpoints inside had registered cameras.
Their square-root Sampson median/p90 are 0.534/1.957 px, with 74/82 within
4 px and 8/82 above 4 px. The other 1,039 raw pairs include 17 unregistered
pairs; 1,022 score, with median/p90 0.376/1.865 px and 947/1,039 within
4 px on the raw denominator. The two all-inside three-view cycles both score
under 2 px. Two cycles are far too few for a foreground three-view quality
claim. Most saved sparse support lies outside these conservative manual
envelopes, and even inside-centre descriptors can include background.
