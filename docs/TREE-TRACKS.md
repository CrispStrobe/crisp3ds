# Fixed-camera tree track diagnostic

This experiment reuses the corrected ten-image scene in
`build-opencv/tree-refine/baseline/scene`. The source camera poses and intrinsics
remain fixed. Before the live run, the following settings are frozen to match
the earlier `tree_refine` pair stage: ORB 3,000/view with the same constructor
arguments, mutual 0.8 Hamming ratio, positive depth, at most 2 px pair
reprojection per view, at least 1° triangulation angle, 500 links per pair,
and 10,000 links total. Links are sorted by descriptor distance before the
same one-feature-per-view union rule. A multiview track additionally requires
each observation to belong to a triangle of retained pair links and some
pair triangulation to project into every track view within 4 px. No pose or
point fitting runs. Two-view components remain a diagnostic population.

The report records original components before acceptance and separately
summarizes rejected classes, original reserved-pixel errors, leave-one-out
projections, per-edge 3D disagreement, and accepted observations. The last two
are conditional on detected links and fixed estimated poses. A small accepted
residual therefore does not measure independent reconstruction accuracy.
Disagreement may arise from repetitive tree texture, camera estimates, or both.

After the frozen ORB diagnosis showed no closed triangles, one SIFT candidate
is predeclared: OpenCV SIFT with 3,000 features/view, mutual 0.8 nearest-neighbor
ratio using L2 distance, and exactly the same fixed-camera pair, cycle, and
multiview geometry tests. It runs once with separate output. This changes the
descriptor family and should not be read as a speed-equivalent comparison.

## Result

| Measured population | ORB | SIFT |
| --- | ---: | ---: |
| Pair links after geometry | 575 | 1,203 |
| Two-view components | 414 | 597 |
| Three-view components | 64 | 160 |
| Four-view components | 11 | 61 |
| Five- and six-view components | 0 | 14 |
| Multiview components without triangle support | 75 | 219 |
| Triangle-supported, >4 px all-view maximum | 0 | 16 |
| Accepted multiview tracks | 0 | 0 |

ORB exactly reproduced the prior 575 links and component lengths. Its nine
previously retained length-three tracks came from open chains: no original
multiview component contains a complete three-edge triangle. For the nine
previously retained tracks, the reserved pixel had median 384 px error and
maximum 1,515 px. Across all 75 ORB chains, separately triangulated linked
edges disagree in 3D by median 4.33 and 90th-percentile 10.96 arbitrary
source units. The 162 valid leave-one-out projections across those chains
have median 55.6 px and 90th-percentile 182.7 px error. Another 74
leave-one-out cases have no valid training triangulation or positive-depth
projection; they are counted as invalid rather than folded into percentiles.
These measurements explain why the original held-out residual was already
large before bundle adjustment: pair geometry does not make transitive links
consistent in a third camera.

SIFT added links and produced 16 cycle-supported components, but their
all-view maximum reprojection error is at least 5.65 px, with median 24.4 px;
none passes the frozen 4 px gate. The 16 components' linked-edge 3D points
disagree by median 12.7 source units. Its 219 open components also show median
53.5 px leave-one-out error (559 valid projections). There is no accepted
observation population whose residual could be reported as accuracy.

As a sensitivity readout on the *same measured observations*, among the 16
SIFT cycle-supported components, 0, 1, and 3 would have all-view maximum
at or below 4, 8, and 16 px respectively. Those higher cutoffs are diagnostics,
not alternate acceptance settings. Among all ORB open chains, zero meet 4 or
8 px. One SIFT open chain does meet 4 px geometry but has no triangle support.
The patterns support inconsistent fixed-camera multiview associations. These
data alone cannot assign the discrepancy uniquely to repetitive image
features or inaccurate supplied camera poses. No pose fitting or dense
reconstruction was performed.

The upstream [conversion notes](https://github.com/Matt1Up/tree-photogrammetry-dataset/blob/main/poses/colmap/README.md)
and [converter](https://github.com/Matt1Up/tree-photogrammetry-dataset/blob/main/scripts/xmp-to-colmap.py),
inspected on 2026-09-26, describe choosing the camera-axis convention by how
many sparse points land in front of cameras and inside images. That is not
an observation-by-observation reprojection check. The notes report zero XMP
distortion coefficients, so this inspection supplies no evidence that omitted
distortion caused our failure. These moving upstream pages are context, not
changes to the pinned local dataset. The next diagnostic should compare
image-derived epipolar geometry with supplied cameras before more pose fitting.

The complete original populations and component identifiers are in
`build-opencv/tree-tracks/orb-v2.json` and `sift-v2.json`. Their paired
`*-integrity.json` manifests include before/after SHA-256 hashes of the TSV,
ten images, executable, and source files; both say `inputs_unchanged: true`.
The reports are 22.7 and 69.3 kB, well under the 100 MiB limit. Free disk
space exceeded 10 GiB.

## Reproduce

Run from the repository root using fresh report names. The runner rejects
existing outputs, checks input hashes around each run, enforces a 120 s
timeout and 10 GiB free-space floor, and sets `TMPDIR` to `.local-tools/tmp`.

```sh
cmake -S scripts/tree_tracks -B build-opencv/tree-tracks/build \
  -DOpenCV_DIR="$PWD/build-opencv" -DCMAKE_BUILD_TYPE=Release
cmake --build build-opencv/tree-tracks/build -j2
build-opencv/tree-tracks/build/tree_tracks --self-test
python3 scripts/tree_tracks/run.py \
  --binary build-opencv/tree-tracks/build/tree_tracks \
  --views build-opencv/tree-refine/baseline/scene/views.tsv \
  --report build-opencv/tree-tracks/new-orb.json
python3 scripts/tree_tracks/run.py \
  --binary build-opencv/tree-tracks/build/tree_tracks \
  --views build-opencv/tree-refine/baseline/scene/views.tsv \
  --report build-opencv/tree-tracks/new-sift.json --sift
```
