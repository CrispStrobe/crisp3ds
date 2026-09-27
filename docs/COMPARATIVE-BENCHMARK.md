# Comparative surface diagnostics: frozen protocol

This extension uses the existing exact-mesh-bound, reference-fitted Sim(3)
transforms and unchanged score samples. It compares three saved candidate
meshes: YCB cracker-box classical `004-finish` (unfiltered), classical
`006-filtered` (48-view photo-support filter), and the saved MVE bunny mesh.
No new alignment or mesh filtering is part of this benchmark. YCB uses 2,048
area-weighted query samples per direction, seeds 2027/2028, at 0.5%, 1%, and
2% of its scanner mesh bounding-box diagonal. Bunny uses 4,096 samples with
the same seeds and relative thresholds. The distance is from a sampled query
point to the closest triangle on the opposite mesh. Thus query integration is
approximate, while target-triangle distance is exact up to floating-point
arithmetic. The old precision, recall, and F-score definitions remain fixed.

Complementary metrics, defined before running the extension:

- Report each direction's mean, RMS, median, p90, and p95 distances. The
  symmetric mean, called Chamfer-L1 mean here, is half the sum of the two
  directional mean distances. Its normalized version divides by the reference
  bounding-box diagonal. These use raw distances, not squared distances; no
  cross-object absolute-unit comparison is implied.
- At each sampled query point, compare the source face normal with the normal
  of its nearest target triangle. Report mean absolute dot product in each
  direction, from 0 (orthogonal) to 1 (parallel or antiparallel). Also report
  mean absolute dot product only for samples within each declared distance
  threshold, with the qualifying fraction alongside it. A nearest triangle at
  an edge can be ambiguous; these are orientation-insensitive local-shape
  diagnostics, not oriented-normal or photometric scores.
- Report each raw triangle mesh's boundary edge count, nonmanifold edge count
  (more than two incident faces), vertex-connected component count, largest
  component face fraction, and Euler V-E+F count. These are structural mesh
  diagnostics. Components connected only at a vertex count as one. Small
  components, open boundaries, or nonmanifold edges do not alone prove a bad
  scan; the scanner reference has its own defects. Topology does not measure
  geometric similarity to the object.

Whole-mesh scoring includes all supported triangle faces, including bases and
background geometry. There is no certified per-view truth mask, so a missing
surface cannot be attributed to a particular image. Low reference-to-output
recall is an estimate of whole-scan surface incompleteness, conditional on the
scanner mesh and reference-fitted alignment. The YCB Google mesh and bunny
Revopoint mesh are separate-sensor shape oracles, not metrology ground truth.
The YCB and bunny instances, units, scanners and pipelines differ; compare
variants within an object and use cross-object results only as failure modes.

The existing evaluator limits still apply: 4,096 query samples per mesh,
1 million vertices and 2 million triangle faces per input, a 512 MiB PLY cap,
and 300 seconds per comparison. No production quality claim follows from these
diagnostics. Numerical results and commands are recorded below after running
this frozen protocol.

## Results

Saved reports (all use the exact prior transforms and the definitions above):
`build-opencv/classical-ycb-foreground-004-finish/comparative-metrics-v2.json`,
`build-opencv/classical-ycb-foreground-006-filtered/comparative-metrics-v2.json`,
and `build-opencv/mve-full-bunny-002-finalized-v2/comparative-metrics-v2.json`.

| Candidate | 1% precision / recall / F | Chamfer-L1 mean / reference diagonal | Accuracy p95 / diagonal | Completeness p95 / diagonal | Normal abs-dot, accuracy / completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| YCB classical 004, unfiltered | 14.79 / 11.23 / 12.77% | 5.70% | 12.48% | 13.13% | 0.533 / 0.493 |
| YCB classical 006, filtered | 42.63 / 34.52 / 38.15% | 2.52% | 6.13% | 12.24% | 0.883 / 0.811 |
| Bunny MVE | 19.29 / 62.40 / 29.47% | 2.74% | 10.46% | 6.21% | 0.483 / 0.480 |

| Candidate | F at 0.5% | F at 1% | F at 2% |
| --- | ---: | ---: | ---: |
| YCB classical 004, unfiltered | 6.57% | 12.77% | 23.98% |
| YCB classical 006, filtered | 19.48% | 38.15% | 63.54% |
| Bunny MVE | 17.78% | 29.47% | 48.14% |

The YCB reference has no boundary or nonmanifold edges, one
vertex-connected component, and Euler count 2. The `004` output has 92
boundary edges and five components, with 90.05% of faces in the largest;
`006` has 44 boundary edges and one component. Neither output has a counted
nonmanifold edge. At the 1% gate, `004` normal agreement among matching
samples is 0.610 accuracy / 0.552 completeness, whereas `006` reaches
0.932 / 0.931. Only 42.63% of output samples and 34.52% of reference
samples qualify in `006`; the high conditional normal score does not fill
missing surfaces. Filtering improved both distance tails and local orientation
agreement, yet completeness p95 remains 12.24% of the reference diagonal.
This supports the observed shape improvement while retaining the existing
quality rejection. The two outputs were fitted separately to the same scan,
so part of the numerical difference can reflect changed reference-fit minima.

The bunny reference has no boundary or nonmanifold edges, one component and
Euler count 2, although four zero-area faces are excluded from geometric
scoring. The MVE output has 6,498 boundary edges, 3,376 nonmanifold edges,
and 780 vertex-connected components; 99.19% of faces are still in its largest
component, so the component count mostly describes small fragments. At the
1% gate, normal agreement is only 0.485 accuracy / 0.464 completeness, despite
62.40% recall. Diffuse reconstruction geometry can lie near the reference
without following its local orientation. Its 10.46%-diagonal accuracy p95
also exposes a long output-to-reference tail that a symmetric mean alone
would obscure. This reinforces the earlier visual quality rejection.

The bunny row describes a different physical object and scanner; its absolute
coordinate-unit distances cannot be compared to YCB distances. Its larger
sample count also changes Monte Carlo precision. All rows remain whole-scan,
post-fit shape diagnostics, not measured physical-scale accuracy. The evaluator
took 11.80 seconds for YCB `004`, 8.48 for YCB `006`, and 134.69 for bunny
on this locally contended machine. Bunny's topology step used 82.55 seconds;
these are scoring times, not reconstruction or alignment runtimes.
