# Real turntable-object evaluation

## Protocol fixed before inspecting the reconstructed mesh

Input: all 73 original `pro/bunny/rgb` photographs from the acquired
[3DLF-Scan subset](OBJECT-DATASETS.md), 1749 × 1155 pixels each. No supplied
camera poses, calibration, masks, depth maps or reference vertices are passed
to reconstruction. The raw run is retained even if a later variant succeeds.

The second run uses the named, image-only `gamma05_clahe2` profile in
`scripts/mve_full/prepare.py`. It preserves originals and records input/output
hashes. No profile parameters are fitted to reference geometry. MVE settings:
maximum 2,100,000 pixels, depth scale 2, four neighbors, two local neighbors,
73 selected views, 70% minimum camera registration, 2 GiB output cap,
20-minute reconstruction deadline and 10 GiB minimum free disk.

The independent Revopoint physical-instance scan is the reference, not the
canonical Stanford design mesh. The scanner mesh has 217,505 vertices and
435,006 triangles, including four zero-area triangles. Its physical units
and cross-sensor transformation have not been independently verified.

The frozen reference-fit diagnostic uses 1,024 area-weighted samples with
seed 2026, 24 proper PCA initializations, four candidates refined for eight
iterations each, then twenty iterations for the best candidate. Both
directions contribute to correspondence fitting and symmetric RMS ranking.
This fits translation, rotation **and scale to the reference**; it cannot
measure independently recovered physical scale or camera accuracy.

Final scoring uses 4,096 samples per mesh, seed 2027 (different from alignment),
and a distance threshold equal to **1% of the reference bounding-box diagonal**.
Distances remain in reference-coordinate units, not asserted millimetres.
Report both directions, precision, recall and F-score. Nearest sampled-point
distances are approximate and include surface-sampling spacing; they are not
exact point-to-triangle distances or an official benchmark score. An additive
reference-versus-reference sampling control uses distinct seeds 2027 and 2028
at the same sample count and threshold without changing the primary protocol.

Inspect both reconstructed and reference surfaces visually. Full-background
or partial-object output is not accepted merely because cameras registered,
the PLY is structurally valid, or a fitted subset lies near the scan. Neither
a reference-fit score nor one successful object establishes production quality.

Runtime includes all reconstruction stages and validation. Report image
preprocessing separately if measured in a repeat rather than the original
run. Largest-process RSS is not the peak simultaneous process-tree sum.
Only Apple M1 CPU execution is currently measured.

## Results

The raw-photo run failed camera initialization after 52.52 seconds. Its failed
artifacts remain at `build-opencv/mve-full-bunny-001`. The contrast-profile
run and subsequent evaluation are recorded separately; no failed baseline
is overwritten or silently omitted.

The contrast variant registered **73/73** cameras and generated **3,286,728**
oriented samples. Stages took 8.39 s import, 77.88 s SfM, 213.73 s depth,
5.04 s point export and 633.82 s FSSR. The original invocation ended after
944.62 s because strict validation rejected zero-area raw triangles before the
native cleanup stage. That integration bug was fixed: intermediate surfaces
may contain degeneracies; final surfaces may not.

A hash-sealed continuation reused these completed stages. Native cleanup left
11 zero-area faces; an exact-zero-area filter removed those faces without
altering vertices or other face records. This produced **589,429 vertices and
1,193,446 triangles** at
`build-opencv/mve-full-bunny-002-finalized-v2/mesh.ply` (SHA-256
`4356427096fc6a80853c895fe626488003bb503c2e92800146ba61c4705f5030`).
The continuation took 16.25 s externally, including provenance checks. It is
**not a fresh replay** of the corrected full pipeline. The failed original
and first cleanup attempt remain unchanged.

Preprocessing was timed separately in a repeat: 26.80 s externally and all
73 output hashes identical to the original preprocessing. Adding these measured
pieces gives about **16.5 minutes**, not a single uninterrupted E2E timing.
The original reconstruction's largest reported RSS was 557,580,288 bytes;
the continuation's was 496,812,032 bytes. Neither is a simultaneous process-tree
peak. Evaluation itself peaked near 931 MB and took 5.47 s.

### Shape result: rejected

The fixed alignment fitted scale 17.23452. At threshold 2.085879777 reference
units, the independent-seed final sampled comparison gave:

| Measure | Reconstruction | Same-reference sampling control |
| --- | ---: | ---: |
| Precision | 12.30% | 86.74% |
| Recall | 18.82% | 85.77% |
| F-score | **14.88%** | 86.25% |

Output-to-reference median/p90 distances were 6.804/17.220 reference units;
reference-to-output median/p90 were 3.604/13.307. These are not millimetres,
not exact triangle distances, and not scores comparable to another benchmark.
The same-reference control is not a numerical correction to the primary score.

Supervisor inspection of three common-scale orthographic projections showed
severe shape disagreement: the reference bunny and its support base are clear,
while the reconstruction is a diffuse, distorted surface without a clean bunny
silhouette. All final coordinates and triangle indices are valid and no zero-area
faces remain, but **structural validity and complete camera registration did not
produce acceptable object geometry**. No KIRI-equivalence claim is justified.

The scan includes a support base; this is a whole-scan diagnostic, not an
object-only official score. Local registration minima, sampling spacing and
reference imperfections still limit the score. They do not justify promoting
the visibly poor result.

Supervisor cross-check: 50 sampled distances in each direction were recomputed
using direct coordinate differences against every target sample, independently
of the norm/dot-product distance implementation. Maximum disagreement was below
4.6e-12 reference units. The fitted rotation has determinant 1.0. These checks
validate arithmetic, not alignment correctness or reconstruction quality.

### Next quality experiments

1. Diagnose camera geometry before spending another dense run: fixed/shared
   intrinsics, plausible distortion, rotation/translation consistency, and
   foreground correspondence support. The free-intrinsics solution registered
   every image but produced widely varying intrinsics; registration count alone
   is not an adequate gate.
2. Separate an explicitly supplied-calibration diagnostic from the ordinary-photo
   lane. Any use of downloaded camera metadata must be declared, never silently
   treated as an image-only improvement.
3. Add image-derived foreground masks and thin-feature checks; keep reference
   geometry out of fitting and mask generation. Measure retained coverage along
   with precision rather than hiding errors through aggressive cropping.
4. Compare a stronger audited sparse backend on the same inputs, preserving
   dense settings, before attributing this failure solely to dense reconstruction.
5. Obtain a second physically scanned, non-Stanford-derived object with clear
   commercial data rights. This bunny is now development evidence, not an unseen
   acceptance set.
