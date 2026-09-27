# YCB shared-gauge sensitivity diagnostic

This protocol is fixed before shared-gauge scores are inspected. The primary
YCB scores in [YCB-COMPARISON.md](YCB-COMPARISON.md) fit a proper Sim(3)
independently for each output. That can choose different rotation/scale minima
even when the pipeline variants share one recovered camera/world coordinate
system. This diagnostic freezes the **006-filtered reference-fit matrix**
(`build-opencv/classical-ycb-foreground-006-filtered/reference-fit-v1.json`)
and applies it unchanged to the 004, 006, 008 and, if a final mesh exists,
009 refined meshes, and the completed 011 continuation. It does not modify
their primary transforms or scores.

The parent fit is hash-bound to the YCB scanner reference and exact 006 mesh.
Every candidate is separately bound by its own mesh SHA-256. Before sharing
the matrix, require byte-identical `cameras.bin`, `images.bin`, and
`points3D.bin` in each candidate's retained COLMAP sparse directory and the
006 parent directory. These hashes establish a common recovered world gauge;
they do not establish physical scale or independent camera accuracy. The
006 matrix was fitted to the same scanner reference, so all shared-gauge
scores remain reference-conditioned shape diagnostics.

Use exactly 2,048 deterministic area-weighted query samples in each direction,
seeds 2027/2028, and the existing fixed 0.5%, 1%, 2% reference diagonal
thresholds (`0.0013931596318764927`, `0.0027863192637529854`,
`0.005572638527505971` reference-coordinate units). Record precision,
recall, F, normalized Chamfer-L1 mean, distance tails and normal agreement.
Also report area-weighted sample covariance eigenvalues and principal-axis
angles for candidate versus 006 in their common native frame, plus the
relative rotation/scale of each candidate's *separately* fitted primary
transform. Principal axes can be unstable when eigenvalues are similar.
No threshold, crop, scale or matrix is adjusted after seeing scores.

An improved shared-gauge score for a candidate with a rotated primary preview
would indicate sensitivity of the per-output fit, not proof of better
geometry. A poor shared-gauge score could reflect geometry differences,
despite common cameras. A missing final 009 mesh is reported unavailable,
never silently replaced by an intermediate. The YCB Google mesh is a
separate-sensor shape oracle, not certified metrology ground truth.

## Results

The frozen 006 transform SHA-256 is
`59d868632a7aadc1c7ccf315d30848b552c992f9357b0b2f35076545e6c8dc28`.
Across 004, 006, 008 and the failed 009 sparse directories, hashes match
exactly: `cameras.bin` `3c4636b5...`, `images.bin` `251cb138...`, and
`points3D.bin` `c11fd9cc...`. Full hashes and each candidate mesh/primary
transform hash are in the saved shared-gauge reports.

| Output | Primary 1% F (independent fit) | Shared 006 gauge 1% precision / recall / F | Shared normalized Chamfer-L1 mean | Primary fit rotation from 006 | Native PCA leading-axis angle from 006 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 004 unfiltered | 12.77% | 12.55 / 38.92 / 18.98% | 16.14% of ref diagonal | 128.9° | 55.3° |
| 006 filtered parent | 38.15% | 42.63 / 34.52 / 38.15% | 2.52% | 0° | 0° |
| 008 native masked | 42.04% | 32.32 / 39.36 / 35.49% | 3.31% | 41.8° | 28.6° |
| 009 full-resolution attempt | unavailable: no final mesh | unavailable | unavailable | unavailable | unavailable |
| 011 full-resolution continuation | 42.06% | 32.57 / 39.06 / 35.52% | 3.36% | 177.8° | 31.4° |

The corresponding 0.5/2% shared-gauge F-scores are 10.62/33.62% for 004,
19.48/63.54% for 006, 18.10/59.39% for 008, and 17.01/59.99% for 011.
Under one gauge, 004's
large extraneous/background surfaces remain visible in the
[common-scale preview](../.local-tools/classical-ycb-004-shared-006-preview.png).
Its recall rises from 11.23% under its own fit to 38.92%, while precision
falls from 14.79% to 12.55%; this is alignment sensitivity, not a recovered
good mesh.

For 008, the independently fitted transform rotates 41.8° relative to 006
and improves the numerical 1% F-score by 6.55 percentage points. The
[shared-gauge preview](../.local-tools/classical-ycb-008-shared-006-preview.png)
shows a roughly axis-aligned box in the common camera frame, whereas the
separate fitted preview appeared diamond-shaped in plane. The primary fit's
higher score therefore should not be read as proof that its object frame is
more plausible. Conversely, the shared-gauge score does not prove 008 has
better geometry than 006: its F is lower than 006's 38.15%, and its mean
distance is larger. The native PCA leading eigenvalues for 008 are
0.1909/0.1666, relatively close; its 28.6° principal-axis difference is
suggestive but unstable under small shape/sample changes.

The completed 011 continuation likewise scores 42.06% under its independently
fitted matrix but 35.52% under the frozen 006 matrix. Its independent fit
differs by 177.8° in rotation from the 006 fit. The [independent preview](../.local-tools/classical-ycb-011-primary-preview.png)
places the broad face diagonally, whereas the [shared-gauge preview](../.local-tools/classical-ycb-011-shared-006-preview.png)
retains the recovered camera-frame orientation. Thus two distinct dense
outputs can each obtain a similar reference-fitted score through markedly
different rotations. The shared-gauge 011 score is also below 006's 38.15%,
so neither 011's primary score nor its denser acquisition alone establishes
better shape in this diagnostic. The 011 native leading covariance eigenvalues
are 0.1939/0.1607; its principal-axis angle is only a weak orientation cue.

Reports: [004](../build-opencv/classical-ycb-foreground-004-finish/shared-006-gauge-v1.json),
[006](../build-opencv/classical-ycb-foreground-006-filtered/shared-006-gauge-v1.json),
 [008](../build-opencv/classical-ycb-native-masked-008/shared-006-gauge-v1.json),
and [011](../build-opencv/classical-ycb-native-masked-011-continuation/shared-006-gauge-v1.json).
All retain the original primary result files. The failed 009 full-resolution
attempt hit its output cap during dense processing and produced no final mesh;
an intermediate cloud would not answer this surface question.
