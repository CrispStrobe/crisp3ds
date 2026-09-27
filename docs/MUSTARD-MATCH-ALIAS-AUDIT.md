# Mustard match-graph alias audit

This TRAIN-only, read-only audit used the sealed fixed-initial exhaustive
COLMAP database (`database.db` SHA-256
`5858be16dfa3f0fc50303d0eb4a001e21ee14036f67755a9997d5dac374b3075`),
its `sfm.log`, producer `result.json`, and the repaired 48-camera sparse
model. SQLite was opened with `mode=ro&immutable=1`; no WAL/SHM sidecars
were present. The complete [audit report](../tests/evidence/mustard-match-alias-audit.json)
is a byte-exact copy of external `mustard-match-alias-audit-003.json`
(SHA-256 `02ea8ca4777f40642c9931f478f017aba89ca7eb0c304f23d236aaf9fb8a9787`).
The [runner](../scripts/object_motion/match_alias_audit.py) binds the database,
producer, log, repair report, three model binaries, and its own source before
and after reading; its SHA-256 is
`36e1698d6cbedd67370733806b3d126db094c66a12e2d2d9b37e8adeb10c9a76`.

The 48 NP3 filename suffixes define a **cyclic acquisition order only**.
They are not calibrated camera angles or supplied poses. Separation is the
minimum number of positions around the 48-image cycle; all 1,128 possible
pairs were present in the exhaustive geometry table.

| Cyclic index separation | Possible pairs | Verified pairs | Verified inlier correspondences | Median full rotation in recovered model |
| --- | ---: | ---: | ---: | ---: |
| Neighbor, 1–4 | 192 | 173 (90.1%) | 8,062 | 17.18° |
| Middle, 5–15 | 528 | 109 (20.6%) | 1,987 | 45.49° |
| Long, 16–24 | 408 | 118 (28.9%) | 2,103 | 11.94° |

The pinned PyCOLMAP 3.11.1 enum labels those 118 long verified edges as
66 `PLANAR_OR_PANORAMIC` and 52 `UNCALIBRATED`; neighbors split 83 and 90,
respectively. These labels describe the two-view geometry classification,
not independently reliable relative poses or a proof that planar ambiguity
caused the final trajectory.

The SfM log's one successful initializer was image IDs `3/6`,
`NP3_012/NP3_036`: cyclic separation 3, 64 verified inliers, and 28.88°
relative full rotation in the saved model. It was **not** initialized from an
opposite-block pair. The repaired model still has 224/939 tracks touching
views separated by at least 16 cyclic positions, including 213 touching
positions at least 20 apart. Long verified edges therefore remain a plausible
source of later cross-block connections, but this audit cannot prove that
they caused the folded trajectory. For example, `NP3_036/NP3_222` has 25
verified inliers across 23 positions and only 2.80° recovered relative
rotation; `NP3_060/NP3_240` has 15 inliers across 24 positions and 1.46°.

The database's `qvec/tvec` pair-pose fields are default identity/zero for
all verified rows (the producer did not request relative-pose estimation).
Thus the rotation column is calculated **only from the final reconstructed
cameras**, not an independent native two-view rotation estimate. It cannot
validate each match geometrically against a separate pose oracle. The
near-coincident opposing-label cameras in the
[trajectory diagnostic](MUSTARD-POSE-DIAGNOSTIC.md) and this graph are
consistent with aliasing, but neither establishes physical pose truth or
reference-mesh accuracy.

A focused next experiment would keep the same TRAIN photos, cached
features, fixed-initial camera, seed, and registration gates, but permit
verified match edges only within four cyclic acquisition positions. Freeze
that **dataset-specific capture-order prior** before running and compare
registration, graph connectivity, held-out-free camera trajectory diagnostics,
and sparse observation support with this baseline. Do not reinterpret a loss
of long edges or a smoother trajectory alone as shape improvement, and do
not use reference mesh or held-out views to select the neighborhood width.
