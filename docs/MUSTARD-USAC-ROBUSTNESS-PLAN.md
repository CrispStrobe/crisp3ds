# Mustard TRAIN-only two-view estimator robustness: frozen pre-run plan

Use only the same two hashed, immutable 48-TRAIN feature and verified-match
databases and exact six filename pairs in `MUSTARD-TWO-VIEW-POSE-PLAN.md`.
Read verified feature-index pairs and normalized measured keypoints through
the existing read-only pose and pair-index loaders. Never read final SfM
cameras, supplied poses, scanner, depth, or held-out views.

Run with `.local-tools/colmap-sparse/venv/bin/python`, whose OpenCV 4.10.0
and NumPy 1.26.4 match the historical pose environment. The runner rejects
another interpreter or OpenCV version. Compare `RANSAC` (historical control), `USAC_DEFAULT`,
`USAC_FAST`, `USAC_ACCURATE`, and `USAC_MAGSAC`. These five methods accepted
the same `findEssentialMat` API in OpenCV 4.10.0 on an analytic nonplanar fixture before
this plan. Use the exact historical assumed intrinsic, normalized threshold
`2/1536`, confidence `0.999`, maximum 2,000 iterations, seeds 17/23/31,
and one OpenCV thread. The historical RANSAC arm calls the unchanged
`mustard_two_view_pose.fit_pair` directly; the new USAC arm applies that
runner's same essential-inlier, cheirality, parallax, interseed-rotation, and
competing RANSAC-homography gates. No method gets a tailored threshold.

For each of the six pair-arm rows, run the full verified set in three fixed
input orders: original SQLite blob order, canonical ascending
`(left_feature_index,right_feature_index)` order, and reverse canonical.
The read-only preflight found every sealed blob is already canonical, so the
third order is necessary to exercise an actual permutation. On the strongest common
middle pair `NP3_012/NP3_336` and both opposing pairs `NP3_030/NP3_198` and
`NP3_036/NP3_222`, run three leave-one-out variants per arm in each input
order. Select the three removed correspondences by ascending SHA-256 of
`b"mustard-usac-robustness-v1|" + (stratum + "/" + left_name + "/" + right_name).encode() + b"|" +
little_endian_uint32_pair_bytes`, breaking hash ties by original row index.
No pose outcome determines the selection. A variant removes the same
feature-index pair from all input orders. There are 12 baseline pair-arm
rows and 18 sampled leave-one-out rows, each in three orders and five methods:
450 three-seed method fits in one bounded batch.

Report each seed's essential inlier count/fraction, cheirality count/fraction,
numeric rotation angle, median rotated-bearing-ray parallax, and status;
report homography support, pair-level abstention/classification, interseed
rotation spread, maximum baseline order difference, and rotation spread under sampled
removals against the same method/seed full fit in the same input order.
Also retain difference from the full blob-order fit as a secondary diagnostic. Preserve
numeric angles on abstained fits only as diagnostics. Primary comparison is
how often each method passes the existing reliability gates and how much its
rotation changes with one verified correspondence or input order, never
closeness to an unavailable ground-truth pose. No method ranking, gate, or
sample changes after observing real fits.

Run only after root reviews the source/API preflight, script hash, and
analytic tests. The runner checks source database and producer-result hashes
before and after, requires at least 10 GiB free on workspace and backup
volumes, has a 55-second process alarm within a 60-second cap, and writes
one fresh JSON file directly on the backup volume under 20 MiB. It never
mutates the sealed inputs or uses large `/tmp` files.
