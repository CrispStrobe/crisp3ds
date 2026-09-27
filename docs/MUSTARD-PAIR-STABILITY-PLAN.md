# Mustard verified-correspondence stability: frozen pre-run plan

Use the exact six TRAIN filename pairs and two sealed database hashes in
`MUSTARD-TWO-VIEW-POSE-PLAN.md`. Report the original full-input fit for all
six pairs in both arms. Perturb only the strongest common middle pair
`NP3_012.jpg / NP3_336.jpg` and both opposing pairs
`NP3_030.jpg / NP3_198.jpg` and `NP3_036.jpg / NP3_222.jpg`, in each arm.
These three pairs were selected by the previous acquisition-order and
verified-count panel rule, before this perturbation run.

For each selected pair-arm row, fit every leave-one-out subset, in original
verified-match blob order. Each removes exactly one complete verified
feature-index correspondence. Also consider every feature-index pair verified
in the other sealed arm and absent in the target arm, in other-arm blob order.
Add each such pair alone to the target's full correspondence set only if its
two indexed keypoint coordinates are exactly equal across the two databases.
Otherwise report it as incompatible and abstain from the addition. Do not
propose unverified matches or reselect pairs using any resulting pose.

Every baseline and perturbation uses the existing `mustard_two_view_pose.py`
estimator and its fixed initial pinhole intrinsic, three seeds, essential
RANSAC, cheirality recovery, homography comparison, and reliability gates.
Record per-seed inlier fractions, cheirality, median inter-ray parallax,
rotation angle, gate status, interseed spread, and rotation difference from
the same-seed full-input estimate even when a gate abstains. Summarize
abstention and rotation spread across perturbations without tuning any gate
or reading final SfM cameras, scanner, depth, or held-out views. Numerical
repeatability across seeds alone is not physical pose validation.

The source DBs and producer reports are hashed before and after through the
existing read-only sealed-input checks. The run requires both disks to have
at least 10 GiB free, writes one fresh JSON directly to the backup data
volume under 20 MiB, and has a 55-second process alarm within the 60-second
external run cap. A reviewed preflight is required before live execution.
