# Mustard OpenMVG photo-only sparse diagnostic

The sealed `openmvg-mustard-photo-sfm-001` stage completed, but its supervisor
status is `failed_registration_or_sparse_gate` because only 46 of 48 TRAIN
views have poses. This document records a **read-only** diagnostic of its
receipt, pinned OpenMVG HTML report, and binary PLY. No SfM/dense stage was
run here; no supplied Berkeley pose, scanner mesh, held-out photo, or reference
geometry was opened.

## What can be verified

The [diagnostic](../scripts/classical_backend/openmvg_mustard_sparse_diagnostic.py)
requires the known `-001` root, the completed five-stage mustard receipt, its
sealed TRAIN-name and stage-report hashes, and matching receipt SHA-256/byte
records for `sfm_data.bin`, `cloud_and_poses.ply`, and
`SfMReconstruction_Report.html`. It rehashes those inputs after reading. It
checks 48 exact distinct TRAIN stems in the HTML; 46 rows with positive
observations and finite nonnegative residuals; 46 poses, nonzero tracks, and
observation count agreement; plus finite PLY coordinates and matching counts
of green camera centers and white landmarks. The command prints JSON and
writes nothing.

The PLY has no view IDs. The named camera-center mapping used for the orbit
numbers is therefore a **provisional source-order inference**. In pinned
OpenMVG v2.1, `sfm_data.hpp` exposes the same const `GetViews()` container to
`Generate_SfM_Report(const SfM_Data&)` and `Save_PLY(const SfM_Data&)`.
`main_SfM.cpp` calls the report writer, const cereal binary save, then PLY
save sequentially on the same SfM data, without an intervening view mutation.
The report writes one row per view in iteration order; PLY writes green
centers in that order only for views with pose and intrinsic. The map is
unordered, so sorting by view ID or angle would be wrong. This diagnostic
zips the **observed report rows in their original order** to the green PLY
block only when observed-row count = pose count = green count, all green
points precede landmarks, and no prior/control colors appear. Otherwise it
reports an explicit orbit abstention. A future ID-bearing cereal-to-JSON
export would remove this source-order assumption and permit stronger pose
checks.

The 6° and 12° chord distributions compare registered angles modulo 360;
180° chords compare opposing views; the 000°–348° chord checks loop closure
against other 12° pairs. These are scale-invariant camera-center shape
diagnostics, not physical camera accuracy. The ideal circular 6°/180° chord
ratio is `sin(3°) = 0.05234`. Broad flags are fixed in code: closure no more
than 2× the other 12° median, 6°/180° median ratio between 0.01 and 0.2,
and opposing p90/p10 at most 2. They do not decide whether a mesh is usable.

## Read-only result

The diagnostic found 48 named TRAIN views, 46 observed/posed views, 222
landmarks, and 1,497 observations. `NP3_042.jpg` and `NP3_048.jpg` are the
two absent views. All exported PLY coordinates are finite, with exactly 46
green centers and 222 white landmarks. The report's per-axis RMSE is 0.517095
pixels, an internal fit statistic. The nominal orbit closes locally:
000°–348° chord divided by other 12° median is 0.377. But median 6° chord /
median opposing chord is 0.595, far above the ideal circular 0.052; opposing
chord p90/p10 is 21.0. Thus two of three provisional orbit-consistency flags
fail. The largest registered angular gap is 24° after excluding held-out
angles and the two missing TRAIN views. These measurements suggest poor
global camera-center coherence under the source-order mapping; they do not
establish which view is wrong or identify a root cause.

The cereal model itself remains opaque: intrinsics, rotations, per-track
membership, and independent reprojections have not been checked. A future
ID-bearing export should be evaluated separately before claiming a definitive
named trajectory. Reference camera or mesh comparison remains outside this
photo-only diagnostic.

The [synthetic tests](../scripts/classical_backend/test_openmvg_mustard_sparse_diagnostic.py)
exercise a full ideal orbit in deliberately unsorted report/PLY order,
ambiguous observed-row count abstention, a shuffled PLY color block, duplicate
names, and nonfinite camera centers.
