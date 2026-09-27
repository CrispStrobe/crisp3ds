# OpenMVS v2.4.0 evaluation fusion trace prototype

Scope: AGPL OpenMVS is an evaluation oracle only. This patch is not a backend
integration and must never run in the sealed `turntable-fresh-openmvs-005`
directory. The separate shallow checkout at
`/Volumes/backups/code/openmvs-v2.4.0-fusion-trace` pins upstream tag commit
`58117204c86bbb11a0b25b26a8987676cf11274d`. The repo patch is
[`openmvs_v240_eval_fusion_trace.patch`](../scripts/classical_backend/openmvs_v240_eval_fusion_trace.patch).

## Source location and trace meaning

In `libs/MVS/SceneDensify.cpp`, `DepthMapsData::DenseFuseDepthMaps` is the
`--fusion-filter 2` path. Its recursive `FusePoint` lambda sees each candidate
DMAP view ID, integer pixel, depth, and confidence before applying its used
pixel, confidence, depth agreement, reprojection, and normal tests. At the
emission gate, `nMinPixelsFuse` accepted pixels and `nMinViewsFuse` distinct
views are required; the output XYZ is the coordinate-wise median of accepted
world points. The existing PLY retains fused XYZ and views but no source pixel
IDs or rejected attempts. `Scene::DenseReconstruction` then may remove points
outside its ROI, changing PLY indices.

The opt-in patch writes `fusion-trace.tsv` in OpenMVS's working folder only
when `OPENMVS_EVAL_FUSION_TRACE=1`. It records the first eight emitted points,
their **pre-ROI** output index and XYZ, seed view/pixel, accepted pixel/view
counts, and up to 512 visited-pixel events per point. Events include the DMAP
view/pixel, raw depth and confidence when read, projected reference depth,
squared reprojection error, normal cosine, and the first reason for acceptance
or rejection. A truncation bit marks incomplete event lists. The checked-in
`validate_openmvs_fusion_trace.py` checks TSV structure and accepted-pixel/view
accounting; it does not claim physical accuracy or verify final PLY retention.

This fixed first-eight sample is a **trace plumbing smoke test**, not evidence
about a wide-geometry tail. Before a later causal diagnostic, freeze a
reference-free sampling rule that retains candidates spread across image views
and scene coordinates, including coordinate extremes. A bounded online
reservoir of candidate event records would avoid tracing every point to disk.
Validate each selected point's XYZ against the final PLY after ROI trimming,
and exclude or explicitly report candidates removed by the ROI. Never select
samples, confidence thresholds, or modes using Google or sensor truth.

## Current build boundary

The patch is source-reviewed and `git diff --check` passes, but it is **not
compile-checked**. The approved CMake object-file preflight stopped before
compilation because local Homebrew lacks `boost_system`; local C++ OpenCV
headers/config are also absent. No dependencies were downloaded, no native
fusion ran, and the sealed 005 source was not copied or modified. The separate
checkout plus incomplete CMake directory uses about 9.4 MiB, below the
200 MiB new-data ceiling; both volumes remained above 10 GiB free.

For any later build or run, recheck exact checkout/patch/source hashes, both
10 GiB floors, a physical 200 MiB total-new-data cap, and a fresh external
working folder. The first accepted control must reproduce the sealed
filter-2 dense PLY before treating traces as evidence. An opt-in tracing
binary must be kept separate from any shipping artifact.
