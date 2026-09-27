# Stock COLMAP positive control and native continuation

## Image-only sparse result

The [reference-first plan](REFERENCE-FIRST-PLAN.md), frozen at `40a254a`, was
executed once using runner commit `8c48b56`. Only the 11 pinned Sceaux JPEGs
entered feature extraction, exhaustive matching and automatic mapping. No
upstream cameras, points or mesh were supplied to reconstruction. Algorithm
defaults were retained; CPU, two threads and seed 20260927 were explicit.

The run completed in **71.489 seconds**: 46.52 feature extraction, 15.278 matching,
8.149 mapping, plus staging/provenance overhead. It registered **11/11 images**
and produced **8,412 points**, with finite poses, intrinsics and positions.
Peak sampled child RSS was 3,677,667,328 bytes; output about 59.8 MB. Source
photos, copied photos and tool hashes stayed unchanged. Tests were completed
before the live run. The [exact report](../tests/evidence/sceaux-stock-sfm-001.json)
has SHA-256 `7fe6d661710e25e55fd6e5d54afab27a2bb517e6b3fe8b6c43fd3c0645e3d44d`.

The evaluation-only [named-camera comparison](../tests/evidence/sceaux-stock-camera-001.json)
fits one proper similarity transform to all 11 supplied OpenMVG camera centers.
Center RMS disagreement is **0.3444% of the reference RMS camera-layout radius**;
orientation median/p95 disagreement is **1.251° / 1.294°**. This supports camera
agreement on this static scene. It does not establish physical pose truth,
intrinsic accuracy, object reconstruction quality or a SOTA ranking.

## Correction: our zero-repeated-image gate was not a COLMAP invariant

The original report remains `status=complete`, `positive_control_pass=false`,
`quality_accepted=false`: 183 tracks repeat an image, failing our predeclared
zero-repeat gate. **The gate's interpretation was wrong**, not evidence that
stock COLMAP necessarily generated corrupt geometry.

COLMAP's author [explains that nearby features can share a 3D track within one
image](https://groups.google.com/g/colmap/c/kcm32t15qEQ). The pinned
[3.11.1 triangulator](https://github.com/colmap/colmap/blob/3.11.1/src/colmap/sfm/incremental_triangulator.cc)
uses geometric residual checks for track continuation/merging. The independent
[track diagnostic](../tests/evidence/sceaux-stock-track-semantics-001.json)
finds 204 extra same-image observations out of 39,974 total. Their median
within-image separation is 0.675 px, maximum 5.878 px; all image backlinks and
database keypoint coordinates agree, and affected tracks have 3–11 distinct
cameras. This is consistent with nearby feature variants, not proof that every
match is physically correct or that a particular orientation mechanism caused it.

OpenMVS 2.4's [COLMAP importer](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/InterfaceCOLMAP/InterfaceCOLMAP.cpp)
retains and sorts these visibility entries; its scene loader does not require
unique image IDs. Multiplicity can affect support counts, so report it and use
distinct views for independent diagnostics. Do not alter valid COLMAP XYZ or
cameras merely to enforce an invented uniqueness requirement.

## Explicit prospective amendment: one unchanged native dense continuation

This amendment is recorded **after inspecting sparse/camera diagnostics and
before any dense output**. It does not rewrite the original gate result or
constitute a new blind sparse test. Eligibility now checks full registration,
finite geometry, valid track references and unchanged source hashes; repeated
image IDs alone are descriptive rather than a corruption condition. The
original 183 repeated tracks are retained without repair or bundle adjustment.

Use stock model `sceaux-stock-sfm-001/models/0` unchanged in the existing
`scripts.classical_backend.run --sparse-model` path, fresh external output
`/Volumes/backups/code/crisp3ds-data/sceaux-stock-dense-001`. Limits: 10 minutes,
1.5 GiB output, 4 GiB sampled child RSS, 16 MiB logs, two threads and both 10 GiB
free-space floors. Native undistortion max dimension 640; native OpenMVS
densify resolution level 2, rough meshing, refinement level 1 / one scale,
then texture. No masks, supplied software cameras, mesh fitting or reference
geometry enter processing. No automatic retry on failure.

This is a **composed image-derived pipeline**, not a fresh uninterrupted e2e
timing. Its low-resolution CPU profile differs from the earlier supplied-camera
control. Keep stage timings and all artifacts, including partial success, and
inspect undistorted camera/image agreement and the bare output mesh. A produced
mesh alone is not accepted quality; no new surface-ranking claim is authorized
by this compatibility trial. The existing runner labels external sparse source
provenance generically; this document and the hash-bound producer report supply
the specific lineage. Its actual worker command retains the venv interpreter
even though toolchain metadata resolves the base Python executable path.
