# Quality-first reconstruction roadmap

Approved 2026-09-27. This is the active execution plan; it supersedes earlier
board-first priorities and MVE-specific backend commitments. Ordinary overlapping
photos of a rigid object are the input, including rotation and translation between
exposures. Calibration/markers are optional aids. Deformation is outside this scope.

## Baseline and definition of success

We have an experimental M1 CPU image-to-mesh path, not competitive object quality.
The 73-view bunny registered every view but failed shape acceptance (14.88% sampled
F-score under the documented diagnostic, not a percentage of KIRI quality).
See [the complete evaluation](BUNNY-EVALUATION.md). MVE remains a control/fallback,
not a required engine. Selected MVE build/smoke/unit checks passed on macOS arm64,
Linux x64 and Windows x64; live reconstruction portability remains unproven.

Primary target: reliable, geometrically faithful meshes from ordinary object photos.
Texture/novel-view appearance, speed and reliability are separate measured axes.
First match a strong established pipeline on several objects; then improve our
specific capture cases. No state-of-the-art or KIRI parity claim without comparable
same-input experiments, retained failures and independent geometry evidence.

## Ordered, scoped tasks

| ID | Work and owner | Deliverable | Acceptance gate |
| --- | --- | --- | --- |
| S01 | Root: freeze roadmap and experiment contract | This plan and linked status | Explicit evidence, limits, unknowns, and distribution boundaries |
| S02 | Sol backend: established classical pipeline | Pinned COLMAP camera estimation/undistortion to OpenMVS import, density, mesh, refinement and texture adapter | Meaningful unit tests; real M1 CPU run; validated artifacts, stages, failures, commands, hashes and timings; CUDA not mandatory |
| S03 | Sol object motion: isolate object-relative poses | Image-derived/manual foreground masks, separate pose-support and reconstruction masks, shared-intrinsics ablations | Same photos and dense settings; no scan-derived masks/seeds; report track spatial support, camera plausibility and registration failures |
| S04 | Sol evaluation: surface distances | Bounded accelerated sample-to-triangle distances, multiple fixed tolerances, analytic tests and self-controls | Correct distances on analytic fixtures; retained alignment/scale disclosure; measured real-reference controls |
| S05 | Root supervised comparison | MVE versus classical backend on YCB and bunny; then independent held-out objects | Mesh inspection, accuracy/completeness, failure rate and complete resource report; do not select from registration or face count alone |
| S06 | Sol backend after S05: geometry improvements | Multi-view consistency, confidence/visibility-aware fusion, bounds, refinement and texture checks | Same cameras for dense-only comparison; no hidden coverage loss or smoothing-only improvement |
| S07 | Sol matching after S03: learned matching experiment | LightGlue feature/matcher adapter with selected code/weights inventory | Improvement on held-out objects, measured M1 runtime/memory; no assumed GPU benchmark transfer |
| S08 | Root plus Sol: bounded GPU quality branch | Geometry-oriented neural reconstruction experiment on Kaggle | Read ../kaggle_usage.md before remote work; same cameras/masks, geometry versus appearance scores; licenses of weights/rasterizer checked |
| S09 | Sol product integration after quality gate | Shared worker/project contract, CLI and Tauri actions, masks, cancel/resume and diagnostics | Actual outputs displayed; no mocked success; stale cache rejection; interruption tests |
| S10 | Sol portability/release | Small real-photo e2e tests on each desktop target, packaging inventory/notices | Native scan validation, not compilation alone; App Store and mobile independent gates |

S02/S03/S04 are the first parallel implementation batch. Root reviews coordinate
conventions, artifact integrity and numerical tests, reruns checks independently,
and records actual live results. Each task owns separate files; no large speculative
rewrite of SfM, stereo or neural rendering. A failed build/run is evidence, not a
completed backend. Stop and report material new authority or infrastructure needs.

## Experiment contract

### Approved comparative execution batch

The next batch compares existing numerical engines rather than adding product UI:

1. Reproduce a bounded upstream OpenMVS example. Preserve supplied-camera stage
   controls separately from image-only reconstruction; upstream meshes are regression
   oracles, not independently measured truth. Pin the sample and executable versions.
2. Diagnose the failed fresh YCB alignment against the successful producer, including
   feature ordering, initialization, matching and random-state differences. Explicit
   successful-pair seeding is an assisted diagnostic, not automatic robustness.
3. Carry photo-derived foreground masks through image undistortion into the native
   dense stage. Verify projection/mask coordinates and native mask-label semantics.
   Do not substitute post-hoc cloud removal for native dense masking.
4. Compare MVE and COLMAP/OpenMVS on the same YCB and bunny image selections; retain
   failed arms and preprocessing differences. Add another engine only if it can be
   provisioned within the local disk/CPU budget and selected licensing constraints.
5. Report bidirectional surface distance distributions, normalized distances,
   precision/recall/F at multiple fixed tolerances, normal agreement and mesh topology,
   alongside registration, failures, stage runtime, memory and disk. A different metric
   must not conceal missing geometry or replace a failed quality gate.

Sol ownership: classical backend/alignment/masks; upstream control provisioning;
surface metrics/benchmark protocol. Root supervises, independently checks numerical
results, runs cross-pipeline comparisons and records the final evidence. Per-run
output caps remain enforced and aggregate free space is checked before native jobs.
MacBook free-space floor is 10 GiB; no GPU availability is presumed. Remote GPU or
large-data expansion requires first reading the existing environment instructions.

Second-object arm frozen before examining results: `classical-bunny-contrast-001`
uses all 73 PNGs in `build-opencv/bunny-gamma05-clahe2`, the exact preprocessed
pixels previously given to MVE. COLMAP uses unknown shared SIMPLE_RADIAL intrinsics,
CPU exhaustive matching, seed 20260927, 8192 requested SIFT features and 1749-pixel
feature/undistortion caps. No supplied poses, masks or reference mesh are inputs.
Automatic initialization; minimum 70% registration. OpenMVS uses the pinned runner's
resolution-level 2 and one-scale mesh refinement, two native threads, 2 GiB output
cap and 20-minute total deadline. This compares complete configured pipelines,
not identical SfM or dense algorithms/settings. New artifacts remain separate from
MVE and all unsuccessful runs. Score any mesh with the existing frozen bunny
alignment algorithm and 4096-query, seed-2027/2028 triangle metric protocol.

Execution checkpoint (2026-09-27): S01 documented; S04 implemented and independently
checked. S02/S03/S05/S06 have real YCB results, including failed controls, a recovered
textured mesh and a fixed-mask surface improvement (12.77% to 38.15% F at 1%).
They are not accepted as complete: fresh alignment replay failed, masks remain
object-specific, surface quality is rejected and held-out/portable live scans are
outstanding. See [YCB-COMPARISON.md](YCB-COMPARISON.md). Stabilizing the replay and
improving object-only depth/support now precede learned/GPU/product expansion.

- Freeze image hashes, selection, resolution, masks, camera model, parameters,
  versions, timeout, memory/output limits and evaluation scope before comparing.
- Preserve originals. No reference scan vertices, reference-derived tracks/masks,
  depths or poses enter image-only reconstruction. A supplied-camera diagnostic
  is a separate lane and must never stand in for end-to-end success.
- Rigid object movement can be represented as object-relative camera movement;
  stationary background features violate that single-motion model. Evaluate their
  exclusion. A rotating board may support poses but must not enter object geometry.
- Share intrinsics only for compatible lens/zoom/image groups. Record implausible
  distortion, focal spread, track distribution, cheirality and parallax. A small
  residual alone does not establish correct cameras.
- Compare dense backends with identical cameras to isolate errors. Preserve failed
  registrations, empty outputs and rejected regions in the report denominator.
- Measure sample-to-triangle accuracy and completeness at 0.5%, 1%, and 2% of the
  reference bounding-box diagonal, with declared samples/seeds and self-controls.
  Thresholds are normalized shape diagnostics, not established physical tolerances.
- Freeze alignment independently of scoring samples; retain unaligned outputs,
  fitted transform and scale. Reference-fitted scale does not prove metric accuracy.
  Declare whole-scan versus object-only scope and handling of bases/unseen surfaces.
- Inspect bare meshes, thin parts, holes and disconnected components. Evaluate
  appearance on held-out views separately; attractive textures can conceal errors.
- Record end-to-end and per-stage wall time, peak memory measurement method,
  hardware, output sizes, determinism and failure rates. Do not mix different lanes.

## Data and compute

Start with the local 60-photo YCB cracker box plus separate Google reference scan,
and the existing 73-photo bunny as a difficult development case. YCB is a shape
oracle, not metrology-certified truth. Bunny has a third-party shape-rights caveat.
See [dataset provenance](OBJECT-DATASETS.md). Expand to textured/flat, low-texture,
thin/cavity, stationary-background turntable and irregular handheld cases. Reserve
additional objects for acceptance rather than repeatedly tuning all available data.

MacBook: small experiments only, preserve at least 10 GiB free, temporary files in
`.local-tools/tmp`, no /tmp bulk work. Initial concurrent batch budget: 3 GiB added
disk, at most one dense/build-heavy task at a time; root coordinates contention.
Use bounded trial timeouts and record load before timing comparisons. VPS bulk
data belongs on /mnt/storage and fast scratch on /mnt/volume1, CPU only. No remote
job is claimed until capability/access preflight and actual execution succeed.

## Distribution and dependency boundaries

Original project code is AGPL-3.0-only after public release. An AGPL desktop/server
backend is now an integration candidate, not necessarily oracle-only. Third-party
licenses do not change, and noncommercial components remain unsuitable for the
commercial path. App Store-compatible dependency selection, packaging, source
obligations and terms require separate review; no root license badge clears them.
Do not relax the existing shipping dependency checker without a reviewed policy
update. Keep experimental tools isolated until promoted explicitly.

## Selection and product gates

1. Establish a reproducible complete classical baseline on M1.
2. Select from same-input shape results on multiple objects, not screenshots or
   camera count. Match the stronger baseline within predeclared per-case tolerance
   on held-out objects; disclose failures rather than hiding them in an average.
3. Only then tune speed, or add neural machinery for demonstrated failure classes.
4. Integrate the selected path into the application with useful recovery feedback.
5. Run comparable KIRI captures/exports before estimating parity. Neural priors can
   infer plausible geometry, not certify hidden surfaces or physical dimensions.

## First-batch evidence and follow-on decisions

The [YCB ablation](YCB-COMPARISON.md) improved COLMAP registration from 2/60 raw
views to 60/60 using photo-derived pose support, with substantially more plausible
intrinsics. This supports foreground-aware camera estimation as a priority; it
does not establish general automatic segmentation or dense mesh acceptance.
The [camera roundtrip](OBJECT-MOTION.md) additionally tests the COLMAP/OpenMVS
adapter's projection conventions. The [surface evaluator](SURFACE-BENCHMARK.md)
now removes nearest-target-sample spacing from the old diagnostic.

After this first surface comparison, consider object-aware image crops that retain
object pixel resolution while reducing wasted background stereo work. Such crops
must update image dimensions, principal point, observations and masks together,
and pass projection-equivalence tests. This is a proposed quality-preserving speed
experiment, not an implemented optimization. Dense silhouettes remain separate
from coarse pose-support masks. Keep the original uncropped run as a control.

## Primary implementation references

- COLMAP camera constraints: https://colmap.github.io/faq.html
- OpenMVS stages and integration: https://github.com/cdcseacave/openMVS
- LightGlue matching: https://github.com/cvg/LightGlue

These identify candidates, not proof of this project's compatibility or quality.
