# Quality-first reconstruction roadmap

## Latest checkpoint: fresh photo-to-mesh on M1, quality rejected

The [60-photo cracker-box experiment](TURNTABLE-COMPLETE-PLAN.md) now has a
genuinely fresh masked JPEG → PyCOLMAP camera reconstruction → high-resolution
OpenMVS dense/mesh/refine/texture run on the M1. All 60 cameras registered and
passed the independent Berkeley rig trajectory gate. Native masks were applied
to all 60 depth maps; the complete textured mesh passed artifact checks and
the 10 GiB disk reserve. This closes the earlier *execution* uncertainty for
this one development object, not the product-quality or licensing gates.

The bare surface remains rejected: reference-fitted F1@1% of the scanner
diagonal is 42.23%; in the older `006` gauge transported using the 60 named
cameras it is 30.92%. The [stage diagnostic](TURNTABLE-COMPLETE-PLAN.md#fresh-complete-openmvs-oracle-and-shape-result)
finds excess breadth already in dense points (rough mesh 31.79%, refined mesh
30.92% in the same gauge). Neither a fourfold depth-resolution increase nor
refinement fixes this object. Google-to-Berkeley cross-scanner registration
remains uncertain, and this repeatedly used object is not held-out acceptance.

Next bounded tasks, ordered by causal value:

1. Freeze a reviewed object-only silhouette/support set on the 48 training
   views of a **second** real object, with an image-only method and an explicit
   thin-feature/label-retention review. Keep its 12 held-out views untouched.
   Compare feature-mask and native dense-mask effects separately on the same
   camera model; do not derive support from scanner/depth ground truth.
2. On the existing cracker-box run, compare native depths to the separately
   retained Berkeley sensor-depth frames using the already frozen pixel/ray
   protocol, including misses and hit-only error. Inspect where the dense
   cloud broadens before choosing a fusion or segmentation change. A result
   from three views remains diagnostic, not general metrology.
3. Test one candidate correction at a time on frozen cameras and photo masks:
   depth confidence/visibility fusion, or better object-only support. Preserve
   the unmodified OpenMVS oracle, rough and refined outputs, resource limits,
   and the camera-transported score. Require improvement in precision **and**
   recall or a justified tradeoff across at least two real objects, not just
   more vertices or a better independent alignment fit.
4. Prove an App Store/commercial-compatible dense backend before product
   integration. OpenMVS is AGPL evaluation-only. The [Mac AliceVision audit](ALICEVISION-MAC-ORACLE.md)
   ran isolated Metal depth kernels but not its photo pipeline; build/source
   dependency closure and a second real-object geometry comparison are gates.
   Keep a separately budgeted GPU/neural appearance lane; splat appearance
   alone does not satisfy the mesh-quality gate.

No KIRI parity claim is possible without the same captures and a matched
held-out geometry/appearance evaluation. The current result is demonstrably a
working *oracle pipeline* and demonstrably below our object-quality target.

## Current acceptance findings and next steps

### Targeted upstream changes: isolate a cause before patching

Maintaining project forks is permitted; upstream configuration is not a product
boundary. The next OpenMVS task is a [paired cached-depth fusion experiment](OPENMVS-PATCH-PLAN.md):
same cameras, masks, images and final depth maps, changing only the fusion mode.
Root reviews source/cache isolation; a separate Sol reviewer checks the contract.
Execution requires two independent copies, unchanged source hashes, bounded
runtime/output, and the 10 GiB local reserve. A point-count increase alone cannot
accept a mode: surface coverage and outlier/error diagnostics must follow.
Only then consider a small pinned fork with rejection counters or an algorithm
fix, preserving an unmodified control and recording the patch's actual effect.

In parallel, test one separately versioned, label-preserving mustard support
candidate on the 48 training photos. Visual acceptance is a separate gate from
successful generation; it is coarse feature support, not dense geometry truth.
The rejected predecessor, held-out split and original photos remain unchanged.

The [coverage-loss](COVERAGE-LOSS.md), [pre-mesh support](DEPTH-SUPPORT-AUDIT.md)
and [rough/refined](ROUGH-REFINED-COMPARISON.md) checks now separate support from
residual accuracy. Refinement offers a measured tradeoff, not a universal gain.
Keep both artifacts and do not select settings solely on hit-only error.

The new-object mask trial exposed an input problem before SfM: numerical area
gates did not detect lost label texture. The mustard candidate is rejected;
drill remains unmasked. Next prepare label-preserving, reviewed foreground
support (manual or independently licensed segmentation), freeze that input,
then test the two-phase alignment policy on the 48 training views only. Preserve
the 12 held-out views for separately disclosed localization/render evaluation.
Do not treat these masks as physical ground-truth silhouettes or reuse them for
geometry acceptance without separate validation. No seed sweep against the
existing sensor depths is approved by this plan.

## Current continuation: comparisons and initialization diagnostics

The [three-branch comparison plan](PIPELINE-COMPARISON-PLAN.md) now defines the
classical, Gaussian and neural-surface lanes, actual-versus-planned oracle status,
and two completed VPS real-object acquisitions. [Initialization recovery](INITIALIZATION-RECOVERY.md)
now demonstrates 60/60 registration with fixed initial intrinsics, followed by
delayed image-only self-calibration, on one cached case. This is not a cold e2e
result. [Recovered dense control](RECOVERED-DENSE.md) now produces a validated
40,450-point dense cloud and 38,440-face rough mesh on M1 CPU; artifact validity
is not geometry acceptance. [YCB expansion](YCB-EXPANSION.md) adds 120 real photographs
and two independent scanner meshes. [48/12 splits](YCB-EVALUATION-PROTOCOL.md)
are frozen and all selected assets rehashed on the VPS; masks and reference
registration remain pending. [AliceVision review](MTL-ALICEVISION.md) corrects the CUDA-only
assumption: upstream SYCL and the separate Metal fork merit bounded builds.
Standalone Metal compute passed, but neither AliceVision backend has run here.
No new neural GPU result is claimed.

## Previous bounded execution: calibrated mesh and sensor-depth checks

After the registration audit and the retained `012` resolution/budget failure,
the next supervised batch is:

1. Sol backend: one fresh `013` run from the sealed calibrated `002` camera
   model, explicit minimum resolution 600, effective-resolution preflight,
   1 GiB output cap, two native threads and ten-minute deadline. Retain the
   failed `012`; do not increase limits or retry within this batch.
   `013` completed dense reconstruction but its camera checker exposed a
   uniform-scale/rounded-short-edge bug. After correcting the checker without
   relaxing tolerance, all 60 retained depth maps passed independently. A
   separately recorded `014` mesh/refine/texture-only continuation is approved
   (200 MiB, five minutes, two threads); no dense recomputation or mutation of
   the stopped `013` is permitted. This is a composed recovery, not fresh e2e.
2. Sol projection review: validate the three already extracted Berkeley
   depth frames against RGB using distinct camera intrinsics/extrinsics;
   record distortion, pixel-origin and unit assumptions and inspect overlays.
3. Sol surface evaluation: implement analytically tested first-hit depth
   comparison, with named-camera similarity alignment only. Freeze photo-only
   object support and missing-ray denominators before evaluating candidates.
   No Google mesh fitting, reference-derived mask tuning or new downloads.
4. Root: review coordinate/provenance gates before any real sensor score,
   inspect completed mesh artifacts, run regression tests and record failures.

Allow at most 20 MiB additional evaluation output alongside the 1 GiB native
run, keep 10 GiB free, and avoid `/tmp`. Sensor-conditioned same-capture checks
are diagnostics, not held-out reconstruction accuracy or certified metrology.
Image-only robustness remains a separate unfinished gate.

Outcome: `013` produced 40,518 dense points; `014` produced 38,550 rough
triangles but timed out during refinement. The frozen three-view rough-mesh
sensor comparison favored earlier `008` on conditional depth residuals
(2.88 versus 5.96 mm hit-only mean), with similar coarse ray coverage
(84.08% versus 83.66%). Calibration assistance is not promoted as a quality
improvement. These are qualified sensor diagnostics, not metrology.

Next scoped priorities: inspect spatial missing-ray patterns using the saved
per-ray data; stabilize image-only camera initialization without supplied
intrinsics; compare rough versus refined outputs within the same camera gauge
before changing refinement settings. A usable rough-mesh/texture path should
remain available independently of expensive refinement, with explicit partial
stage status. Do not infer KIRI parity or optimize against only these three views.

Approved 2026-09-27. This is the active execution plan; it supersedes earlier
board-first priorities and MVE-specific backend commitments. Ordinary overlapping
photos of a rigid object are the input, including rotation and translation between
exposures. Calibration/markers are optional aids. Deformation is outside this scope.

## Baseline and definition of success

We have an experimental M1 CPU image-to-mesh path, not demonstrated competitive
object quality. The 73-view bunny registered every view but has not passed shape
acceptance. Its historical 14.88% sampled F-score is not a reliable intrinsic
quality estimate after the [registration audit](REGISTRATION-AUDIT.md), and is
never a percentage of KIRI quality. See [the earlier evaluation](BUNNY-EVALUATION.md).
MVE remains a control/fallback,
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

The [subsequent comparative batch](BENCHMARK-RESULTS.md) adds multiple surface
metrics, an upstream supplied-camera software control, native depth-mask proof,
full-resolution YCB and a second classical object. Explicit seed selection recovers
a fresh 60/60 box camera run; generic retries recover 73/73 bunny cameras, but neither
establishes universal automatic robustness. Four times as many YCB depth pixels
does not materially improve its score. Bunny whole-scan and post-hoc object-ROI
fits expose registration failure, so their F-scores are not a reliable standalone
pipeline ranking. The next evaluation task is independently validated registration
(including partial overlap and known-frame/landmark controls), alongside isolated
camera-model and depth-support ablations. Do not keep changing reference crops or
alignment settings until a preferred pipeline scores well.

### Next supervised batch: distinguish calibration and registration failures

1. **Registration audit (Sol surface agent):** keep the published aligner and
   prior scores unchanged. Use known proper similarities of asymmetric synthetic
   geometry, independent surface samples, missing support surfaces, outliers and
   symmetry stress cases. Add self-transformed real scanner geometry with known
   transforms. Measure transform error and independent paired-point residuals,
   not only the same nearest-point objective used to fit. A stress-case failure
   is evidence, not a reason to tune to the real candidate's F-score.
2. **Intrinsics ablation (Sol camera agent):** reuse the exact foreground feature
   database in fresh copies. Freeze a camera-model comparison before running it;
   keep supplied Berkeley intrinsics in a separate calibration-assisted lane.
   Verify distortion parameter order, pixel-center conventions and disabled
   intrinsic refinement. Supplied camera poses and scanner meshes stay out of
   reconstruction. Compare registration, image residuals and camera-reference
   diagnostics; distinguish fitted residuals from held-out evidence.
3. **Bounded dense control (Sol backend agent, after root review):** if a camera
   trial passes artifact/projection checks, run the existing CPU OpenMVS chain
   at the earlier low-resolution profile with correctly undistorted photo masks.
   Record complete provenance and input hashes without mislabeling calibrated
   inputs as image-only. Reuse helpers rather than implementing another engine.
4. **Root review:** check independent numerical oracles, inspect projections and
   bare geometry, rerun unit/live checks and record both successful and failed
   controls. A known-intrinsics gain is a diagnosis, not a new general-photo
   quality claim. No promotion into the application until the quality gate passes.

This batch uses already-local data and native tools. New retained output budget
is at most 1.5 GiB, with a hard 10 GiB free-disk reserve, two native threads,
bounded stages and no bulk work in `/tmp`. No previous artifacts are overwritten
or deleted to make a run look successful. Further reconstruction jobs are serial.

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
