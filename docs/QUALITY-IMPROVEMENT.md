# Quality and performance improvement workstream

Started 2026-09-26 following the user's approval. This is an implementation plan, not a claim that the production dense stage or full scanner already works.

## Success definition

User priority clarified: materially better **quality**, not merely faster execution. The stereo profile speedup is not the quality milestone. The current follow-up has three independent tasks: mask-aware MVE sampling with outside-mask mutation tests; same-fixture depth-resolution/pose-source ablations; and fixed-population quality acceptance scoring. The original source, images, baseline outputs and oracle-pose distinctions must be preserved.

The experimental quality comparison will report full-object and boundary accuracy, missing-inclusive errors and coverage. A provisional relative improvement gate is useful for rejecting misleading changes but is not a physical product tolerance. Costlier computation is acceptable in these bounded experiments if it improves the quality metrics. Ground-truth camera poses are diagnostic inputs only; an oracle-pose result must never be presented as an improvement delivered by the application.

Improve usable, measured surface accuracy and completeness at a documented runtime/memory cost. Do not optimize a single stereo score by filling unsupported regions. Keep metric accuracy, coverage, interpolation, sharp-feature preservation, run time and memory separate. Physical tolerances and object size remain to be supplied by the user; do not invent a product accuracy promise.

The four current Middlebury scenes are development diagnostics. They have already informed engineering choices and must not subsequently be relabeled unseen acceptance scenes. Preserve their baseline artifacts. Synthetic tests establish conventions and controlled behavior, not real photogrammetry quality. New measured turntable scans and a held-out multi-view reference are needed before production acceptance.

## Scoped implementation tasks

### Foreground follow-up after the tree experiment

Fixed-camera separation is now implemented as a research diagnostic: the
same frozen poses and screens yield 666 baseline-region tracks versus 707
freshly rematched foreground tracks, with 160 versus 173 tracks spanning at
least three views. Independent geometric/integrity verification passes;
foreground image support still loses one view. See
[the protocol and limitations](FIXED-CAMERA-OBJECT.md). This does not measure
physical surface quality or establish a production backend winner.

The small [ETH3D pipes reference acquisition](RIGID-REFERENCE-PLAN.md), safe
extraction, coordinate validation, frozen four-view sparse experiment, and
separate exact laser proximity scoring are complete. The 258 image-derived
points have median/p90 proximity **12.8/90.3 mm**, with a **7.45 m** worst
outlier despite subpixel reprojection error. See [scoring](PIPES-SCORE.md)
and [protocol](PIPES-EVALUATION-PROTOCOL.md). No laser geometry or supplied
SfM seeds enter reconstruction. Supplied camera poses remain an oracle input;
proximity is not official accuracy or completeness. Noncommercial dataset
terms require separate use review.

Next scoped quality tasks:

1. Freeze an image-only ambiguity or third-view verification policy before
   a new run; preserve the current outputs and record every rejection.
   The inspected outlier crops suggest repeated-hardware mismatches, not
   simply insufficient parallax. Do not tune against individual laser errors.
2. Score retained geometry and lost support separately. The existing 52-point
   three-view subset has a better extreme tail but a worse median than all
   258 points; requiring three views alone is not a demonstrated improvement.
3. Add visibility-aware dense evaluation and a separate acceptance scene.
   Pipes has now informed development. One-way sparse proximity cannot
   establish complete surfaces or justify meshing unsupported areas.

The [foreground ablation](COLMAP-FOREGROUND.md) is rejected as an overall
improvement: 5/10 rather than 9/10 cameras and 23/82 rather than 8/82
missing-inclusive bad foreground pairs. Its lower residuals on surviving
predictions are not a surface-quality result. Only two frozen three-view
cycles are wholly inside the manual envelopes, so this dataset cannot yet
provide a strong object-specific multi-view validation population.

1. **Camera/support separation:** retain the full-frame camera baseline as
   an explicitly unverified diagnostic pose source; triangulate only
   object-supported tracks with camera parameters fixed. Keep background
   tracks out of dense object seeds. Compare fixed foreground observations
   and coverage without changing the frozen validation matches. This tests
   architecture, not physical camera accuracy.
2. **Rigid multi-view reference:** acquire a small, licensed reference with
   at least three views and trustworthy calibration/geometry; freeze inputs,
   thresholds, and a held-out scoring population before experiments. Keep
   new local downloads small, reserve 10 GiB, and stage large references on
   the VPS when available. Foliage and post-hoc envelopes are development
   stress tests, not substitutes for rigid measured targets.
3. **Calibrated turntable acceptance:** use the existing board-pose path on
   an actual rotating-board scan, record calibration and scale evidence,
   and evaluate object geometry against dimensions not used by fitting.
   Only advance dense fusion/meshing once camera coverage and object support
   pass a declared gate. Do not infer exact motor angles or weaken filters.

| ID | Deliverable | Acceptance and dependencies |
| --- | --- | --- |
| Q02 | Full-image, fixed interior and shared-support stereo diagnostics; categorical error maps | Hand-counted fixtures; distinguish missing from wrong; label shared-support selection bias; retain all-image score; no inferred occlusion/confidence labels without data |
| Q03 | Predeclared baseline/fast/quality SGBM experiments and explicit two-direction consistency | Default predictions unchanged; correct signed right disparity and zero handling; record parameters and rejection counts; same inputs; no GT-informed matching |
| Q04 | Per-stage timing and repeated profile runner | Warm-up separated, repeated observations, binary/input hashes, environment and thread count, reproducible commands; resource estimates clearly separate from measured RSS |
| Q05 | Calibrated object-frame rectification and geometric search bounds | Analytic horizontal/vertical/rotated camera tests; units, distortion, disparity sign and parallax; reject degeneracy and unsupported operations explicitly |
| Q06 | Cross-view consistency and conservative fusion experiment | Known 3+ view geometry, injected noise/outliers, masks and depth discontinuities; report accepted/rejected observations; no fabricated photo-derived quality claims |
| Q07 | Actual image-derived multi-view depth experiment | Connect rectification, image matching, masks and object-frame fusion; compare against known rendered surfaces and subsequently measured real geometry; quantify missing coverage as well as error |
| Q08 | Dedicated multi-view backend comparison | Selected MVE remains a CPU candidate; require auditable files, valid sparse seeds and actual mask enforcement. Compare the same cameras/images and metric surface outputs; no automatic selection by repository badge |
| Q09 | Measured capture and geometry refinement | Fixed capture settings; real calibration and board poses; object-only reprojection residuals; refine only with controlled gauge/scale and verify against independent dimensions |
| Q10 | Surface reconstruction and targeted optimization | Only after reliable depths: normals and edge-aware fusion/meshing; profile before caching/parallel/GPU changes; quality-equivalent benchmark and cancellation/resource budgets |

Initial parallel assignments: a Sol diagnostics agent owns Q02; a Sol stereo agent owns Q03–Q04; a Sol geometry agent owns Q05–Q06. The supervisor owns integration, independent verification and deciding whether the measurements justify Q07–Q10. A failed experiment is recorded, not silently promoted to a production capability.

## First diagnostic result

Q02 is implemented and independently rerun. See [QUALITY-DIAGNOSTICS.md](QUALITY-DIAGNOSTICS.md). Between 62% and 98% of SGBM's missing predictions on these four scenes lie in its left search-support strip. This is an algorithmic support limitation, not evidence that those pixels have been correctly reconstructed. On the fixed interior ROI, Venus bad-2/missing is 2.70% for SGBM versus 2.72% for ELAS, whereas Piano is 19.19% versus 13.22%. The backend-quality gap is scene-dependent; aggregate full-image scores alone were concealing this distinction.

The supervisor artifact is `.local-tools/quality-diagnostics-supervised/diagnostics.json`; categorical maps and the original full-image denominators are retained. No matcher parameters were changed to obtain these diagnostic results.

## Implemented experiments and decisions

Q03–Q04 are implemented: four fixed stereo profiles, explicit signed two-direction consistency, stage timing, process peak RSS, controlled serial OpenCV execution and a repeated, hash-checked runner. On four development scenes the `fast` profile reduces median matching time by about 24–30% and modestly reduces missing-inclusive error. The stricter `quality` profile loses too much coverage to select as a default. Full-direction HH helps Piano but costs time/memory and worsens the other scenes. Preserve the original default baseline and expose these only as explicit test profiles. See [STEREO-PROFILES.md](STEREO-PROFILES.md).

Q05–Q06 are implemented as a separate test library: distortion-aware rectification with signed horizontal/vertical bounds; object-frame triangulation and geometric rejection; visibility-aware, mask-checked track fusion with centroid revalidation. Q07 has an image-derived synthetic experiment, not a general production multi-view pipeline. Two image-pair estimates from three rendered views retain 7,424/7,802 eligible samples after consistency checking; p95 depth error is 0.818 mm. Pair-only p95 errors before rejection are 0.975 and 0.931 mm, but on the same retained subset they are 0.704 and 0.931 mm. Averaging does not beat the better pair on this shared subset; part of the aggregate gain comes from rejection. These idealized planar surfaces do not establish photographed accuracy.

The image fixture still applies masks after SGBM and cannot guarantee that background pixels do not affect path costs. It does not yet connect its pair estimates to the general `fuse_tracks` API. See [DEPTH-GEOMETRY.md](DEPTH-GEOMETRY.md). Those integration/mask gates, Q08 dedicated-backend comparison, Q09 measured captures/refinement and Q10 mesh/performance integration remain open; successful synthetic tests do not close them.

Q08 now has an actual selected-MVE dense execution and independent supervisor replay, documented in [MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md). It reconstructs a 400×300 reference depth map from the separate rendered sparse fixture, using estimated marker poses and object feature seeds. Known-plane radial scoring gives 94.20% object coverage and 3.10 mm matched MAE, with many valid background depths. This is useful integration evidence but not a head-to-head comparison against the Q07 fixture, which has different cameras, surfaces, image scale and pose inputs. Q08 remains open for comparable masked outputs and measured data.

## Quality-first follow-up decisions

The mask-aware MVE variant passes background-mutation isolation tests but fails completeness and boundary acceptance at both tested scales; do not promote it. Higher-resolution processing reduces matched surface error, while the known-pose oracle shows a substantial remaining pose/seed gap. Joint board-anchored camera/track optimization reduces L1 finite-rectangle MAE from 2.130 to 1.580 mm and missing-inclusive bad-2 from 51.74% to 27.23%, with slightly higher coverage. This candidate passes the provisional surface gate but fails its separately frozen marker-RMS safeguard. Keep the rejected result and original settings; do not relax the safeguard after seeing the score. See [QUALITY-GATES.md](QUALITY-GATES.md).

Next bounded quality tasks, in order:

1. **Q09a: measured observation uncertainty and refinement acceptance.** Estimate marker/feature localization uncertainty on independently rendered and real calibration captures; define uncertainty-weighted residuals and board/scale acceptance before running a new candidate. Compare fixed baseline and candidate on held-out views and measured dimensions. Do not choose weights using object ground truth.
2. **Q08a: preserve object boundaries under masking.** The current complete-patch mask discards boundary support. Evaluate normalized partial-support matching or another mask-native backend with a declared minimum support, strict outside-mask mutation invariance, unchanged truth population, and explicit silhouette/depth-step acceptance. Merely dilating the mask into stationary background is not an acceptable fix.
3. **Q07a: multi-view quality acceptance before meshing.** Reconstruct more than one reference view, test depth consistency in the object frame and fuse only supported observations. Report raw and fused accuracy/completeness separately; test nonplanar surfaces, thin features and textureless/specular failures, then a measured turntable object. No inpainting or mesh smoothing may count as observed coverage.

These tasks remain open; the current results are controlled development evidence, not a complete reconstruction engine or a real-object accuracy claim.

### Current bounded implementation batch

Three supervised Sol tasks now address the failed real-tree experiments without
changing the shipped reconstruction capability:

| Task | Scope | Acceptance |
| --- | --- | --- |
| Track diagnosis | Preserve keypoint identities, inspect cycle support and triangulation consistency in the same ten photos; no camera fitting | Known-good and false-chain fixtures; report all rejected populations and pre-filter residuals; filtered residuals are not independent accuracy |
| Image-derived fusion | Feed independently matched depth maps from distinct camera centers into the existing conservative fusion function | Known nonplanar/depth-step geometry, explicit left/right checks, outside-mask mutation invariance, raw and fused missing-inclusive/boundary scores; no ground-truth candidate inputs |
| Full-pipeline oracle preflight | Bounded local executable/dataset/resource checks for a separate COLMAP + OpenMVS run | Unit-tested unavailable/failure states, reproducible manifest; no download, install, upload or claim of a completed oracle run |

The older pairwise rendered-image experiment remains unchanged as a baseline.
This batch uses existing photos and builds, targets less than 300 MiB of new
artifacts, and preserves the 10 GiB free-space floor. Surface reconstruction
and production integration remain gated on actual depth quality.

Batch outcome: [track diagnostics](TREE-TRACKS.md) find no accepted ORB or
SIFT multiview components at the frozen geometry gate. [Image-derived fusion](DEPTH-IMAGE-FUSION.md)
now exercises both filtering and equal-weight two-candidate fusion, but neither
improves missing-inclusive whole-object or boundary error on the fixture.
The [full-pipeline preflight](PIPELINE-ORACLE-READINESS.md) verifies the data but
finds comparator executables unavailable. No product reconstruction milestone
is advanced. Next: robust image-derived epipolar checks against the supplied
poses, followed by a separately bounded camera-estimated oracle lane; do not
weaken track gates or promote filtered-only error reductions.

### Camera-versus-correspondence isolation

The follow-up freezes SIFT matching before any supplied-camera geometry filter,
reserves every fifth descriptor-ordered correspondence, fits a fundamental
matrix and homography using training correspondences only, and evaluates both
the supplied-camera and fitted epipolar models on the same untouched reserved
population. All 45 pairs remain in the report, including unavailable cases;
pair 1/4 is preselected for detail, not selected after scoring. Homography
support is degeneracy context, not evidence of correct 3D geometry.

A separate analytic camera-import audit checks image identity, intrinsics,
quaternion/translation conventions and resizing against pinned source records.
An independently coded evaluator recomputes epipolar matrices and scores from
the saved correspondences. No supplied pose is overwritten, no held-out pixel
is used for refitting, and no result is called physical accuracy.

Outcome: the independent import audit passes, while image-fitted epipolar
models beat the supplied camera model on held-out matches in all 30 eligible
pairs. A calibrated two-view fit also improves image residuals, but fewer than
half its training inliers triangulate in front of both recovered cameras.
See [TREE-EPIPOLAR.md](TREE-EPIPOLAR.md). Do not replace the source poses with
that fit. The next scoped experiment is a camera-estimated sparse lane with
multi-view cheirality, parallax and independent observation checks before dense
reconstruction; the supplied tree poses must not be treated as geometry truth.

### Camera-estimated sparse research lane

**Audit correction:** the first implementation of the proposal below was
invalid as a held-out benchmark: six-column affine SIFT keypoints were treated
as position/scale, causing orientation leakage, and the focal heuristic was
incorrectly marked as a prior. The lane is quarantined. See
[the upstream usage audit](COLMAP-USAGE-AUDIT.md). Run conventional full-feature
COLMAP baselines before interpreting this custom experiment; original JPEGs
with EXIF are available locally and should not be excluded for disk reasons.

Use an isolated, pinned PyCOLMAP CPU wheel to test the same ten reduced photos
with no supplied extrinsics and no supplied focal values. This is an external
research oracle, not a shipped backend or a completed photo-to-mesh comparison.
Initialize one shared SIMPLE_RADIAL camera with heuristic focal length
`1.2 × 768 = 921.6 px`, image-center principal point, and zero radial distortion;
allow focal/radial refinement. Freeze settings before the first reconstruction.

Reserve deterministic feature groups per image before matching or mapping;
duplicate orientation descriptors at the same feature location/scale must not
cross the train/validation split. Preserve original IDs and the database row
mapping for an independent checker. These are withheld observations from the
same photographs, not an independent capture or physical ground truth.

The run must retain failures, registered-camera counts, tracks, positive-depth
and parallax diagnostics, and training versus held-out errors separately.
A reusable subprocess guard caps time, logs and output space and preserves
the 10 GiB disk reserve. Downloads are limited to 250 MiB, the isolated
environment to 750 MiB, and run output to 1 GiB. No OpenMVS, global install,
remote upload, or automatic product integration is included.

## Interpretation rules

### Follow-up after the COLMAP usage audit

The clean observation-holdout run now passes its frozen epipolar screen, but
its three-view prediction p90 is 16.261 px. The next diagnostic preserves all
181 closed cycles and the original 106 usable lexical-first-pair predictions.
It records per-edge residuals, ray angles, depths, first-pair reprojection and
all three triangulate-two/predict-one orientations. An independent NumPy
calculation checks the existing closest-ray implementation. A local contact
sheet displays all 21 original >4 px cases plus deterministic low-error
controls; visual inspection does not turn suspected mismatches into truth.

One exploratory comparison is fixed before inspecting its result: select the
two cameras with the longest centre-to-centre baseline, with lexical tie
breaking. This selection uses camera geometry only, not pixel residuals or
the future third-view pixel. Report missing-inclusive bad-4 on the *same*
106 originally usable cycles as well as all-cycle availability. Because this
policy may predict a different third image, record target changes and do not
present it as an accuracy improvement on identical target observations.
This already-seen scene is development data, not a fresh acceptance set.

1. Replay mapping only from an immutable copy of the original-photo database.
   Keep feature extraction, matches, camera initialization and mapper settings
   unchanged. Record registration/refinement snapshots, point counts, camera
   parameters and filtering diagnostics. This isolates where points disappear;
   it is not a new quality benchmark.
2. Independently verify surviving model snapshots and distinguish filtering,
   numerical failure and unobserved possibilities. Default logs alone did not
   identify the point-removal step. Do not weaken reprojection/parallax gates
   to obtain a larger cloud.
3. Repair holdout construction with deterministic spatial connected components
   over feature centres within 0.25 pixels, including six-column affine rows.
   Transitive and boundary-neighbour orientation clones must remain together.
   Test separately before reconnecting the quarantined holdout runner.
4. Only after a predeclared, successful exclusion audit, evaluate camera
   predictions on reserved observations. The existing full-feature COLMAP
   models are training baselines, not clean validation models or metric truth.

The mapper replay is bounded to 300 seconds and 1 GiB, with the existing
10 GiB free-space floor. No new photographs, dependencies, or remote jobs
are part of this follow-up. Any camera-model or prior ablation must explicitly
account for the already-verified pair geometry in a reused database.

- Full-image bad-2 includes missing predictions. A lower matched-pixel MAE can come from rejecting difficult pixels. Report both and compare on fixed regions as well as explicitly biased common prediction support.
- ELAS's MIDDLEBURY preset interpolates aggressively. Full coverage is not evidence that all predictions have independent image support. The adapter currently cannot label the exact interpolated pixels.
- Confidence must be available at reconstruction time, never derived from ground truth. Ground-truth error rankings are diagnostic only, not confidence estimates.
- For synthetic fusion tests, oracle depths test geometry/fusion only. A separate image-derived test is required to test matching.
- No measured runtime comparison between native ARM and Rosetta is treated as a pure algorithm comparison. Report the execution architecture.
- Preserve geometry and masks through resizing/rectification; evaluation-only masks do not prevent stationary background or marker-board contamination during matching.
- A finished mesh can conceal missing observations. Preserve raw accepted points and coverage evidence, and do not smooth away LEGO edges to improve appearance.

## Storage and build boundary

### Bounded real-image follow-up

The next comparison separates three questions: (1) fix the COLMAP/OpenCV half-pixel import convention and rerun the frozen tree baseline; (2) test conflict-free multi-view tracks plus gauge-anchored camera/point refinement against that corrected baseline; (3) rectify the same fixed tree pair for SGBM versus the existing external ELAS oracle. Feature-derived checks are diagnostic, not measured geometry ground truth. Existing failed outputs remain intact. Full-pipeline estimates against COLMAP+OpenMVS, Metashape and RealityScan must be labeled unmeasured until an identical-input end-to-end experiment is performed; stereo errors do not establish finished-mesh rankings.

Results: the corrected baseline remains empty, the refinement candidate is rejected, and fixed-pair stereo produces nonempty disparity but has no dense truth. See [TREE-REFINE.md](TREE-REFINE.md) and [TREE-PAIR-ORACLE.md](TREE-PAIR-ORACLE.md). Next bounded tasks, in order:

1. Validate tracks across three views with reprojection/epipolar and cycle checks; diagnose supplied camera consistency without fitting held-out pixels. Preserve the current failed track population as a regression case.
2. Connect image-derived pairwise disparity to conservative multi-view fusion. Test on a known nonplanar synthetic surface first, then this real scene; require masks, left/right and cross-view consistency, raw support counts and unfilled coverage. No geometry accuracy claim from agreement alone.
3. Run a separate COLMAP + OpenMVS research oracle on the same small capture, after pinning its test-only dependencies and verifying storage/build limits. This is a new experiment, not a completed comparison or shipping choice. Obtain independent measured object geometry before ranking surface accuracy.

The user approved small Mac tests and identified a CPU-only VPS with larger storage plus Kaggle guidance in `../kaggle_usage.md`. The current local batch targets an eight-to-twelve-image connected neighborhood from the CC BY 4.0 tree dataset, at most 250 MB of source photographs and 1 GiB total new data/artifacts, retaining at least 10 GiB free. No whole-dataset clone or large archive download is part of this batch. VPS and Kaggle packaging must exclude credentials, unrelated projects and existing build trees; no remote run is claimed merely because a package was prepared.

Source inspection found that the tree's COLMAP export has blank image observations and empty point tracks. Its zero point-error fields are placeholders, not measured residuals. Camera poses and point coordinates alone therefore cannot serve as an observed MVE seed bundle. The test must derive real image correspondences, triangulate with the supplied estimated poses, and validate projection conventions. The source coordinate scale is unverified. Results may report image residuals, valid depth coverage and cross-view consistency, but not measured millimetre accuracy or independent ground-truth agreement.

Keep scratch in `.local-tools/tmp`, generated experiments in ignored build directories and datasets in `.local-tools/test-data` on this Mac. Check free space before substantial batches and preserve at least 10 GiB. On the VPS, use the user's `/mnt/storage` for large data and `/mnt/volume1` for fast work once that environment is explicitly available. No global package installs or large new downloads are needed for the initial experiments.

Experiments remain separate from the shipped core until their contracts and quality gates pass. GPL development oracles remain external executables. No new AGPL oracle or unknown-license code is introduced by this workstream.
