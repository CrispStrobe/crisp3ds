# Implementation status

## Public repository and project license

On 2026-09-27, at the owner's request, the repository became public and original
project code was licensed under AGPL-3.0-only. Third-party licenses are unchanged;
this does not automatically clear App Store distribution. The new
[Windows CI retry](https://github.com/CrispStrobe/crisp3ds/actions/runs/36275507360)
started successfully after publication, superseding the billing-startup block
recorded below. Its compilation, smoke and unit checks passed. Together with
earlier macOS arm64/Linux x64 jobs, the selected MVE build matrix now passes;
real-photo reconstruction on Windows/Linux remains unverified.

## Active quality-first implementation

### Current continuation: exact crop recovers partial dense geometry; poses remain suspect

The [frozen common-crop trial](MUSTARD-CROP-PLAN.md) preserves all original
object pixels and the same 48 estimated camera poses, 939 3D points and 4,734
tracked observations. Its shared 222×302 crop translates pixel coordinates
and principal points only; lossless PNG output, serialized projections/rays,
and unchanged 3D-point bytes are checked before native reconstruction.

The live M1 CPU run improves dense support from **1 to 24 depth maps** and
from **0 to 24,657 fused output points** without changing matching, poses or
native thresholds. All 24 available maps are 222×302, their cameras agree with
the cropped model, and none of their 460,545 positive depth pixels lies outside
its corresponding mask. However, **24 of 48 views still lack depth maps**.
The unchanged full-coverage audit correctly stops before meshing. This is
partial dense progress, not a complete pipeline or surface-quality pass.
See [crop results and evidence](MUSTARD-CROP-RESULTS.md).

The independent [pose trajectory diagnostic](MUSTARD-POSE-DIAGNOSTIC.md)
finds opposing-label views near-coincident in both position and orientation:
000/180 differ by 2.67°, 060/240 by 1.46°, and 090/270 by 2.20°. Some adjacent
capture views instead jump by roughly 85–92°. These observations indicate
possible pose aliasing; neither registration count nor dense-point count
establishes correct object geometry. Improving and independently validating
alignment is now more important than adding meshing, texture or neural stages
on these same cameras. No reference mesh or held-out image entered the trial.

Local regression: **540 Python tests run, 8 skipped, 532 passed** in 28.66
seconds; **11/11 native CTests passed**. The crop test includes a portable
synthetic native fixture and a local real-model serialization check. The short
native CTest run overlapped the dense trial, so its recorded timing is not a
controlled performance comparison. Both CI workflows passed at prior commit
`2737d04`; this batch requires its own CI run. About 23 GiB internal and 18 GiB
external storage remain free, above both 10 GiB floors.

### Previous continuation: sparse integrity repaired; dense trial fails neighbor support

The one frozen sparse-track repair completed on the M1 in 1.05 seconds
(sampled worker RSS 174,928 KiB). It preserved all 48 cameras and 939 points,
removed 81 redundant same-image observations, and left the 897 originally
clean points' XYZ and tracks unchanged. The saved model has 4,734 observations
and zero repeated-image tracks. Root independently reloaded the binaries and
recomputed residuals for all 4,815 original observations: mean changes from
1.083929 to 1.084016 px, p95 from 2.758900 to 2.805505 px. These slightly worse
common-population residuals pass the frozen 10% safeguard; this is **structural
repair, not a demonstrated accuracy improvement**.

The [root review](../tests/evidence/mustard-sparse-repair-root-qa.json) accepts
only eligibility for one bounded, full-resolution masked OpenMVS dense/rough
mesh trial. The [pre-score protocol](MUSTARD-DENSE-BENCHMARK-PLAN.md) freezes
whole-mesh evaluation against the independent Google scanner shape oracle,
with separate alignment and scoring samples. The reference never enters
reconstruction. Native limits are 600 seconds total, 4 GiB child RSS, 3 GiB
output, two threads and 10 GiB free on both disks. No refinement or texture
stage is included. Neither attempted continuation produced an eligible mesh;
surface quality is not yet measured.

The first continuation stopped before OpenMVS: pinned COLMAP retained the
zero-distortion `SIMPLE_RADIAL` model while exporting 1279×1023 images for its
1280×1024 camera. The failed run is preserved; it is not a dense result. The
correction for this exact-zero-distortion profile uses original, byte-identical
1280×1024 photographs and an equivalent `PINHOLE` camera representation instead
of resampling already distortion-free images. Nonzero distortion cannot use
that identity path.

The corrected fresh continuation passes the 48-image identity-copy, camera,
mask and import checks. However, OpenMVS selects a neighbor for only one view
and produces only its depth map: 20,460 positive depth pixels, all inside its
mask. Its selected neighbor has no depth map, so fusion returns **zero points**.
The wrapper correctly rejects this nominally exit-zero native stage. Neither
run reaches meshing, and no reference-mesh score is reported. See the
[dense failure evidence and next experiments](MUSTARD-DENSE-RESULTS.md).

Read-only pose inspection also shows suspicious near-overlap between some
views separated by 180 degrees in the dataset and a large camera outlier.
Pinned native source also reveals a coverage-weighted neighbor score and an
absolute score floor of 2.0 in `InitViews`, separate from its area filter. All
48 views passed initial sparse-neighbor selection; most then lost every
eligible dense neighbor. Tiny foreground coverage can therefore contribute
even with enough sparse overlap. The relative contributions of coverage and
pose error remain unmeasured. **48 registered images is not a validated camera solution.**
No thresholds were relaxed and no new reconstruction was run after this
failed dense trial.

Final local regression: **531 Python tests run, 8 skipped, 523 passed** in
28.98 seconds; **11/11 native CTests passed**. These include a tiny native
zero-distortion handoff regression and a nonzero-distortion rejection case.

### Previous continuation: all 48 cameras recovered; surface accuracy still untested

The M1 CPU correction trial completed in 27.78 seconds but only three of five
masks passed its checks. Its parent remains failed/partial. Root accepted those
three masks separately; a frozen, versioned tighter-ROI recipe then produced
the remaining two in 8.34 seconds (sampled peak RSS 1,430,416 KiB). Root reviewed
both broad-context sheets and the new complete 48-view sheet. Acceptance is
for **coarse camera-alignment support**, not exact silhouettes or mesh accuracy.

The new candidate preserves 43 original masks and selects the five reviewed
repairs with per-view source, prompt, producer and QA hashes. Its
[independent full-set review](../tests/datasets/sam21_mustard_candidate48_v2_review.json)
binds every mask and the contact sheet. A fresh train-only stage contains
48 exact photographs and masks (48,794,098 copied bytes) on the external SSD;
no held-out photographs, supplied poses or reference geometry enter it.
The [paired comparison](MUSTARD-SFM-COMPARISON.md) ran with identical frozen CPU
COLMAP settings, raw then masked, and no dense reconstruction. **Both failed.**
Raw images produced no model (45.915 seconds summed worker stages); masked
images produced only a rejected two-camera, 69-point candidate (22.995 seconds).
Neither is a usable 48-view reconstruction. The masked candidate's focal length
fell from the heuristic 1536 px to 336.16 px and its radial parameter reached
5.653, beyond the mapper's configured extra-parameter limit of 1.0. This is
evidence of a self-calibration failure to investigate, not proof that fixing
intrinsics will recover accurate geometry.

The reusable `--sfm-intrinsics-policy fixed-initial` ablation then recovered
**37/48 cameras (77.1%), 717 points and 3,437 track observations** in 27.136
seconds summed worker time, with the same masks, matching, seed and acceptance
thresholds. Its actual camera parameters remain `[1536,640,512,0]`; these are
initial estimates, not supplied calibration. Eleven consecutive late views
(276–348 degrees in the selected split) remain unregistered. The producer's
sparse gate passed, but deeper inspection found **31/717 tracks with multiple
observations from the same image**. Diagnostics must disclose that integrity
warning and distinguish observation counts from distinct-view support.
No dense reconstruction, reference-mesh accuracy or KIRI-quality claim follows
from this result. The baseline failures remain preserved.

A read-only boundary audit found only 20, 22 and 11 unique existing-point
correspondences for the first three missing views, below the unchanged 30-inlier
pose gate; later missing views had no verified edges to the registered model.
Sequential matching also had not attempted wrap-around pairs. A separate final
ablation changed **only matching to exhaustive**, retaining fixed initial
intrinsics and all other settings. It recovered **48/48 cameras, 939 sparse
points and 4,815 observations**, with no missing views and camera parameters
still `[1536,640,512,0]`. Summed worker time was 38.558 seconds. Regression tests
overlapped these runs, so these timings are not a controlled speed benchmark.
The final model has 753/939 tracks supported by at least three distinct views;
median distinct-view support is four. Recomputed observation reprojection
median is 0.868 px and p95 2.759 px (4,815 finite observations). **42/939 tracks
contain repeated-image observations**, so track-integrity warnings remain.
Full registration is an alignment milestone, not 100% geometric accuracy:
the next gate is sparse-track integrity followed by dense/mesh evaluation
against independent reference geometry. No additional reconstruction trials
or dense stages were run after this ablation.

Local regression: **514 Python tests run, 8 skipped, 506 passed** in 29.67
seconds; **11/11 native CTests passed**.
Both CI workflows passed at the intrinsics-policy commit `7416af7` and the
diagnostics/evidence commit `8f2b0df`.
Large data and AI weights remain on the SSD, with the 10 GiB free-space floors
unchanged. Sparse results are not yet a surface-quality measurement.

### Previous continuation: five-view correction ready, RAM-gated

The separately frozen [board-negative experiment](M1-SAM.md) adds the
original-image negative point `(620,330)` only to the five rejected training
views. Root and the independent reviewer confirmed its checkerboard placement
in all five original photos. The new runner preserves the prior 48-mask
artifact and records four-point provenance for any newly generated masks.

The [candidate composer](../scripts/classical_backend/mustard_candidate48.py)
requires a successful bounded run and separate five-view visual acceptance,
then copies 43 unchanged masks plus five corrected masks into a new,
unreviewed inventory. Staging requires another independent full-48 review.
Tests cover incorrect lineage, changed source files, partial results and QA
rejection; 17 focused tests and 81 classical-backend tests passed. Existing
three-point runtimes and the paired COLMAP settings are unchanged.

**No correction inference, composition, staging or SfM ran this turn.**
Actual input preflight passed, but the permitted 60-second observation found
only 3,664,560–3,916,176 KiB of Mac free+inactive memory, below the frozen
4,194,304 KiB launch reserve. The proposed output remains absent. Resume the
single bounded trial only after memory headroom recovers; do not lower the
runtime or disk safety floors. Both CI workflows passed at `338c1fd`; these
new changes require their own post-push CI result.
Final full local regression: **484 tests run, 8 skipped, 476 passed** in
27.95 seconds, with small test scratch inside the repository and large
datasets/weights remaining on the external SSD.

### Current continuation: full-training M1 mask path

The photo-only stage copied and verified exactly 48 mustard TRAIN photographs
to the external SSD (48,696,280 bytes including metadata). No held-out photos,
reference geometry, masks or camera poses were transferred by this stage.
Its report SHA-256 is
`d6e72143fc02ee3d18dd8961ae508417bf122cbdbc24ebc4465e207a30997861`.
The reviewed Mac continuation reuses the sealed 16 masks and can infer the
remaining 32 in bounded four-frame CPU batches. Fake-worker tests cover
interruption, resumption to the full inventory and tampering; full visual QA
is still required before the paired sparse comparison.

Both CI workflows passed at `ff64c6b`, fixing a test that incorrectly relied
on a real `/Volumes/backups` mount on CI. The first local continuation suite
ran 468 tests with eight errors and eight skips: five existing storage-reserve
checks and three external-volume temporary-executable timeouts. This is not
a passing regression result. After verified relocation of the completed
806 MB bunny run, all five reserve-sensitive tests passed. The stereo matrix
timeouts were isolated to rewriting a copied executable as a script at the
same path on macOS; fresh per-script paths fix the test fixture without
changing production code or timeouts. All four focused matrix tests passed.

The first M1 continuation stopped at the unchanged 2.5 GiB available-memory
floor after **32.93 seconds**: 16 imported plus seven newly inferred masks,
23 sealed frames in total. No complete inventory was published. Root viewed
the RGB and cleaned-mask sheets: the new masks retain bottle, cap and labels,
with some light boundary fringe; this is partial coarse-support evidence,
not full-set acceptance or a reconstruction-quality result. A second
invocation passed a stricter 4 GiB launch-headroom check, reused all 23 sealed
frames, and completed the remaining 25 in **91.93 seconds**. Peak sampled
worker RSS was **1,428,528 KiB**. All 48 frames and receipts validated, with
the original runtime limits unchanged.

**Full visual QA rejected this mask set.** Views 318, 330, 336, 342 and 348
retain checkerboard background adjoining the bottle; largest-component
cleanup does not separate that contact. Root inspected the complete RGB,
raw and cleaned sheets and recorded the [hash-bound rejection](../tests/datasets/sam21_mustard_m1_full_review.json).
No SfM or dense stage consumed the masks. Next is a separately frozen
image-only negative-prompt candidate for those five views; do not overwrite
this result, silently discard failed views, or claim improved mesh accuracy.

Final local regression with small internal test scratch: **473 tests run,
8 skipped, 465 passed** in 30.10 seconds. Large inputs, masks, environments
and weights remain on the external SSD. Both CI workflows passed at
`246c3e8`; the subsequent documentation/test-fix commit still requires its
own final CI result.

### Latest continuation: M1 CPU inference demonstrated

Following the user's compute-policy clarification, inference moved off the VPS.
A fresh external-only SAM environment (`sam21-m1-002`) completed all 16 setup
stages with 808,199,810 bytes of output. The pinned source, checkpoint, wheels,
training photos and sealed mask references were verified before use. Setup001
is preserved as a pre-install failure caused by inconsistent inventory path
sorting; all 78 transferred vendor files matched the VPS byte-for-byte, and a
regression test now covers that ordering rule. No Kaggle job was submitted.

The bounded **M1 CPU** three-view trial completed in **15.612 seconds**, peak
sampled worker RSS **1,222,672 KiB**. Root independently decoded and compared
all six raw/clean candidate/reference mask pairs: **pixel equality and IoU 1.0
for every pair**. PNG file hashes differ, so this is not byte-identical output.
Root also inspected the RGB/raw/clean sheet and accepted only three-view M1
parity and coarse bottle support. The [QA record](../tests/datasets/sam21_mustard_m1_parity_review.json)
binds the setup, runtime and review artifacts. See [M1 SAM](M1-SAM.md).

This demonstrates a real M1 inference path, **not improved mesh accuracy**.
The remaining 32 training masks have not been generated, and full48 QA and
the paired SfM comparison remain outstanding. All environment/weight/output
files are on the external SSD; internal space stayed above the 10 GiB floor.
Final local regression: **465 tests run, 8 skipped, 457 passed**. Both CI
workflows passed at `788b812` (including the SQLite-handle Windows fix); the
M1 implementation still needs its own post-push CI result.

### Latest continuation: resumable masks and sparse comparison prerequisites

The prompt hash failures on Windows at `1f3a077` were caused by checkout CRLF
conversion: independently converting the two LF fixtures reproduced both exact
CI failure hashes. Commit `3d91758` pins their checkout line endings without
changing the byte-hash checks. Commit `4e0b22f` also makes fixture and
`.gitattributes` changes trigger the quality-contract matrix.

A separate resumable SAM runner now uses four-frame batches, strict import of
the 16 sealed results, per-frame publication receipts, and cross-invocation
hash checks. Six focused tests cover interruption and tampering. Remote
read-only preflight verified the frozen 48-photo inventory and exactly 16
sealed prior masks; the interrupted third batch's six unsealed pairs are
excluded. No full-mask-set acceptance follows from this check.

The train-only staging gate now requires all 48 photo hashes, a complete mask
inventory and a separately hash-bound full visual-review decision. The
classical runner monitors internal reserve as well as external output reserve
when output devices differ. A read-only sparse diagnostic adds match-graph
components including isolated images, registered names, track distributions,
and finite/invalid per-observation reprojection counts. Native analytic
PyCOLMAP tests exercise camera projection and binary-model/SQLite integration.
See the [comparison plan](MUSTARD-SFM-COMPARISON.md). No real staging, paired
SfM comparison, or new geometry score is claimed.

VPS available RAM fluctuated from about 1.4 GiB to 3.3 GiB and back below the
unchanged 2.5 GiB inference gate. Mac internal free space fell to about
10.47 GiB. The first full regression encountered seven existing disk-headroom
guard failures (452 tests, 8 skips); these guards require approximately
10.5 GiB. Test temporary files were moved to the external drive, and the stereo
matrix tests now honor that explicit `TMPDIR` override. Final rerun: **452 tests,
8 skipped, the remaining 444 passed**; production resource limits are unchanged.

The final immediate VPS launch gate was 1,301,820 KiB, below 2,621,440 KiB;
no continuation inference ran and its proposed output remains uncreated. The
user then clarified that future inference is **M1 or Kaggle only**, with the
VPS restricted to smaller CPU tasks. A bounded M1 CPU environment and a
three-view cross-host comparison are the next implementation task, with all
large files on the external drive. The Linux-specific RAM monitor must not
be reused as though it measured available memory on macOS.

### Latest continuation: prompt correction and masked Brush execution

The storage relocation regressions are fixed: Windows requires a writable flush
handle and normalized symlink-target comparison during rollback. The quality
harness at `9205b26` [passed on Windows, macOS and Ubuntu](https://github.com/CrispStrobe/crisp3ds/actions/runs/36306164359).

Explicit foreground/background points corrected the known SAM target-switching
failure in three reviewed mustard training photos. All three masks retained the
bottle and labels, excluded the board, and needed zero component cleanup. This
is an accepted **three-view coarse-support smoke**, not an exact-silhouette or
full-orbit result. Root reviewed common-point placements on all 48 training
photos before approving a separate bounded candidate. That run stopped at the
third batch's 90-second limit after 219.53 seconds overall: 16 views have sealed
results; six additional raw/clean file pairs lack a completed batch manifest and
are unusable. The full candidate remains **incomplete and not reconstruction-ready**;
there was no retry or resource-gate relaxation. Held-out images, reference
geometry and sensor depth remain excluded from mask generation and review.
Root independently checked the partial manifest/sheet hashes and reviewed all
16 sealed cleaned masks: the bottle, cap and labels remain, with detached
background specks removed where present. This accepts only partial coarse-support
visual evidence, not exact silhouettes or the incomplete full candidate. The
next mask-run improvement is hash-verified resumable smaller batches, retaining
the same prompts and acceptance rules rather than weakening resource limits.
See [the SAM trial](SAM-MASK-TRIAL.md) for the candidate's current status.

The [Brush mask bridge](BRUSH-PREFLIGHT.md) fixes a release-specific filename
contract: v0.3.0 needs `NP3_000.png`, not `NP3_000.mask.png`. All 123 prepared
RGB/model/mask files were independently rehashed. The masked 20-step M1 smoke
completed in 7.099 seconds and exported 3,939 finite splats. No loader telemetry
confirmed mask attachment, and no held-out appearance or mesh-quality score was
measured. This remains execution evidence, **not a quality improvement**. The
completed run monitored external disk space continuously; the launcher now also
monitors internal free space for future runs. No new weights were downloaded.

Local regression: **432 Python tests run, 8 skipped, the remainder passed**.
The [paired mustard SfM plan](MUSTARD-SFM-COMPARISON.md) freezes the next raw-versus-
masked alignment comparison, but requires accepted full48 masks and additional
staging/diagnostic safeguards before execution. No new dense mesh-quality result
was produced in this continuation.

### Latest continuation: neural prerequisites and controlled fusion

Four completed reconstruction runs were checksum-verified and relocated to the
external drive, recovering 5.42 GiB internally. The [storage policy](STORAGE.md)
separates experiment artifacts (`/Volumes/backups/code`) from AI weights
(`/Volumes/backups/ai`), retaining a 10 GiB internal reserve.

The formerly disk-blocked 016/017 cached-depth fusion comparison now completed
in 9.514 seconds. Root independently checked both PLY payloads: filter 2 emitted
40,894 points, versus 120,147 for filter 1. All copied/source inputs and the
binary/result seals remained unchanged. This demonstrates a fusion-dependent
point-count difference, **not improved geometry**. The frozen two-pixel
point-support diagnostic on 6,144 sensor rays found only 13 net added hits
(5,512 to 5,525), with shared-hit mean depth disagreement worsening from
9.060 to 10.913 mm. Root independently recomputed these counts and residuals.
Filter 1 is **not promoted**; filter 2 remains the default. These are not mesh
ray-intersection scores or certified physical accuracy. See the
[fusion investigation](OPENMVS-PATCH-PLAN.md) for diagnostics and limitations.

The [SAM 2.1 tiny smoke](SAM-MASK-TRIAL.md) ran on three frozen mustard training
photos on the CPU VPS. A largest-component cleanup removed detached specks in
that smoke, but the broader frozen test exposed semantic target switching:
many mid-orbit masks select the bright background instead of the bottle.
Root independently reviewed the raw and cleaned sheets and **rejected this
candidate for reconstruction**, including masks passing numerical checks.
The first two batches attempted 32 views: 19 passed numerical cleanup and 13
failed its removal guard. The RAM gate stopped before batch 3. Explicit
foreground/background prompts on representative training-only failure views
are the next scoped experiment; relaxing the component threshold is not a fix.

The [Brush ARM64 smoke](BRUSH-PREFLIGHT.md) passed the sealed input-lineage and
resource gates and executed the fixed 20-step CLI configuration on M1 in
6.482 seconds. Root independently validated the 3,939-splat binary PLY. The CLI
emitted no backend log, so Metal device selection was not independently observed.
No held-out render or mesh quality was measured, and no mask adapter was used:
this is execution-only evidence, not a quality baseline. A splat output is not
a triangle mesh. The [MapAnything Apache lock](MAPANYTHING-PLAN.md)
verified pinned metadata only: no 4.91 GB checkpoint download or inference ran.
These components do not yet establish a new end-to-end quality result.
The pipeline registry now lists ten candidates and three executed families;
the third is Brush's execution-only smoke with an empty quality-evidence list,
not a third scored reconstruction baseline.

Local verification for this continuation: **415 Python tests run, 8 skipped,
the remainder passed**, plus **4 foundation and 11 OpenCV-enabled native tests**.
Cross-platform CI has been extended to cover Brush, model-inventory and storage
contracts; a live M1 smoke is not evidence of native training on other platforms.

### Latest continuation: iPhone RGB capture contract and Apple support probe

The [iPhone 13 mini review](IPHONE13MINI-REUSE.md) confirms the RGB-first plan:
no rear LiDAR dependency; front TrueDepth is a distinct optional sensor path.
ObjectScanner offers reusable capture patterns but uses Apple reconstruction,
and its turntable capture gate needs separation from on-device reconstruction
for a capture/export-only client. No upstream code was copied this turn.

Supervised Sol implementations add a [JPEG capture manifest](RGB-CAPTURE-CONTRACT.md)
and an [optional Apple-only reconstruction launcher](APPLE-OBJECT-CAPTURE.md).
Root independently revalidated the three-real-photo import proof: original bytes
unchanged, no depth required, no metric-scale assertion. Motion is a declaration;
optional camera metadata remains projection-unverified, especially under EXIF
rotation. This is not yet an iOS camera app or a backend pose-import bridge.

The original Swift adapter compiled on the M1 Mac and the final bounded support
probe returned `PhotogrammetrySession.isSupported=true`. No USDZ reconstruction
ran; support availability is not shape-quality evidence. The compile/cache plus
probe artifacts stayed under 100 MiB, preserving the 10 GiB disk reserve.

The [pinned msplat-ios audit](MSPLAT-IOS-AUDIT.md) identifies a real Metal trainer
from precomputed COLMAP cameras, with coordinate-normalization and split-policy
integration work. It is not an RGB-to-mesh engine. The registry now has nine
candidates, still only two executed reconstruction families; no msplat build,
M1/iPhone training or new reconstruction-quality result is claimed.

Final local regression: **382 tests run, 8 skipped, remaining passed**. Registry
validation passes (nine candidates, two executed). Prior `144e0ff` passed both
CI workflows; this continuation still needs its own CI result.

### Latest continuation: fusion isolation ready, execution disk-blocked

Supervised Sol work adds a [source-reviewed OpenMVS fusion comparison](OPENMVS-PATCH-PLAN.md)
and a bounded executor. It copies the same sealed 008 cameras/images/masks/final
depth maps into independent arms; only fusion filter 2 versus 1 changes.
Independent review confirmed relative scene paths and the pinned cached-depth
code path. Tests reject changed maps, config overrides and changed binaries.
The fresh execution preflight stopped before creating either arm: free space
was 12,369,154,048 bytes versus 12,473,860,096 required for the 10 GiB reserve,
two 700 MiB caps and a 256 MiB buffer. No native process, source rebuild, fork
algorithm change or quality comparison ran. No data was deleted to make room.

The separate mustard row-envelope candidate completed once on the VPS for all
48 training photos. Root reviewed both sheets: label bands are preserved, but
checkerboard background leaks into multiple masks (particularly NP3_282).
The [successor QA record](../tests/datasets/ycb_mustard_envelope_review.json)
therefore rejects it for reconstruction. Both failed candidates remain intact;
no held-out/reference inputs were used. Mustard and drill are still not ready
for object-only reconstruction. Explicit board exclusions or validated
segmentation are next; another blind colour-threshold sweep is not planned.

Local regression: **369 tests run, 8 skipped, remaining passed**. Both CI
workflows passed for prior commit `f8b2a42`; this continuation requires its own
CI result. No reconstruction-quality gain is claimed.

### Latest continuation: isolate coverage loss and evaluate refinement

The [paired coverage audit](COVERAGE-LOSS.md) identifies 215 fixed sensor rays
lost by 015 versus 008, none gained; 175 losses fall in the already defined
boundary band. On their 4,951 common hits, depth disagreement worsens from
2.754 to 2.922 mm. The [pre-mesh depth audit](DEPTH-SUPPORT-AUDIT.md) separately
compares 189,982 fixed original-photo probes: native depth support falls from
91.271% to 90.584%. Both warped-mask exclusion and missing allowed depth increase.
These differently defined probes do not identify a unique cause or prove that
meshing alone explains the larger sensor-ray coverage loss.

A [same-camera rough/refined comparison](ROUGH-REFINED-COMPARISON.md) of existing
008 outputs gives a mixed result: refinement reduces ray coverage from 84.08%
to 82.28%, but improves shared-hit mean disagreement from 2.807 to 2.636 mm.
The all-supported-ray within-5-mm rate rises slightly, 70.72% to 71.16%.
Root reproduced the old rough score and independently recomputed paired counts
and residuals. No new native reconstruction or parameter search was run.

All 48 training views of each new object were rendered on the VPS and visually
reviewed without opening held-out images or references. One frozen mustard
[coarse mask candidate](YCB-OBJECT-MASKS.md) generated 48 masks but **failed
visual acceptance**, cutting coloured-label strips despite numerical checks.
The [rejection record](../tests/datasets/ycb_mustard_support_review.json) prevents
treating it as approved preparation. No reconstruction used those masks. Drill
masks remain manual/segmentation work; neither new training package is ready.

Local regression: **354 tests run, 8 skipped, remaining passed**. The Windows
SYCL executable-name check was fixed in `c77a298`; both its cross-platform quality
and foundation workflows pass. This continuation needs its own CI. Large new outputs stayed
on the VPS; local additions were diagnostics and small review sheets.

### Previous continuation: recovered dense geometry and frozen new-object splits

The [recovered-camera dense control](RECOVERED-DENSE.md) completed on M1 CPU:
40,450 dense points and 38,440 rough faces from the sealed 005 sparse model.
All 60 depth cameras and warped masks pass; root independently reproduced the
checks and found no zero-area rough triangles. Densification took 47.081 s and
rough meshing 1.541 s. This is a cached/composed-stage run, not cold e2e timing.
Total new output was 523 MB; about 11.3 GiB remained free. No refinement or
texturing was attempted. Subsequent fixed-ray sensor scoring **does not show a
quality gain over earlier 008**: 015 has 80.58% ray coverage and 2.922 mm hit-only
mean depth disagreement versus 84.08% and 2.881 mm for 008. Within 5 mm among
all supported rays is 68.10% versus 70.72%. The supplied-intrinsics 014 control
remains worse on depth residuals. Root reproduced old scores exactly and
independently recomputed pooled new statistics from saved per-ray data. This is
an assumption-qualified diagnostic, not whole-object accuracy or KIRI parity.

The new [dataset protocol](YCB-EVALUATION-PROTOCOL.md) freezes 48 training and
12 held-out photographs per new YCB object, rejects duplicate photo content
and reference leakage, and keeps masks explicitly pending. Root rehashed all
62 selected files per object on the VPS; both passed. This is verified input
preparation, not completed new-object reconstruction.

[SYCL preflight](ALICEVISION-SYCL-PLAN.md) makes AliceVision's missing toolchain
and build-contract checks explicit without downloading/building dependencies.
It remains unbuilt and unbenchmarked. Prior Windows CI found a test fixture
SQLite handle left open; `da560e9` fixes explicit closing and adds a regression
test. New CI results remain pending; local success is not Windows validation.

Final local regression: **339 tests run, 8 skipped, remaining passed**; registry
validation and diff checks pass. Prior `2836886` quality checks passed on Linux
and macOS but failed on the Windows fixture issue fixed by `da560e9`.

### Previous continuation: initialization recovery, Metal audit and VPS data

Supervised Sol implementations and bounded live tests now provide:

- [Image-only initialization recovery](INITIALIZATION-RECOVERY.md): the new
  free-intrinsics seed failed at 2/60 views; freezing heuristic intrinsics
  registered 60/60 with 3,921 points. Subsequent global bundle adjustment retained
  that support and refined focal length from 1536 to 1103.95 px. Training median
  reprojection error fell from 0.3696 to 0.3533 px. This is one cached sparse
  experiment, not cold-run reliability or improved mesh accuracy.
- A test accidentally changed the original database's bytes through a writable
  native open. Exact historical bytes could not be restored. The audit and
  explicit rebound-cache provenance are retained; no exact replay is claimed.
- [AliceVision review](MTL-ALICEVISION.md): upstream SYCL merged in May 2026,
  correcting our CUDA-only characterization. The separate native Metal fork is
  promising but has significant build/error-reporting caveats. A standalone M1
  Metal compute oracle passed 4/4 values; neither full backend has run here.
- [Two new VPS acquisitions](YCB-EXPANSION.md): mustard and drill, 60 real photos
  each plus independent scanner meshes. Roughly 1.4 GiB stays on `/mnt/storage`;
  only manifests and two previews were copied locally. Masks, frozen evaluation
  splits and cross-sensor registration remain unfinished. No new quality score.

Root regression: **321 tests run, 8 skipped, remaining passed**. About 12 GiB
remains free on the Mac. No neural training or Kaggle quota was consumed.
Prior checkpoint `301768e` passed both CI workflows; this batch needs its own CI.

### Previous continuation: comparison contracts and camera bridge

The [three-branch plan](PIPELINE-COMPARISON-PLAN.md) and hash-bound
[pipeline registry](../benchmarks/pipeline-evidence.json) distinguish two
executed pipeline families, two audited candidates and three planned neural
branches. MicMac/AliceVision have not produced reconstruction results here.
The [Nerfstudio camera bridge](NEURAL-CAMERA-BRIDGE.md) now has analytic
projection/rejection tests and a live 14-view supplied-camera export; no
neural package was installed or trained. Root independently checked all
14 poses against normalized PyCOLMAP rotations: exact agreement.

The [initialization audit](INITIALIZATION-DIAGNOSTIC.md) identifies severe
focal/distortion drift in the failed image-only two-view model and ranks
alternative seeds from image correspondence cycles, without motion-label or
reference-geometry ranking. No new mapping run or quality improvement is
claimed. The [spatial sensor audit](SENSOR-SPATIAL.md) partitions the existing
fixed rays; root independently reconciled all support/hit/threshold counts.

Local regression: **305 tests run, 8 skipped, remaining passed**. Separate
opt-in real-data/stereo-CLI checks: **16 run, 1 skipped, remaining passed**
(overlapping tests, not additive). Registry validation and diff checks pass.
New real-object source availability/sizes were checked, but no new datasets
were downloaded; VPS acquisition and neural GPU runs remain pending.

Both the [quality harness](https://github.com/CrispStrobe/crisp3ds/actions/runs/36292531222)
and [foundation matrix](https://github.com/CrispStrobe/crisp3ds/actions/runs/36292531153)
passed at prior checkpoint `1b21c1f`. This continuation needs its own CI result.

### Previous checkpoint: calibrated mesh and sensor depth

Latest checkpoint: the corrected calibrated dense profile completed with
40,518 points. A separately recorded continuation produced a 38,550-face
rough mesh; refinement exceeded its five-minute cap, so no new refined or
textured calibrated artifact exists. The strict all-view checker was corrected
to use OpenMVS's uniform intrinsic scale despite short-edge rounding; all 60
depth-map cameras and masks pass without relaxed tolerances.

The new [three-view sensor-depth comparison](SENSOR-DEPTH-BENCHMARK.md)
uses fixed shared rays and camera-only reference alignment, not mesh fitting.
On stage-matched rough meshes, earlier `008` has 2.88 mm hit-only mean absolute
depth disagreement and 84.08% ray coverage; calibrated `014` has 5.96 mm and
83.66%. Neither is a whole-object quality percentage. Calibration/pose/mask
and depth-rectification assumptions remain unresolved; see the
[projection check](BERKELEY-PROJECTION-CHECK.md). The new calibrated path is
therefore **not** a demonstrated surface-quality improvement. Image-only
initialization and missing surface coverage remain priorities.

Current local regression discovery: **285 tests run, 8 skipped, remaining
tests passed**. Root independently recomputed the saved per-ray summaries and
checked ray/triangle intersections and distorted RGB projection against separate
numerical implementations. Rough output and all failed attempts remain preserved;
approximately 12 GiB is free, above the 10 GiB reserve.

Both the [quality harness](https://github.com/CrispStrobe/crisp3ds/actions/runs/36290465780)
and [foundation matrix](https://github.com/CrispStrobe/crisp3ds/actions/runs/36290465759)
passed on all three platforms at prior checkpoint `50b68ae`; these are not
portable real-photo reconstruction results.

### Earlier checkpoint: registration and calibration (`50b68ae`)

The new [known-transform registration audit](REGISTRATION-AUDIT.md) isolates
an evaluator defect: independently sampled copies of identical scanner geometry
can be aligned 16.67–25.25 degrees incorrectly, and partial coverage can make
the current objective prefer wrong poses. Historical shape scores remain
recorded but cannot establish intrinsic pipeline rankings. This does not
accept any reconstruction or explain away all geometry defects. A bounded
intrinsics-assisted camera control now recovers 60/60 views and 3,942 points,
with 1.34-degree median orientation disagreement against the supplied camera
reference. It remains separate from image-only results. The original threaded
calibrated verification crashed; a serial two-arm replay retained an image-only
2/60 failure rather than retuning its seed. The subsequent dense control passed
undistortion, mask warping and import, but exceeded its 1 GiB output cap while
writing full-resolution depth maps; no complete cloud or mesh was produced.
See [camera trials](CALIBRATION-ABLATION.md) and
[native-control evidence](CALIBRATED-CONTROL.md). No calibrated mesh-quality
improvement is claimed.
The [sensor-depth feasibility check](BERKELEY-DEPTH-FEASIBILITY.md) extracted
three frames, but does not yet provide independently registered shape scores.

That checkpoint's local Python discovery passed **267 tests (8 skipped)**.
The new native-resolution preflight has boundary/budget regression tests;
its corrected explicit-minimum profile has not yet been run through meshing.
Approximately 12 GiB remains free, above the 10 GiB reserve. Prior application
build/test results below are unchanged; this batch changes experimental Python
adapters, diagnostics and documentation, not the production reconstruction gate.

Earlier comparative checkpoint: [three real-photo benchmark lanes](BENCHMARK-RESULTS.md)
now compare MVE and classical COLMAP/OpenMVS on two objects, plus a supplied-camera
OpenMVS control on Sceaux. The box reaches 42.06% independently fitted F at 1%
reference diagonal, but only 35.52% using the prior unchanged alignment; raising
depth resolution did not improve that reported shape diagnostic. An independent Berkeley-camera
diagnostic adds trajectory/orientation evidence, not Google-mesh accuracy.
The supplied-camera Sceaux control reaches 94.04% rough / 93.52% refined F
against an upstream **software** mesh, not physical ground truth. The full
reports retain failed registrations, budget failures and composed continuations.

Automatic camera initialization remains fragile: generic retries recover all
73 bunny cameras, whereas a fresh 60/60 box replay still requires an explicit
seed pair. The latest box/native-mask meshes remain rejected. No KIRI parity is
established; the C++/Tauri production reconstruction action remains gated.

Both unmasked and native-masked bunny classical paths now complete through
textured OBJ export using that recovered model, with explicit continuation/cached
camera provenance. Native mask effects are checked in all 73 depth payloads.
The [whole-scan and post-hoc object-ROI evaluations](BUNNY-CLASSICAL-COMPARISON.md)
expose unreliable geometric registration; their F-scores do not establish a
pipeline ranking. This narrows the next evaluation work, not a quality acceptance.

Earlier full-application validation: all 11 native CTests, 30 desktop tests, 43 CLI/web
contract cases, the web build and live Chromium interaction test passed again.
The complete Python discovery passed 245 tests (8 skipped), including the new
SQLite handle-lifetime regression test in the 33-test classical suite.
The dependency metadata checker matches 530 exact entries. New quality-harness
tests cover multiple surface metrics, camera-reference composition, native mask
warping and upstream controls. The disk reserve remains 10 GiB.
A CI-only test dependency on ignored local provenance was previously replaced
with a checked-in, hash-traced mapper-options fixture. At commit `975cd8a`, both the
[foundation matrix](https://github.com/CrispStrobe/crisp3ds/actions/runs/36278628665)
and [quality harness matrix](https://github.com/CrispStrobe/crisp3ds/actions/runs/36278628682)
passed. This verifies builds/contracts, not portable real-photo quality.

The expanded quality harness now also passes on macOS, Linux and Windows at
`0f470ad` ([run 36287594304](https://github.com/CrispStrobe/crisp3ds/actions/runs/36287594304)).
Its first Windows attempt exposed SQLite connections left open after a lookup;
explicit closure on success and error fixed the issue without skipping the test.
The [foundation matrix at that same commit](https://github.com/CrispStrobe/crisp3ds/actions/runs/36287594254)
also passes on all three platforms. These remain build/contract checks, not
real-photo reconstruction or packaging validation on Windows and Linux.

The user approved [SOTA-ROADMAP.md](SOTA-ROADMAP.md). Supervised Sol tasks implement
the classical CPU path, object-motion experiments and benchmark tooling. Measured
progress is not product acceptance. MVE remains a control, not a mandatory engine;
backend quality gates precede Tauri reconstruction integration.

### Earlier checkpoints (superseded where the latest reports differ)

The first new live control, [60-photo raw YCB](YCB-COMPARISON.md), failed MVE
camera initialization in 267.14 seconds (not a timeout), with no mesh produced.
The raw COLMAP/shared-intrinsics arm registered only 2/60 views. Changing feature
eligibility to photo-derived pose-support masks registered 60/60, with 3,939
points and a near-planar, consistently ordered orbit. Root independently loaded
both models. This is a camera-estimation improvement, not mesh acceptance; the
foreground model subsequently completed the recovered surface/texturing trial above.
OpenMVS v2.4.0's pinned arm64 release tools start on M1; this alone does not
establish a working pipeline, binary dependency clearance or cross-platform use.

The [new surface scorer](SURFACE-BENCHMARK.md) measures sample-to-triangle
distances instead of distances to another finite point sample. The unchanged
bunny scores 29.47% F at the same 1%-diagonal threshold, using the existing
reference-fitted transform. This is a metric correction, not improved geometry;
the mesh remains rejected. Root independently compared the distance calculation
against 4,181 scalar point/triangle pairs (maximum squared-distance discrepancy
2.78e-16). The YCB same-mesh control gives 100% F at its tested tolerances.

Root independently reran the isolated Python suite: 191 tests passed with 8 skips;
11 native CTests, 30 desktop tests and the desktop production build also passed.
The generic system Python suite lacked PyCOLMAP; the complete pass uses the
existing isolated environment with pinned NumPy, Pillow, PyCOLMAP and OpenCV.
The new [quality-harness workflow](https://github.com/CrispStrobe/crisp3ds/actions/runs/36277497472)
passed on macOS, Linux and Windows at `bb3c80c`. Heavy photo tests are separate
from these contract/unit checks. The dependency metadata checker also passed
against 530 exact entries; experimental binaries remain outside shipping approval.

## CPU photo-to-mesh integration and real object reference

The [current milestone](CPU-E2E-PLAN.md) now targets ordinary overlapping photos
with unknown poses, not a mandatory marker-board workflow. Two supervised Sol
agents added a selected SIFT-only MVE runner and a bounded real-object downloader.
MicMac was rejected for the product path after finding noncommercial and GPL
sources in its normal build; see [pinned source evidence](MICMAC-BACKEND.md).

The [full CPU pipeline](MVE-FULL.md) has run on Apple M1: ten real CC BY tree
photos, no supplied poses, nine registered cameras, 322,133 oriented samples and
a 35,785-vertex / 69,897-triangle mesh. Two runs produced byte-identical final
meshes in 51.15 and 51.64 seconds. The second run's external `/usr/bin/time -l`
reported maximum resident set size 186,826,752 bytes; this is not a simultaneous
whole-process-tree memory sum. Root independently parsed all geometry, checked
finite vertices, face indices and nonzero areas, and inspected a vertex-colored
and untextured projection. The surface is coarse, partial, and includes background;
it is an end-to-end milestone, not accepted object quality.

The [3DLF-Scan reference](OBJECT-DATASETS.md) is now actually local: 73 real
turntable photographs plus a separately captured Revopoint mesh and metadata,
116,672,689 bytes total. It is published under CC BY 4.0; third-party rights in
the Stanford-derived printed shape still need review before commercial asset
redistribution. It is not a bundled or shipping-approved dataset. Only selected ZIP members were fetched,
not the full 9.8 GB archive. The scan has 217,505 vertices and 435,006 triangles,
including four zero-area triangles. Its frame is not registered to the photos.
Supplied poses, depth, masks and scan geometry do not enter our image-only run.

The first raw bunny run failed after 52.52 seconds: sparse reconstruction could
not initialize a camera pair from only 80 feature tracks. This failure is retained
at `build-opencv/mve-full-bunny-001`; no mesh or accuracy score is claimed. A fixed
image-only contrast-profile retry is recorded separately, with unchanged originals.

That retry registered **73/73 views**, but its [measured shape result](BUNNY-EVALUATION.md)
is **poor and rejected**. A preserved, hash-checked continuation produced a
589,429-vertex / 1,193,446-triangle final mesh after removing 11 zero-area faces
left by native cleanup. The original run took 944.62 s, continuation 16.25 s,
and separately repeated preprocessing 26.80 s. This is roughly 16.5 minutes of
measured pieces, not a fresh uninterrupted replay of the corrected runner.
At the predeclared 1%-of-reference-diagonal threshold, the reference-fitted
sampled F-score is **14.88%**, versus **86.25%** for the distinct-seed reference
sampling control. Visual inspection confirms severe shape disagreement.
This is neither physical-scale accuracy nor acceptable production object quality.

This remains a separate backend CLI, not the C++ `reconstruct` command or a
working desktop reconstruction button. A selected CMake build now compiles all
six tools on M1 and completes a full tree run in 54.44 s (9/10 views,
69,676 final faces). Live testing found and fixed mixed Mono codec headers and
Homebrew libraries; compile-only checks had missed that issue. App Store runtime
packaging remains unverified. No new native dependency has been marked
shipping-approved. A scoped remote platform matrix is now being tested;
do not infer full reconstruction portability from compile/usage checks.

The remote macOS arm64 and Ubuntu x64 build/smoke/unit jobs passed. Windows
reached native compilation after fixes for patch line endings and compiler
discovery, then failed on `min`/`max` macro collisions. The `NOMINMAX` fix is
committed, but its [retry](https://github.com/CrispStrobe/crisp3ds/actions/runs/36275102048)
was refused **before job startup** because GitHub reports a billing/payment or
spending-limit issue. Windows verification remains blocked; no billing settings
were changed. That historical startup block was subsequently resolved by public
visibility; run 36275507360 passed as recorded above. Neither mobile nor App Store
packaging is verified.

Latest local checks: **16 backend tests, 22 object-dataset tests, 11 native CTests
and 2 selected-MVE CTests pass**. About 22 GiB remains free. Large photos and meshes
are excluded from the private source backup.

A fresh replay of the corrected CMake path, including final sanitation, also
passed and produced the exact same mesh SHA as the preceding CMake run. Its
wall time was 263.53 s under substantial observed background load (load average
13.8 during execution), versus 54.44 s earlier; do not present this as a controlled
speed comparison. No zero-area faces needed removal in either tree mesh.

A second object is now prepared from the explicitly **CC BY 4.0 YCB dataset**:
60 original NP3 RGB photos, 1280 × 1024, covering 360° in 6° increments, and a
separate Google-scanner mesh with 32,770 vertices / 65,536 triangles. The two
source archives total 680,498,737 bytes and the selected files 66,848,804 bytes.
Full archive and per-file hashes are recorded; every photo decodes and the
reference mesh has valid finite geometry and no zero-area triangles. It remains
**unreconstructed**, ready for the next camera/foreground quality experiment.
The scan is an unregistered, separate-scanner shape reference, not metrology GT.
See [dataset provenance and preparation](OBJECT-DATASETS.md).

Source and written results are now backed up in the private GitHub repository;
see [backup scope](REPOSITORY-BACKUP.md). Large datasets and raw generated run
artifacts remain local. The [backend experiment plan](BACKEND-EXPERIMENTS.md)
adds a CPU full-mesh oracle and later neural comparisons, with a common
confidence-aware depth interface. These are planned experiments, not newly
implemented reconstruction capabilities.

## Additional-view recovery: marginal gain, hard cases remain unresolved

The [anchor recovery experiment](PIPES-RECOVERY.md) uses all ten additional
prepared photos, keeping one original feature per point and requiring at least
two mutually consistent new views. It re-estimates **46/258** points in
**7.889 seconds**, leaving 212 exact baseline fallbacks (186 insufficient
new-view support, 26 pair conflicts). These ten photos are now training data,
not held-out validation; this is not a same-input algorithm comparison.

Exact laser scoring of changed geometry took **11.691 seconds**. Across the
fixed 258 anchors, within-20-mm support increases only **151 to 152**; median
proximity remains **12.798 mm**, p90 changes **90.280 to 89.721 mm**, and the
worst error remains **7.45 m**. Of 46 re-estimates, 24 move nearer and 22 farther.
The three inspected extreme outliers are unresolved: anchor 104 finds one new
view; anchors 212 and 214 find none. No production promotion is justified.

Root independently replayed all 150 accepted feature coordinates in seven
images and checked projections with a separate quaternion calculation:
positive depths throughout and maximum reprojection 1.899 pixels. Three
changed-point nearest distances were separately checked against the full scan.
Two Sol agents implemented runner/evaluator and another performed read-only
review. **110 targeted Python tests and 11 native CTests pass**. No installs,
downloads or remote jobs; approximately 26 GiB remains free.

Next priority is improving the image correspondence evidence itself, then
testing it on a separate measured scene. More photos alone under the same
SIFT policy did not recover the severe failures. Preserve both the baseline
and this additional-data result rather than silently replacing the former.

## Larger-context check: useful outlier signal, not a production filter

The [frozen context experiment](PIPES-CONTEXT.md) completed in 2.708 seconds.
It recomputes larger-neighborhood descriptors without changing geometry and
retains 153/258 points, versus 231 for the unchanged-size all-pair control.
All three inspected extreme outliers are rejected; the largest retained
laser proximity drops from 7.45 m to 0.410 m. Retained median/p90 are
9.843/83.315 mm, but 56 of the original 151 points within 20 mm are discarded.
Original-population near-reference support therefore worsens to 95/258.
This rejection policy is **not promoted**. Test confirmation/recovery next,
not just stricter deletion. Pipes remains development evidence.

Two Sol agents implemented the verifier and evaluator; a third reviewed the
logic. Root added the unchanged-size control before execution, reviewed the
code, ran the experiment, and independently recomputed the scores. The
evaluator's exact-score/verdict sealing was tightened after review. Latest
targeted regression run: **100 Python tests and 11 native CTests pass**.
Approximately 26 GiB remains free; no new datasets, installs, or remote jobs.

## Measured pipes baseline: geometric outliers remain

The ETH3D pipes archives are now safely extracted and validated. The frozen
[four-view protocol](PIPES-EVALUATION-PROTOCOL.md) uses supplied cameras and
image-only SIFT matches, never laser points or supplied SfM seeds for fitting.
The [live sparse run](PIPES-SPARSE.md) produced **258 points, 52 with at least
three views, and 568 observations in 2.817 seconds**. The fourth selected
image contributes no accepted observations. Independent forward projection
checks all 568 positive depths and reproduces the saved residuals.

The separate [laser proximity scorer](PIPES-SCORE.md) exhaustively compares
each point against all 11,482,717 measured reference points in float64,
applying the published scan transform once without ICP or scale fitting.
Median distance is **12.8 mm**, p90 **90.3 mm**, and maximum **7.45 m**;
**151/258 (58.5%)** are within 20 mm. Scoring took 68.605 seconds with
approximately 546 MiB peak RSS. These are one-way point proximity diagnostics,
not official ETH3D accuracy, completeness, or end-to-end application accuracy.
The supplied cameras are an oracle input, not recovered scanner poses.

The first FLANN scorer failed its independent nearest-neighbor audit; its
failed run remains preserved and no score was published. Exhaustive scoring
replaced it without relaxing tolerances. Two initial sparse executions failed
before matching because OpenCV's GCD backend did not honor a positive thread
limit; the successful execution explicitly disables that worker pool.

Supervisor Chromium inspection loaded all six original-photo crops for the
three largest outliers. Repeated valves/collars appear to support false
correspondences despite subpixel reprojection errors; this is a visual
diagnosis, not ground-truth correspondence labeling. The post-hoc three-view
subset removes the extreme tail but retains only 52 points and has a worse
median distance (17.1 versus 12.8 mm overall). No filter is promoted.

Next: predeclare image-only ambiguity/third-view verification experiments,
report rejected support and coverage alongside geometric errors, and reserve
another reference for acceptance. This scene has now informed development.
Then evaluate dense geometry with appropriate visibility and completeness
accounting. No production backend or full photo-to-mesh quality claim changed.

Verification: **88 Python tests and 11 native CTests pass**; live extraction,
reconstruction, exact scoring, and browser contact inspection completed.
Approximately **26 GiB** remains free. No packages or remote jobs were added.
ETH3D's noncommercial asset restrictions remain subject to use review.

## Fixed-camera object support and measured-reference acquisition

Historical checkpoint; extraction and evaluation status below is superseded
by the measured pipes baseline above.

The [fixed-camera diagnostic](FIXED-CAMERA-OBJECT.md) retains all nine baseline
camera poses and uses the same triangulator/screens on two sets of verified,
object-region training matches. Existing matches yield **666 tracks (160 with
three or more views)**; foreground rematching yields **707 (173)**. Accepted
observations increase from 1,562 to 1,658, but image support drops from nine to
eight: the sole baseline observation in Tree-46 is absent after rematching.
Camera count remains nine by construction; this is not new pose recovery.

Root's independent verifier checks source hashes, graph membership, track
accounting, original feature IDs, heldout exclusion, positive depths,
reprojection, parallax, conditioning, and the fixed-ray least-squares solution.
It checks rejected-track accounting but does not independently reclassify
every rejection reason. The live run took 2.702 seconds. Neither the larger
point count nor unchanged heldout camera scores establishes surface accuracy;
the custom triangulation screens also differ from COLMAP's mapper, so the
earlier 151-point subset is not a comparable quality baseline.

The [ETH3D pipes reference](RIGID-REFERENCE-PLAN.md) is now downloaded in an
isolated research directory: 199,173,411 bytes compressed, 286,754,145 bytes
declared after extraction, comprising 14 images with supplied calibration
and a separate evaluation laser cloud. It is **not extracted or evaluated**.
CC BY-NC-SA restrictions require review for commercial use; this is not a
shipping asset. The initial Python CA failure was resolved using the system
CA bundle with TLS verification kept enabled; the empty failed destination
is preserved. No global packages or production dependencies changed.

The original download inventory's directory flag was corrected in a separate
hash-bound `inventory-reviewed.json`; original archives and manifest remain
unchanged. The reviewed archive members are 19 regular files and six
directories. Nothing has been extracted. A fresh fixed-camera replay produced
byte-identical point/report files. Final verification: **73 Python tests and
11 native CTests pass**; approximately 25 GiB remains free.

## Foreground evidence: baseline is mostly background; filter not promoted

The [manual tree-envelope audit](FOREGROUND-ROI.md) finds only **151/1,773**
baseline points with every observation centre inside the target envelope,
**82/1,121** held-out pairs with both centres inside, and **2/181** closed
cycles with all three inside. Root inspected all ten overlays before any
support scoring; the polygons and exact PNG hashes are frozen. These are
approximate post-hoc development annotations, not segmentation ground truth.

The [foreground-centre training experiment](COLMAP-FOREGROUND.md) rematched
after removing outside training centres, preserving the exact held-out
observations and pairs. It finished in **3.53 seconds** using cached features:
**5/10 cameras, 192 points**, versus the baseline's 9/10 and 1,773. On the
same 82-pair foreground population, >4 px errors or unavailable predictions
worsen from **8 to 23**. Lower finite residuals do not compensate for missing
coverage. This experiment is **not promoted** and does not establish a better
object reconstruction. Independent verification passes database/track/split
integrity with 594 positive-depth observations and zero split overlap;
quality acceptance remains false.

Current regression verification: **56 Python tests and 11 native CTests
pass**. No new image downloads, dependencies, remote jobs, or shipping
backend changes were made; approximately 25 GiB remains free.

Next work should separate reliable camera estimation from object-only
reconstruction support: calibrated board anchors for the real turntable,
fixed-camera object matching/triangulation for controlled diagnostics, and a
small rigid multi-view reference with trustworthy independent geometry.
Do not weaken COLMAP's filters to turn this failure into a nominal success.

## Three-view failure audit

The [per-cycle audit](THREE-VIEW-AUDIT.md) and independent NumPy checker
reproduce the frozen errors; root's fresh audit report is byte-identical.
No triangulation-convention bug was found. The 21 >4 px cases span 16 raw
match components, and only two lie in components with conflicting same-image
features. Cycle closure and conflict checks alone therefore do not explain
or remove the failures.

An exploratory camera-only longest-baseline policy gives 12 missing-inclusive
bad cases rather than 21 on the fixed 106 cycles, but changes the predicted
image in 89 cases and introduces one nonpositive-depth result. It is **not**
a paired same-pixel accuracy improvement or a production change.

The local contact sheet loaded all 93 image crops in Chromium. Supervisor
spot-checks reveal background building/grass/sky support in several gross
failures and even the lowest-error control. These are visual observations,
not ground-truth labels. Foreground-supported geometry now needs explicit
measurement before a scene-wide camera screen can justify object meshing.
Current verification: **45 Python tests and 11 native CTests pass**.

## Clean held-out camera screen: pass, not surface acceptance

The new [clean observation-holdout run](COLMAP-HELDOUT-V2.md) registered 9/10
images and retained 1,773 points. Root independently verified the database,
feature exclusion, tracks and 7,046 positive-depth observations. Of 1,121 raw
held-out matches, 1,104 are scoreable; median/p90 epipolar error is
0.385/1.867 px. All predeclared development camera-screen checks pass with
zero spatial leakage. This replaces neither the full-feature baselines nor
the invalid first holdout artifact.

The three-view diagnostic still has a large error tail: 106 usable cycles,
median/p90 third-view reprojection 1.355/16.261 px, 21 above 4 px. Another
73 registered cycles are low-parallax and two lack registered cameras.
No surface or production-quality claim follows from the epipolar pass.
Current root verification: **35 Python tests and 11 native CTests pass**.

## Mapper replay and feature-split repair

The [mapper-only diagnostic](COLMAP-POINT-LOSS.md) reuses a copied original-photo
database without extraction or rematching. It records 123 points after initial
pair refinement and 16 after the third image; the final model remains 3 cameras
and 16 points. Root's fresh replay produced byte-identical final model files,
and an independent audit verified 41 positive-depth observations and reciprocal
track links. This is not a reproduction of the original 7-camera/zero-point
trajectory: the random seed is reset at a different pipeline stage. The exact
cause of the original point loss remains open, and no filter was weakened.

The repaired [feature splitter](FEATURE-HOLDOUT.md) groups spatial connected
components rather than interpreting affine columns as scale. A live read-only
audit and independent supervisor repeat agree exactly on 53,546 original
features, 44,062 groups and 10,664 withheld rows, with zero train/held-out
centres within 0.25 px. Source database hashes were unchanged. This proves
split integrity, not reconstruction accuracy; the old contaminated run is
still invalid and its runner remains quarantined.

## Latest sparse-oracle audit correction

The first isolated PyCOLMAP 3.11.1 CPU experiment registered 3/10 reduced
tree images and 12 points, but **its held-out benchmark is invalid**. Our
wrapper misread six-column affine keypoints as position/scale rows, leaking
duplicate orientations between training and validation. It also marked a
heuristic focal length as a known prior. Earlier claims of clean feature-group
exclusion are retracted. The failed experiment is preserved; it must not be
used to judge stock COLMAP or to supply production camera poses. See
[the upstream-checked sparse experiment](COLMAP-SPARSE.md) and
[independent audit](SPARSE-ORACLE-VERIFY.md). Full COLMAP + OpenMVS remains
unrun, and no shipping dependency approval is implied by installing PyCOLMAP
in an isolated research environment.

Following the [upstream usage audit](COLMAP-USAGE-AUDIT.md), a conventional
full-feature CPU baseline on the same reduced images registered **9/10 cameras
and 2,369 points** in 19.496 seconds. Its 10,310 observations have positive
depth and reciprocal track links; training reprojection median/p90 is
0.568/1.270 px, not independent accuracy. The original-JPEG/EXIF baseline
finished in 149.037 seconds with 7 cameras but zero retained points, an
unresolved failed reconstruction. Neither result is a finished scan.
Root verification passes 23 Python tests and all 11 native CTests.
An unchanged-settings fresh replay again registered the same nine images,
with 2,376 points and 10,425 positive-depth observations. Model consistency
passes, but it is not a bitwise-identical reconstruction. About 25 GiB remains
free; the isolated environment/data is 134 MiB and sparse run artifacts total
60 MiB. No new photographs or remote jobs were used.

Updated 2026-09-26 after expanded stereo/oracle evaluation. Sol agents implemented the test-only evaluator, dataset tooling and independent unit/live tests under main-agent supervision. Review covered numerical/input boundaries, metric denominators, PFM conventions, resource bounds, output preservation, benchmark provenance and repeatable CLI-to-browser behavior.

| Task | Delivered behavior | Limit |
| --- | --- | --- |
| F01 | Versioned project contract, JSON Schema, board layout/distortion model, pose and sparse report contracts, shared validation cases and coordinate convention | Depth and persisted artifact contracts remain to be added |
| F02 | C++20 static core, C ABI, CLI, manifest validation, synthetic projection/triangulation diagnostic, explicit unavailable full reconstruction | Image poses require the optional OpenCV build; no dense reconstruction or mesh export |
| F03 | Tauri 2 host, project library, calibration-model and marker-grid editors, explicit corner JSON, CLI pose-report inspection | Native worker not connected; image paths only in GUI; reports imported for inspection and cleared on edits |
| F04 | Exact locked package metadata inventory, selected-license policy checker, parser tests, native CI matrix and macOS web/host checks | No full source-file/artifact audit, license/source screen or release source archives yet |
| R03 MVE spike | Selected headless MVE libraries/tool built and actually executed on three rendered views with estimated poses and sparse seeds; loaded-camera convention check and radial-depth scorer; supervisor replay reproduced counts/errors | One reference depth map, synthetic only; 36,927 valid depths outside object planes demonstrate missing object isolation; default-build GPL utility excluded; no shipping approval |
| R01 implementation | Pinned OpenCV 4.12.0 PNG/JPEG ArUco detection → planar pose candidates/refinement → metric pose/residual JSON through CLI and C ABI | Synthetic-image verification passes; no measured real dataset; only DICT_4X4_50, planar square markers, known opencv-radtan calibration |
| R02 implementation | Binary-mask validation, eroded-mask ORB matching, conflict-free tracks, metric triangulation, structured CLI/C ABI sparse report | Synthetic-image verification only; 64 views, acquisition-order pair window 2, single-scale features, no bundle adjustment |
| D02 partial | Validated sparse-report import, interactive point cloud, millimetre bounds, residuals and view coverage | Manual inspection only; no worker or mesh viewer; imported results remain unverified and clear on edits |
| R03a evaluation | Separate OpenCV stereo evaluator, calibrated depth conversion, external-prediction scoring, four measured stereo references, checksummed fetcher, separate libELAS oracle and project-authored Census comparison; fixed-profile quality/performance runner and spatial error diagnostics | Mirrored development scenes, no held-out backend winner; upstream file equivalence unverified; test-only oracle is not an approved shipping dependency |
| R03b/e experiments | Calibrated horizontal/vertical rectification and geometric search bounds, object-frame triangulation, visibility-aware track fusion with masked-centroid revalidation; three-view rendered-image stereo experiment | Separate test library, not production dense reconstruction; image fixture masks after matching, not within matcher costs; generic depth-to-fusion integration and measured object acceptance remain open |

## Verification evidence

- Release CMake build and 4 CTest tests pass locally on Apple Silicon.
- Core agent additionally ran AddressSanitizer/UndefinedBehaviorSanitizer with all 4 tests passing.
- 43 manifests/mutations agree between native and TypeScript validation, for both native build configurations.
- 30 web project/report validation tests pass; TypeScript checking and Vite production build pass.
- Chromium smoke checks passed for project creation, image-path entry, reload, JSON download, saved-project switching, valid import, invalid-import preservation and narrow layout. Downloaded JSON was accepted by the CLI. No browser runtime errors were observed in those checks.
- The locked dependency metadata gate passes for 530 entries (including seven optional native source entries); its 3 parser-policy tests pass. This is not a shipped-source or legal audit.
- `cargo check --locked` and `cargo build --locked` pass for the Tauri 2.11.6 host on macOS. The native application has not been interactively launched or packaged.
- The optional OpenCV build passes 10 CTests, including stereo-unit tests with assertions enabled in Release builds, prepared-input CLI parity/provenance checks, and the new geometry/fusion and image-derived depth regressions. Actual rendered PNG/JPEG images are detected and solved; tests also check blank frames, wrong dimensions, oversized inputs, path escape and failure of a mixed valid/invalid image run. Synthetic PNG translation was approximately `[-55.118,-54.980,530.695]` mm against `[-55,-55,530]` mm ground truth; RMS reprojection residual was about 0.608 px. These are fixture results, not promised scanner accuracy.
- The independent three-view sparse fixture produces 1,189 points, all near the two known textured panels, including 465 three-view tracks. Depth error is 2.05 mm median and 5.70 mm at the 95th percentile. Tests check eroded-mask containment and structured failures for missing masks, wrong dimensions, nonbinary masks, escaped symlinks, empty masks and degenerate baseline. Images, truth and quality reports are retained under `build-opencv/synthetic-sparse-fixture/`.
- Independent Chromium smoke checks loaded the actual sparse CLI report, verified visible point-cloud rotation, retained a valid report after foreign-project rejection, left stage metadata unchanged, cleared results after calibration edits, and checked narrow layout without runtime errors.
- `otool -L` on the optional CLI shows only AppKit, libc++ and libSystem dynamic dependencies on this Apple Silicon host. OpenCV and selected codecs are built statically from the pinned source archive.
- Independent Chromium checks passed for selecting the calibration model, generating/saving board geometry, reload/export, invalid-board-edit preservation, importing the actual CLI report, retaining a report after foreign-project import rejection, leaving stage metadata unchanged, clearing stale results after calibration edits, and narrow layout.
- `scripts/live_browser_test.mjs` now reproduces sparse CLI-to-browser checks, including pixel changes after orbit/zoom, report rejection/preservation, calibration invalidation and 390px layout. Both agent and supervisor ran it successfully with Playwright 1.63.0 / Chromium 153.0.8010.12; it cleans up its Vite process.
- Python discovery reports 31 tests with 30 passing and one ETH3D noncommercial-data check intentionally skipped when dataset and stereo-CLI opt-ins are enabled. Checks include hand-calculated metric/depth expectations, PFM conventions, dataset conversion, quality diagnostics, profile failure reporting and benchmark failure/timeout/input-mutation handling. Separate ELAS and Census known-shift checks also pass. ETH3D assets are separately gated and are not part of the default benchmark.
- A live measured-reference Piano stereo run yielded 83.86% coverage, 31.32% all-reference bad-2/missing rate and 1.59 px matched MAE at derived 705×480 resolution. It caught a PFM scale interpretation bug that synthetic unit-scale files did not expose. Corrected results repeat across agent/supervisor runs. See [DENSE-EVALUATION.md](DENSE-EVALUATION.md) for exact settings, provenance and limitations; this is not a passing production-quality gate.
- The expanded common-input matrix completed all 12 engine/scene combinations in both agent and supervisor runs. The supervisor report is `build-opencv/benchmarks/run-20260926T105006Z-1737152e/benchmark.json`: fixed search ranges 0–79 for Piano and 0–95 for Cones/Teddy/Venus. ELAS all-reference bad-2/missing rates were 18.36%, 10.76%, 15.04% and 3.70%, respectively. Its interpolation-heavy preset yields full reference coverage; that is not equivalent to every pixel being independently supported. See [ORACLES.md](ORACLES.md) for comparative results and limits.
- The selected MVE tool reconstructed 44,152 depth pixels at 400×300 from three rendered images and estimated sparse geometry. A supervisor fresh-scene replay produced identical counts and radial-depth errors: 94.20% of object-plane rays had a depth, with 3.10 mm matched mean absolute error. This is a different fixture from the pairwise depth test and must not be ranked directly against it. Two independent converter/scorer unit tests pass. [MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md) documents the radial-vs-axial depth correction, synthetic scoring scope and unmasked background output.

CI configuration includes Windows/Linux/macOS core builds. Those remote jobs have not been run from this workspace. Local browser checks do not establish native WebKit interactions or mobile support. Native parser rejects duplicate object keys; the browser's JSON.parse currently keeps the last value. Native validation must remain authoritative at the future worker boundary.

## Next delivery

Small real-image follow-up: [ten pinned tree photographs](REAL-DATA.md), 202.1 MB including metadata, now feed a [test-only COLMAP-pose/ORB/MVE path](TREE-DENSE.md). The source export has no observation tracks and no verified metric scale; actual image matching produced 575 independent two-view seeds, with 1.055 px loaded-camera RMS reprojection. The fixed dense test **failed**: reference views at 768×512 produced zero and two valid pixels respectively. Both agent and supervisor fresh-scene replays reproduced this result. No physical accuracy or usable real reconstruction is claimed. This case demonstrates why a successful process exit and synthetic tests are insufficient acceptance. CPU VPS and private Kaggle [staging tools](REMOTE-QUALITY.md) are local preparations only; no remote run, upload or GPU quota consumption occurred.

Further real-image diagnostics corrected the COLMAP/OpenCV principal-point convention; a fresh baseline still yields 0/2 depth pixels. The [multiview refinement trial](TREE-REFINE.md) is rejected: only nine of 423 retained tracks span three views, reserved-observation RMS is already 574.5 px before refinement and worsens to 839.3 px despite better training fit. Dense output remains effectively empty. This identifies inconsistent multiview support, not its unique cause. Separately, the [fixed-pair SGBM/ELAS comparison](TREE-PAIR-ORACLE.md) produces nonempty disparity on the same corrected cameras: SGBM has 21.31% finite coverage and ELAS 100% including filling. These are image-space diagnostics without dense ground truth or verified metric scale; neither establishes a usable mesh. [The three-pipeline comparison](PIPELINE-COMPARISON.md) separates this evidence from unmeasured estimates against COLMAP+OpenMVS, Metashape and RealityScan.

Supervisor reruns reproduce the tree stereo metrics and refinement residuals. The fixed-pose multiview-track control produces 0/0 depth pixels, versus 7/0 after refinement: neither is accepted. Geometry and anchored-optimization self-tests pass, as do the 19 existing acquisition, tree and remote-staging Python checks. The latter require discovery from `scripts/remote_quality` because their imports are local. No new photographs or external services were needed; about 29 GiB remains free.

Three additional exporter tests pass: a valid fresh export, refusal to overwrite,
nonfinite-RMS rejection and changed-anchor rejection (22 Python tests total in
this supervisor verification batch).

The next supervised quality batch diagnoses the real-tree track failure and
connects image-derived depth maps to the actual fusion function. [Track
diagnostics](TREE-TRACKS.md) reproduce all 575 ORB links: all 75 multiview
components are open chains, with no triangle support. A single frozen SIFT
trial gives 1,203 pair links and 16 cycle-supported components, but none meets
the unchanged 4 px all-view gate. More matches therefore do not establish
valid multiview geometry; supplied-camera accuracy versus correspondence
failure remains unresolved.

The new [image-to-fusion regression](DEPTH-IMAGE-FUSION.md) uses independently
matched camera-a and camera-b depth maps, with shared-photo correlation stated,
outside-mask mutation checks, left/right checks and fixed full-foreground
scoring. On 9,519 sampled foreground pixels, bad-or-missing >2 mm counts are
813 raw, 827 after filtering and 854 after two-candidate equal-weight fusion.
Boundary counts also worsen (206, 215, 232). Neither variant is accepted as
a quality improvement; a passing wiring regression is not a quality pass.

[Oracle readiness](PIPELINE-ORACLE-READINESS.md) verifies the ten pinned images
and records a CPU-only run plan. All six COLMAP/OpenMVS commands are absent
from PATH; no full-pipeline oracle was installed or run. Supervisor verification
passes all 11 native CTests and four new preflight unit tests; the live preflight
checks source hashes and preserves a 10 GiB reserve under its proposed 4 GiB
output allowance. This is admission planning, not an implemented runtime quota.

Independent supervisor ORB and SIFT reruns reproduce the complete diagnostic
JSON exactly, with all recorded inputs unchanged. The final fusion-report
revision also passes its targeted test. About 29 GiB remains free; no new
photographs, external installs or remote jobs were needed.

The next [camera audit](CAMERA-AUDIT.md) independently verifies the ten pinned
image/camera associations, quaternion-to-matrix conversion, intrinsics resize,
MVE metadata and analytic relative geometry; it finds no local conversion
defect. That does not validate the supplied physical poses.

The [epipolar experiment](TREE-EPIPOLAR.md) matches images before any supplied-pose
filter and reserves every fifth descriptor-ordered match. Of 45 pairs, 30 meet
the frozen sample gate; image-fitted fundamental matrices have lower held-out
median error in all 30. The unweighted median of pair medians is 7.305 px for
supplied cameras versus 0.282 px for train-only image fits. On the same 1,693
held-out matches pooled across those pairs, 475 versus 1,580 are within 2 px.
These are full-frame epipolar diagnostics, not object surface accuracy.

On preselected pair 1/4, a separately fitted calibrated essential model lowers
held-out median/p90 error from 1.831/5.323 px to 0.252/0.778 px, but only 146 of
309 training inliers triangulate with positive depth in both recovered cameras;
six of those are beyond the fixed 50-unit-baseline distance cutoff, leaving
140 accepted by `recoverPose`. The candidate is not accepted as a replacement
camera pose. Low image residual alone does not validate a 3D reconstruction.

Supervisor replay reproduces every pair result and match file exactly. An
[independent standard-library checker](EPIPOLAR-VERIFY.md) verifies all 45
pairs and the calibrated follow-up from saved observations and matrices.
Ten new Python unit tests, the analytic camera-audit self-test, and all 11
native CTests pass. The Python image experiment uses the pre-existing
`/usr/local/bin/python3` (3.11), OpenCV 4.10 and NumPy 1.26.4, distinct from
the native OpenCV 4.12 build; no dependencies were installed. Full-pipeline
quality and physical camera calibration remain unproven.

Supervisor verification for this batch: eight acquisition/patch-diagnostic tests, six tree import/dense-failure tests and five remote-staging safety tests pass (19 total). A live fetch re-verifies the pinned files without redownloading photographs. The independent live MVE replay correctly exits with `dense_failed_empty_reference` and retains both failed depth maps. Photos plus the original, agent replay and supervisor scenes occupy roughly 232 MiB on disk; about 29 GiB remains free. A read-only, unwarped-patch diagnostic finds much weaker agreement across the three selected neighbors than within recorded image pairs; it is not a substitute for MVE's warped-patch objective or proof of a particular failure cause.

Quality-first follow-up: [the pose/resolution ablation](QUALITY-ABLATION.md) reduces synthetic matched surface error from 3.081 mm at L2 to 2.130 mm at L1. Known-camera oracle poses reach 0.660 mm at L1; this is a diagnostic opportunity, not an implemented pose improvement. [Independent acceptance checks](QUALITY-GATES.md) reject the mask-aware MVE experiment at both resolutions: improved object isolation comes with lost boundary coverage, while common-support accuracy barely changes. An independent correctly rasterized marker fixture supports subpixel refinement, but conflicts with the older renderer's result and is not real-capture validation. Joint observed-track/board bundle adjustment is the next experimental candidate. None of these experiments enables production reconstruction.

The first fixed [Ceres bundle-adjustment candidate](BUNDLE-QUALITY.md) improves L1 finite-rectangle surface MAE from 2.130 to 1.580 mm, missing-inclusive bad-2 from 51.74% to 27.23%, coverage from 98.51% to 98.62%, and boundary bad-2 from 62.67% to 50.00%. Ground-truth geometry is used only in scoring, not optimization. It passes the provisional surface gate but **remains rejected** by the separately declared marker-fit safeguard: marker RMS rises from 1.10791 to 1.11717 px. A supervisor optimizer replay reproduced the rejection; independent world scoring reproduced the surface metrics (`build-opencv/bundle-quality/world-supervisor.json`). A separate `EIGEN_MPL2_ONLY` build reproduces the result to numerical precision; this is test-only dependency evidence, not shipping approval. This is a promising development result on two synthetic panels, not an accepted production improvement or measured scanner accuracy.

Quality follow-up verification: supervisor Python discovery ran 43 tests (42 passed, one intentional dataset-license skip), with three additional ablation tests and three guarded optimizer malformed-input tests passing. Eight non-fixture-regenerating CTests passed; the pose/sparse CTests were not rerun during these experiments to preserve their frozen inputs. Supervisor runs also passed optimizer Jacobian/joint-recovery self-tests, the independent eight-view marker render, and mask parity/background-mutation checks. No UI code changed in this follow-up.

The active scoped quality work is recorded in [QUALITY-IMPROVEMENT.md](QUALITY-IMPROVEMENT.md). Fixed-profile development experiments show a roughly 24–30% serial matching-time reduction and modest missing-inclusive error improvement with `fast`; this is not a production backend selection. Preserve the default baseline for regression comparisons. The new synthetic depth fixture also separates coverage, edge error, per-pair error and consistency-filtered results; its idealized geometry is not a physical tolerance claim.

R01 and R02 still need measured turntable-image acceptance; camera-calibration estimation is not implemented. R03a establishes a real stereo baseline; R03b–e cover controlled rectification, reconstruction-mask propagation, dedicated-backend comparisons and oriented fusion as scoped in [DENSE-EVALUATION.md](DENSE-EVALUATION.md). MVE is not mandated. R04 completes a measured CLI scan through meshing. Capture requirements are in [CAPTURE-DATASET.md](CAPTURE-DATASET.md); implemented contracts and test scope are in [POSE-MILESTONE.md](POSE-MILESTONE.md) and [SPARSE-MILESTONE.md](SPARSE-MILESTONE.md).

No rig photographs, camera-calibration dataset, measured turntable object, or hardware protocol were supplied. The downloaded stereo fixture does not fill that gap. The full scan, rig control, mesh viewer, worker cancellation/resume, installers, native mobile reconstruction and browser WASM remain open work. The foundation must not be described as a complete photogrammetry engine.

The inspected 3DLiveScanner repository offers targeted capture/replay/texturing references, not a replacement portable photo-to-depth engine. See [REPOSITORY-REUSE.md](REPOSITORY-REUSE.md). No code or SDK binaries from that project have been integrated.
