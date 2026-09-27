# Fresh turntable pipeline and license boundary

Frozen before the next live run, 2026-09-27. The subject is the same original
60 Berkeley NP3 cracker-box photographs and the existing photo-derived coarse
masks. Reconstruction inputs exclude the Berkeley pose/calibration metadata and
Google reference mesh. All large artifacts and scratch go to
`/Volumes/backups/code/crisp3ds-data`, with at least 10 GiB free on both the
Mac internal volume and external SSD throughout.

## Fresh image-only camera stage

Replay the one earlier successful turntable configuration without altering the
source JPEGs/masks: shared SIMPLE_RADIAL intrinsics, mask at feature extraction,
SIFT max 1,800 features / 1,200 image pixels, sequential matching with overlap
8 (no loop detector), automatic incremental mapping, seed 20260927, two CPU
threads. Preserve every option and input hash. The fresh result, rather than the
old 60/60 model, is the only candidate for continuation. A failed replay stays
failed; do not splice old poses into it.

The first acceptance gate is 60/60 registration with finite, structurally valid
tracks and plausible camera intrinsics. Independently compare all named cameras
to sealed Berkeley rig metadata *after* mapping: require center RMS <5% of the
reference camera-layout radius and p95 orientation <10°. These thresholds are
diagnostic, not claims of physical pose truth, and the rig metadata never enters
reconstruction. Retain the full evaluation and all failed/partial artifacts.

This implements masks and a sequential overlap graph. Meshroom's documented
2 px minimum 2D motion filter is a relevant design idea but is not assumed to
exist in the selected PyCOLMAP API. Our result must not be labeled an exact
Meshroom or AliceVision turntable implementation.

## Native complete oracle path

Only after the camera gate, use the *same fresh model* through undistortion,
mask reprojection into those exact undistorted cameras, native dense stereo,
depth fusion, meshing, refinement and texturing. Verify 60 mask/image pairs and
native DMAP exclusion outside the masks. Freeze a high-resolution profile before
launch; object regions occupy only a few percent of the original 1280×1024
images, so a 320 px whole-image depth map is not a credible quality test.
Inspect the bare mesh and compare to the independent scanner reference using
unchanged pre-existing metric settings, keeping whole-reference and object
coverage visible. A finished OBJ/texture or 60 cameras alone is not acceptance.

The locally available OpenMVS v2.4.0 binary is **AGPL-3.0** and is used only as
an evaluation oracle. Its algorithms and result formats can inform an original
adapter or independently implemented compatible alternative; its code and
binary are not approved for the commercial/App Store package. COLMAP itself is
BSD licensed, subject to an exact linked-dependency audit. Selected MVE is
BSD licensed but has not achieved the required object quality. AliceVision core
is MPL-2.0; a full current Apple Silicon dense build is unverified. The
project's present AGPL license also requires an explicit future licensing
decision for the intended App Store distribution.

The previous composed PyCOLMAP→OpenMVS masked run `classical-ycb-native-masked-008`
already reached texturing on this old model. Do not repeat it merely to count
another complete run. The new test addresses **fresh image-only repeatability**
and a camera-gated high-resolution masked continuation.

## Fresh sparse execution, 2026-09-27

The unchanged historical foreground worker was rerun from the original 60 JPEGs
and coarse photo-derived masks in the fresh external folder
`/Volumes/backups/code/crisp3ds-data/turntable-sparse-profile-001`. Its
[`turntable-profile.json`](/Volumes/backups/code/crisp3ds-data/turntable-sparse-profile-001/turntable-profile.json)
records 2/60 registered images, 127 points, 40.425 s wall time, and a verified
but rejected binary model. The source inputs and pinned PyCOLMAP 3.11.1 binary,
all mapping/feature/matching options, feature counts (40,600), 444 pair rows,
and automatically selected seed images #33/#28 match the earlier 60/60 case.
Keypoint/descriptor bytes and verified two-view geometry differ between the
fresh and earlier databases despite matching feature counts. The fresh seed
model ended at `f=3023.57 px, k=-7.122`; every proposed third camera failed
registration. This is a real repeatability failure in fresh feature/match/SfM
execution, not a resource or wrapper failure. The failed result remains intact.

A separately labeled development diagnostic copied that **freshly extracted**
database, held its image-only starting `SIMPLE_RADIAL` camera
`[1536,640,512,0]` fixed during automatic-pair mapping, then ran one global
BA that refined focal length, distortion, and extrinsics with principal point
fixed. PyCOLMAP changed the copied SQLite file bytes during mapping while its
six logical tables and schema remained identical; the first
`turntable-sparse-fixed-cache-002` attempt stopped on an overstrict byte-level
assertion after making a 60-view model. The corrected fresh output
[`turntable-sparse-fixed-cache-003/turntable-profile.json`](/Volumes/backups/code/crisp3ds-data/turntable-sparse-fixed-cache-003/turntable-profile.json)
sealed 60/60, 3,923 points, and 12.106 s worker time, with final
`f=1126.16 px, k=-0.0890`. The separate post hoc
[`independent-camera-gate.json`](/Volumes/backups/code/crisp3ds-data/turntable-sparse-fixed-cache-003/independent-camera-gate.json)
passed the frozen rig thresholds: center RMS/radius 2.0014% (<5%) and p95
orientation 3.016° (<10°). This cached-DB diagnostic is not the one-command
fresh pipeline and contributes no Berkeley metadata to reconstruction.

The next [`turntable-sparse-fresh-fixed-004/turntable-profile.json`](/Volumes/backups/code/crisp3ds-data/turntable-sparse-fresh-fixed-004/turntable-profile.json)
is a genuinely fresh one-command photo/mask → feature extraction → sequential
matching → fixed-initial-intrinsics automatic mapping → delayed BA run. It
registered 60/60 with 3,917 points in 42.375 s; the final shared camera is
`f=1108.61 px, k=-0.04217`. Source hashes, model hashes, and unchanged logical
feature/match tables are sealed. Its image-only orbit diagnostic has 59/59
steps in one direction and radius coefficient of variation 0.00690. These are
structural checks only. A separate post hoc
[`independent-camera-gate.json`](/Volumes/backups/code/crisp3ds-data/turntable-sparse-fresh-fixed-004/independent-camera-gate.json)
(SHA-256 `1656ac8aea32650953f4ec83413cd9022f28ad85ef69cf91af0c46026f868214`)
compared the sealed 60 named cameras to Berkeley rig metadata and passed the
frozen gate: center RMS/radius `0.01894131485` (<0.05) and p95 orientation
`2.89594377°`
(<10°). Berkeley data remained outside extraction, matching, mapping, and BA.
This camera agreement does not establish mesh accuracy; the masked native dense
continuation and bare-shape review remain separate gates.
