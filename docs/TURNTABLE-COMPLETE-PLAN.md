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
