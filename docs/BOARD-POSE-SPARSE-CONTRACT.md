# Fixed-board-pose to object-only COLMAP sparse contract

Status: synthetic implementation only. No sealed mustard board poses, photos,
masks, or tracks have been ingested; no sparse model has been exported.

## Coordinate and input contract

Each exact TRAIN JPEG filename identifies one original RGB image, one
`<filename>.png` binary/object mask, its SHA-256 hashes, dimensions, zero-
distortion `PINHOLE` `(fx, fy, cx, cy)`, and `camera_from_board=(R,t)` where
`x_camera = R x_board + t`. `R` must be in SO(3); the checkerboard frame is the
COLMAP world frame because the board rotates rigidly with the target. This
does **not** assert that the checkerboard axes/scale are the Berkeley object
reference frame. The current board diagnostic reports translation in arbitrary
checkerboard-square units and retains a global 180-degree corner-label gauge;
neither metric scale nor gauge is resolved by this importer. Its current
`fx=fy=1536, cx=640, cy=512` and zero distortion are assumptions, not
independently validated calibration; live ingestion requires that review.
Calibration and 2D track coordinates use OpenCV integer
pixel centers, so both principal point and keypoints receive `+0.5` when
emitted to COLMAP. Camera center is `-R.T @ t` in board coordinates.

Object-only 2D tracks are a **separate required input**. Camera poses alone
cannot create 3D points. A track is accepted only when all observations are
inside the corresponding masks, unique, finite, in bounds, in at least three
images, and triangulate in front of every camera with a **maximum pairwise**
viewing-ray angle of at least 1 degree (not a minimum over all view pairs)
and at most 2 pixels reprojection error per view. At least eight
tracks are required. These are conservative ingestion gates, not tuned on
mustard results. Inputs with lens distortion need an explicit, separately
tested rectification/calibration contract; this module rejects them by schema.

The module verifies exact source/mask directory inventories and SHA-256,
JPEG/RGB and PNG/grayscale formats, dimensions, K, pose handedness, and track
integrity before building an in-memory PyCOLMAP 3.11.1 reconstruction. It
uses the existing pinned `.local-tools/colmap-sparse/venv/bin/python`; no
feature extraction, pose re-estimation, dataset reference, scanner, depth,
or dense stage is part of this contract.

## Route and export boundary

An explicit `write_sparse_model` call can emit the registered COLMAP sparse
model to a fresh directory after checking more than 10 GiB free on both the
source and output volumes. It leaves partial output for inspection if a write
fails. Neither tests nor any checkerboard diagnostic call this export path.

COLMAP sparse is the lowest-risk intermediate because PyCOLMAP directly
represents the fixed cameras and verified tracks. OpenMVS can later consume a
COLMAP undistortion/`InterfaceCOLMAP` handoff, but that is a separate dense
integration decision. The current classical runner explicitly rejects
`--pose-mask-dir` combined with `--sparse-model`, so its existing dense path
must not be assumed to apply masks to this model. MVE would require a new
adapter and is not used here.

Before any live integration: freeze exact TRAIN photo/mask/track manifests
and hashes, verify the orbit diagnostic exports `camera_from_board` with the
stated convention and calibrated zero-distortion K, then obtain explicit
approval for a fresh output path, volume checks, time/output cap, and a
read-only review of the resulting sparse reprojection/track geometry. Do not
use Berkeley-provided poses, scanner, depth, or held-out frames as inputs.
