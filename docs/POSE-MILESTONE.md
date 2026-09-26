# R01: calibrated image pose milestone

Implement a real image-to-pose path using saved photographs and already known camera calibration. This milestone does not calibrate the camera from a chessboard image set, reconstruct surfaces, or control the rig.

## Scoped work and supervision

| Task | Owner | Acceptance |
| --- | --- | --- |
| R01a | Sol dependency agent | Optional exact-source OpenCV CPU build, selected codecs/modules, native license metadata and configuration checks; default foundation stays dependency-free |
| R01b | Sol core agent | Additive project validation, calibrated ArUco image detection, object-to-camera pose/report API and CLI, explicit per-image failures |
| R01c | Sol app agent | Full marker geometry and calibration-model editor/validation; existing project metadata stays loadable; setup status does not imply executed pose |
| R01d | Supervising agent | Review cross-validator agreement, pose conventions, actual source/build options, generated image and CLI output; run independent integration checks and update delivery status |

## Contract decisions

Continue schemaVersion 1 with optional fields. Legacy summaries remain valid project records; pose execution needs the complete geometry. Add `calibration.distortionModel: "opencv-radtan"`, with 4, 5 or 8 OpenCV-order radial/tangential coefficients. Unsupported models must not be silently interpreted. Pose estimation requires exact calibrated image dimensions and does not resize images.

Add `board.dictionary: "DICT_4X4_50"` and `board.markers: [{id,cornersMm}]`. Initial execution supports ArUco only; AprilTag summary metadata remains loadable. Marker IDs are unique integers within the dictionary. Each marker supplies four finite measured board-space corners in decoded top-left, top-right, bottom-right, bottom-left order. Initial boards are planar (z=0), with square markers whose size agrees with markerSizeMm. In-plane rotation is valid. Printed grid coordinates use x right and y down; z follows the right-hand rule. A grid origin can be the top-left outer marker corner; it need not be the rotation axis.

Report pose in the established convention `Xcamera = R * Xboard + t`, with row-major R, t in millimetres, reprojection RMS/max in pixels, observed marker IDs and explicit image failures. Report provenance must identify the implementation and settings. The board frame moves with the object; the camera remains fixed relative to the room. Its pose relative to that moving frame changes per image.

## Required evidence

1. Render an actual marker-board image from known camera intrinsics and a tilted known pose. Detect the rendered markers and estimate the pose through the production path. Check rotation, metric translation and reprojection tolerances.
2. Exercise errors: absent markers, invalid layout, wrong dimensions, missing/unreadable images, path escape including symlinks, and insufficient/ambiguous observations where relevant.
3. Run both the dependency-free and OpenCV-enabled build tests. Check web/native manifest agreement for the additive fields and preserve old fixtures.
4. Confirm the CLI reports real pose results independently of the GUI. Full reconstruction must remain unavailable.
5. Record that synthetic success is not real-capture validation. A measured turntable dataset must pass before R01 is considered fully validated.

## Dense-backend freedom

No pose or board contract should depend on MVE. The next dense-backend experiment will compare practical candidates, including an OpenCV calibrated pairwise stereo baseline, against actual quality and masking needs. See [BACKEND-EVALUATION.md](BACKEND-EVALUATION.md).
