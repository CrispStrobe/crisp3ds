# Recovered image-only YCB camera binding

`scripts/object_motion/recovered_camera_reference.py` creates a **post hoc,
camera-only** proper Sim(3) from the delayed self-calibrated image-only recovery
005 world frame to the per-angle Berkeley table frame. It uses matching NP3
camera names and centers; neither Google reference geometry nor observed sensor
depth enters the fit. It is an evaluation coordinate binding, not a pose seed
for recovery or a mesh registration.

The input gate binds the exact completed 005 report (SHA-256
`80241c6c8d72097ce38a9a046aadfa495c1027bbfbd05a6a9480ad09b02c7e10`)
to completed fixed-initial-intrinsics recovery 004 (SHA-256
`9a7ddeac8b99ab3ec41a512aec0b852b555aa6176860d2047460e3f87f910b63`).
It checks 004's original 60 photo hashes against both the source photo
manifest and photo-derived preparation manifest; verifies 004, the preserved
input copy, and 005 refined camera/pose/point binaries; and rehashes all 61
Berkeley calibration/pose metadata members against the prior camera-reference
report. The sensor-depth report verifier now has a distinct recovered schema
branch, with these same exact 005/004/photo/model and metadata bindings. The
existing original and calibrated branches retain their prior checks.
For recovered native rough meshes, the sensor evaluator also requires the
producer's completed undistort-stage hashes to match all three current
`dense/sparse` binaries and binds its source-model hashes and 005/004 report
lineage. Every candidate now snapshots all three dense/sparse binaries before
and after ray scoring; the pre-existing camera-pose equality gate remains.

The final bounded camera-only output, after adding post-fit rehashes of the
actual binaries and metadata, is
`build-opencv/object-motion/recovered-camera-reference-002/report.json`
(SHA-256 `67a1f52556ef954e82970e6bcb5349d2a9c7af9e56ba9d63d72caa96ca68f89d`,
21,367 bytes). The earlier 001 remains as pre-postcheck diagnostic evidence,
not the chosen report for sensor scoring. The final output binds 60 named
views. Center-fit RMS is **0.0080693** in the
Berkeley table units (metres under the current metadata convention), and
rotation-error p95 is **2.84445°**. This is a camera agreement diagnostic;
the fit does not establish the accuracy of a reconstructed surface.

The Berkeley table-pose refinement state remains unknown. NP3 RGB-to-depth
projection, the assumed rectified depth pinhole model, the chosen IR depth
scale, photo-mask support, visibility and reference-pose errors still affect
any later sensor residual. A global camera-center Sim(3) conflates camera and
reference-pose error with geometry; it is not physical metrology or a Google
scanner registration. No recovered-mesh sensor score was run by this adapter.
