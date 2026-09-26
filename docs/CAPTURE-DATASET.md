# First reconstruction dataset

The first engine acceptance test needs photographs of a measured object and the calibration for that exact camera/lens/focus/image size. Synthetic geometry validates conventions and arithmetic; it does not validate dense reconstruction quality.

Use diffuse, steady lighting, fixed exposure/white balance/focus, an object that does not move relative to the board, and a camera that remains fixed during each ring. Begin with a textured matte object of known dimensions; add glossy LEGO and low-texture cases after the baseline works. Save original still images without editing or resizing.

The known-size marker board must rotate rigidly with the object and remain sufficiently visible for per-image pose estimation. Record the marker dictionary/family, IDs, measured corner coordinates, board layout, and units. Marker size alone is not enough to describe a multi-marker board. The current optional pose backend accepts ArUco DICT_4X4_50, planar square markers with explicit corners, and known OpenCV radial-tangential calibration; see [POSE-MILESTONE.md](POSE-MILESTONE.md).

Capture overlapping views around the full turn, recording commanded movement separately from observed poses. Use a second camera elevation for better top coverage. Do not assume the underside is observed. Initial planning target: 36–64 images, adjusted after examining overlap and blur; this is a capture suggestion, not an accuracy guarantee. The current sparse baseline accepts at most 64 views and 256 million aggregate pixels per project, pairs only nearby manifest entries, and does not automatically downsample. Keep entries in acquisition order and choose calibrated image dimensions within that budget.

Keep a calibration image set, object image set, object masks, measured reference dimensions, and a short description of the rig. Masks must exclude the marker board and stationary background. Do not substitute turntable motor angles for measured camera poses.

Acceptance record: pose reprojection errors in pixels; number of accepted views and triangulated tracks; visible-surface coverage and missing regions; dimensions in millimetres compared with measurements; runtime, peak memory and repeatability. Choose numerical acceptance thresholds against the intended object size and measurement needs before judging the backend. A watertight interpolated mesh is not proof that its surface was observed.

## Quality-improvement capture protocol

Prepare separate development and acceptance objects before tuning. Suggested cases are a matte textured object with independently measured dimensions, a glossy LEGO assembly, a low-texture surface, and a thin feature/depth discontinuity. These are proposed captures, not datasets already present in this workspace.

For each, retain a capture manifest with the camera/lens and focus setting, calibrated resolution, exposure/white-balance settings, marker-board dimensions, intended image order, camera elevation, lighting arrangement and observed capture failures. Record at least three independently measurable lengths with measurement uncertainty and the location of each measurement. Do not infer dimensional truth from nominal LEGO dimensions alone if the assembled object can have gaps or flex.

Repeat the scan independently to expose setup/pose repeatability rather than rerunning the same photographs. Assess the surface only where the reference and cameras actually observe it; list top/underside gaps separately. Retain the unfilled point cloud and observation counts next to any smoothed or watertight output.

For the first acceptance run, choose physical accuracy/completeness and end-to-end resource thresholds in advance. Error in stereo pixels is a useful component diagnostic but does not replace dimensional error in millimetres. The software team has requested the user's object-size and detail requirements; physical pass thresholds remain pending rather than being retrofitted to the first output.

No real capture dataset was supplied. R01 and R02 have synthetic-image implementation evidence; real-capture acceptance and R03–R04 remain open. The generated test board images must not be used as camera calibration or as evidence of physical scanner accuracy. Sparse masks must be matching-size, 8-bit grayscale PNGs containing only 0 and 255; see [SPARSE-MILESTONE.md](SPARSE-MILESTONE.md).
