# YCB cracker-box camera-calibration ablation

This is a **diagnostic**, not an image-only production result. Sixty foreground-supported `NP3_000..354` RGB photos supply all keypoints, descriptors, raw correspondences, and estimated poses. One arm additionally supplies Berkeley's independent NP3 RGB **intrinsics and distortion**, but never its camera poses, depth, object model, or reference mesh to SfM. The calibrated lane must therefore be labeled *oracle-assisted intrinsics; image-estimated poses*. The Google 64k mesh frame is not established by this metadata; see [YCB-REFERENCE-FRAMES.md](YCB-REFERENCE-FRAMES.md).

## Frozen protocol

The source is the 60-camera image-derived foreground SfM database at `build-opencv/object-motion/foreground/database.db` (SHA-256 `a36cc364bf9b7d7a8357c12ee3619002dd2c85696f09b5921d1f9e926f2e3ca7`). Its photo manifest SHA-256 is `0b78469039d11516d0ac96b642553099267cd97c9a6439a1ec4b2fcb133615bd`. Both arms copy that database, retain the same 60 image names and 444 raw-match pairs, delete old two-view geometries, and estimate each pair's geometry serially from the cached keypoint XY and raw match indices. Feature extraction and descriptor matching are not repeated. Before/after feature and raw-match table SHA-256 digests are equal in the sealed producer reports. These are *training correspondences*, not a held-out test set.

Both fresh `002` arms use PyCOLMAP 3.11.1, fixed seed `20260927`, the preselected `NP3_192.jpg` / `NP3_162.jpg` initialization pair, one model, mapper two threads, 590-second / 100-MiB per-arm ceilings, and a 10-GiB free-disk floor. The image-only arm retains the original single `SIMPLE_RADIAL` camera. The calibrated arm changes only the shared camera to fixed `FULL_OPENCV` and re-estimates two-view geometries after that change. All global/local bundle-adjustment intrinsic refinement and absolute-pose intrinsic refinement are disabled for the calibrated arm. Its camera vector is `[fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6]`, with the supplied nonzero `k3=-0.1771694841` preserved and `k4..6=0`; it is **not** an eight-parameter OPENCV approximation. The calibration H5 SHA-256 is `15f724c10131129183e97cb3ae6fdb3f13ff7ad8c5a50812313f06652a7f6c07`.

The main calibrated arm preselects `cx,cy + 0.5`, assuming the supplied OpenCV calibration uses integer pixel centers whereas COLMAP uses half-pixel centers. The Brown–Conrady formula was checked numerically against pinned PyCOLMAP, but the dataset's pixel-origin convention is not independently documented, so that `+0.5` remains an explicit assumption. No unshifted sensitivity arm or mesh-score-based selection was run. Input masks are coarse image-derived **pose support**, not exact object silhouettes.

## Observed sparse outcomes

| Attempt | Two-view re-verification | Cameras / 60 | Sparse points | Interpretation |
| --- | --- | ---: | ---: | --- |
| `001/image_only_reverified` | PyCOLMAP `verify_matches` worker | 60 | 3,937 | Contextual image-only replay, not the matched serial-method control. |
| `001/calibrated_shifted` | PyCOLMAP `verify_matches` worker | — | — | Native SIGSEGV during verification; no model. Retained log and DB. Eight SIFT-worker threads appeared despite the requested mapper/OMP limits. One isolated pair later succeeded, which does **not** establish the crash's cause. |
| `002/image_only_reverified` | Serial per-pair estimator | 2 | 126 | Frozen paired-control arm failed the ≥3-camera sparse gate. No seed or threshold retuning. |
| `002/calibrated_shifted` | Same serial per-pair estimator | **60** | **3,942** | Calibrated-intrinsics-only sparse producer passed the artifact gate. |

The serial verifier produced 444 geometry rows for each `002` arm. In the image-only arm these are 315 `UNCALIBRATED`, 128 `PLANAR_OR_PANORAMIC`, and one zero-inlier row; the calibrated arm has 280 `CALIBRATED`, 28 `UNCALIBRATED`, 135 `PLANAR_OR_PANORAMIC`, and one zero-inlier row. This is evidence that changed intrinsics affect two-view geometry and registration on these particular cached matches, **not** proof that calibrated reconstruction geometry is more accurate. The `001` image-only worker completed 60/60, so the `002` image-only failure also shows method sensitivity; its result cannot be used as a clean calibration-only causal contrast. No paired final-mesh quality claim follows from `002`.

The calibrated model's 24,413 *training* reprojection observations have median 0.357 px and p95 1.634 px; its 3,942 tracks have median maximum-pair parallax 20.0°. These statistics describe internal fit/coverage, not held-out accuracy. After a proper camera-center Sim(3) to Berkeley's independently supplied pose metadata, center-fit RMS is 0.00848 in the metadata's native units, leave-one-out RMS 0.00879, and orientation-error p95 1.84°. The contextual `001` image-only replay had center-fit RMS 0.00803 and orientation-error p95 2.85° under a *different* verifier. These mixed changes do not certify either camera or mesh as ground truth.

## Artifacts and replay

The sealed calibrated report is `build-opencv/object-motion/calibration-ablation-002/calibrated_shifted/report.json`, with binary model at its `model_dir`; `cameras.bin`, `images.bin`, and `points3D.bin` SHA-256 values are respectively `c7913b3ffa0e6bdef9d37a360adc43367ca43f131971de02c6e38a8920a1cf54`, `21c8aa1740287bcbfecec87fceca6d8957dd1e315178be210cecadf8862ab788`, and `8f4c8ca118d9f8677822e9c168c93c337bf8b3feac3b1d3d0c664ac78c4a8721`. Its schema is `ycb_calibrated_intrinsics_sfm_v1`, lane `calibrated_intrinsics_only`; the report also binds every photo, source DB, calibration, model file, runner, PyCOLMAP binary, table digest, options, and camera-oracle metadata. The failed/insufficient attempts retain `trial.log` and `trial_status.json` in their respective directories. An external-model consumer must verify the report and model hashes before use, and must retain the oracle-assisted lane label.

From the repository root, with the pinned local environment and preexisting data:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.object_motion.test_calibration_ablation
.local-tools/colmap-sparse/venv/bin/python -m scripts.object_motion.calibration_ablation --arm calibrated_shifted --output build-opencv/object-motion/calibration-ablation-REPLAY
```

The runner requires a fresh arm output directory and does not overwrite these retained trials. A successful registration count is only a sparse feasibility gate; it is neither surface accuracy nor evidence of a general foreground-segmentation solution.
