# Berkeley NP3 depth-to-RGB projection check

**Gate result:** three existing cracker-box RGB/depth pairs at 0°, 120°, and 240° have plausible depth-edge alignment under one frozen sensor calibration chain. This supports an **assumption-qualified, sensor-conditioned diagnostic**, not certified depth metrology, a Google-mesh alignment, or an image-only reconstruction result. No reconstructed mesh was scored or changed here.

The inspected [report](../build-opencv/object-motion/berkeley-projection-check-002/report.json) has schema `berkeley_rgb_depth_projection_check_v1`, status `assumption_qualified_projection_plausible`, and `pixel_mapping_validated=true` **only in that limited sense**. It binds the already pinned Berkeley archive SHA-256 `15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5`, calibration H5 SHA-256 `15f724c10131129183e97cb3ae6fdb3f13ff7ad8c5a50812313f06652a7f6c07`, the three exact depth/RGB/photo-derived-pose-mask file hashes, the helper code hash, and the numeric matrices. The earlier `001` report and overlays remain intact; `002` additionally shows projected depth edges *outside* the pose mask.

## Depth grid, units, and frame chain

Each `/depth` dataset is 480×640 little-endian uint16 with zero meaning no sample. The [official BigBIRD data description](https://rll.berkeley.edu/bigbird/access.html) calls these *raw depth maps in 100 µm units*. The three frames have 249,111 / 248,130 / 249,295 valid samples; median corrected depths are 0.811 / 0.820 / 0.822 m. The local Berkeley calibration has `NP3_ir_depth_scale=1.0016965552242483` and `NP3_ir_depth_bias=0`, while `NP3_depth_scale=1` and `NP3_depth_bias=0`. We use `z_m = raw × 0.0001 × NP3_ir_depth_scale` as an explicit interpretation of those supplied fields. The original YCB-hosted point-cloud script was not reachable during this check, so the choice of IR rather than generic depth scale is not independently established by a primary processing-code source. Since both biases are zero here, bias-unit semantics are **not** tested. The generic `depth_scale` was not applied.

The 640×480 grid is **assumed** to support pinhole depth rays: `X_depth = z[(u-cx_d)/fx_d, (v-cy_d)/fy_d, 1]` with `NP3_depth_K`, without applying `NP3_ir_d`. The [Berkeley BigBIRD calibration thesis, §4.2.3](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-142.pdf) establishes the depth intrinsic matrix's convolution-window offset, **not by itself that these depth arrays are distortion-free**. Locally `fx_depth=568.644` versus `fx_IR/2=568.644`, but `cx_depth=306.191` versus `cx_IR/2=310.991`, the predicted ≈−4.8 px shift. `NP3_ir_d` is appreciably nonzero (`[0.0828,-0.5686,0.00733,0.00136,2.15246]`); whether depth backprojection should undo it remains unresolved. A full independent factory calibration/rectification audit was not available. The frozen pinhole diagnostic must retain this systematic-error caveat; plausible overlays do not resolve it.

The 3-D point goes from the depth/IR camera to NP3 RGB through

`H_RGB_from_IR = H_NP3_from_NP5 × inverse(H_NP3_ir_from_NP5)`.

This yields ≈−25.87 mm x translation in the supplied metre-scale rig frame and a proper near-identity rotation. The RGB point is projected with the full five-coefficient `NP3_rgb_d` Brown–Conrady model and `NP3_rgb_K` onto the 1280×1024 distorted RGB photo. Depth pixels `(u,v)` and RGB pixels are treated as OpenCV integer-centered samples. Our 0.5-pixel comparison is a *diagnostic*, not proof of origin convention. The 3-D ray is never backprojected with the RGB K or 1280-pixel IR K.

## Independent pixel sanity check

The helper projects >2 cm valid-neighbor depth discontinuities to each RGB photo. [Overlay 0°](../build-opencv/object-motion/berkeley-projection-check-002/NP3_000_overlay.png), [120°](../build-opencv/object-motion/berkeley-projection-check-002/NP3_120_overlay.png), and [240°](../build-opencv/object-motion/berkeley-projection-check-002/NP3_240_overlay.png) draw **magenta for all projected depth edges** and **green for the subset inside the frozen coarse photo-derived pose mask**. The magenta edges visibly follow the checkerboard perimeter, cracker-box edges, and some fixtures across three distinct object orientations. This is important: a mask-clipped visualization alone could hide a wrong transform. Disocclusion, invalid depth, shadows, Canny texture edges, and low depth resolution cause visible misses; the overlays do not establish subpixel registration.

| Frame | Projected depth edges | Inside pose mask | RGB Canny distance, all edges median / p90 | Mask-clipped median / p90 | +0.5 depth-center clipped median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0° | 1,645 | 194 | 2.87 / 58.56 px | 1.91 / 4.23 px | 2.87 px |
| 120° | 1,425 | 116 | 4.11 / 82.78 px | 0.96 / 5.67 px | 1.16 px |
| 240° | 1,506 | 201 | 3.69 / 71.44 px | 2.32 / 5.06 px | 2.87 px |

The very large all-edge p90 reflects depth discontinuities on weakly textured walls and sensor-invalid regions, not just registration error. The clipped subset is only 8–12% of projected edges, so its smaller distances must **not** be advertised as global reprojection accuracy. Identity-extrinsic and no-RGB-distortion alternatives are logged as negative/sensitivity controls, but no variant was selected using mesh residuals. Five analytic tests cover integer/half-pixel addressing, rigid translation, positive-depth rejection, RGB radial distortion, and valid-neighbor discontinuities; all pass. The observed edge comparison favors integer centers, but texture, sampling, and mask clipping make that assumption-qualified.

## Permitted use and remaining ambiguity

For a separate RGB-D diagnostic, measured depths can be projected to RGB and tested against the existing *photo-only* pose mask, with a predeclared conservative interior erosion and explicit valid/covered-ray denominators. The pose mask is not an exact object silhouette; camera disocclusion and nearby board/background can remain, so report foreground support and missing-ray coverage instead of silently calling it pure object surface. BigBIRD's archived segmentation masks are described as generated from object models and must **not** enter this image-only lane. This check does not license a mesh-to-Google reference alignment or optimization of camera/depth conventions against mesh score.

The [Berkeley thesis, §4.3.2](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-142.pdf) reports ambiguities in solvePnP table poses and subsequent plane equalization. We have not established whether these particular `H_table_from_reference_camera` files represent raw or refined poses, nor measured the residual sensor depth bias. Any global camera-conditioned mesh-vs-depth residual can therefore mix reconstruction geometry, camera pose, calibration, depth noise, and segmentation errors. Do not correct those poses using the evaluated mesh or describe the metric as ground-truth accuracy.

For the calibrated-intrinsics reconstruction specifically, the separate [camera-only alignment report](../build-opencv/object-motion/calibrated-camera-reference-001/report.json) (schema `ycb_calibrated_berkeley_camera_oracle_v1`, SHA-256 `e46f791bb8fdb36891e1f70df4b18608197fc5f656d9ce87f2cf7dfc43ec86ff`) fits a proper named-center Sim(3) from the exact calibrated `002` sparse model, not from any surface. It rehashes that producer's three binary model files and all 61 calibration/pose metadata members; its center-fit RMS is 0.00848 in the Berkeley metadata's native units and orientation-error p95 is 1.84°. A calibrated mesh diagnostic must use this frozen mapping, not the image-only model's different gauge, and must not refit it to depth residuals.

Reproduce the **projection check only**, using the preexisting local data and pinned environment:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.object_motion.test_berkeley_projection_check
.local-tools/colmap-sparse/venv/bin/python -m scripts.object_motion.berkeley_projection_check --output build-opencv/object-motion/berkeley-projection-check-REPLAY
```

The output directory must be fresh. No additional dataset download, model-derived mask, pose-assisted SfM, or mesh score is performed.
