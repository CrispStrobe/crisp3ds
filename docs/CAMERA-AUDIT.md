# Independent camera convention audit of the pinned tree scene

The read-only checker at `scripts/camera_audit/check.py` compared the ten selected entries in `.local-tools/test-data/tree-subset/manifest.json` with the pinned COLMAP camera and image files, the JPEG headers, the resized PNG headers, `views.tsv`, `conversion.json`, and each MVE `meta.ini`. Its final fresh report is `build-opencv/tree-refine/camera-audit-v3.json`. The report includes SHA-256 hashes of all 38 input files it read and is about 8 KiB. Free disk space after the run was about 28.7 GiB, above the 10 GiB reserve. The checker caps input and report sizes, enforces the reserve, and refuses optimized Python because assertions are part of its validation.

The checker uses quaternion vector rotation via cross products, independently of the importer's explicit matrix formula. Synthetic checks cover a nonidentity quaternion and translation, `C = -Rᵀt`, different x/y resize factors with a noncentered principal point, a projected optical-axis point, epipolar geometry, analytic horizontal rectification, and deliberately mismatched image/name and image/camera IDs. No source or scene files were changed.

## Proven results

- All ten selected image IDs and names map to the expected COLMAP image and camera records. Their JPEG dimensions agree with the selected camera dimensions, and their file sizes and SHA-256 hashes agree with the pinned manifest. The source camera IDs happen to equal the image IDs for these ten entries; this is checked as a record association, not assumed by the algebra.
- All ten TSV rows use the expected row-major world-to-camera rotation, COLMAP translation, image order, output PNG dimensions, and OpenCV intrinsics. The maximum absolute difference in those four numeric groups was zero at the available text precision. The MVE rotation and translation differ from the TSV values by at most `4.98e-13`, consistent with formatting. The MVE focal length, pixel aspect, and normalized principal point match the converted intrinsics.
- The resize mapping is `fx' = fx × W'/W`, `fy' = fy × H'/H`, `cx' = cx × W'/W − 0.5`, `cy' = cy × H'/H − 0.5`. This accounts for COLMAP's half-pixel origin and OpenCV's integer-centered pixel coordinates under a center-aware resize. For the pinned 768×512 images, the resulting principal point is `(383.5, 255.5)` in OpenCV coordinates and `(0.5, 0.5)` in MVE normalized coordinates.
- For each of 45 camera pairs, the relative pose `R_b R_aᵀ`, `t_b − R_b R_aᵀ t_a` agrees algebraically with the two camera centers. Projected synthetic world points satisfy the fundamental-matrix epipolar constraint; the maximum normalized algebraic residual was `8.20e-15`. An independently constructed horizontal rectified frame gives equal vertical coordinates for the two projections. The pair oracle's use of OpenCV `stereoRectify(R,T)` and of each `P` matrix's left 3×3 block when reprojecting a point already in its own camera frame is consistent with this geometry. The check does not reproduce OpenCV's chosen rectification output pixel for pixel.

These results establish that the local COLMAP-to-OpenCV-to-MVE conversion and the pair geometry are internally consistent. They do not validate the upstream XMP-to-COLMAP axis choice or the physical camera poses. The pinned COLMAP `images.txt` has no measured `POINTS2D` observations for these views, and the local subset has no `points3D` file. The large multiview residuals reported in [TREE-TRACKS.md](TREE-TRACKS.md) can still reflect incorrect feature associations, inaccurate supplied poses, scene motion, or a combination. No coordinate flip follows from this audit.

## Reproduce

Use a fresh report path; the checker refuses to overwrite an existing report. It uses only the Python standard library and reads the already pinned local files.

```sh
python3 scripts/camera_audit/check.py --self-test
python3 scripts/camera_audit/check.py \
  --dataset .local-tools/test-data/tree-subset \
  --scene build-opencv/tree-refine/baseline/scene \
  --report build-opencv/tree-refine/new-camera-audit.json
```
