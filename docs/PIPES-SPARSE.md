# ETH3D pipes four-view fixed-camera sparse baseline

`scripts/pipes_sparse/run.py` is a small research baseline for the *training* views `DSC_0634.JPG` through `DSC_0637.JPG`. It uses supplied ETH3D `PINHOLE` intrinsics and world-to-camera poses as fixed inputs. The laser scan, scan alignment, and supplied `points3D.txt` never enter the reconstruction. Consequently, this lane tests matching and triangulation conditional on supplied cameras; it does not measure camera estimation or independent geometric accuracy.

The input is `build-opencv/pipes-prepare/run-001/prepare-metadata.json`, produced by the validated extraction step. Its `images` list covers all 14 scene images; the baseline sorts by image name and takes the first four, requiring the exact four names above. Each selected JPEG is read through its allowlisted `pipes/images/dslr_images_undistorted/<name>` path and checked against both the record digest and the prepared file digest map. The code reads no extracted calibration observation rows, SfM points, PLY, or MLP.

The frozen settings are SIFT at most 4096 keypoints per resized image, strict Euclidean ratio `<0.8` in **both** nearest-neighbor directions, and supplied-pose square-root Sampson distance `<=2` pixels. All six view pairs are matched. Full connected components with repeated image names are rejected. Remaining tracks use the audited `scripts.fixed_camera.run.reconstruct` world-ray intersection: maximum acute ray parallax at least 1 degree, finite normal-matrix condition at most `1e8`, positive depth and reprojection error at most 4 pixels in *every* supporting view. Counts for accepted tracks supported by three or four views are reported separately. Parameters are fixed before any laser scoring.

Each input is resized to a maximum long edge of 1024 with its native aspect ratio. The actual integer output width and height set separate `sx` and `sy` intrinsic scales. The image coordinates are in ETH3D/COLMAP's pixel-edge frame: the first pixel center is `(0.5, 0.5)`. OpenCV `KeyPoint.pt` is converted by adding 0.5 before epipolar geometry and triangulation. No offset is added to the scaled intrinsics; OpenCV resize maps pixel edges by the actual scale. The report records source and resized camera parameters, actual scales, poses, image hashes, package versions, binary/source hashes, every screen threshold, pair counts, component rejection counts, and the separately counted three-view tracks.

`points.json` contains all accepted world-frame `xyz` values and their supporting resized-image SIFT feature coordinates, per-observation reprojection errors, parallax, and condition values. The inherited `original_feature_id` field is the index of the resized-image SIFT feature in this baseline, **not** an ETH3D supplied SfM observation ID. `report.json` includes accepted observations for each of the four views, including zero counts, so weak support from the fourth view is explicit. `report.json` and `points.json` are independent of any later scan scorer. Any distance to the reference scan must be computed in a separately recorded step, without feeding it back into this baseline or changing the thresholds.

Use `/usr/local/bin/python3` (tested Python 3.11.1, OpenCV 4.10.0, NumPy 1.26.4, Pillow 9.4.0) and a fresh output directory. The launch command is `/usr/local/bin/python3 -m scripts.pipes_sparse.run --data build-opencv/pipes-prepare/run-001 --output <fresh-output>`. Its parent process invokes `scripts.research_job.guard.run_child` with `timeout_seconds=300`, `max_output_bytes=128*1024**2`, `reserve_bytes=10*1024**3`, and `max_log_bytes=10*1024**2`. The guard creates the output and task-local temporary directory. This macOS GCD OpenCV build ignores positive thread requests and reports eight workers; the child calls `setNumThreads(0)` to disable that pool, requires `getNumThreads()==1`, and records the requested API value and actual count. Numerical library thread caps are set to two. The runner checks the 10 GiB floor and its own 128 MiB JSON cap. Do not launch before the prepared data and launch plan have been reviewed.

The local checks are:

```sh
/usr/local/bin/python3 -m unittest scripts.pipes_sparse.test_run -v
```

They cover actual width/height scales and pixel-center conversion with a projected point, two-way ratio ambiguity, component conflict rejection and fixed-ray reconstruction, and the four-image input allowlist in the presence of unused scan and SfM files.

## Frozen local run

[`build-opencv/pipes-sparse/run-003/report.json`](../build-opencv/pipes-sparse/run-003/report.json) and [`points.json`](../build-opencv/pipes-sparse/run-003/points.json) are the first completed sparse outputs. The guarded run succeeded in 2.817 seconds; the child reported 327,974,912 bytes peak RSS and one actual OpenCV worker after requesting API value 0 (GCD pool disabled). It found 341 epipolar-gated pair edges, 262 track components, and accepted 258 points with 568 observations. Of the accepted points, 206 have two-view support and 52 have three-view support. Four components failed the fixed geometry screens: three high reprojection errors and one nonpositive depth. An independent NumPy forward-projection check of all 568 accepted observations found positive depth and a maximum error of 1.682379729 pixels, agreeing with the stored reprojections to `1e-8`.

| Selected view | Accepted observations | Pairwise epipolar edges involving this view |
| --- | ---: | ---: |
| `DSC_0634.JPG` | 193 | 223 |
| `DSC_0635.JPG` | 249 | 304 |
| `DSC_0636.JPG` | 126 | 155 |
| `DSC_0637.JPG` | 0 | 0 |

The frozen fourth view contributed no geometry-consistent match to this four-view baseline. This is an observed support limitation; no images were substituted or thresholds changed. `run-001` and `run-002` remain preserved as pre-SIFT failures of the OpenCV GCD thread-count check (0.322 and 0.220 seconds respectively); neither produced sparse points. The revised thread setup disables that backend's worker pool and verifies its reported count of one. Matching and triangulation settings stayed fixed across these attempts.

Sealed SHA-256: `points.json` `49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0`; `report.json` `00a71eac608d806a0aff5ebb1aad64a45edc47ed3d956f65c8461cee6e8fc82e`. Reference-scan scoring belongs to a separate job and is not part of these reconstruction results.
