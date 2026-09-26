# Object-motion sparse experiment

This experiment asks whether 60 original YCB `003_cracker_box` turntable photos support a coherent *object-frame* sparse reconstruction when one physical NP3 camera stays fixed. The 60 JPEGs are the Berkeley RGB-D camera's 0°–354° sequence in 6° steps. Reconstruction uses only those RGB photos. The Google scanner mesh, supplied depth, poses, and dataset masks do not enter preparation, matching, or mapping. The raw and foreground trials share one `SIMPLE_RADIAL` intrinsic camera and the same SIFT, sequential matching, and incremental mapping options. See [the pinned photo provenance](OBJECT-DATASETS.md).

The fixed camera sees at least three different motion groups: a stationary light tent, a turntable checkerboard that moves with the box, and the box itself. Ordinary SfM treats a consistently matched group as static and can return an apparently registered model in *that group's* frame. A registration count therefore cannot establish that the box was reconstructed. This experiment measures photo-derived pose-support track coverage, mixed/outside tracks, camera intrinsics, and camera-center movement in the estimated object frame. It does not claim physical scale or mesh quality.

## Reproduce

From the repository root, after [the 60-photo dataset preparation](OBJECT-DATASETS.md):

```sh
python3 -m unittest scripts.object_motion.test_object_motion -v
python3 scripts/object_motion/prepare.py --output build-opencv/object-motion/prepare-001
.local-tools/colmap-sparse/venv/bin/python scripts/object_motion/run.py \
  --manifest build-opencv/object-motion/prepare-001/manifest.json \
  --output build-opencv/object-motion --arm both --timeout-seconds 590
.local-tools/colmap-sparse/venv/bin/python scripts/object_motion/diagnose.py \
  --model build-opencv/object-motion/foreground/models/0 \
  --manifest build-opencv/object-motion/prepare-001/manifest.json \
  --output build-opencv/object-motion/foreground/orbit-diagnostics.json
```

The preparation requires Pillow in its Python interpreter; the local PyCOLMAP 3.11.1 virtual environment runs SfM and diagnostics. Each trial is a fresh directory with a 590-second wall deadline, 10 GiB free-space floor, 1 GiB output cap, and bounded 4 MiB combined log. The wrapper returns nonzero for process failure, timeout, disk-floor stop, no sparse model, or fewer than three registered cameras. It does not overwrite existing trial directories. Large generated data lives in ignored `build-opencv/object-motion/`. No package is installed globally.

`prepare.py` verifies all 60 bytes and hashes against the local pinned YCB manifest. Its fixed search window (x 390–850, y 200–700) finds reddish packaging pixels, spans them row by row, and bridges narrow pale-label gaps. It saves each white=eligible, black=excluded PNG in COLMAP's `image.jpg.png` convention and records scanlines, hash, source photo, and angle. These are **coarse pose-support masks**, derived from RGB photos and inspected on sample angles 0°, 90°, 180°, 270°. They are not accurate object silhouettes, do not define mesh clipping, and cannot establish object-only reconstruction. A checkerboard patch can still fall inside a scanline hull, especially near 180°. A future mesh mask needs a separately documented, stricter silhouette contract and evaluation.

The raw arm extracts SIFT from each whole original photo. The foreground arm gives COLMAP the image-derived mask during feature extraction. Both use `CameraMode.SINGLE` with an uncalibrated initial focal factor of 1.2, 1200-pixel maximum SIFT image size, 1800 features per view, CPU two-thread extraction/matching/mapping, and sequential matching with eight neighboring frames. Mapping is incremental, one model maximum. This is a bounded *pose matching* comparison, not a proof that image masking improves final geometry. Sequential matching omits the 354°/0° wrap pair, and distant repeated views are not exhaustively matched; both limits are recorded in the run options.

`summary.json` includes every registered image's source angle, number of triangulated observations inside and outside the same photo-derived support mask, total all-inside/mixed/all-outside 3D tracks, focal divided by image width, principal-point bounds, and estimated camera centers. Focal `0.3–3` image widths is only a broad sanity flag. Center spread and neighboring 6° center steps use arbitrary similarity-scale units; a positive spread may be compatible with an object-frame orbit, but no supplied pose is used as ground truth. The separate `orbit-diagnostics.json` projects the center path to its PCA plane and reports angular sweep, directional consistency, radius variation, and loop closure. This is a path-consistency check, not calibrated pose truth. All-inside tracks can still originate from checkerboard leakage or accidental matches, and low support can result from conservative masks. The sparse model's points and cameras are *pose diagnostics*, not a watertight object or quality score.

## Measured local result, 27 September 2026

Both arms used the exact same 60 original NP3 JPEG bytes. The raw arm took 178.44 s. Mapping wrote a binary model, then the first diagnostic reporter exited on a PyCOLMAP API mistake (`Image.num_points3D` is a property). The original `raw/trial_status.json` remains `failed`. A separately recorded diagnostic continuation read only the completed binary model, verified the original input manifest, and sealed SHA-256 hashes for `cameras.bin`, `images.bin`, and `points3D.bin` in `raw/summary.json`. This was **not a successful 60-view sparse reconstruction**: only 2/60 photos registered, with 297 points. Of its tracks, 54 are wholly within the carton support mask, 6 mixed, and 237 wholly outside; 114 of 594 track observations fall inside. Its fitted focal length is 4,584.95 px (3.582 image widths), beyond the broad plausibility flag, with radial parameter −5.3701. This indicates the raw model mostly follows unrelated scene evidence and is unsuitable for an object pipeline.

The foreground pose-support arm took 96.19 s and registered **60/60** photos, covering every 6° source angle. It produced 3,939 sparse points: 3,909 all-inside tracks, 29 mixed, and 1 all-outside. Of 24,396 track observations, 24,362 are inside the coarse support mask. Its shared focal length is 1,077.58 px (0.842 image widths), principal point (640, 512) lies inside the 1280×1024 photos, and radial parameter is −0.00723. The center path's PCA plane contains 99.9977% of center variance; its signed angular sweep is −356.54°, all 59 consecutive steps progress in one direction, median step is −6.12°, and projected radius coefficient of variation is 0.00617. The first/last center distance is 0.568 times the median neighbor step. These figures are consistent with an estimated object-frame orbit, subject to the mask and SfM assumptions; they are not comparison with supplied poses or proof of accurate object geometry.

For context, raw extraction stored 157,019 keypoints and 197,144 putative matched feature pairs across 444 nonempty pairs. Foreground extraction stored 40,600 keypoints and 55,325 putative matches across the same 444 nonempty sequential pairs. That reduction is the intended intervention. The foreground model's sealed binary hashes, photo hashes, PyCOLMAP version, and exact options are in `foreground/summary.json` and `foreground/provenance.json`; its orbit report is `foreground/orbit-diagnostics.json`. The raw counterpart retains its failed trial status and an explicit `recovery.json`. The successful pose support makes the foreground model eligible for a separately labeled dense backend trial; no mesh silhouette, dense coverage, scale, or shape score has been established by this sparse experiment.

Tests: six preparation/wrapper contract tests pass under system Python, and four PyCOLMAP diagnostic/interface tests pass under the local virtual environment, including read-only checks against both saved real models. No global installation or source-image mutation was performed.

## OpenMVS camera conversion check

After the classical adapter imported the sealed foreground model and undistorted all 60 photos, its `scene.mvs` was reverse-exported by OpenMVS 2.4.0's `InterfaceCOLMAP` to a fresh ignored TXT reconstruction. This checks the live Mac M1 camera conversion only. It does not inspect shape or assert that sparse point IDs survive export. From the repository root, when `classical-ycb-foreground-002/scene.mvs` exists:

```sh
mkdir -p build-opencv/object-motion/roundtrip-001
.local-tools/classical-backend/bin/InterfaceCOLMAP \
  -i "$PWD/build-opencv/classical-ycb-foreground-002/scene.mvs" \
  -o "$PWD/build-opencv/object-motion/roundtrip-001/export" \
  --working-folder "$PWD/build-opencv/object-motion/roundtrip-001" \
  --max-threads 2 --binary 0
.local-tools/colmap-sparse/venv/bin/python scripts/object_motion/check_interface.py \
  --source-model build-opencv/classical-ycb-foreground-002/dense/sparse \
  --exported-model build-opencv/object-motion/roundtrip-001/export/sparse \
  --output build-opencv/object-motion/roundtrip-001/report.json
```

The [local roundtrip report](../build-opencv/object-motion/roundtrip-001/report.json) records 60 matching camera basenames and 3,939 sparse points in both representations. Across 3,000 projections of common source-world points, the maximum pixel difference was 0.00413 px. Maximum intrinsic change was 0.00376 px, center change 1.32×10⁻⁵ arbitrary units, and rotation-matrix Frobenius change 3.66×10⁻⁶; all passed declared 0.01 px projection/intrinsic and 10⁻⁴ center/rotation bounds. The TXT export rounds decimal camera parameters, explaining the small differences. The report hashes both sparse model file sets. It establishes adapter coordinate consistency at these bounds on this local M1 run, without validating dense geometry or physical scale.
