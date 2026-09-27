# Camera-only COLMAP → Nerfstudio bridge

`scripts/neural_bridge/colmap_to_nerfstudio.py` exports reviewed COLMAP
`cameras.txt` / `images.txt` **PINHOLE** or **SIMPLE_PINHOLE** cameras into
Nerfstudio `transforms.json`. It copies no photos or masks, installs no neural
package, trains nothing, and neither estimates nor improves poses. It rejects
distorted models rather than silently dropping their distortion. Every image
must be assigned exactly once to explicit, nonempty, disjoint train,
validation, and test lists. [Nerfstudio's parser](https://docs.nerf.studio/_modules/nerfstudio/data/dataparsers/nerfstudio_dataparser.html)
recognizes `train_filenames`, `val_filenames`, and `test_filenames`; when
explicit split keys are used, the requested split must be present.

COLMAP text stores OpenCV **world-to-camera** quaternion/translation, with
camera +X right, +Y down, +Z forward. The bridge inverts that rigid pose and
right-multiplies `diag(1, −1, −1, 1)` to emit OpenGL **camera-to-world**, +X
right, +Y up, +Z back, in the *unchanged* COLMAP world gauge. This matches
[Nerfstudio's COLMAP exporter](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_utils.py)
and [camera-axis convention](https://docs.nerf.studio/quickstart/data_conventions.html).
No world-axis rotation, centering or scale is baked into the file. For any
subsequent gauge-sensitive comparison, explicitly disable Nerfstudio's
automatic orientation, centering and pose scaling (`orientation_method=none`,
`center_method=none`, `auto_scale_poses=False`, `scale_factor=1`); the JSON
also sets `orientation_override=none`. Merely exporting the file does not
control a future trainer's parser settings.

Quaternions within the narrow accepted norm tolerance are normalized before
conversion. In the live Pipes text, norms range from 0.999999607 to 1.000000705;
using them unnormalized produces differences up to 1.62e-6 in transform entries.
Root's independent check using PyCOLMAP's normalized rotations matches all
14 exported transforms exactly. This is a defined rounding correction, not a
pose refinement or a measured reconstruction improvement.

The bridge uses the COLMAP numeric `cx,cy` without a ±0.5 adjustment, as in
Nerfstudio's own COLMAP exporter. [Nerfstudio's documentation](https://docs.nerf.studio/quickstart/data_conventions.html)
describes a pixel-center convention distinct from OpenCV's; this is an
interop assumption to check if subpixel projection matters, not a claimed
resolution of the convention difference. Analytic tests independently
project a nonidentity known pose through both conventions.

Input names must be source-root-relative POSIX paths with no traversal,
absolute path, ambiguous segment or symlink escape. The generated image and
optional mask paths are relative to the fresh output directory and resolve
to files inside the declared roots. The exporter checks actual decoded
image dimensions; optional masks must exist for **every** frame and be
8-bit one-channel PNGs of matching dimensions with only 0/255 pixels.
The source text, every referenced image/mask, the generated transforms and
bridge code receive SHA-256 provenance. Text inputs are hashed before parsing
and every source is rehashed before writing. The output is limited to 2 MB,
10,000 frames, 64 MB input `images.txt`, bounded decoded raster dimensions,
and a 10 GiB free-disk floor; output must be a fresh directory. The emitted
`camera_model` is global `OPENCV` with zero distortion and complete per-frame
intrinsics, matching Nerfstudio's one-global-model rule.

`--provenance supplied|estimated` is mandatory. The label reports **camera
source**, not a quality grade. With estimated cameras, an all-image SfM run
may already have used test/validation photos to estimate those poses: image
lists alone do not establish held-out camera estimation or independent
generalization. With supplied cameras, evaluation is conditional on those
poses, not camera recovery. No neural reconstruction integration or accuracy
result follows from this export.

## Local smoke export

The fresh [export directory](../build-opencv/neural-camera-bridge-001/transforms.json)
contains a 14-frame `transforms.json` and
[provenance](../build-opencv/neural-camera-bridge-001/provenance.json) from the
existing supplied-camera ETH3D Pipes undistorted PINHOLE text model. The
explicit filename-order split is ten training (`DSC_0634`–`DSC_0643`), two
validation (`DSC_0644`–`DSC_0645`), and two test (`DSC_0646`–`DSC_0647`)
images. The files together are 17,432 bytes; no images were copied.
SHA-256: transforms
`d62e2cc4b1f31ccd2265c2125b56d63f5f992d334b180617526086689261f145`,
provenance
`cc94fa784bf3b626fe9066529b1d1f05cb690e175e0db3d7aa67f1ee89a51461`.
All 14 image paths resolve; exported camera rotations have determinants
within floating-point tolerance of +1. Four focused tests pass. This is a
format and pose-convention smoke test only; no Nerfstudio package, parser,
training run or rendered output has been tested. Supplied ETH3D poses are
not an image-only camera-estimation result.
