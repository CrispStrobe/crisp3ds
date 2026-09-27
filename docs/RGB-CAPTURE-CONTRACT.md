# Original-RGB capture bundle (JPEG, no depth required)

The [capture-bundle helper](../scripts/capture_bundle/rgb_capture.py) accepts ordinary original JPEG photos as a reconstruction *input inventory*. It hashes and decodes each source for validation, records its stored pixel dimensions and EXIF orientation, and writes one small JSON manifest. It never transposes, rewrites, copies, masks, calibrates, estimates poses, or runs SfM. No depth sensor, LiDAR, ARKit session, or pre-calibration is required to make a valid RGB-only bundle. HEIC/HEIF is **not** accepted or claimed decoded here; use original JPEGs (for example, iPhone photos captured/exported as JPEG without further image changes). This helper has not been tested on an iPhone 13 mini device.

The bundle is separate from [`project.schema.json`](project.schema.json), which describes a later reconstruction project with millimetre units and stage statuses, and from the [COLMAP-to-Nerfstudio bridge](../scripts/neural_bridge/colmap_to_nerfstudio.py), which requires an already estimated/validated COLMAP camera model. Creating a capture bundle does **not** fill either later contract or assert that camera calibration, track geometry, mesh quality, or metric scale exists.

Declare `capture_motion` as one of `stationary_object_moving_camera`, `moving_object`, or `unknown`. This is a **user declaration**, not an image-derived motion classification (`capture_motion_verified=false`). Moving-object capture is allowed as RGB input: the object can rotate/translate against a stationary background. For that case, optional ARKit world camera poses are device trajectory metadata, **never direct object poses**. Even for a stationary object, raw camera poses require an explicit coordinate conversion and compatibility test before any backend use. `pose_use_policy` and `projection_compatibility` make this restriction machine-readable. No metric scale is asserted without an independent reference (`metric_scale_asserted=false`), even if optional device translations are tagged in metres.

Example using three original JPEGs already in an image directory:

```sh
python3 -m scripts.capture_bundle.rgb_capture create \
  --images-root /path/to/original-jpegs \
  --image IMG_0001.JPG --image IMG_0002.JPG --image IMG_0003.JPG \
  --capture-motion moving_object \
  --output /path/to/fresh/capture.json
python3 -m scripts.capture_bundle.rgb_capture validate \
  --images-root /path/to/original-jpegs \
  --manifest /path/to/fresh/capture.json
```

The paths are source-root-relative. The creator rejects traversal, absolute paths, symlinks, duplicate/case-colliding names and duplicate file contents; it bounds count, bytes and decoded pixels, hashes photos before/after decode and again before publication, and uses exclusive creation for the manifest. Validation re-decodes and rehashes originals and checks exact manifest fields. The original JPEG and its EXIF bytes remain unchanged; `exif_orientation` is recorded, while `stored_width`/`stored_height` refer to untransposed encoded pixels. GPS/private EXIF metadata remains in the originals even though this manifest does not export those fields, so handle source files accordingly.

Optional JSON inputs to `create` are `--intrinsics-json` and `--poses-json`. Intrinsics must explicitly use `coordinate_space="stored_jpeg_pixels"`, match every original stored size, name a source, and provide finite positive focal lengths, principal point and either no distortion or finite OpenCV radial/tangential coefficients (4, 5, or 8). Poses must give every filename, a finite proper homogeneous 4×4 transform, a source, translation units (`meters`/`unknown`), and an explicit space (`arkit_world_camera_to_world` or `opencv_world_to_camera`). These checks prove **syntax and numeric plausibility only**. In particular, EXIF orientation and ARKit/display-frame camera axes are not transformed into the stored JPEG pixel axes; projective compatibility is marked **unverified** for all optional camera metadata. A downstream reconstruction must establish that mapping and object/world-frame semantics rather than importing these matrices directly.

The current helper uses Pillow for JPEG decoding, a 10 GiB local free-space floor before creating the small manifest, and a maximum of 1,000 images, 100 MB per JPEG, 50 megapixels per image and 2 MB manifest. It does not require network access or install packages globally.

Local read-only image proof: `NP3_000.jpg`, `NP3_006.jpg`, and `NP3_012.jpg` from the existing YCB training-photo directory were inventoried as `moving_object` into [`capture.json`](../build-opencv/rgb-capture-proof-001/capture.json), SHA-256 `cc9d54c29c6fcbf2109e8a0df480c0045a386c13546158e04bf11bdfdda70e1d`. The separate `validate` command re-decoded/rehashed all three and reported `valid`, zero copied/modified images, unverified motion and projection compatibility, and no metric-scale assertion. This only proves JPEG ingestion on three existing non-phone photos; it is not an iPhone test or a reconstruction result. No depth, supplied pose, reference geometry, or held-out view was read.
