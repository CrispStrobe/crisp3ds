# Photo-only bunny foreground masks

This is a narrowly object-specific image segmentation experiment for the 73 preprocessed bunny photos in `build-opencv/bunny-gamma05-clahe2/`. The images show a dark bunny against bright cloth and a bright turntable. Seven spread-out views were inspected before fixing the recipe, and no camera pose, reference scan, reconstruction, or mesh score was consulted. The method is not a general foreground segmenter, and these are coarse **mesh-support masks**, not calibrated object silhouettes or proof of shape quality.

The frozen recipe in [`bunny_masks.py`](../scripts/object_motion/bunny_masks.py) converts each image to grayscale, selects pixels below 170 inside x = 15–85% of image width and y < 96.1% of image height, then retains the largest 8-connected component. It neither fills internal holes nor dilates the result. The central and lower guards reject distant background and most of the turntable; they can also remove true object pixels at the crop, and connected contact shadows can remain. This choice favors retaining the ears and body over an aggressive silhouette. It is specifically tied to this dark-on-light photo set. It is not the red-box pose-support mask from the YCB experiment.

Generate fresh output and run the small synthetic contract tests from the repo root with the existing local Python environment:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.object_motion.test_bunny_masks -v
.local-tools/colmap-sparse/venv/bin/python scripts/object_motion/bunny_masks.py
```

The measured output is [`build-opencv/bunny-masks-001/manifest.json`](../build-opencv/bunny-masks-001/manifest.json), SHA-256 `e5b242f837ee0971408b976ee9fed45e8ee3c660e56fda3f0db2e7a6749c6521`. It records exact source-photo and mask hashes for 73 original-coordinate, 1749×1155, binary 0/255 PNGs in [`masks/`](../build-opencv/bunny-masks-001/masks/). Each mask has COLMAP's `IMAGE_NAME.png` convention, such as `frame_0024.png.png`. Its `images[]` records have exact `name`, `image_sha256`, and `mask_sha256` fields accepted by the generic classical dense-mask preparation helper, plus dimensions and component diagnostics. The mask PNG payload totals 396,226 bytes. Foreground area fractions range from 13.37% to 19.03% (median 16.09%). The three synthetic tests passed.

The generated masks were visually inspected on front, side, back, and return views (`0000`, `0024`, `0048`, `0072`). The ears and body are present, but a white contact-shadow wedge remains at the feet in some views. All 73 components touch the 96.1% lower guard, so the mask boundary there is intentionally artificial and may cut thin lower geometry. Any masked OpenMVS result should be compared with the unmasked bunny baseline as a separate, explicitly labeled arm, using the same recovered cameras and dense settings. Do not retune this mask from reference-scan scores.
