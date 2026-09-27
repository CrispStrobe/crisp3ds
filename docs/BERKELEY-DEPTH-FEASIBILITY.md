# Berkeley RGB-D depth as a camera-registered validation lane

This is a feasibility inspection, **not a mesh score**. No depth data or
supplied poses entered the image-only YCB reconstruction. The existing
[named-camera audit](../build-opencv/object-motion/ycb-camera-reference-003/report.json)
already fits a proper Sim(3) from 60 reconstructed RGB camera centers to the
supplied Berkeley turntable/camera centers, without the Google mesh. It finds
center RMS 0.00872 in Berkeley coordinates and median rotation residual
2.48°. This fit can anchor a separate camera-conditioned geometry check that
does not optimize a mesh transform against the surface being evaluated.

The pinned Berkeley RGB-D archive SHA-256 is
`15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5`.
Only three explicitly named members were extracted into the ignored
`build-opencv/ycb-depth-feasibility-001/003_cracker_box/` directory:

| Angle/member | Uncompressed bytes | SHA-256 | Nonzero pixels / 307,200 | Nonzero raw p1 / median / p99 |
| --- | ---: | --- | ---: | ---: |
| `NP3_0.h5` | 616,544 | `7f1c08739faaf718fbce7a43402e18375273c1e96dd1b55e922244d04bd8b6be` | 249,111 | 6100 / 8093 / 10678 |
| `NP3_120.h5` | 616,544 | `08f25e0eacecbd201e6a869cdd849aaca3c932ca6da65eaec8a4e5b75beb6bda` | 248,130 | 6100 / 8189 / 10711 |
| `NP3_240.h5` | 616,544 | `ad930515be084927319eaa28dbef386f6743127f02d3f124b2ab5a41896a6b8a` | 249,295 | 6122 / 8209 / 10711 |

Total extracted payload is 1,849,632 bytes, below the 5 MB cap. Each HDF5
file has exactly one `/depth` dataset, unsigned little-endian 16-bit with
shape 480 × 640, and no attributes. Zero is common (18.85–19.23% of pixels)
and must be treated as missing. The official [BigBIRD data description](https://rll.berkeley.edu/bigbird/access.html)
states raw depths are in 100 µm units, so typical raw values 6,100–10,700
correspond to 0.61–1.07 m. The archive's calibration gives
`NP3_depth_scale=1`, `NP3_depth_bias=0`; these should not be mistaken for a
conversion from the published 100 µm raw unit to metres. Depth intrinsics
are `fx=fy=568.644`, `cx=306.191`, `cy=228.480` pixels. RGB and IR/depth
intrinsics and extrinsics are distinct; using `NP3_rgb_K` or
`H_NP3_from_NP5` to backproject this `/depth` array would misregister it.

For each angle, the documented transform chain is
`H_NP3_ir_from_table = H_NP3_ir_from_NP5 × inverse(H_table_from_reference_camera)`.
The depth/IR extrinsic is used because the raw depth image is on the depth
camera grid; `NP3_depth_K`, not the 1280-pixel IR K, is used for rays. As a
basic coordinate sanity check, the supplied table origin projects to
approximately pixel `(315.3, 270.3)` and positive depth 0.86305 m at all
three angles. Around that pixel, 21×21 observed windows have nonzero median
depths 0.8328, 0.6894, and 0.7182 m at 0°, 120°, and 240° respectively;
the 0° center pixel itself is missing. These values are plausible foreground
surfaces in front of the turntable, but are **not** sufficient to certify
pixel-level RGB/depth registration or pose accuracy.

A future bounded check could place each reconstructed triangle mesh into the
Berkeley table frame using the already frozen named-camera Sim(3), then render
predicted first-hit depth into each NP3 depth/IR camera. Compare only valid
raw-depth pixels with explicit visibility and missing-ray denominators;
report signed and absolute residuals, percentiles and per-view coverage.
Before any score, verify raster orientation, depth pixel convention and the
IR/depth transform with an independent projection/round-trip fixture; predeclare
object-versus-table handling. The archive's RGB segmentation masks are
generated from object models, according to the official BigBIRD description,
and live on the RGB grid, not the depth grid. Using them as an object-only
oracle requires depth-to-RGB reprojection and explicit model-derived-mask
disclosure; silently resizing them would be invalid. An unmasked whole-scene
depth score would instead penalize absent table/background geometry. This
lane prefers camera-derived similarity over unconstrained mesh-to-reference
fitting, but neither guarantees metrology. It is sensor-conditioned and
shares captures with the RGB images; it is not held-out capture,
Google-mesh alignment, or certified metrology.

No reconstructed mesh was scored or altered during this inspection.
