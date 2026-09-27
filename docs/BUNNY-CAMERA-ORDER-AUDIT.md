# Bunny camera order audit

The 73 contrast-prepared `frame_####.png` filenames are **not acquisition
order**. `scripts/mve_full/prepare.py` sorted the original `bunny_N_rgb.png`
paths lexicographically before assigning sequential frame names. The frozen
`build-opencv/bunny-gamma05-clahe2/prepare-manifest.json` maps each frame to
its original photo. For example, `frame_0001` is `bunny_10`, `frame_0011` is
`bunny_1`, `frame_0033` is `bunny_3`, and `frame_0034` is `bunny_40`. Thus an
orbit check run in `frame_####` order falsely flags large adjacent jumps.

The supplied PRO `poses_metric.json` contains 73 `T_CO` entries under the
original numeric photo names. It describes a depth-only turntable ICP
trajectory, not measured camera truth. Its neighboring rotations in numeric
`bunny_N` order range from 4.8613° to 5.0095° (median 4.9329°); this supports
the numeric sequence as the acquisition order. These poses and the Revopoint
mesh were **not inputs** to either reconstruction. The source metadata and
manifest were used only to interpret the frozen output filenames afterward.

Read-only evaluation used the unchanged image-only
`scripts/object_motion/orbit_plausibility.py` function on 72 ordered slots,
original indices `bunny_0` through `bunny_71`, mapping each to its frame name
through the preparation manifest. The current gate requires an even number of
slots. The 73rd photo (`bunny_72`) remains registered in both models, but is
excluded from this 72-slot diagnostic. No parameter or threshold was changed.

| Frozen image-estimated model | Ordered 72-slot verdict | Largest adjacent center / orientation step | Winding | Reversed significant steps | Inward fraction |
| --- | --- | --- | --- | --- | --- |
| COLMAP retry model, 73 registered | pass, 72/72 tested | 0.1804 median radius / 10.2414° | 360° | 0 | 1.0 |
| MVE contrast model, 73 registered | pass, 72/72 tested | 0.1969 median radius / 10.2608° | 360° | 0 | 1.0 |

COLMAP cameras were read from the frozen `classical-bunny-retry-002/probe/models/0`
PyCOLMAP reconstruction. MVE cameras were read from `meta.ini` in each frozen
`mve-full-bunny-002/scene/views/view_####.mve`; MVE's `rotation` and
`translation` are world-to-camera, so centers were computed as `-Rᵀt`.
The MVE metadata agrees with the corresponding `synth_0.out` camera record.
Both trajectories pass only broad necessary full-turn checks. This does not
validate focal length, distortion, sparse correspondences, exact pose,
physical scale, or mesh shape. In particular, MVE's 73/73 result still has a
visibly distorted mesh; the COLMAP/OpenMVS mesh scores remain limited by
independent scan-registration ambiguity.

The false-order check (frames `0000`–`0071` in filename order) failed for
both a near-opposite adjacent jump and zero winding. It is **invalid camera
evidence** and must not be used in a quality comparison. The corrected
COLMAP check has no failed criterion and no collapsed opposite pairs; the
corrected MVE check has the same outcome.

SHA-256 evidence binding this diagnostic:

| Input | SHA-256 |
| --- | --- |
| `build-opencv/bunny-gamma05-clahe2/prepare-manifest.json` | `eca0bfa60badd7fa5c9b5311e7f413a3f337159c158f856066086e44b0a90ecd` |
| `pro/bunny/pcd/poses_metric.json` in the 3DLF subset | `446a3124234704ba649d8b16ed9938835d9e21d16764b9768e72c3a99e417ef7` |
| COLMAP `cameras.bin` | `beed7404b81fe3869a661a0cdf2700d3d293e4aca8dd9c93f359acf7fcdecef5` |
| COLMAP `images.bin` | `9cfc1fab830d3b36a631a9e37258720d648586fb5ceee3db8bb9b5cbfa9d5f83` |
| COLMAP `points3D.bin` | `c24084a823a6588c312e75b094e5ebb6ea84a3fb805d4c5bc8474bd8d6e60a84` |
| MVE `scene/synth_0.out` | `4980a71bbd78760b0eaa0cdc319e0cd7cac4d6c537db7d364c5292` |
| Orbit diagnostic code | `d9762661facc1925f9ad998a0fd1d5013d1d4345787af21542e393bbf9cfb762` |

For future bunny SfM arms, derive ordered slots from the preparation manifest
and original numeric names, freeze that mapping before camera evaluation,
then check trajectory, intrinsic stability, foreground sparse-track support,
and reprojection error before spending on dense meshing. Supplied PRO poses
may be used only as a clearly labeled post hoc reference comparison after
confirming coordinate conventions and resolution scaling; they must not enter
the image-only reconstruction or parameter selection. Keep scanner mesh
alignment and surface metrics separate from camera plausibility, because the
existing registration audit does not identify the correct cross-sensor pose.
