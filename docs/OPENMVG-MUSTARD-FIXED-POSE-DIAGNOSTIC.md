# Fixed-intrinsic mustard OpenMVG named-pose diagnostic

The sealed fixed-intrinsic OpenMVG `-003` JSON export was checked **read-only**.
The [diagnostic](../scripts/classical_backend/openmvg_mustard_fixed_pose_diagnostic.py)
opens only its receipt and `sfm_camera_poses.json`; it does not run SfM, dense
reconstruction, the converter, or a reference comparison. It reuses the
v2.1 cereal [view/pose parser](../scripts/classical_backend/openmvg_mustard_json_pose_validator.py)
for explicit filename → `id_pose` → extrinsic-key links and proper-rotation
checks. The 48 TRAIN filenames provide angle labels; no supplied poses or
scanner data enter the measurements.

The receipt SHA-256 is
`ba5450e939b824f1c529b53e65538d4d38ab5bb3bd8bf7f073b38015e2b50088`;
the 66,159-byte JSON SHA-256 is
`a9f6b6aa9ce63e163a9f1a324768a2593fefcdf10e3f149dc61c4ccd5d6c1854`.
The wrapper pins those hashes, source model SHA-256
`4147cd6ea9e0848dd97210701535371c100425125dc83df07a67482ebd8eca55`,
source receipt/log hashes, the existing converter SHA-256, exact `-V -I -E`
command, and exported counts/names. It rehashes receipt and JSON after
validation. Four [synthetic tests](../scripts/classical_backend/test_openmvg_mustard_fixed_pose_diagnostic.py)
cover valid linkage, SHA/source tampering, nonfinite poses, a changed fixed
intrinsic, and a broken view-to-pose link.

## Read-only result

The JSON contains exactly 48 TRAIN views, 46 distinct finite named poses with
proper orthonormal rotations, and one intrinsic. `NP3_042.jpg` and
`NP3_048.jpg` lack poses. Thus the predeclared **48/48 registration gate still
fails**. The intrinsic remains exactly the image-only fixed heuristic:
focal length 1536 px, principal point (640, 512) on 1280×1024 photos, and
radial k1 = 0. These numbers confirm that this arm held those parameters;
they do not validate them against physical calibration.

The same previously defined TRAIN-label chord diagnostics used for the free-intrinsic
arm give median 6°/opposing chord ratio **0.3528** (ideal circular orbit
0.0523), opposing-chord p90/p10 **43.14**, and 000°–348° closure divided by
other 12° median **0.6587**. Closure passes its broad local flag; the adjacent
and opposing flags fail. For context, the prior free-intrinsic JSON arm had
ratios 0.5954, 21.01, and 0.3767 respectively. Fixing the intrinsic removes
that arm's focal/distortion runaway, but it does not produce a coherent
full-turn camera-center pattern under these image-only diagnostics. The
absolute camera scale, orientation against a capture rig, track membership,
independent reprojection, and object geometry remain unverified here.
