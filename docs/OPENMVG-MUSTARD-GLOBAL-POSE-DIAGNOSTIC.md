# GLOBAL mustard OpenMVG named-pose diagnostic

The sealed GLOBAL `-001` JSON pose export was checked **read-only** with the
[diagnostic](../scripts/classical_backend/openmvg_mustard_global_pose_diagnostic.py).
It opens only the export receipt and `sfm_camera_poses.json`, reusing the
reviewed v2.1 cereal [view/pose parser](../scripts/classical_backend/openmvg_mustard_json_pose_validator.py)
for explicit filename → `id_pose` → extrinsic-key links, intrinsic fields,
and proper-rotation checks. TRAIN photo filenames supply angle labels. No
SfM/dense stage, converter, supplied camera, scanner, reference geometry, or
held-out photo was run or read here.

The wrapper pins receipt SHA-256
`ac20487ca1932e7e42046c43b5fbae768383051f53a61729f4bb5d49dac7bebd`,
66,870-byte JSON SHA-256
`6382b68e07815d730f5b428e23004bac99a210fced8b5c9a5706e8503a2da709`,
source GLOBAL model SHA-256
`eb5d96ce4379ee32bdef3f946f833fd2c7a85b0a5c818faa4ce799766f74fc00`,
source receipt/log hashes, the existing converter SHA-256, exact `-V -I -E`
command, and the exported count/name inventory. It rehashes receipt and JSON
after validation. Four [synthetic tests](../scripts/classical_backend/test_openmvg_mustard_global_pose_diagnostic.py)
cover a valid ID-linked orbit, receipt/model hash changes, broken view-to-pose
links, malformed rotations, and a signed missing-name mismatch.

## Read-only result

The JSON has exactly 48 TRAIN views, 46 distinct finite named poses with
proper orthonormal rotations, and one intrinsic. `NP3_252.jpg` and
`NP3_276.jpg` lack poses, so the predeclared **48/48 registration gate still
fails**. The intrinsic is `pinhole_radial_k1`: focal length **1589.1185 px**,
principal point **(705.9023, 538.8405)** on 1280×1024 photos, and radial
k1 **−0.78077**. The generic posthoc sanity flags in the shared parser pass;
that is not a calibration or physical-accuracy result.

The predeclared TRAIN-label center-chord diagnostics yield median 6°/opposing
ratio **0.3033** (ideal circular 0.0523), opposing-chord p90/p10 **35.61**,
and 000°–348° closure/other-12° median **2.220**. All three broad orbit
flags fail. This is an internal photo-only trajectory-shape diagnostic,
not a reference camera score or object-geometry result.

The earlier incremental model also has 46/48 poses but misses `NP3_042` and
`NP3_048`; GLOBAL misses `NP3_252` and `NP3_276`. Only 44 names are common
to their posed sets. Its reported chord ratios (0.5954 adjacent/opposing,
21.01 opposing spread, 0.3767 closure) therefore use a **different pair
population**. Raw ratio differences are not a paired improvement claim or a
controlled parameter-only comparison. Both models fail complete registration
and the broad global-orbit checks; track membership, independent reprojection,
reference alignment, and mesh quality remain outside this diagnostic.
