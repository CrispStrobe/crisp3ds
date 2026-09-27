# Original-photo depth-support diagnostic: masked 008 vs recovered 015

The read-only [paired audit 003](../build-opencv/classical-depth-support-003.json) asks whether 015's loss of object coverage is already visible **before meshing**. It does not use the Berkeley depth frames, scanner mesh, poses, or any reference geometry. Its two inputs are the sealed masked native depth runs [008](../build-opencv/classical-ycb-native-masked-008/result.json) and [015](../build-opencv/classical-ycb-recovered-015/result.json), with the same hash-verified original photo-derived coarse pose masks. It neither reruns SfM/dense reconstruction nor changes any native output.

For every 1280×1024 source image, the probe set is frozen at original integer RGB pixels `x ≡ 2 (mod 4), y ≡ 2 (mod 4)` where that shared source mask is nonzero. There are **189,982 paired probes** across 60 views. For each arm independently, the auditor sends each original pixel center `(x+0.5, y+0.5)` through its binary `SIMPLE_RADIAL` camera's PyCOLMAP `cam_from_img`, then through its verified native DMAP K into nearest depth-map pixel (`floor(projected_index+0.5)`). The K includes the importer/resize `−0.5` origin convention. A separate Newton radial inverse agrees with PyCOLMAP within `1e-7` normalized units for every live probe; synthetic projection/radial/rounding goldens test the transform. This compares *the same original pixels*, not raw depth pixel counts from differently sized/undistorted maps. It still compares different estimated cameras' rays, not a common world-space surface.

The auditor independently rechecked all 60 DMAP cameras per arm (K/R/C and zero positive depth outside OpenMVS masks), source→undistorted pose equality, the native warped-mask hashes, source photo/mask manifest, and source/dense model hashes. It classified each original probe as out of native frame, excluded by that arm's warped native mask, allowed-but-no-positive-depth, or positive-depth. The analyzer and every consumed model, mask, DMAP, and report were SHA-256 checked before/after; the result records an SHA-256 of each view's ordered little-endian original probe IDs. Runtime was bounded to 120 seconds, output to 20 MiB, and free disk to at least 10 GiB. Audit 001 was an unwritten diagnostic failure (the DMAP loader returned a relative image path rather than a basename). Audit 002 completed; final audit 003 added post-read upstream-report and post-hash deadline guards, then reproduced its counts exactly.

| Same original-mask RGB probes | 008 | 015 |
|---|---:|---:|
| Positive native depth | 173,399 / 189,982 (91.271%) | 172,094 / 189,982 (90.584%) |
| Projected outside native frame | 0 | 0 |
| Excluded by warped native mask | 2,453 | 3,223 |
| Mask allowed, but no positive depth | 14,130 | 14,665 |

The paired contingency is **both 171,018; only 008 2,381; only 015 1,076; neither 15,507**. Thus 015 has a **net 1,305-probe / 0.687-percentage-point** depth-support deficit on this frozen lattice. Its marginal deficit divides into **770** more warped-mask exclusions and **535** more allowed-but-missing depth samples. This identifies a real pre-mesh support difference, but not a unique cause: camera intrinsics/poses, undistortion, mask resampling, and stereo estimation all differ between these composed runs. A positive depth says nothing about depth correctness, surface completeness, metric scale, or shape accuracy. The result is compatible with the separate sensor-ray hit deficit but cannot by itself explain every missed sensor ray or prove a mesh-stage defect. No quality acceptance or parameter tuning follows from this diagnostic.

Reproduce against the same ignored local inputs with:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.classical_backend.depth_support_audit \
  --output build-opencv/classical-depth-support-NEW.json
```

The checked-in code pins the exact 008/015 run-result and source-manifest hashes, requires a fresh output path, and records all input bindings. Final audit 003 SHA-256: `4611960b0dfb45c0571e4af82a02aaa9c57d2c11356cf2e030daeff640c5b5d6`.
