# Depth reconstruction quality audit

2026-10-09. A severe failure is reproducible in the ordinary band matcher with
perfect cameras and synthetic textured photographs. This is a concrete reason
to change matching support geometry, rather than continue smoothing damaged
meshes. It does not establish the cause of every error on real objects.

## Independent controls

`crates/dense/tests/depth_geometry_audit.rs` constructs its own analytic scene.
No scanner geometry or dataset-provided cameras, masks or depths enter it.

The coordinate control intersects independently calculated double-precision
world rays with boxes, then checks production projection/unprojection across
48 cameras, unequal focal lengths and off-centre principal points. World-point
error is below 2e-6 scene units and pixel error below .0002 px. This establishes
the tested pinhole convention, not accuracy of recovered real cameras, image
undistortion or every crop/pyramid transform.

Exact analytic depths of a box with a recess and raised feature pass through
production GPU fusion and meshing. The manually constructed outer hull contains
the features but does not encode the recess. Rim removal is zero and artificial
support flattening disabled; other fusion/mesh settings are defaults. At three
independent mesh-ray intersections:

| Surface | Analytic front Y | Exported mesh front Y |
| --- | ---: | ---: |
| Flat face | -.800000 | -.812322 |
| Recess floor | -.600000 | -.604516 |
| Raised feature | -.980000 | -.994825 |

Voxel size is .0208333; each error is below one voxel. The mesh is closed, genus
zero, with 74,772 triangles. Three rendered views were inspected: both features
survive, with softened edges and slight face rippling. This isolates fusion and
meshing with ideal evidence; it does **not** validate their response to noisy,
correlated, missing or occluded stereo depths, or the photo-to-hull front end.

![Exact-depth fusion and mesh control, three views](../tests/evidence/depth-geometry-audit.png)

## A matching failure with perfect cameras

Three parallel cameras observe a textured flat plane at camera depth 2.0.
Reference/source brightness arrays are generated directly from the plane's
texture, with exact integer disparities at truth. Geometry, photographs and
the selected centre pixel's initial depth stay fixed. Only surrounding initial
depths change from 2.0 to 2.12.

| Matching control | Correct surrounding prior | Biased surrounding prior |
| --- | ---: | ---: |
| Ordinary band refinement, cost aggregation sigma 2 | 1.999909 | 1.894572 |
| Ordinary band refinement, cost aggregation disabled | 1.999933 | 1.894488 |
| Existing independent coherent plane scoring | 2.000000 | 2.000000 |
| Independent plane scoring after the band's result | 1.999909 | 1.998706 |

The band matcher moves a correct centre pixel about **5.27%** away from truth.
This happens before cross-view filtering, fusion or mesh smoothing. The
coherent plane control fixes this controlled failure; it is not evidence of
restored Bunny/drill detail or a six-object adoption result. It uses six
coordinate iterations, stride 1 and fixed fronto-parallel normals appropriate
to this synthetic plane; it is not a claim about the full existing slanted
configuration on real objects.

The mechanism is visible in `shaders/warp.wgsl` and `stereo/matcher.rs`: one
hypothesis adds the same inverse-depth offset to **each patch pixel's own prior
depth**. NCC then scores that deformed patch and attributes its peak to the
centre pixel. Neighbouring prior errors therefore alter the centre's data cost.
For this control the offset that corrects the surrounding prior predicts
`1 / (1/2 + (1/2 - 1/2.12)) = 1.892857` at the already-correct centre, close to
the observed failure. Aggregating cost slices indexed by these offsets adds
further coupling between corrections with different depth baselines.

Warping an entire prior surface can be useful for a coupled surface update, but
its patch score must not be treated as independent evidence for each pixel's
depth. Disabling cost aggregation reproduces the failure, isolating it from
that additional smoothing. The current implementation demonstrably fails that distinction. The
same mechanism can plausibly suppress relief or create false relief; its
contribution to actual Bunny/drill defects still requires a controlled replay.

## Comparison with established pipelines

These are implementation comparisons, not an external benchmark on our images.
Revisions and reproduction values are in
[`depth-geometry-audit.json`](../tests/evidence/depth-geometry-audit.json).

| Stage | Established implementation | Current ordinary Crisp3DS path |
| --- | --- | --- |
| Photometric support | COLMAP derives a patch homography from the centre's candidate depth and normal. | Band refinement warps patch pixels using their individual prior depths plus a common inverse-depth offset. |
| Geometric evidence | COLMAP's geometric stereo evaluates source-depth forward/backward reprojection during optimisation. | Ordinary band matching is photometric; a later filter counts relative depth agreement at rounded projections. |
| Fusion checks | COLMAP checks depth, projection and normals, requires multiple supporting samples and takes median coordinates. OpenMVS dense fusion also tracks depth/confidence/normals, reprojection thresholds and multi-view support. | TSDF receives accepted depths with uniform near-surface votes and explicit free-space/behind-surface votes; ordinary fusion does not receive photometric confidence, source identities or normals. |
| Volume integration | Open3D integrates calibrated depth into a TSDF with a truncation band; its classic implementation also converts Z differences to ray distances. | Calibrated camera-Z differences are integrated with a fixed voxel truncation and an extra weaker behind-surface band, followed by hull-aware field completion and meshing. |

Sources: [COLMAP patch scoring and geometry](https://github.com/colmap/colmap/blob/4ae1aafb16d5eae3d54552b2fc896a4fc9d0f7f4/src/colmap/mvs/patch_match_cuda.cu),
[COLMAP fusion](https://github.com/colmap/colmap/blob/4ae1aafb16d5eae3d54552b2fc896a4fc9d0f7f4/src/colmap/mvs/fusion.cc),
[OpenMVS densification/fusion](https://github.com/cdcseacave/openMVS/blob/a3b6d0d99800a65af6b5f6b20dcc1df4c9e0e0c4/libs/MVS/SceneDensify.cpp),
[Open3D integration](https://github.com/isl-org/Open3D/blob/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/cpp/open3d/pipelines/integration/UniformTSDFVolume.cpp).
No upstream algorithm code was incorporated. OpenMVS is a comparison source,
not a permitted source-code adoption under the owner's permissive-code rule.
Source licenses do not establish patent clearance; prior research restrictions
remain in effect.

## Consequence for further work

First replace the band's patch cost with a coherent candidate surface cost at
the refinement stages where detail is recovered. Adding only a final slanted
polish leaves earlier matching and filtering damage in the initial surface.
Use these independent controls before testing own recovered Bunny/drill inputs;
compare raw, filtered and merged depths with fixed evaluation alignments.

Then test geometric reprojection and uncertainty-aware fusion as separate
changes. Extra rejection alone can make an already patchy map worse; support
and confidence must remain attached to the evidence. A point-cloud fusion
control can isolate TSDF carving from matching errors. Neither path should
claim quality from a closed mesh, increased coverage or a texture covering it.
Dense defaults still require the existing six-object numerical and visual gate
plus browser execution/memory checks. No defaults change in this audit.

Run the controls serially with a GPU:

```sh
CRISP3DS_GPU_TESTS=1 RAYON_NUM_THREADS=2 cargo test --release \
  --manifest-path crates/dense/Cargo.toml --test depth_geometry_audit \
  -- --nocapture --test-threads=1
```

`CRISP3DS_AUDIT_OUTPUT` optionally retains a newly created mesh directory.
Without it, the test deletes its own mesh output. Camera-ray checks run even
when GPU tests are disabled. The band failure is reported diagnostically; the
test does not assert that the erroneous result is acceptable.
