# Architecture: a modular photo-to-mesh pipeline

Crisp3DS turns photos of an object into a closed mesh. The reconstruction is
split into modules with fixed interfaces, so that each module can have several
interchangeable providers and each platform uses the ones it can run. This
document is the plan of record; status columns say what exists today.

## Stages and their interfaces

```
photos + lens calibration
   │
   ├─ masks ─────────── one object mask per photo
   ├─ cameras ───────── one pose per photo, sparse points
   ├─ undistortion ──── pinhole photos and masks
   │        ▼
   │   scene (the neutral format below)
   │        ▼
   ├─ stereo ────────── mask repair, silhouette hull, matching, fusion
   ├─ surface ───────── closed mesh
   └─ check ─────────── agreement with the photos, preview
```

Everything above the scene is provider-based. Everything below it is one
implementation, the native crate `crates/dense`, with the Python package
`scripts/turntable_mesh` kept as its reference.

### The scene: what providers hand to the dense stages

A directory, the same one `dense_pipeline --inputs` and `crisp3ds-dense run
--inputs` take:

| File | Content |
| --- | --- |
| `cameras.json` | `{"views": [{name, source, image, mask, width, height, k, rotation, translation}]}`. `rotation`, `translation` are world-to-camera. `k = [fx, fy, cx, cy]` with half-pixel-centre principal point. Images are undistorted pinhole images. Paths are absolute or relative to the directory |
| `masks/` | 8-bit masks, object above 127, in the undistorted frame |
| `sparse_points.npy` | N×3 points; used only to locate the object |

The scene has no physical scale unless a provider says so (planned field
`scale: {"unit": "mm", "source": ...}`; today none sets it). Handedness is that
of the photos.

### Module contracts

| Module | Input | Output | Must also report |
| --- | --- | --- | --- |
| Masks | photos | one 0/255 mask per photo, named like the photo, in the photo's own (distorted) frame | a contact sheet; photos where much of the dark region was dropped |
| Cameras | photos, lens calibration (`crisp3ds_lens_calibration_v1`), optionally masks | poses for the registered photos, sparse points, the lens model actually used | registered count, reprojection statistics, orbit sanity; pass or fail against the gates in `docs/PHOTOS-TO-INPUTS.md` |
| Undistortion | photos, masks, lens model | pinhole photos and masks, updated `k` | — |

A provider is selected by name. It declares the platforms it supports; the
application offers only those. All providers write the same event log
(`docs/ENGINE-CONTRACT.md`, stages `masks` and `cameras`).

## Providers

### Cameras

| Provider | How | Desktop | Phone | Browser | Status |
| --- | --- | --- | --- | --- | --- |
| `alicevision` | external executables: features, matching, global SfM with a fixed lens | yes | no | no | Works (Python orchestration; native orchestration being ported) |
| `colmap` | external executable: feature extraction, sequential matching, mapper with fixed intrinsics; model converted to the scene | yes | no | no | Planned next |
| `import` | read an existing solution (AliceVision `.sfm`, COLMAP text or binary model) | yes | yes | yes | Planned; `.sfm` reading exists |
| `markers` | printed mat with fiducials on the turntable; pose from marker corners | yes | yes | yes | Planned. Also gives physical scale and settles handedness |
| `turntable` | our own solver for ordered turntable photos with a known lens: features, tracks, bundle adjustment from a turntable initial guess | yes | yes | yes | Planned; prototype first, then Rust |
| `device` | poses recorded by ARKit or ARCore during capture | no | yes | no | Later, with capture |

Camera recovery by AliceVision's global SfM is not repeatable (about 0.7
degrees and 1% of the orbit radius between runs on identical features, same
reprojection error). Any provider is therefore judged by the reconstruction it
leads to: dense stages on its cameras, scanner scores on the test objects.

### Masks

| Provider | How | Desktop | Phone | Browser | Status |
| --- | --- | --- | --- | --- | --- |
| `threshold` | dark object on a light backdrop: threshold, largest component, hole cleanup; the stereo stage's multi-view repair does the rest | yes | yes | yes | Exists in Python; being ported; whether it suffices without a network is being measured |
| `sam` | SAM 2.1 with automatic prompts | yes | later | later | Works through PyTorch (external). Native route: ggml (candidate home: CrispEmbed, which already has SAM ViT-B encoders and WASM builds) or ONNX Runtime |
| `import` | masks the user already has | yes | yes | yes | Works |

### Undistortion

One native implementation (radial `radialk3` today; the mask path already
exists and is pixel-identical to OpenCV's remap). Photos are being moved to the
same code, so no camera provider has to produce undistorted images.

## Dense stages

| Stage | Reference (Python) | Native (`crates/dense`) | Parity |
| --- | --- | --- | --- |
| Scene from an AliceVision solution | `dense_all_views_inputs.py` | `inputs` | identical cameras and masks on two objects |
| Stereo | `multiscale_stereo.py` | `stereo`, WebGPU compute | scanner F1 within 0.002 on four objects; about 7× faster on an M1 |
| Surface | `tsdf_hull_mesh.py` | `mesh` | within 0.002; about 4× faster, a third of the memory |
| Check | `mesh_photo_check.py` | `check` | identical scores |
| Driver | `dense_pipeline.py` | `run`, library API, C interface | synthetic scene verified; real objects in progress |

Rules of the port are in `crates/dense/README.md`: same files, same settings,
same events, and a native stage replaces its reference only after reproducing
the scanner scores within 0.003.

## Platforms

| | macOS, Windows, Linux | iOS, Android | Browser |
| --- | --- | --- | --- |
| Compute | native crate through Metal, DirectX 12, Vulkan | native crate through Metal, Vulkan (not yet built or measured) | WebAssembly and WebGPU (build check in CI; not yet run) |
| Cameras | `alicevision`, `colmap`, `import`, `markers`, `turntable` | `markers`, `turntable`, `device`, `import` | `markers`, `turntable`, `import` |
| Masks | `threshold`, `sam`, `import` | `threshold`, `import`; `sam` once native | same |
| Front end | Studio (Tauri) | Studio (Tauri mobile) | Studio (static web app) |

The front end talks to the pipeline through the event log and artifact files
only (`docs/ENGINE-CONTRACT.md`), whether the pipeline runs in the same
process, in a local engine, or on another machine.

## Order of work

1. Native driver and library API (in progress); Studio links the crate and
   stops starting a Python engine.
2. Photos step restructured around the provider interfaces, with native
   orchestration, native undistortion of photos, `alicevision` and `threshold`
   as first providers, and the measurement of whether masks need a network.
3. `colmap` camera provider and the `import` provider, validated on the four
   test objects through the dense stages.
4. Browser build of the dense stages.
5. `markers` provider (mat design, detection, pose, scale).
6. `turntable` provider: prototype against the test objects, then Rust.
7. Native SAM (ggml or ONNX), if step 2 shows masks need it.
8. Photo upload and capture for phones; signed releases.

Development tools that stay in Python: the scanner evaluator
(`scan_evaluate.py`), the turntable diagnostic (`turntable_rig.py`) and the
synthetic test scene.

## What this does not solve

- Thin parts far from the turntable axis are still short or lost.
- Reconstruction from a dozen photos with unknown poses is not solved; all
  results use about 70 photos.
- The test objects share one camera, lens, backdrop and material.
- External camera providers bring their own licenses (AliceVision MPL-2.0,
  COLMAP BSD-3); bundling them with an application needs the same license
  review as everything else shipped.
