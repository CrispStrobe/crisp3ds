# Three reconstruction branches and evidence expansion

Updated 2026-09-27. This is an execution queue, not a claim that all candidates
are integrated, commercially cleared, or state of the art. See the machine-readable
[evidence registry](../benchmarks/pipeline-evidence.json), [roadmap](SOTA-ROADMAP.md)
and [sensor protocol](SENSOR-DEPTH-BENCHMARK.md).

## What has actually helped

| Pipeline | Work actually done here | Next useful comparison |
| --- | --- | --- |
| COLMAP → OpenMVS | Native M1 CPU sparse/dense/mesh/texture runs; retained failures; supplied- versus estimated-camera diagnostics | Primary portable classical baseline; stabilize image-only initialization, compare rough/refined in one camera frame |
| MVE | Real photo reconstructions and cross-platform selected-build checks | Independent classical implementation control, not the mandatory production backend |
| MicMac | Exact-source/build-path license inspection only; no executable built or run | Potential independent CPU matching/density oracle, conditional on resolving selected-build rights |
| AliceVision | Build/dependency feasibility review only; no reconstruction run | Independent SfM/MVS/mesh/texture comparison on NVIDIA hardware |
| Nerfstudio / gsplat | Camera-format bridge with analytic tests and 14-view live export; no package execution, training, render or mesh result | Reproducible Gaussian appearance baseline, then separately validated depth-to-mesh extraction |
| GauStudio / NeuS | Candidate review only; no execution | Mesh extraction from a trained Gaussian scene / image-and-mask-conditioned implicit surfaces respectively |

MicMac's audit saved us from accepting its top-level license as the whole build's
license; it has **not** improved a measured reconstruction. The pinned ordinary
build includes NEC noncommercial code, GPL components and LGPL ANN; see
[the file-level audit](MICMAC-BACKEND.md). Our AGPL change permits considering
GPL/AGPL experiments and desktop/server integration, but does not remove the
noncommercial restriction or settle App Store distribution.

Upstream AliceVision's [macOS build instructions](https://github.com/alicevision/AliceVision/blob/develop/INSTALL_macOS.md)
say its DepthMap library is CUDA-only. A CPU/macOS partial build is not a complete
M1 dense backend. A separate Metal port would need its own provenance, build and
quality evaluation; we have not validated one.

## Keep the three branches, share the experiment inputs

The shared input package must contain original-image hashes, explicit training and
held-out IDs, camera convention and distortion, masks with provenance, object frame,
scale evidence, and a resource budget. Hold out views **before** alignment for a
strict end-to-end novel-view test; poses for scoring require separately disclosed
localization. Do not label an all-image SfM run a fully held-out image experiment.

1. **Classical mesh:** image-derived cameras/tracks → OpenMVS density → rough
   mesh → optional refinement → texture. Keep a usable rough-mesh path even when
   refinement times out. This is the local M1 CPU baseline.
2. **Gaussian:** the same permitted camera/input split → Splatfacto/gsplat fitting
   → held-out renders and splat artifact. Add depth rendering and confidence-aware
   fusion as a separate experiment with analytic camera/depth golden tests. A
   splat PLY is not a triangle mesh. Render quality cannot satisfy mesh acceptance.
3. **Neural surface:** the same camera/input split → NeuS image/mask fit → extracted
   mesh. First compare with fixed cameras to isolate the surface estimator; then
   repeat with image-estimated cameras. Normal/depth priors are an additional
   ablation, never sensor truth. Do not start by reimplementing a neural renderer.

Object-relative cameras and foreground geometry masks are mandatory for turntable
comparisons. A rigidly rotating/translating object can be represented in its own
frame; a static background cannot be treated as co-moving with it. None of these
backends makes camera alignment or reflections automatically correct.

## Ordered comparison work and gates

| Task | Owner / target | Required output and gate |
| --- | --- | --- |
| C01 | Sol + root, local small | Spatial residual/missing-ray report and image-only seed diagnostic; no new reference fitting or native retry loop |
| C02 | Sol + root, local/CI | Typed evidence registry with tests: analytic golden, independent shape, sensor diagnostic, software agreement and rendering remain distinct |
| C03 | Sol + root, VPS | Acquire two new YCB objects below; hash source and selected members, inspect reference completeness, freeze masks/splits before reconstruction |
| C04 | Sol + root, CPU VPS | Same-input COLMAP/OpenMVS and MVE; freeze one deterministic seed policy, retain all failures, record total and per-stage time/RSS/bytes |
| C05 | Sol + root, Kaggle | Pin and audit Nerfstudio/gsplat including extensions; camera-conversion goldens, one bounded fit, held-out renders; no mesh claim yet |
| C06 | Sol + root, Kaggle | Audit selected GauS-style extractor or replacement; first-hit depth convention tests, extracted mesh and fixed-camera geometry comparison |
| C07 | Sol + root, Kaggle | NeuS geometry baseline and AliceVision classical GPU control, one frozen scene each before expanding |
| C08 | Root, acceptance | At least three untouched objects and retained failure rate; publish accuracy/completeness at fixed tolerances, normal/edge error, runtime and memory separately |

C05–C07 require the remote preflight in [REMOTE-QUALITY.md](REMOTE-QUALITY.md)
and the local usage guide before any upload. No neural run, remote upload or GPU
quota consumption has occurred. Current Mac free space is about 12 GiB: preserve
10 GiB, put large new archives on `/mnt/storage`, scratch on `/mnt/volume1`.

These are established comparison families, not a demonstrated 2026 SOTA ranking.
For any future SOTA claim, record the dated benchmark leaderboard/publication,
official evaluator, exact split, permissible inputs, and comparable compute;
select a leading reproducible method only after its code/weights rights review.
KIRI needs same-capture outputs and measured end-to-end time, not paper scores.

## Additional data: concrete queue, not downloaded goldens

The [official YCB release](https://ycb-benchmarks.s3.amazonaws.com/index.html)
provides real multi-view RGB/RGB-D and separate Google-scanner meshes under
CC BY 4.0. Start with `006_mustard_bottle` (curved, less textured regions) and
`035_power_drill` (handle/concavities), in addition to our cracker box. Proposed
selection: 60 original NP3 frames per object; verify their existence and capture
coverage during extraction. A single elevation does not cover undersides.

Read-only HTTP HEAD checks on 2026-09-27 returned 200 for these official assets:

| Object / asset | Bytes | ETag (not SHA-256) |
| --- | ---: | --- |
| mustard `berkeley/006_mustard_bottle/006_mustard_bottle_berkeley_rgbd.tgz` | 657272400 | `c15b25428f8eea3432996eb2b4565de3-79` |
| mustard `google/006_mustard_bottle_google_16k.tgz` | 10703139 | `36d5d7cea8336172a8cdc34b253459d8-2` |
| drill `berkeley/035_power_drill/035_power_drill_berkeley_rgbd.tgz` | 631773983 | `c9e9201a2de0ad6c1b075feebeaedffb-76` |
| drill `google/035_power_drill_google_64k.tgz` | 12777450 | `b6802a421551a54684887a809b28f5ec-2` |

URLs use `https://ycb-benchmarks.s3.amazonaws.com/data/` plus the path above.
Total source transfer is 1,312,526,972 bytes before extraction and reconstruction.
No bytes accepted yet; checksums, reference mesh quality and cross-sensor frame
registration remain unverified. Review labels/trademarks before redistribution.
The different mesh resolutions are acquisition candidates, not equal-resolution
goldens; convergence of sampled/reference geometry must be checked before scoring.

[DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) is particularly relevant:
49/64 real views and structured-light references, with official visibility masks
and evaluation code. Its sample archive is 6.3 GB: VPS only. The page says freely
available but does not state an explicit commercial license; resolve exact terms
before acceptance. Importantly, DTU's `Surfaces` download is reconstructed MVS
geometry, **not** its structured-light reference `Points`.

[Tanks and Temples](https://www.tanksandtemples.org/download/) supplies scanner
reference geometry for training scenes and a separate set of COLMAP results.
Its data terms restrict use to noncommercial purposes; do not assume a commercial
product benchmark is cleared. Keep it pending explicit intended-use review.
Continue existing ETH3D/Middlebury diagnostics, but they do not replace unseen
object-level tests. Existing bunny and YCB scanner-frame alignment limitations
remain in [OBJECT-DATASETS.md](OBJECT-DATASETS.md).

## License boundaries checked against upstream

[gsplat](https://github.com/nerfstudio-project/gsplat) and
[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE)
have Apache-2.0 principal licenses; gsplat's documented acceleration is CUDA,
not an M1 Metal training implementation. [NeuS](https://github.com/Totoro97/NeuS)
has an MIT principal license. Exact environments and assets still need review.
[GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio) explicitly excludes its
rasterizer from MIT. [2DGS reference code](https://github.com/hbb1/2d-gaussian-splatting/blob/main/LICENSE.md)
limits use to research/evaluation and requires permission for commercial use.
A gsplat-based reimplementation is a separate integration, not a drop-in license
change. [OpenMVS](https://github.com/cdcseacave/openMVS) is AGPL; the public
project's AGPL policy does not automatically approve App Store bundling.
