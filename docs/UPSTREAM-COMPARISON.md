# Sceaux software-oracle agreement

This comparison is frozen before scoring the saved Sceaux meshes. The local
MVE run uses only the 11 verified Sceaux photographs and its own recovered
cameras. Its candidate is `build-opencv/mve-upstream-sceaux-001/mesh.ply`.
The reference is the pinned upstream OpenMVS v2.3
`.local-tools/upstream-control/openmvs-sceaux-v23/scene_dense_mesh.ply`,
derived from the sample's supplied OpenMVG cameras and an OpenMVS pipeline.
It is a **software oracle for agreement**, not physical ground truth or a
certified accuracy target. The two runs also use different SfM and dense
settings, and may cover different portions of the castle/background scene.

If the upstream PLY has unsupported property aliases, export geometry without
changing vertices or triangles into a fresh
`build-opencv/upstream-comparison/` PLY using the strict
`scripts/classical_backend/geometry.py` exporter. Record source and normalized
SHA-256. Fit the existing fixed Sim(3) procedure to that normalized oracle:
1,024 area-weighted samples per mesh, seed 2026, 24 proper PCA axis/sign
seeds, four trimmed candidates and 60-second cap. Use a separate 2,048-sample
surface comparison with seeds 2027/2028 at 0.5%, 1%, and 2% of the oracle
bounding-box diagonal. Record bidirectional distances, F-scores, normalized
Chamfer-L1 mean, normal agreement and topology with the current bounded
evaluator. The transform fits translation, rotation and scale to this oracle;
none of these results verifies recovered physical scale.

Inspect a common-scale orthographic preview after fitting. A low score may
come from pipeline failure, registration minima, unequal scene scope, or
differences in reconstruction settings. A high score would mean agreement
with one software output, not truth. Neither outcome promotes the local MVE
pipeline or upstream mesh to production quality. Do not compare Sceaux's
absolute distances with the separate YCB or bunny object scans.

The local OpenMVS v2.4 CPU 640px lane has a validated rough mesh from its
saved mesh stage, while the original run failed at refinement. Score that
retained rough mesh separately with the same frozen fit and thresholds; label
it a partial-stage artifact. A later fresh, hash-verified refinement
continuation, if successful, receives its own row and provenance.

As an independent check on the MVE mesh's poor surface fit, a post hoc
camera-center diagnostic pairs all 11 recovered MVE cameras by JPEG name with
the cameras exported from the pinned upstream `scene.mvs`. Both are converted
from world-to-camera rotations/translations to centers. Fit one proper Sim(3)
with Umeyama to the named centers, report fit and leave-one-out residuals,
then score the same MVE mesh at the same 2,048 samples and three thresholds
under that camera-derived transform. The supplied upstream cameras are a
software-stage oracle; this transform is never an input to MVE reconstruction
and is not physical pose ground truth. A camera fit cannot prove surface
quality, but can separate a poor surface-reference alignment from a larger
camera-frame discrepancy.

## Results

The upstream mesh's float32 PLY aliases required strict geometry-only export.
The export preserved its 363,631 vertices, 726,814 triangles and zero
zero-area faces, promoting coordinates exactly to float64. Source SHA-256 is
`cb884fac57c4cd42f4d06a737a0060821ef5f684810e75fdb62b0782da8b3c1e`;
normalized SHA-256 is
`4a5a858dadfc00672d8b08353364bc95ddc62066d69261bfa66b012b64410d8d`.
The [export record](../build-opencv/upstream-comparison/sceaux-oracle-export.json),
[frozen fit](../build-opencv/upstream-comparison/mve-reference-fit-v1.json),
and [surface report](../build-opencv/upstream-comparison/mve-surface-metrics-v1.json)
bind the exact meshes. The upstream reference diagonal is 11.5199794
unverified scene units; thresholds are 0.0575999, 0.1151998 and 0.2303996
in those units. The fitted output-to-oracle scale is 0.5688854.

The MVE image-only run registered 11/11 photographs and produced 20,913
vertices and 41,339 faces. Its fit did not meet the final relative-change
convergence flag, so a local registration minimum remains plausible.

| Candidate versus upstream mesh | F at 0.5% | F at 1% | F at 2% | 1% precision / recall | Normal abs-dot accuracy / completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| MVE image-only Sceaux | 11.50% | 23.58% | 42.45% | 17.82% / 34.81% | 0.522 / 0.521 |
| OpenMVS v2.4 CPU rough mesh, supplied cameras (partial stage) | 89.27% | 94.04% | 96.54% | 99.12% / 89.45% | 0.917 / 0.889 |
| OpenMVS v2.4 CPU refined continuation, supplied cameras | 89.16% | 93.52% | 96.07% | 99.17% / 88.48% | 0.912 / 0.863 |

MVE's normalized symmetric mean distance is 3.63% of the oracle diagonal.
Output-to-oracle median/p95 distances are 0.384/1.782 scene units;
oracle-to-output are 0.181/1.056. The upstream mesh has 460 boundary edges,
zero nonmanifold edges and one vertex-connected component. MVE has 731
boundary edges, 80 nonmanifold edges and 37 components, with 98.63% of its
faces in the largest component. These topology counts describe each mesh's
structure, not agreement between their geometry.

The [common-scale XY/XZ/YZ preview](../.local-tools/upstream-sceaux-mve-preview.png)
shows major scene-shape disagreement: the oracle has extended facade and
ground structures, while the transformed MVE points form a diffuse, oblique
cluster in the same panels. The score is a poor software-oracle agreement
diagnostic. Different reconstructed scene coverage and possible fit minima
remain confounders, so it cannot isolate which pipeline stage caused the gap.
The upstream model is no more a physical truth mesh than the MVE output.

The local OpenMVS rough mesh was retained before the original run failed at
refinement. Its [geometry export](../build-opencv/upstream-comparison/openmvs-cpu-rough-export.json)
binds source SHA-256 `9114dbf32b51ffbf820d6812053daf3b90ea50ed06f3cee5fcc6b075df9ad574`
to normalized SHA-256
`fa778afb1071defe5d364a700c33e6e4d3c61181beb19d9f2ade62f35701116e`.
The [fit](../build-opencv/upstream-comparison/openmvs-cpu-rough-reference-fit-v1.json)
has scale 0.9916333; the [surface report](../build-opencv/upstream-comparison/openmvs-cpu-rough-surface-metrics-v1.json)
uses the same 2,048 query samples and thresholds as MVE. Its normalized
symmetric mean distance is 0.321% of the oracle diagonal. Output-to-oracle
p95 is 0.0644 scene units; oracle-to-output p95 is 0.3074, reflecting
incomplete areas despite high agreement on retained surfaces. The
[common-scale preview](../.local-tools/upstream-sceaux-openmvs-cpu-rough-preview.png)
shows the main facade and ground structure closely matched, with upper
regions missing or displaced. The supplied-camera stage shares its scene
input lineage with the upstream software sample; the 94.04% score is a
regression agreement result, not independent geometry accuracy. This row
is **not** a completed local OpenMVS run.

A separate hash-verified continuation repaired the refinement handoff and
completed refinement/texturing from the retained scene; the original failed
run remains failed. Its refined PLY has 30,491 vertices and 60,727 faces,
source SHA-256
`60d5487c49a0147344739db35b4388c5fbcffa3367f772b2d1d333d15bb8e4ed`.
The [export record](../build-opencv/upstream-comparison/openmvs-cpu-refined-export.json),
[reference fit](../build-opencv/upstream-comparison/openmvs-cpu-refined-reference-fit-v1.json),
[surface report](../build-opencv/upstream-comparison/openmvs-cpu-refined-surface-metrics-v1.json),
and [preview](../.local-tools/upstream-sceaux-openmvs-cpu-refined-preview.png)
retain that provenance. Its fitted scale is 0.9915639 and normalized symmetric
mean distance is 0.362% of the oracle diagonal. The 1% F-score is slightly
lower than for the rough mesh under these independent sample-based fits;
refinement is not shown to improve coverage here. Preview still agrees on
the main structure and misses portions of the upper scene. This is a
composed-stage success, not a fresh uninterrupted full run.

### Named-camera alignment diagnostic

`scripts/upstream_control/camera_compare.py` reads the recovered MVE bundle
and checks its 11 world-to-camera poses against each named view's metadata.
It compares their camera centers with the 11 centers in the upstream
`scene.mvs` export (`images.txt`, SHA-256
`e210acf8eb4c99107b8d8bec0f4a29c35703c82062a461f665dbcc0d01c77b06`).
The proper Sim(3) fit has scale **1.038998** and camera-center residual RMS
**0.0333** upstream scene units, or 0.29% of the upstream mesh diagonal.
Leave-one-camera-out RMS is **0.0419** units. These are relative agreement
to supplied scene cameras, not camera ground truth; center agreement does
not check rotations or intrinsics.

Scoring the same unchanged MVE mesh under this *camera-derived* transform
gives 0.5/1/2% F-scores **18.60/24.31/30.72%**. At 1%, precision is only
**14.01%** while recall is **91.75%**. The [camera-aligned report](../build-opencv/upstream-comparison/mve-camera-aligned-metrics-v1.json)
records output-to-oracle median/p95 distances **1.210/4.639** units, versus
oracle-to-output **0.030/0.143**. Its normalized symmetric mean is 6.78% of
the oracle diagonal, and [its preview](../.local-tools/upstream-sceaux-mve-camera-aligned-preview.png)
shows much extra diffuse geometry around an approximate oracle surface.
By contrast, the surface-fitted scale was 0.568885 and selected a more
balanced but still poor overlap. The strong camera-center agreement makes
a pure camera-center registration failure unlikely as the sole explanation;
poor/extra surface geometry is a substantial contributor. This diagnostic
uses supplied upstream cameras only after MVE reconstruction, and neither
pose set is metrology truth.
