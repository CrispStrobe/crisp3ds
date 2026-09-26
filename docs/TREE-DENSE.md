# Real-tree COLMAP → selected MVE test

This is a test-only dense feasibility run on real photos. It produced a nearly
empty depth result: zero valid pixels in one reference view and two in the
other. It does **not** demonstrate usable tree reconstruction, physical
accuracy, object isolation, or a production backend decision.

## Source and limits of the geometry

The pinned source is [Single Tree Photogrammetry Dataset by Matthew Guertin
(2020)](https://huggingface.co/datasets/Matt1up/tree-minnetonka-photogrammetry),
revision `5f9de5e4a1be429b192a928cf1359c066dadb4b3`, CC BY 4.0. Its
locally hashed [subset manifest](../.local-tools/test-data/tree-subset/manifest.json)
selects ten 5464×3640 JPEGs, 202,091,069 bytes including metadata. Selection
used camera-center distance and optical-axis similarity, because the source
`images.txt` has empty image observation lines and `points3D.txt` does not
contain tracks. The source point cloud was not used. The ten photos include
tree, background, ground, and sky; no object mask is supplied.

The source COLMAP camera models are `PINHOLE`. The importer rejects distorted
and malformed camera models. Source camera positions and point-cloud scale are
estimates. There is no verified length unit or physical reference, so all
reconstructed depth values are in arbitrary source coordinates. No millimetre
or ground-truth accuracy is claimed.

## Test-only conversion

The importer validates the manifest hashes, photo sizes, pose records, camera
models, and dimensions. It resizes each image to 768×512, with focal lengths
scaled separately on x and y. COLMAP places the top-left pixel center at
`(0.5, 0.5)`, while OpenCV uses `(0, 0)`, so source principal points are now
transformed as `c' = c_COLMAP × scale - 0.5`. For selected MVE, the normalized principal
point is `(c' + 0.5) / output_dimension`; MVE's loaded K then adds the 0.5
pixel-center offset. The source COLMAP quaternion is world-to-camera and is
converted directly to MVE's row-major world-to-camera rotation. Its translation
is copied, with no physical-unit conversion.

The original results below used an incorrect extra half-pixel before scaling.
The corrected fresh baseline in `build-opencv/tree-refine/baseline` still
produces zero and two valid depth pixels. The small import correction therefore
does not resolve this failure. See [refinement follow-up](TREE-REFINE.md) and
the separate [fixed-pair stereo diagnostic](TREE-PAIR-ORACLE.md).

Since source tracks are absent, the test-only OpenCV 4.12 helper detects up to
3,000 ORB features per reduced photo, uses mutual 0.8-ratio Hamming matches,
triangulates with the fixed supplied poses, and retains pair observations with
positive depths, ≤2 px reprojection in each image, and ≥1° triangulation angle.
It caps each pair at 500 and the whole scene at 10,000. These are independent
two-view pair seeds; duplicates across pairs are possible and they are not
claimed to be multi-view feature tracks. This fixed method generated 575
seeds and 1,150 measured image observations. The imported reprojection RMS was
1.0547 px, maximum 1.9997 px. The selected MVE library loaded 10 views and
575 features; its independently loaded camera reprojection check was 1.0547 px
RMS, maximum 1.9997 px in
[camera-check.json](../build-opencv/tree-dense/camera-check.json).

The subset, converted scene, and run output use 213,205,331 bytes together in
the replay, below the 1 GiB batch cap. The disk retained over 29 GiB free,
above the 10 GiB reserve. The input photos remain in the ignored
`.local-tools/test-data/tree-subset` directory, and generated scenes remain
under ignored `build-opencv/tree-dense`. No production core files were changed.

## Fixed dense run and replay

The predeclared run used reference views 1 (`The_Tree-36.jpg`) and 4
(`The_Tree-35.jpg`), MVE scale 0 (768×512), three global and three local
neighbors, and a 120 s wall timeout per view. The [first run
logs](../build-opencv/tree-dense/dmrecon-1.log) and [replay
status](../build-opencv/tree-dense/replay/dense-status.json) use the selected
MVE commit `bf2279f161ba962072ecac85224c15e82bc5f52e` binary described in
[MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md). The replay created a fresh scene from
the same pinned inputs, with identical seed and loaded-camera checks. No
settings were tuned after seeing the first result.

| Reference | MVE candidate features processed | Optimized features | Positive depth pixels / 393,216 | Full-image coverage |
| --- | ---: | ---: | ---: | ---: |
| 1, `The_Tree-36.jpg` | 463 | 0 | 0 | 0% |
| 4, `The_Tree-35.jpg` | 514 | 1 | 2 | 0.000509% |

The first run and replay gave identical valid-pixel counts; the replay depth
files were SHA-256 `a4368b7e7c75c515576bb40132ee465fd461a3e79ccffea50c554122a1c88069`
and `00e4cc0ee5d7890f8d5a26e1789f468193d540428e85ae915457cae4395ab800`.
An additional [supervisor run](../build-opencv/tree-dense-supervisor/dense/dense-status.json)
used another fresh import and the same fixed settings. It independently
loaded 575 seeds / 1,150 observations with 1.0547 px RMS and yielded the same
two depth hashes. The hardened runner returned a nonzero exit with
`dense_failed_empty_reference` after preserving both outputs and recorded
`inputs_unchanged: true`.
The positive values for view 4 were 6.4219 and 6.4621 in arbitrary source
units. The logs report 33 ms and 28 ms reconstruction in the first run;
wall times were 0.075 s and 0.077 s. Replay wall times were 0.040 s and
0.039 s, with child peak RSS at most 20.3 MiB as reported by Darwin
`getrusage`; this is a process-level measurement for this tiny failed run,
not an estimate of memory for a successful dense run.

The [depth evaluation](../build-opencv/tree-dense/replay/depth-evaluation.json)
counts finite positive pixels across the full image. [View 1
preview](../build-opencv/tree-dense/replay/depth-preview-1.png) is empty;
[view 4 preview](../build-opencv/tree-dense/replay/depth-preview-4.png) contains
two pixels. There is no overlapping valid output for cross-view depth
consistency. The sparse reprojection check establishes that the imported
camera convention and two-view observations agree at image precision; it
does not establish that MVE can optimize dense patches here. MVE reported
463/0 and 514/1 processed/successful feature optimizations, which directly
explains the empty output. The source's estimated poses/intrinsics, foliage,
and background may contribute, but this run does not isolate a cause.

## Reproduction

Run from the repository root with the pinned test subset and selected MVE
binary already present. Use fresh scene/output paths; the scripts refuse to
overwrite existing evidence.

```sh
cmake -S scripts/tree_dense -B build-opencv/tree-dense-seed \
  -DOpenCV_DIR="$PWD/build-opencv" -DCMAKE_BUILD_TYPE=Release
cmake --build build-opencv/tree-dense-seed -j2
python3 -m unittest scripts/tree_dense/test_import_scene.py \
  scripts/tree_dense/test_run_dense.py -v
python3 scripts/tree_dense/import_scene.py \
  --dataset .local-tools/test-data/tree-subset \
  --scene build-opencv/tree-dense/new-run/scene \
  --seed-tool build-opencv/tree-dense-seed/tree_dense_seed --max-width 768
python3 scripts/tree_dense/verify_scene.py \
  --inspector build-opencv/mve-spike/check_scene \
  --scene build-opencv/tree-dense/new-run/scene \
  --output build-opencv/tree-dense/new-run/camera-check.json
python3 -m scripts.tree_dense.run_dense \
  --binary .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --scene build-opencv/tree-dense/new-run/scene \
  --output build-opencv/tree-dense/new-run
python3 -m scripts.tree_dense.evaluate_dense \
  --scene build-opencv/tree-dense/new-run/scene \
  --output build-opencv/tree-dense/new-run
```

The selected MVE scene format and synthetic control are covered separately in
[MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md). This real-tree test shows that passing
sparse loading and reprojection checks is insufficient for dense acceptance.
