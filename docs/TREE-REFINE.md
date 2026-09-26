# Corrected tree camera convention and one multiview refinement trial

This test used the same pinned ten tree photos and estimated COLMAP poses as
[TREE-DENSE.md](TREE-DENSE.md). It provides no ground-truth scale or object mask.
Depth coverage is measured over each full 768×512 image, including background.

## Camera convention correction

The original tree importer treated COLMAP principal points as OpenCV integer
pixel-center coordinates. COLMAP's convention instead puts the upper-left
pixel center at `(0.5, 0.5)`; the corrected importer now converts to OpenCV
coordinates before resizing: `cx_cv = cx_colmap × scale_x − 0.5` and likewise
for y. It then writes MVE normalized principal points `(cx_cv + 0.5)/width`.
The previous scene and results remain preserved and are labeled with their
old convention. The [COLMAP FAQ](https://colmap.github.io/faq.html) defines
the source pixel convention. The geometry unit test checks the conversion.

Fresh corrected baseline:
`build-opencv/tree-refine/baseline/scene`. The MVE run kept reference views
1 and 4, scale 0, three global and three local neighbors, and no masks.
It loaded 575 pair seeds, and gave 0 and 2 valid depth pixels respectively.
Thus the half-pixel correction alone did not make this scene usable.

## Frozen multiview candidate

The separate `scripts/tree_refine` helper uses the same resized images, fixed
ORB configuration (up to 3,000 features/view), mutual 0.8-ratio matching, and
two-view positive-depth, 2 px, ≥1° tests as the original seed helper. It keeps
keypoint IDs, merges pair links only when a component has at most one feature
per view, and rejects ambiguous merges. A 3+ view track reserves its last
observation from triangulation, geometry filtering, and Ceres fitting. The
reserved pixel did participate in descriptor matching and track formation;
its residual is a conditional check, not an independent test set.

The one predeclared optimizer uses only measured image pixels, fixes all six
pose parameters of cameras 0 and 1, fits other cameras and 3D track points,
uses a 1 px Huber loss and at most 30 Ceres iterations. Fixing two entire
cameras resolves the similarity gauge and retains their estimated baseline,
but forbids correction of either anchor pose. The scale remains arbitrary.
Camera 7 has no fitted track support and retains its source pose. The nine
supported cameras form a training graph connected to both anchors.

| Diagnostic | Corrected pair baseline | Multiview tracks, fixed poses | Multiview tracks + BA |
| --- | ---: | ---: | ---: |
| Sparse points in MVE | 575 | 423 | 423 |
| Training RMS, 846 observations | — | 1.029 px | 0.606 px |
| Reserved RMS, 9 observations | — | 574.5 px | 839.3 px |
| Positive depth pixels, view 1 / 393,216 | 0 | 0 | 7 |
| Positive depth pixels, view 4 / 393,216 | 2 | 0 | 0 |

Pair tests yielded 575 links, but only 423 retained tracks: 414 length-2 and
9 length-3. Sixty-six multiview components failed the fit-to-training-view
screen. There were no same-view merge conflicts in this run. The enormous
reserved error **before** optimization shows that even the initial multiview
associations are inconsistent. BA makes that error worse while lowering the
training error. The fixed-pose multiview export also yields no valid pixels,
which separates a track/seed issue from BA pose changes. View 2's camera
center moved 3.70 arbitrary source units after BA; both anchored poses and
their 0.0954-unit baseline stayed identical within floating-point roundoff.
All three scenes fail the nonempty-both dense acceptance condition. No
accuracy improvement is claimed. Reserved observations were exported into
both multiview MVE bundles to keep complete tracks; their residuals are held
out **only from BA**, and the dense runs are not held-out validation.

The source scene, baseline dense status, fixed-pose track scene, guarded BA
candidate scene, and their dense statuses are under `build-opencv/tree-refine/`.
The guarded candidate scene's
`refinement-report.json` records track counts and residuals. MVE's independent
loader read ten views and 423 features from the candidate scene. Its loaded
principal points are `(384, 256)` in 768×512 images, as expected for the
corrected convention. The generated data are under 100 MiB total, below the 500 MiB
new-data limit; disk free space remained about 29 GiB, above the 10 GiB reserve.

## Reproduce

Run these from the repository root with the already pinned dataset and local
MVE/OpenCV/Ceres builds. Choose fresh output paths; the scripts refuse to
overwrite scenes or refinement output. The three `run_dense` commands are
expected to exit nonzero with `dense_failed_empty_reference` while preserving
their result JSON and depth files.

```sh
python3 -m unittest scripts/tree_dense/test_import_scene.py -v
python3 -m unittest scripts/tree_refine/test_export_scene.py -v
cmake -S scripts/tree_refine -B build-opencv/tree-refine/build \
  -DOpenCV_DIR="$PWD/build-opencv" \
  -DCMAKE_PREFIX_PATH="$PWD/.local-tools/bundle-quality/ceres-install-mpl;$PWD/.local-tools/bundle-quality/eigen-install" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-opencv/tree-refine/build -j2
build-opencv/tree-refine/build/tree_refine --self-test
python3 scripts/tree_dense/import_scene.py \
  --dataset .local-tools/test-data/tree-subset \
  --scene build-opencv/tree-refine/new-baseline/scene \
  --seed-tool build-opencv/tree-dense-seed/tree_dense_seed --max-width 768
python3 -m scripts.tree_dense.run_dense \
  --binary .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --scene build-opencv/tree-refine/new-baseline/scene \
  --output build-opencv/tree-refine/new-baseline/dense
build-opencv/tree-refine/build/tree_refine \
  build-opencv/tree-refine/new-baseline/scene/views.tsv \
  build-opencv/tree-refine/new-fixed-tracks.txt --no-ba
python3 -m scripts.tree_refine.export_scene \
  --baseline build-opencv/tree-refine/new-baseline/scene \
  --refined build-opencv/tree-refine/new-fixed-tracks.txt \
  --scene build-opencv/tree-refine/new-fixed/scene
python3 -m scripts.tree_dense.run_dense \
  --binary .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --scene build-opencv/tree-refine/new-fixed/scene \
  --output build-opencv/tree-refine/new-fixed/dense
build-opencv/tree-refine/build/tree_refine \
  build-opencv/tree-refine/new-baseline/scene/views.tsv \
  build-opencv/tree-refine/new-refinement.txt
python3 -m scripts.tree_refine.export_scene \
  --baseline build-opencv/tree-refine/new-baseline/scene \
  --refined build-opencv/tree-refine/new-refinement.txt \
  --scene build-opencv/tree-refine/new-candidate/scene
python3 -m scripts.tree_dense.run_dense \
  --binary .local-tools/mve-spike/mve-bf2279f161ba962072ecac85224c15e82bc5f52e/apps/dmrecon/dmrecon \
  --scene build-opencv/tree-refine/new-candidate/scene \
  --output build-opencv/tree-refine/new-candidate/dense
```

The held-out result suggests false multiview associations or pose inconsistency
in this foliage-heavy sequence. These alternatives are not separated by this
trial. A useful next test would need independently validated tracks and camera
poses before dense reconstruction; fitting the same sparse pixels harder is
unlikely to resolve the current failure.
