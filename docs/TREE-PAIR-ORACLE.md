# Fixed real-tree stereo pair oracle

This local experiment finds substantial image-space disparity on one pair of
the ten real-tree photos. It is evidence that dense matching can produce
nonempty output on these rectified inputs. It is **not** a depth-accuracy,
physical-scale, object-isolation, or full reconstruction result. The earlier
selected-MVE attempt returned 0 and 2 positive pixels for references 1 and 4;
its multi-view optimization, camera import, and output contract differ from
this one-pair stereo test, so those counts are context rather than a direct
algorithm ranking.

## Fixed inputs and geometry

The source is the pinned ten-photo subset described in [TREE-DENSE.md](TREE-DENSE.md).
This run uses the fresh corrected scene at
`build-opencv/tree-refine/baseline/scene`, where COLMAP's principal points are
converted to OpenCV pixel-center coordinates before resizing to 768×512.
The source cameras are `PINHOLE` with zero distortion. Supplied world-to-camera
poses and resized intrinsics are held fixed; their scale is arbitrary.
Original view 1 is `The_Tree-36.jpg`, and view 4 is `The_Tree-35.jpg`.

OpenCV 4.12 `stereoRectify` and `initUndistortRectifyMap` produce the two
rectified grayscale PGM files. The helper detects the rectified baseline axis
and transposes both images for a vertical baseline. For this pair the axis is
horizontal, and it swaps left/right because the original view order yields
negative disparity. The matcher convention is then positive
`d = x_left - x_right`. The C++ self-test checks positive-disparity sign,
horizontal swapping, vertical transposition, projection without a doubled
baseline, and the resize principal point convention.

The scene's 177 measured ORB pair seeds were split deterministically before
any range selection: 141 training and 36 held-out. Projecting **training
3D seeds only** through the fixed cameras gives a 11.45–101.01 px disparity
range. With an 8 px upper allowance and 16-pixel alignment, both matchers
searched 0–111 px. The bound is drawn from independently triangulated sparse
seed geometry, not depth ground truth or the held-out correspondence
disparities. The scene seeds were triangulated from image matches with the
same estimated poses; they are therefore a weak image check, not independent
ground truth.

Before dense matching, the held-out rectified vertical residuals were median
0.99 px, 90th percentile 2.87 px, maximum 3.96 px. The fixed gate requires
median ≤2 px and 90th percentile ≤4 px; a grossly misrectified scene stops
before either matcher runs. The retained source matches themselves allow up
to 2 px reprojection error in each input view, explaining residuals above
one pixel. These residuals do not establish exact camera calibration.

## Matching and checks

Both matchers read byte-identical `left.pgm` and `right.pgm`, with the same
0–111 px search. OpenCV StereoSGBM uses block 5, P1 200, P2 800,
`disp12MaxDiff=1`, `preFilterCap=31`, uniqueness 10, speckle window 100,
speckle range 2, and `MODE_SGBM`. The separate, unmodified local libELAS
oracle uses its `MIDDLEBURY` preset and full-resolution output; see
[its adapter notes](../scripts/oracles/elas/README.md). libELAS is GPL and is
kept outside the production code. The local comparison redistributes no
oracle binary.

| Check | StereoSGBM | libELAS |
| --- | ---: | ---: |
| Finite, nonnegative output / 393,216 pixels | 83,789 (21.31%) | 393,216 (100%) |
| Strictly positive output pixels | 81,663 | 392,119 |
| Held-out correspondences with valid disparity | 26/36 | 36/36 |
| Median absolute disparity difference on valid held-out matches | 0.59 px | 1.07 px |
| Valid held-out matches within 2 px | 23/26 | 24/36 |
| Left/right agreement ≤1 px among valid output | 48,804/83,789 (58.25%) | Not exported by adapter |

The held-out check samples each disparity map at an ORB image point and
compares it with the matched point's horizontal displacement. It validates
some high-texture locations only. It has no depth truth, no verified metric
scale, no tree mask, and no independent dense correspondence reference.
StereoSGBM's right pass is computed with flipped, swapped images on the same
search interval; its consistency count is a stricter subset than mere finite
coverage. The libELAS adapter exports only left disparity. Its 100% finite
coverage includes interpolated or filled regions and should not be equated
with 100% correct geometry. Both algorithms see tree, background, ground,
and sky. Neither output has been tested for fusion into an isolated tree.

The [machine-readable report](../build-opencv/tree-pair-oracle-guarded/report.json)
records full settings, held-out residuals, hashes of the corrected scene,
two source images, matcher binaries, common PGM inputs, and disparity outputs.
Its run directory is 5.3 MiB. Input hashes were checked both before and after
execution. Fixed-settings reruns gave byte-identical PGM and disparity hashes.
The binary and upstream libELAS provenance are local; no large download or
new dependency was added.

## Reproduce

The supervisor independently reran the guarded runner at
`build-opencv/tree-pair-oracle-supervisor/report.json`: input hashes remained
unchanged and all coverage, held-out and left/right counts above reproduced.

From the repository root, after creating the corrected scene with the tree
importer and building the separate local ELAS oracle:

```sh
cmake -S scripts/tree_pair_oracle -B build-opencv/tree-pair-oracle-tool \
  -DOpenCV_DIR="$PWD/build-opencv" -DCMAKE_BUILD_TYPE=Release
cmake --build build-opencv/tree-pair-oracle-tool -j2
python3 scripts/tree_pair_oracle/run.py \
  --scene build-opencv/tree-refine/baseline/scene \
  --tool build-opencv/tree-pair-oracle-tool/tree_pair_oracle \
  --elas .local-tools/oracles/elas/build/elas_oracle \
  --output build-opencv/tree-pair-oracle-new-run
```

The runner refuses to overwrite an output directory and stops if the
rectification gate fails or inputs change during the run. Each child process
has a timeout. Outputs are PGM, PFM, CSV correspondences, geometry JSON, and
a report JSON. They stay under ignored `build-opencv`.
