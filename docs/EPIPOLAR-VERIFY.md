# Independent epipolar report verification

The fixed-camera SIFT diagnostic saves all mutual matches in descriptor query
order, with every fifth match held out. Its report is checked independently by
`scripts/epipolar_verify/verify.py`, which uses only the Python standard library
and does not fit a fundamental matrix or import any producer code.

From the repository root:

```sh
python3 -m unittest discover -s scripts/epipolar_verify -v
python3 scripts/epipolar_verify/verify.py \
  --report build-opencv/tree-epipolar-capped/report.json \
  --views build-opencv/tree-refine/baseline/scene/views.tsv \
  --essential build-opencv/tree-epipolar-capped/essential-report-v2.json
```

The verifier checks all 45 view pairs, saved match order and train/holdout
partition, the reported input hashes against the current files, and the
reported camera arrays against the source TSV. For each pair it recomputes
the supplied fundamental matrix from world-to-camera rotation and translation
and pixel intrinsics, comparing matrices up to scale and sign. It then
recomputes square-root Sampson distances and all reported held-out summary
counts, fractions, medians, and nearest-rank 90th percentiles for the supplied
and fitted matrices. A Sampson denominator at or below `1e-12` is invalid;
invalid observations remain in the held-out denominator for within-threshold
fractions. A zero or nonfinite fundamental matrix is rejected.

The optional essential report check recomputes the pair 1/4 calibrated
fundamental matrix from the saved recovered rotation and unit translation,
checks its essential matrix and pose against the source cameras, and scores
all 83 held-out matches independently. It also checks that the listed
training inliers and positive-depth indices are valid subsets of training
matches. It does not estimate an essential matrix or recover a pose.

The check can detect arithmetic, camera-convention, split, and reporting
errors in saved artifacts. It cannot establish the provenance of fitted
matrix parameters from the artifact alone; the training-only fit must also
be inspected in the producer code. It does not establish ground-truth camera
accuracy or correspondence correctness.
