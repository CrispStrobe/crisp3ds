# Bounded surface comparison

`scripts/object_dataset/surface_metrics.py` scores two registered triangle meshes in
both directions. It draws deterministic, area-weighted query points from each
surface. For every point, an AABB tree finds the closest point on an actual
triangle of the other mesh. The tree prunes only boxes whose lower bound is
already farther than a found triangle, so the target distance is exact up to
floating-point arithmetic (with a 1e-12 barycentric boundary tolerance).
The surface integral, including threshold precision and recall, remains a
Monte Carlo estimate from at most 4,096 query points per mesh. Query seeds and
counts are recorded. Precision is the output-to-reference fraction within the
threshold (accuracy); recall is the reference-to-output fraction
(completeness); their harmonic mean is F-score. Distances and thresholds use
the reference mesh's declared coordinate units.

The script accepts 1–16 distinct positive absolute thresholds, at most 1
million vertices and 2 million faces in each 512 MiB PLY, and a 300-second
scoring deadline. A median-split NumPy AABB tree uses 16 triangles per leaf;
SciPy and other spatial-index dependencies are unnecessary. Triangle faces
with exactly zero area are counted and excluded from both sampling and target
distance queries. Meshes containing nontriangular faces are rejected. All
vertices must be finite and every face index valid. The report
records mesh hashes, alignment/scale provenance, direction summaries, stage
times, score thresholds, and caps. Output JSON is written only to a fresh path.

Example, using the frozen bunny transform (the thresholds are 0.5%, 1%, and
2% of its reference bounding-box diagonal):

```sh
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 scripts/object_dataset/surface_metrics.py \
  --reference .local-tools/test-data/3dlf-scan-bunny/revopoint/bunny/fuse_mesh_rgb.ply \
  --output build-opencv/mve-full-bunny-002-finalized-v2/mesh.ply \
  --transform build-opencv/mve-full-bunny-002-finalized-v2/reference-fit-v1.json \
  --threshold 1.0429398885 --threshold 2.085879777 --threshold 4.171759554 \
  --samples 4096 --seed 2027 \
  --save-report build-opencv/mve-full-bunny-002-finalized-v2/surface-metrics-4096-v2.json
```

The transform must be a proper uniform-scale Sim(3) mapping output coordinates
into reference coordinates, and its JSON must bind the exact two PLY SHA-256
hashes. This case uses the pre-existing reference-fitted transform, whose
scale is 17.23452. It describes shape after fitting to the very reference
used for scoring. Neither that scale nor these coordinate-unit distances
establish independently recovered metric dimensions. The separately scanned
Revopoint mesh is a shape oracle for this physical print, not certified
metrology truth. It includes a support base, and the output contains diffuse
extra geometry. The precision/recall asymmetry reflects that whole-mesh
comparison. The score is not a product-quality certification.

## Local measured results

The 4,096-sample bunny report is
`build-opencv/mve-full-bunny-002-finalized-v2/surface-metrics-4096-v2.json`.
The reference has four zero-area faces, excluded from scoring; the prepared
output has none. Units are unverified reference-coordinate units.

| Threshold | Accuracy / precision | Completeness / recall | F-score |
| ---: | ---: | ---: | ---: |
| 1.0429398885 (0.5%) | 10.79% | 50.46% | 17.78% |
| 2.085879777 (1%) | 19.29% | 62.40% | 29.47% |
| 4.171759554 (2%) | 36.11% | 72.19% | 48.14% |

Output-to-reference mean/RMS/median/p90 distance is
8.123/10.974/6.232/17.378. Reference-to-output is
3.291/5.449/1.020/10.169. Stage times on this machine were 1.53 seconds
triangle preparation, 11.26 tree construction, 1.68 sampling, and 31.30
distance queries, 45.77 seconds total. These are scoring times, separate from
reconstruction and from the frozen alignment fit. The older 14.88% F-score in
`BUNNY-EVALUATION.md` used nearest *sampled target points* and is a different
estimator; the new 29.47% 1% score removes target-sampling spacing error. The
increase is a change in metric, not an improvement in the reconstruction. Both
show poor shape fidelity relative to the scanner mesh. Repeated query sampling
or confidence intervals would be needed to quantify Monte Carlo uncertainty.

For a numerical self-control, the YCB cracker-box reference mesh was compared
with itself at 512 independent query points per direction:

```sh
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 scripts/object_dataset/surface_metrics.py \
  --reference .local-tools/test-data/ycb-cracker-box/reference/google_64k_geometry_f64.ply \
  --self-control --threshold 0.001 --threshold 0.01 --samples 512 --seed 2027 \
  --save-report .local-tools/test-data/ycb-cracker-box/reference/surface-self-control-512.json
```

Its two F-scores were 100%; mean directional distances were below 4e-18
reference-coordinate units, consistent with floating-point zero. This checks
the distance/index path and independent query draws, not a reconstruction.
The YCB Google scanner mesh is likewise a separate-sensor shape oracle, not
independent cross-sensor scale calibration or certified metrology truth.

Analytic and randomized checks run with
`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m unittest scripts.object_dataset.test_surface_metrics`.
