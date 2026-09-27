# Classical bunny stage comparison

Protocol frozen before inspecting scores. Compare the classical unmasked
rough mesh from `classical-bunny-retry-dense-002` and its bounded refinement
continuation (if completed) against the same Revopoint scanner mesh used in
[SURFACE-BENCHMARK.md](SURFACE-BENCHMARK.md). The classical rough run exhausted
its refinement deadline after successfully producing a mesh; it is a
**partial-stage output**, not a completed run. The continuation's refined
output, if valid, is a separate stage result of the same reconstruction,
not a new image-only pipeline.

Normalize each native mesh once through `geometry.export_geometry`, preserving
all triangle faces and recording source and normalized SHA-256. Fit each
normalized mesh independently to the scanner reference with the unchanged
`align.py` 1,024-sample, seed-2026, proper-Sim(3) procedure. Score each with
`surface_metrics.py` using 4,096 area-weighted query samples per direction,
seed 2027, exact nearest-triangle distances, and the frozen 0.5/1/2% scanner
bbox-diagonal thresholds 1.0429398885, 2.085879777, 4.171759554 reference
units. Record P/R/F, normalized mean distance, p95 and topology. Do not tune
the fit, thresholds, mesh, crop, or samples after seeing results. Compare the
existing MVE full-pipeline mesh by its already published 4,096-sample report;
do not rescore it merely to change its outcome.

These are reference-fitted shape diagnostics, not independently calibrated
metric dimensions or certified ground truth. The Revopoint scan has a support
base and different surface coverage; accuracy/precision and
completeness/recall should be read separately. A rough-versus-refined score
change is a stage ablation, not proof of an algorithmic or product-quality
improvement. Stochastic query uncertainty is not estimated by one seed.

## Results

The original whole-scanner results retain the scanner disk and every output
face. Each classical stage has its own independently fitted reference Sim(3).
The rough mesh is partial-stage output; the refined mesh is a completed
continuation from that exact rough mesh, not a fresh end-to-end run.

| Whole-scanner output | Faces | F at 0.5% / 1% / 2% scanner diagonal | 1% precision / recall | Normalized Chamfer-L1 mean | Fit scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| [MVE full pipeline](../build-opencv/mve-full-bunny-002-finalized-v2/comparative-metrics-v2.json) | 1,193,446 | 17.78 / 29.47 / 48.14% | 19.29 / 62.40% | 2.74% | 17.2345 |
| [Classical rough partial stage](../build-opencv/classical-bunny-retry-dense-002/rough-surface-metrics-4096-v1.json) | 266,899 | 8.54 / 16.65 / 34.91% | 22.58 / 13.18% | 4.10% | 72.4598 |
| [Classical refined continuation](../build-opencv/classical-bunny-finish-004/refined-surface-metrics-4096-v1.json) | 42,919 | 8.93 / 17.73 / 35.33% | 23.29 / 14.31% | 4.13% | 73.6297 |
| [Classical native-masked rough](../build-opencv/classical-bunny-native-masked-005/rough-surface-metrics-4096-v1.json) | 252,486 | 8.69 / 17.11 / 32.71% | 19.75 / 15.09% | 4.00% | 86.0056 |
| [Classical native-masked refined](../build-opencv/classical-bunny-native-masked-005/refined-surface-metrics-4096-v1.json) | 41,283 | 8.89 / 17.40 / 32.95% | 19.95 / 15.43% | 4.07% | 86.1268 |

At the 1% threshold, rough→refined is only +1.08 F percentage points while
faces fall by 84%. No quality improvement follows from this single sampled,
independently registered stage comparison. The refined directional p95
distances are 15.15 (output to scan) and 32.67 (scan to output) scanner units;
rough values are 14.45/31.69. Both classical
[rough](../.local-tools/classical-bunny-rough-primary-preview.png) and
[refined](../.local-tools/classical-bunny-refined-primary-preview.png)
previews show a bunny-like body but severe orientation/support-disk mismatch.
These fits and surfaces do not establish valid cross-sensor correspondence;
their lower F-scores **cannot be interpreted as a pipeline quality ranking**.
The mismatch cannot be assigned solely to the aligner or solely to geometry.
The score stage took 49.2/59.5 seconds for unmasked rough/refined and 32.5
seconds for masked refined on this host, excluding reconstruction and fitting;
contention varied, so these are evaluator timings, not pipeline speed rankings.
Native masking changes the input support but not the frozen cameras or native
depth/mesh/refine profile. Its whole-scan fitted F remains around 17%, and
the [rough](../.local-tools/classical-bunny-masked-rough-primary-preview.png)
and [refined](../.local-tools/classical-bunny-masked-refined-primary-preview.png)
previews still cannot establish valid cross-sensor correspondence.

## Post-hoc scanner-only object ROI protocol

This complementary development diagnostic was fixed after inspecting the
whole-mesh rough preview but before any ROI candidate scores. The scanner
support disk is a broad XZ-plane surface at low Y: 63.7% of scanner triangle
area lies between Y=-38.57 and -28.74, another 8.0% from -28.74 to -23.83,
then the next Y bin falls to 2.3%. Retain a scanner triangle **only if all
three vertices have Y > -24**, selected from this scanner-only area-density
valley. Reindex retained vertices and add no cap. The same exact hash-bound
reference ROI is used for every candidate; retain *all* triangles of every
candidate. This cut may remove bunny contact/lower-body surfaces along with
the plate, and the remaining scan is not a certified object-only ground truth.
The [reference-only preview](../.local-tools/bunny-scanner-roi-v1-preview.png)
must be inspected before scoring candidates.

Fit every whole candidate to this ROI independently using the unchanged
1,024-sample proper-Sim(3) algorithm. Score with 4,096 samples per direction,
seed 2027, in two predeclared panels: (A) 0.5/1/2% of the *cropped* scanner
bbox diagonal; (B) the unchanged original absolute thresholds
1.0429398885/2.085879777/4.171759554 scanner units. A and B answer
different tolerance questions; neither replaces the original whole-scan
score. No candidate-dependent reference selection, candidate crop, or
score-based alignment choice is permitted. The ROI was designed post hoc on
this dataset, so it is not held-out validation.

The exported ROI is [hash-bound here](../build-opencv/bunny-scanner-roi-v1/export-report.json):
SHA-256 `113c1a230a88559760fb7dbedf3e9e09fda9b099ebb4bf11953d5c860d33accc`,
197,747 of 435,006 original faces, bbox diagonal 122.8185228524 scanner
units. Panel A thresholds are 0.6140926143, 1.2281852285 and 2.4563704570
scanner units. Panel B retains the original absolute values; all six were
computed in one run per output on the same 4,096 draws.

| Whole candidate against ROI | Panel A 1%-of-ROI-diagonal precision / recall / F | Panel B original 2.085879777-unit precision / recall / F | ROI-normalized Chamfer-L1 mean |
| --- | ---: | ---: | ---: |
| [MVE](../build-opencv/bunny-scanner-roi-v1/mve-surface-metrics-4096-v1.json) | 12.38 / 62.74 / 20.68% | 20.31 / 70.61 / 31.55% | 3.44% |
| [Classical rough](../build-opencv/bunny-scanner-roi-v1/classical-rough-surface-metrics-4096-v1.json) | 12.94 / 13.33 / 13.13% | 22.73 / 21.83 / 22.27% | 4.90% |
| [Classical refined](../build-opencv/bunny-scanner-roi-v1/classical-refined-surface-metrics-4096-v1.json) | 13.79 / 14.18 / 13.99% | 21.19 / 22.24 / 21.70% | 4.74% |
| [Classical native-masked rough](../build-opencv/bunny-scanner-roi-v1/classical-masked-rough-surface-metrics-4096-v1.json) | 19.12 / 20.51 / 19.79% | 31.81 / 33.13 / 32.46% | 3.55% |
| [Classical native-masked refined](../build-opencv/bunny-scanner-roi-v1/classical-masked-refined-surface-metrics-4096-v1.json) | 19.56 / 20.29 / 19.92% | 31.84 / 32.91 / 32.36% | 3.55% |

The [MVE ROI preview](../.local-tools/mve-bunny-roi-preview.png) remains
amorphous and includes much extra surface: high ROI recall coexists with low
precision and no clear bunny silhouette. Classical
[rough](../.local-tools/classical-bunny-rough-roi-preview.png) and
[refined](../.local-tools/classical-bunny-refined-roi-preview.png) meshes
look bunny-like, but their independently fitted orientations remain
inconsistent with the paired scanner projections. Whether that stems from
registration, different geometry, or both is unresolved; the ROI scores do
not give a trustworthy intrinsic shape ranking. Scanner disk removal alone
does not resolve correspondence or missing/extra coverage. No further
post-hoc registration tuning was done for this batch. A held-out, independently
registered object or checked landmarks would be needed to separate causes.
Native-masked [rough](../.local-tools/classical-bunny-masked-rough-roi-preview.png)
and [refined](../.local-tools/classical-bunny-masked-refined-roi-preview.png)
show a clearer bunny outline and higher fitted ROI F than unmasked, but the
YZ view remains inconsistent and there is extra undersurface. These are
conditional fit-and-surface observations, not a validated mask-quality gain.
