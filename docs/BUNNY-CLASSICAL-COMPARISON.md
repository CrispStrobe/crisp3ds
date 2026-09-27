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

Pending the frozen scoring runs.

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
