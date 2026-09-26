# Real turntable-object evaluation

## Protocol fixed before inspecting the reconstructed mesh

Input: all 73 original `pro/bunny/rgb` photographs from the acquired
[3DLF-Scan subset](OBJECT-DATASETS.md), 1749 × 1155 pixels each. No supplied
camera poses, calibration, masks, depth maps or reference vertices are passed
to reconstruction. The raw run is retained even if a later variant succeeds.

The second run uses the named, image-only `gamma05_clahe2` profile in
`scripts/mve_full/prepare.py`. It preserves originals and records input/output
hashes. No profile parameters are fitted to reference geometry. MVE settings:
maximum 2,100,000 pixels, depth scale 2, four neighbors, two local neighbors,
73 selected views, 70% minimum camera registration, 2 GiB output cap,
20-minute reconstruction deadline and 10 GiB minimum free disk.

The independent Revopoint physical-instance scan is the reference, not the
canonical Stanford design mesh. The scanner mesh has 217,505 vertices and
435,006 triangles, including four zero-area triangles. Its physical units
and cross-sensor transformation have not been independently verified.

The frozen reference-fit diagnostic uses 1,024 area-weighted samples with
seed 2026, 24 proper PCA initializations, four candidates refined for eight
iterations each, then twenty iterations for the best candidate. Both
directions contribute to correspondence fitting and symmetric RMS ranking.
This fits translation, rotation **and scale to the reference**; it cannot
measure independently recovered physical scale or camera accuracy.

Final scoring uses 4,096 samples per mesh, seed 2027 (different from alignment),
and a distance threshold equal to **1% of the reference bounding-box diagonal**.
Distances remain in reference-coordinate units, not asserted millimetres.
Report both directions, precision, recall and F-score. Nearest sampled-point
distances are approximate and include surface-sampling spacing; they are not
exact point-to-triangle distances or an official benchmark score. An additive
reference-versus-reference sampling control uses distinct seeds 2027 and 2028
at the same sample count and threshold without changing the primary protocol.

Inspect both reconstructed and reference surfaces visually. Full-background
or partial-object output is not accepted merely because cameras registered,
the PLY is structurally valid, or a fitted subset lies near the scan. Neither
a reference-fit score nor one successful object establishes production quality.

Runtime includes all reconstruction stages and validation. Report image
preprocessing separately if measured in a repeat rather than the original
run. Largest-process RSS is not the peak simultaneous process-tree sum.
Only Apple M1 CPU execution is currently measured.

## Results

The raw-photo run failed camera initialization after 52.52 seconds. Its failed
artifacts remain at `build-opencv/mve-full-bunny-001`. The contrast-profile
run and subsequent evaluation are recorded separately; no failed baseline
is overwritten or silently omitted.
