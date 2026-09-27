# Paired coverage-loss audit: rough008 versus rough015

This is a read-only analysis of the sealed
`build-opencv/sensor-depth-004/report.json` (SHA-256
`9027600ccac5488ec422a6adc3736979eb126a7b33d9d8d19d851b107327cc5c`).
`scripts/object_dataset/coverage_loss.py` uses exactly the prior 2,048 selected
depth rays at each of three angles (0°, 120°, 240°), the same observed depth,
and the same fixed photo-derived coarse support and 16-pixel-eroded interior.
It introduces no rays, masks, alignment, model fit or threshold selection.

Each ray is classified as hit by both rough meshes, lost (008 hit/015 miss),
gained (008 miss/015 hit), or missed by both. The fixed 4×4 depth-image grid
uses 160×120-pixel cells; every cell and whole view has coarse, eroded-interior
and boundary-band (coarse minus interior) rows. Five- and ten-millimetre rates
use **all supported sampled rays** in their denominators, counting a miss as
not within threshold. Residual distributions are computed **only on the paired
both-hit rays**, separately from transition counts, to avoid a coverage-driven
change in the residual population masquerading as a geometry improvement.
The report checks that all four hit states partition each support and that
pooled state counts equal the sum over views.

The bounded output is `build-opencv/coverage-loss-001/report.json` (SHA-256
`fe56eeb9dc4a8dfd2d892e3c82582063ec391fc7ac860a68db060aaec5b8fa3c`,
318,172 bytes; source and analyzer hashes embedded). Pooled results:

| Support | Rays | Both hit | 008-only hit (lost) | 015-only hit (gained) | Both miss | 008/015 hit fraction | Paired both-hit absolute mean, 008→015 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse | 6,144 | 4,951 | 215 | 0 | 978 | 84.08% / 80.58% | 2.754 → 2.922 mm |
| Eroded interior | 3,787 | 3,466 | 40 | 0 | 281 | 92.58% / 91.52% | 2.316 → 2.451 mm |
| Boundary band | 2,357 | 1,485 | 175 | 0 | 697 | 70.43% / 63.00% | 3.777 → 4.019 mm |

Lost-ray counts by view are 68, 95 and 52 at 0°, 120° and 240° respectively;
no gain appears in any of the three views. Of the 215 lost rays, 175 are in the
non-eroded boundary band. The 5 mm coarse missing-inclusive supported rate is
70.72% for 008 versus 68.10% for 015. Cell-level counts and shared-hit
residual distributions are retained in the JSON for independent inspection.

These are sampled, object-support-conditioned sensor residuals. They are not
whole-surface completeness or physical ground truth. The Berkeley depth/RGB
projection, IR scale, table poses, camera-only Sim(3), support mask and
first-hit rendering each carry unresolved error; the 008 and 015 camera models
also differ. Spatial coincidence of lost rays with a boundary band does **not**
show that a particular reconstruction stage or masking choice caused the loss.
