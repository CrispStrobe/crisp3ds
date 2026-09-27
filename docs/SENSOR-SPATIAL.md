# Frozen spatial audit of the three-view sensor diagnostic

This is a **read-only partition** of the already frozen
`build-opencv/sensor-depth-003/report.json` (SHA-256
`40b1d821b9e7c8a9eaddf6946815922af93c3d72bacee25cfff0006dd3b0e8a7`).
It does not project new pixels, cast new rays, fit cameras or meshes, change
support, select a threshold from results, or run native reconstruction.

For each of the existing 0°/120°/240° frames, partition the 640×480 **depth
image** into a fixed 4×4 grid of 160×120-pixel cells, indexed row then
column. Each selected linear pixel ID maps to exactly one cell by integer
division; no cell boundary follows the object or a residual. Report both
the existing coarse photo-mask support and its existing fixed 16-RGB-pixel
eroded-interior subset. In each cell/panel/candidate, report supported ray
count, first-hit count, no-hit count, first-hit fraction, and the fraction of
**all supported rays** with a first hit within 5 mm and 10 mm. Missing rays
fail these thresholds; they are not assigned a depth or a zero residual.
Also report hit-only absolute mean/median/p95 where hits exist, explicitly
separate from the missing-inclusive rates. Empty cells have null rates and
residuals. The thresholds and support are inherited from the parent score.

Before calculation, verify the exact parent report hash/schema/status,
2,048 sorted unique IDs per view, identical observed depths and interior
membership across the two candidates, finite positive observed depths,
finite positive or null predictions, and the parent selection ID hash.
The output binds the parent hash, source mesh/report hashes and this script
hash. ≤1 MB JSON, ≥10 GiB free disk. No new dependency.

The bins describe *sampled, photo-support-conditioned rays*, not equal-area
object surface or all sensor pixels. The image-space population varies by
view and region; 2,048 rays were selected uniformly from each view's eligible
mask, so regional counts are unequal and small cells can be noisy. Erosion
changes the population. A depth miss can arise from support/projection,
visibility, camera, or mesh geometry. The supplied table poses, camera-only
Sim(3), assumed rectified depth grid/no second IR distortion, and IR-scale
interpretation remain uncertain. This audit is not physical ground truth or
a new quality acceptance gate.

## Results

The one read-only [spatial report](../build-opencv/sensor-spatial-001/report.json)
completed at 100,840 bytes, SHA-256
`e4ef2feaf5af58b24d9f9988c6e364364a207bba787f2e19332659ec3c0d0738`.
For both candidates and all views, cell counts sum back to the parent's 2,048
selected rays, parent first-hit counts, and parent interior counts. No parent
ray or score was changed.

Support is highly localized: four central cells contain all selected rays at
0° and 120°; at 240°, those cells contain 2,036 of 2,048 rays and one
upper-left-adjacent cell has only 12. The other 11–12 cells are empty and
cannot support any spatial quality inference. As a concrete missing-inclusive
example, at 120° in cell (row 1, column 2), 1,097 selected rays were
supported; the fraction with a first hit within 5 mm was 84.9% for rough 008
and 11.5% for rough 014. In 0° cell (row 2, column 1), however, rough 014
had 18.0% versus 7.3% for 008 among 205 supported rays. The diagnostic is
therefore spatially uneven, not a claim of pointwise dominance. The 12-ray
240° cell has no stable comparative meaning.

These cells are depth-image locations, not matched patches of object surface
across views. Poor local coverage or agreement may reflect photo-support
boundaries, depth/RGB reprojection, missing surfaces, different visibility,
camera/turntable pose, or shape. No cause is identifiable from these bins
alone, and no physical ground-truth ranking follows.
