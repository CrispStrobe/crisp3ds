# MVE quality ablation on the rendered sparse fixture

This is a diagnostic of the existing, **unmasked** MVE dense path on the immutable
`synthetic-sparse-r02` fixture. It does not establish measured-capture accuracy or
justify a production backend. The six conditions were declared before scores were
viewed: estimated poses and sparse seeds, subpixel ArUco poses with observed-track
retriangulation, and known fixture poses with observed-track retriangulation, each
at MVE scales 2 and 1. The known-pose condition is a diagnostic ceiling only; it
uses test truth unavailable in production. No ground-truth object point enters any
condition's seed generation.

Run from the repository root (requires the existing pinned OpenCV and MVE builds):

```sh
PYTHONPATH=scripts/quality_ablation python3 -m unittest scripts/quality_ablation/test_quality.py -v
python3 scripts/quality_ablation/run.py --output build-opencv/quality-ablation-new
```

The output directory must be fresh. The runner records SHA-256 of the fixture
images, sparse report, truth scoring file, scripts, selected MVE source files,
and MVE executable. It checks those inputs before and after each dense run,
enforces a 10 GiB free-space reserve, times out subprocesses, and writes a
failure status if one fails. `manifest.json`, `results.json`, `status.json`, and
per-run logs live in the ignored output directory. Temporary files are directed
to `.local-tools/tmp`. The selected MVE pin is
`bf2279f161ba962072ecac85224c15e82bc5f52e`; the exact build and license
record is in [MVE-DENSE-SPIKE.md](MVE-DENSE-SPIKE.md).

The baseline uses `sparse-report.json`'s estimated object-to-camera poses and
1,189 sparse points. Both altered-pose cases solve every 3D seed anew from the
same 2,843 observed ORB pixels by least-squares intersection of camera rays.
The subpixel case redetects the same four board markers with OpenCV
`CORNER_REFINE_SUBPIX`, then uses the production IPPE+LM pose solver and the
fixture calibration. It changes no dense image, track observation, neighbor
count, or MVE option. All cases use the same unmasked 1600×1200 grayscale PNGs,
master view 2, two neighbors, two local neighbors, and scales 2 (400×300) and
1 (800×600).

## Scoring interpretation

The fixture's known camera projects each output pixel onto the nearer finite
object rectangle, defining the same truth pixel footprint for all cases at a
given scale. Each valid MVE radial depth becomes a 3D world point using that
case's camera pose. The primary error is its absolute world-Z distance to the
visible truth plane, in mm. This is point-to-plane geometry, not an invalid
comparison of radial depth with camera-Z. Because point-to-plane distance does
not penalize lateral displacement beyond a finite rectangle, the result also
reports Euclidean distance to the **finite** truth rectangle. These are
visible-panel diagnostics and cannot measure back surfaces.

Coverage divides matched object pixels by all truth object pixels. Matched bad-5
is the fraction of matched pixels with error strictly over 5 mm. Missing-inclusive
bad-5 counts a missing truth pixel as bad. Edge coverage and edge error use the
truth pixels whose ray hit is within 3 mm of a rectangle boundary; edge truth is
used only in scoring. Valid depths outside both object rectangles are counted,
but are not attributed solely to stereo error because masks are not enforced.

## Results

See [`results.json`](../build-opencv/quality-ablation-final/results.json) for exact
values, per-run peak resident bytes, wall time, percentiles, finite-rectangle
scores, paired-grid scores, camera errors, and seed reprojection checks. Rounded
primary scores:

| Camera / seeds | Scale | Truth pixels | Coverage | Point-to-plane MAE | P95 | Missing-inclusive bad-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Estimated / estimated | L2 | 7,672 | 94.17% | 3.081 mm | 5.531 mm | 12.81% |
| Estimated / estimated | L1 | 30,514 | 98.51% | 2.130 mm | 3.919 mm | 2.54% |
| Subpixel / retriangulated | L2 | 7,672 | 94.15% | 3.481 mm | 5.989 mm | 15.85% |
| Subpixel / retriangulated | L1 | 30,514 | 98.53% | 2.567 mm | 4.464 mm | 3.80% |
| Known pose / retriangulated | L2 | 7,672 | 93.87% | 1.473 mm | 3.918 mm | 7.98% |
| Known pose / retriangulated | L1 | 30,514 | 98.60% | 0.660 mm | 1.896 mm | 1.54% |

Resolution helps even after restricting both scales to the same L2 cells where
both produce valid depth: estimated MAE is 3.047 mm at L2 versus 2.131 mm for
the mean valid L1 child errors. That paired subset is selected by validity and
is a comparison diagnostic, not a replacement for full-coverage scores. The
known-pose ceiling at L1 removes about 1.47 mm of matched point-to-plane MAE
relative to the estimated baseline; therefore camera/seed uncertainty is a
material contributor at this resolution. The test does not uniquely apportion
that gap between pose and seed errors because they change together to maintain
consistent tracks. It also does not prove that any particular pose algorithm
will close the gap in production.

Subpixel refinement worsened this fixture at both scales. The rendered marker
texture places pixel centres on the declared physical marker corners; an actual
marker has its outer boundary roughly half a source pixel beyond those centres.
This convention can bias corner-refinement comparisons. Independent renders
with correct outer-boundary geometry and varied subpixel position and blur are
needed before accepting or rejecting subpixel refinement for measured images.
The original fixture, conversion scripts, MVE source, and original spike scores
were not changed.
