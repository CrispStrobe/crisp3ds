# Frozen held-out three-view audit

This read-only diagnostic uses `heldout-v2-001/holdout.json`, its COLMAP text
model, and the independent `verification-supervisor.json`. It reads the ten
source 768×512 PNGs for a local visual sheet. It performs no feature detection,
matching, fitting, pose adjustment, camera adjustment, or geometry filtering.
Input hashes are checked before and after; output is a fresh directory under
`build-opencv/three-view-audit`, capped at 50 MiB with 10 GiB free reserved.

All 181 closed triangles in the raw held-out match graph are enumerated in
lexical `(image name, original feature ID)` order. For each cycle the report
saves all three source observations, the three per-edge square-root Sampson
distances, and registration status. For registered cycles it triangulates the
first two lexical views by the closest-point midpoint of their undistorted
world rays. The exact original availability rule stays frozen: a ray angle
below 1° is unavailable. Otherwise report closest-ray gap, first-pair
reprojection in each image, third-view depth and reprojection. Nonpositive
depths and arithmetic failures have explicit status. This lexical result must
reproduce the independent verifier's 181 closed / 179 registered / 73 low
parallax / 106 scored counts and summary; a mismatch fails the audit.

As a diagnostic with no selection, the same computation is repeated for all
three orientations: AB→C, AC→B, and BC→A. Every orientation and its status
are reported, including ones with low parallax or poor reprojection. These
additional residuals do not replace the frozen lexical result and are not
used to choose a favourable triangulation. Counts of repeated feature IDs
across cycles and cycle overlap indicate that the 181 cycles are correlated,
not 181 independent observations.

An exploratory camera-only policy selects the two registered cameras with
the longest centre-to-centre baseline in each cycle (ties by lexical image
names), triangulates their two held-out observations, and predicts the
remaining image. It uses camera centres alone for selection; it never uses
feature angle, reprojection, or the third pixel. All 181 statuses are
reported. On the **fixed 106 cycles eligible under the original lexical
rule**, report finite median/p90, within-4 count, and missing-inclusive
bad-over-4 count. Also report how often this policy predicts a different
target image. Its errors are not paired accuracy measurements on identical
target pixels when the target changes. This already-seen development data
is not a new acceptance gate.

The raw held-out match graph is also summarized by connected component.
Report components containing more than one feature ID from the same image,
and how above-4-px lexical failures share graph components and individual
features. Such conflicts can arise from incorrect candidate matches; no
camera or semantic cause is assigned automatically.

The local HTML sheet shows every registered frozen lexical cycle with finite
third-view error above 4 px (expected 21 from the saved verifier result) and
the ten lowest-error finite cycles as deterministic controls, ties broken by
cycle ID. Each row shows 192×192 crops centred on the three held-out feature
locations, with a red crosshair; the third crop also marks the lexical
prediction in blue when it lies inside the crop. It references the original local PNGs and does
not classify visual content or declare any match semantically wrong. The
sheet is an inspection aid, not a new score or ground truth.

```sh
python3 -m unittest scripts.three_view_audit.test_run
python3 -m scripts.three_view_audit.run \
  --output build-opencv/three-view-audit/run-001
```

## Supervisor verification and visual inspection

The fresh `supervisor-001/report.json` is byte-identical to `run-001/report.json`.
A separate NumPy least-squares implementation, using native PyCOLMAP pose
loading and its own triangle enumeration, agrees on all 181 cycle references.
Across the 106 usable cases, the maximum third-view error difference is
3.12e-10 pixels and maximum point-position difference is 1.77e-10 scene units.
The standalone checker and four tests are in `scripts/three_view_numpy/`;
the supervisor report is `build-opencv/colmap-sparse/three-view-numpy-supervisor-001/report.json`.
This rules out the suspected closest-ray implementation error on this fixture.

Chromium loaded all 93 images across the 31 selected contact rows. Root
spot-checked cycles 69, 39, 3, 60, 154 and 88: repeated building features,
grass, and sky/horizon regions appear in these cases, including the lowest-error
control. These are outside the target tree. Several gross failures are
consistent with ambiguous background correspondences; visual inspection is
not a physical correspondence oracle and cannot assign one cause to all 21
failures. See the retained
[visual notes](../build-opencv/three-view-audit/run-001/visual-review-supervisor.md).

Next quality work should measure foreground support explicitly, exclude sky
and irrelevant background **before** matching in a new experiment, and test
view selection on a controlled rigid-object capture. Keep the full current
results unchanged; post-hoc removal would invalidate the comparison.

## Frozen result

The read-only report is
[`run-001/report.json`](../build-opencv/three-view-audit/run-001/report.json),
with the [local contact sheet](../build-opencv/three-view-audit/run-001/contacts.html).
All saved model, manifest, PNG, and verifier source hashes matched before and
after. Six synthetic tests passed. The lexical calculation reproduced the
saved verifier exactly: 181 closed cycles, 179 registered, 73 below 1°,
106 scored; third-view median/p90 1.355/16.261 px and 85/106 within 4 px.
Thus 21 scored cycles exceed 4 px; the sheet includes all 21 and ten
deterministic low-error controls. Local browser inspection confirmed the
original PNG references load and the red/blue crosshairs render.

The 21 high-error cycles are heterogeneous. Ten have first-pair parallax
below 2° despite clearing the 1° availability threshold. Their median
first-pair ray angle is 2.088° versus 2.870° for the other 85 scored
cycles. The median AC edge square-root Sampson distance is 3.218 px in
high-error cycles versus 0.698 px in the others; 8/21 high-error cycles
nevertheless have all three edge distances at most 2 px. Thirteen of the
21 have both first-pair reprojection residuals at most 2 px. These patterns
show that low parallax and inconsistent candidate links can both contribute;
they do not isolate a single camera or feature failure mode.

Across all 543 two-ray orientations, 380 score, 153 are below 1°, 6 have
unregistered cameras, and 4 have nonpositive depth. For the exploratory
longest-camera-baseline policy, the fixed set of 106 originally eligible
cycles has 105 scored and one nonpositive-depth case; 94/106 are within 4 px
and 12/106 are over 4 px or unavailable. Its finite median/p90 are
0.665/4.045 px. It predicts a different target image for 89/106 cycles,
including all 21 original high-error cycles. These residuals are therefore
not a same-target comparison and do not replace the lexical result.

The 181 cycles use only 370 distinct held-out feature observations: 80
observations recur across cycles, 88 cycles contain a recurring observation,
and 165 cycle pairs share at least one observation. The raw candidate graph
has 750 connected components, three with multiple feature IDs from the same
image. Two of the 21 high-error cycles belong to those conflicted components;
the 21 errors span 16 components, with no component containing more than
three. The geometry and contact sheet remain diagnostics on shared images,
not independent physical ground truth.
