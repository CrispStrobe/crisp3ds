# Stereo benchmark quality diagnostics

`scripts/analyze_stereo_benchmarks.py` examines an already completed matrix from
[ORACLES.md](ORACLES.md). It uses only the Python standard library and reads the
recorded truth and disparity PFM files. It checks their SHA-256 hashes against
`benchmark.json`, checks dimensions, and verifies full-image counts against the
evaluator's recorded counts. It does not run or tune a matcher.

```sh
TMPDIR=.local-tools/tmp python3 scripts/analyze_stereo_benchmarks.py \
  build-opencv/benchmarks/run-20260926T105252Z-5527a422/benchmark.json \
  --output-dir .local-tools/quality-diagnostics-20260926T105252Z-v2
```

The output directory must be new. The script checks that at least 10 GiB of
free space remains after estimated output bytes before writing. Images above
10 million pixels are rejected. `diagnostics.json` contains counts and rates;
each scene also has one binary PPM (`P6`) category map per engine. In the maps,
gray means invalid truth, blue means missing prediction on valid truth, green
means error ≤2 pixels, and red means error >2 pixels. Black means excluded by
an optional evaluation mask. The maps are categorical, not calibrated
continuous error heatmaps.

Counts use the evaluator's validity rule: truth and prediction are valid when
finite and ≥0; zero is valid. PFM rows are bottom first. Scale sign selects
byte order, and scale magnitude is ignored, matching the evaluator and
Middlebury disparity convention. A missing prediction and an incorrect finite
prediction are distinct categories. `bad2_all_valid` divides their sum by valid
truth pixels; MAE and RMSE divide by finite matched predictions only. Rates
without a denominator are JSON `null`.

The `full` region uses all pixels accepted by the optional prepared evaluation
mask. `left_search_strip` is x below the scene's common disparity search count.
The fixed geometric `interior` spans x from `max(ceil(width/10), search_count)`
to `width-ceil(width/10)` and y from `ceil(height/10)` to
`height-ceil(height/10)`, with exclusive upper bounds. The strip and interior
measure location effects; neither identifies an actual occlusion, interpolation,
or confidence mask. `common_full` and `common_interior` keep only truth-valid
pixels where **every** engine has a valid prediction. Their shared denominator
permits error comparison on identical pixels, but this population is selected
by engine outputs and is therefore selection biased. It must not replace
full-image coverage and bad2 rates.

## Recorded four-scene result

These counts come from the 2026-09-26 benchmark in [ORACLES.md](ORACLES.md),
analyzed into `.local-tools/quality-diagnostics-20260926T105252Z-v2/`.

| Scene | Full valid truth | SGBM missing | SGBM missing in left strip | Common predicted truth | Interior valid truth | Interior bad2 all: SGBM / ELAS / Census |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cones | 163,321 | 40,819 | 35,998 | 119,701 | 90,066 | 8.96% / 5.86% / 11.06% |
| Piano | 318,804 | 51,452 | 31,656 | 262,894 | 205,764 | 19.19% / 13.22% / 26.96% |
| Teddy | 165,344 | 42,045 | 35,930 | 120,789 | 89,959 | 8.65% / 5.44% / 17.31% |
| Venus | 166,222 | 37,644 | 36,768 | 125,442 | 89,670 | 2.70% / 2.72% / 21.81% |

The left search strip accounts for about 88%, 62%, 85%, and 98% of SGBM's
missing predictions in Cones, Piano, Teddy, and Venus respectively. The
remaining missing pixels and all incorrect finite pixels still matter. On the
common full-image prediction set, SGBM/ELAS/Census bad2 matched rates are
5.07%/4.91%/8.52% for Cones, 17.77%/13.95%/27.04% for Piano,
7.12%/8.67%/18.39% for Teddy, and 1.65%/2.17%/19.92% for Venus. These
common-set rates improve for some engines because hard or unsupported pixels
are excluded. They cannot establish that an engine will predict those pixels
well in a real scan.

These are rectified stereo-pair disparity diagnostics, not turntable-scan,
object-dimension, or reconstructed-surface accuracy. The scripts have no
engine-provided confidence or actual interpolation masks to inspect.

Run the focused tests with:

```sh
TMPDIR=.local-tools/tmp python3 -m unittest scripts/test_analyze_stereo_benchmarks.py -v
```
