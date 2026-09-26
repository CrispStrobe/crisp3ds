# ETH3D pipes: one-way sparse-to-laser distance

The first measured readout is the [frozen four-view sparse baseline](PIPES-EVALUATION-PROTOCOL.md), scored **after** its `points.json` and `report.json` have been sealed. The scorer is `scripts/pipes_score/run.py`; it never writes to the prepared reference or sparse run. Its single output is `score.json` in a new output directory.

Run with `/usr/local/bin/python3 -m scripts.pipes_score.run --prepared build-opencv/pipes-prepare/run-001 --sparse <frozen-sparse-run> --output <new-score-run>`. The output path must not exist and its parent must already exist. The repository research-job guard runs the scorer child with a hard 300-second timeout, a 256 MiB output cap and a 10 GiB free-space reserve; it writes `status.json` and `child.log`. The worker additionally checks elapsed time during large loops, disables OpenCV parallelism with `setNumThreads(0)` and requires an effective count of one, rejects sparse JSON over 128 MiB, and rejects a score report over 128 MiB. It reads at most 12 million scan vertices. The scorer records its process peak RSS and platform-reported unit; a supervisor may additionally record system-level peak memory.

The reader validates the prepared PLY metadata and reads only the first `vertex` element, whose `x/y/z` properties are binary little-endian floats. Other PLY elements, including the trailing camera element, are excluded. Each scan vertex is checked for finite coordinates and transformed once using the `MLMatrix44` attached to `scan1.ply`; the transform must be a proper rigid transform and agree with `prepare-metadata.json`. The sparse points are already in the supplied camera-world frame. There is no scale fitting, ICP, output filtering or laser-informed point selection.

The denominator is every accepted sparse point in `points.json`, matched to the count in `report.json`. A missing or empty sparse result fails. Every query is compared with **every** transformed scan vertex in 250,000-vertex chunks using float64 arithmetic; the smallest squared Euclidean distance is square-rooted to metres. Equal distances choose the lowest scan vertex index. This uses bounded memory and does not rely on an approximate search setting. Synthetic tests compare all queries on a small random cloud to independent all-pairs distances, check a known rigid translation and rotation, and place the true nearest point across a chunk boundary. The earlier `score-001` attempt failed its FLANN-versus-full-scan audit by 0.000246 m on one query; that method is deprecated, and its failed output is not a score.

`score.json` records source, prepared-reference, sparse-input, protocol and library-binary hashes; the applied matrix; scene unit and library versions; peak process RSS; the count of laser vertices and sparse points; every sparse-point distance and nearest vertex index; and counts/fractions at `0.01`, `0.02`, `0.05` and `0.10 m`. It also reports median, p90, p95 and maximum distance. The scorer hashes all inputs again before writing; the SHA-256 of `score.json` is printed in `child.log` after the exclusive write. A frozen score report can be reproduced from the hashes and code revision.

## Frozen first-run result

The guarded [score-002 status](../build-opencv/pipes-score/run-002/status.json) is `succeeded` (68.605 s). It scored all 258 accepted points from `pipes-sparse/run-003` against all 11,482,717 eval-scan vertices. The [score artifact](../build-opencv/pipes-score/run-002/score.json) records 572,391,424 bytes of peak process RSS and the source, input and protocol hashes. The one-way distances were median **0.0128 m**, p90 **0.0903 m**, p95 **0.1515 m**, and maximum **7.4513 m**.

| Distance to nearest measured scan point | Accepted sparse points within threshold | Fraction of all 258 |
| --- | ---: | ---: |
| ≤ 0.01 m | 119 | 46.1% |
| ≤ 0.02 m | 151 | 58.5% |
| ≤ 0.05 m | 188 | 72.9% |
| ≤ 0.10 m | 238 | 92.2% |

The first [score-001 status](../build-opencv/pipes-score/run-001/status.json) is `nonzero_exit`: its FLANN candidate missed the independently computed nearest distance by 0.000246 m on point 74. That failed attempt produced no `score.json` and is excluded. The exhaustive scorer in score-002 changed the search implementation to satisfy the existing exact-distance protocol; it did not change the sparse points, thresholds, or reference geometry.

This is **one-way proximity to measured points**. Unscanned geometry can make a valid reconstruction look distant, and proximity alone does not certify visibility or correct surface ownership. There is no scan-to-reconstruction denominator, four-view completeness, official ETH3D accuracy, or F1 score in this first run. The [ETH3D paper](https://www.eth3d.net/data/schoeps2017cvpr.pdf) explains its laser-beam free-space handling and voxel normalization; the [official evaluator](https://github.com/ETH3D/multi-view-evaluation) implements those definitions separately.

These results do not establish production surface quality. This run is sparse, only four selected views were used, visibility remains unresolved, and the long distance tail includes severe outliers. The measured fractions are diagnostic evidence for geometry work, not an acceptance pass.
