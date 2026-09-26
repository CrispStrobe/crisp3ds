# Pipes larger-context descriptor check

This bounded experiment tested whether doubling SIFT keypoint size helps flag
suspect correspondences among the **258 already reconstructed points**. It used
the same four images, resized views, detected feature centers and angles, and
strict mutual 0.8 L2 ratio matching. A point passed only when every pair of its
original observations still matched. No point, camera, or image coordinate was
changed. The frozen policy and evaluation limits are in
[PIPES-CONTEXT-PROTOCOL.md](PIPES-CONTEXT-PROTOCOL.md).

The completed run is [run-001](../build-opencv/pipes-context/run-001/verdicts.json)
(SHA-256 `8cd82ec832ac83ca75e4892598c04d6869631a7c79a3304bcafe04ade0c1f1a1`).
It retained **153/258** points. Applying the same all-pair rule to the original,
unchanged-size descriptors retained **231/258**; original tracks only required
connected matches, so this control separates the stricter rule from the size
change. All 3,595 detected features received context descriptors, and the run
reproduced all six original pair-match counts before evaluating the candidate.

Using previously computed laser proximity distances for the same frozen points,
the retained set has median **9.843 mm**, p90 **83.315 mm**, and maximum
**410.055 mm**. The unchanged-size all-pair control has median **12.305 mm**,
p90 **93.693 mm**, and maximum **7,451.33 mm**. The candidate rejected all
three previously inspected outliers, but also discarded **56 of the 151**
original points within 20 mm of the reference. These distances are proximity
diagnostics, not correspondence labels or a new reconstruction accuracy result.
Selection cannot recover missing geometry, and this inspected dataset is not an
independent acceptance test. There is no production promotion from this run.

| Policy | Points retained | Median proximity | p90 proximity | Within 20 mm / original 258 |
| --- | ---: | ---: | ---: | ---: |
| Frozen original | 258 | 12.798 mm | 90.280 mm | 151/258 |
| Original-size all-pair control | 231 | 12.305 mm | 93.693 mm | 137/258 |
| Twice-size all-pair candidate | 153 | 9.843 mm | 83.315 mm | 95/258 |

The candidate retains 323 observations, including only 17 three-view points
(baseline: 568 observations and 52 three-view points). At 20 mm its
original-population bad-or-missing fraction increases from 41.5% to 63.2%.
This is not a completeness metric, but makes the lost support explicit.
Next work should test recovery or independent confirmation of uncertain
correspondences, rather than simply deleting them or sweeping this threshold.

The separate evaluator completed at
`build-opencv/pipes-context/evaluation-001/evaluation.json`, SHA-256
`dcdf0c1665e1f73fd5d3b543eddaa59d1e552db2b01f0509b6425ea85b243851`.
It pins the exact previous score artifact and requires the pre-scoring verdict
hash, validates pair witnesses and counters, and checks inputs before/after.
Root independently recomputed candidate/control counts and proximity
statistics from the sealed files with Python's standard library; they agree.
The guarded descriptor run took 2.708 seconds and recorded 311,017,472 bytes
peak RSS. Evaluator `status: pass` means input/integrity validation, not a
quality gate pass. Five verifier and seven evaluator unit tests pass.

The verdict file records per-point pair witnesses, missing-descriptor reasons,
and SHA-256 hashes before and after for the four images, prepared manifest,
frozen points and report, protocol, implementation, and runtime files. It also
checks the Python, OpenCV, NumPy, and Pillow versions and runtime binaries
against the frozen baseline. These checks establish replay consistency for this
machine and input set; they do not show that the policy generalizes to other
scenes or runtimes.

To reproduce in a **fresh** output directory, with the prepared inputs and
same runtime available:

```sh
/usr/local/bin/python3 -m scripts.pipes_context.run \
  --output build-opencv/pipes-context/run-002
```

The runner enforces a 300-second limit, a 128 MiB output cap, and a 10 GiB free
space reserve. Its default inputs are `build-opencv/pipes-prepare/run-001` and
`build-opencv/pipes-sparse/run-003`; the verdicts do not read the laser score.
