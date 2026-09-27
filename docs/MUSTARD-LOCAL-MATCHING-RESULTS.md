# Mustard local-matching trial: incomplete orbit

The live runner is frozen at commit `a35bcd1`, SHA-256
`5d1cff55cbd6ddf5e7e27c63358fbe988608bb9cf3f51de7400a244d5b2c030d`.
The later database-guard changes were not used for this experiment.

The one [predeclared local-four graph trial](MUSTARD-LOCAL-MATCHING-PLAN.md) did **not** recover a valid full training orbit. Its mapper exited zero and preserved one 24-camera, 528-point candidate, but that is below the unchanged 34-of-48 producer registration gate. The producer `result.json` remains `status=failed` (SHA-256 `c1187d6434216e173b11863220ca2db374077812385573bf9bcc80df7058217c`); this candidate is diagnostic, not an accepted sparse model. No dense reconstruction, reference fit, held-out evaluation, or further matching trial followed.

The copied database used exactly the 48 accepted TRAIN images and the frozen symmetric cyclic distance ≤4 in their **sorted selected-name positions**. It retained 192/1,128 pair rows in both match and verified-geometry tables, including 173 nonempty verified pairs. The graph stayed connected across all 48 image names. The mapper used the same fixed-initial intrinsics, feature/descriptors, seed, automatic initialization and options as the exhaustive comparison. Both native mapper logs chose **the same initial image IDs 3/6**, `NP3_012.jpg` and `NP3_036.jpg`; the local rule already included this pair. Thus the outcome does not support a claim that removing a bad *nonlocal initial pair* repaired the model. The local trial's mapper log SHA-256 is `6be67c601400d451e6bdcf813a1a0d53c62ef37ac055dba5f70881dfe04b9958`.

The original trial additionally failed its strict *byte-for-byte copied-database* postcheck: the database hash changed after the mapper read it, while independently checked logical image, camera, feature and retained-pair rows stayed identical. The exact trial remains failed; a later semantic-seal guard is a code correction for future work, not a retroactive rerun or quality promotion. The [database diagnostic](../tests/evidence/mustard-local4-database-diagnostic.json) records that distinction. It does not change the more important 24/48 registration failure.

An [independent read-only model/trajectory diagnostic](../tests/evidence/mustard-local4-failed-diagnostic.json) binds both result/database/model hashes and the sparse-metric/trajectory code hashes before and after inspection. The selected local model registers two partial arcs, `NP3_000..108` and `NP3_300..348`, leaving 24 TRAIN views absent. The registration row below describes the full runs; all pair, residual and trajectory rows restrict analysis to the 24 names shared with the exhaustive model:

| Diagnostic population | Exhaustive baseline | Local-four candidate |
| --- | ---: | ---: |
| Total registered TRAIN views (of 48) | 48 | 24 |
| Verified pair edges among common 24 | 129 | 83 |
| Verified correspondences among common 24 | 5,705 | 4,774 |
| Points with common-view observations | 658 | 528 |
| Finite common-view reprojections | 2,885 | 2,737 |
| Mean / p95 reprojection error, pixels | 1.028 / 2.754 | 0.851 / 2.515 |
| Median short-adjacent center step / subset radius | 0.15792 | 0.15771 |
| Median short-adjacent orientation step | 6.43° | 6.23° |

The error figures are **not paired errors on identical 3D points or observations**: the local model retained fewer, different tracks, so its lower residual can result from selection. It also has 28 repeated-image tracks, an integrity caveat. Both common-24 trajectory summaries use only 23 genuine short adjacent edges; the 192-label-degree `NP3_108→NP3_300` gap is excluded rather than falsely treated as an adjacent step. There are **zero opposing-view pairs** within those 24 names. By contrast the full exhaustive 48-camera model has 24 opposing pairs and a median opposing-center distance of only 0.179 of its median planar radius, a fold/alias warning. The local candidate cannot show that this full-orbit defect was removed, because its missing arc removes the test.

This was a graph-factor ablation, not an object-quality benchmark. The restricted verified graph remains connected across all 48 training images while the mapper registers only 24: **connectivity is insufficient for geometrically consistent camera recovery**. A lower residual on a smaller selected population cannot validate the cameras or shape. The next work should improve and independently validate training-image correspondences and camera coverage, not merely lower neighbor or registration thresholds. Any new arm needs its own frozen hypothesis and checks before dense continuation; the separate Google mesh must not become a tuning objective.
