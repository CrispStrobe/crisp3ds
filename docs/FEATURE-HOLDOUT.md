# Future global feature holdout

`scripts/colmap_sparse/feature_split.py` prepares a feature-observation split
for a future COLMAP sparse run. It does not reconstruct anything. The first
`run-001` split is invalid because it confused a field of a six-column affine
keypoint with SIFT scale; this module does not repair that historical run.

COLMAP keypoint rows may have 2 columns (`x,y`), 4 columns
(`x,y,scale,orientation`), or 6 columns (`x,y` and a 2×2 affine matrix).
All fields must be finite and rows must have one supported width. Spatial
grouping uses only the first two fields, so orientation and scale variants
cannot be assigned separately. A spatial hash finds every pair of centres
within **0.25 px**, including pairs on opposite grid-cell boundaries. Union
find takes the transitive closure, so a chain of nearby rows stays in one
component even if its endpoints are farther than 0.25 px apart.

For each image, SHA-256 ranks components using a fixed seed, image name, and
the component's sorted centre coordinates. The lowest `round(0.20 × group
count)` components are held out. The result is an immutable object containing
original row IDs, component memberships, and per-row assignments. Input row
permutation changes IDs but leaves the assignment at each location unchanged.
The percentage applies to components, so the percentage of individual
keypoint rows can differ. A future mapper must remove **all** held-out rows
from its keypoint and descriptor tables before any pair matching, and keep
the original-to-compact row map for independent verification.

The read-only dry-run audit uses seed
`colmap-spatial-components-v1-20260926` on the existing full-keypoint database
at `build-opencv/colmap-sparse/resized-supervisor-001/database.db`. It checks
the proposed split with the independent verifier's same 0.25 px neighbour
search, records before/after database hashes, source hashes, all original row
assignments, and per-image counts. The output is limited to 5 MB. The audit
requires at least 10 GiB free space and creates its output exclusively. It
does not alter the database, build a model, or establish pose accuracy.

The completed dry run is
[`feature-split-audit-001.json`](../build-opencv/colmap-sparse/feature-split-audit-001.json)
(253 KiB). Across ten images it read 53,546 keypoint rows, formed 44,062
spatial components, and proposed 10,664 held-out rows. The independent
neighbour check found zero train/held-out centres within 0.25 px. The
database SHA-256 was identical before and after. This validates the proposed
assignment boundary only; the assignment has not been used by a mapper.

```sh
python3 -m unittest scripts.colmap_sparse.test_feature_split
python3 -m scripts.colmap_sparse.feature_split \
  --database build-opencv/colmap-sparse/resized-supervisor-001/database.db \
  --seed colmap-spatial-components-v1-20260926 \
  --output build-opencv/colmap-sparse/feature-split-audit-001.json
```
