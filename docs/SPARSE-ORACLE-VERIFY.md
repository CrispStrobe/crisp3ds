# Independent sparse reconstruction audit

`scripts/sparse_verify/verify.py` reads the image-derived COLMAP **text** model,
the mapper database, and a pre-mapping feature/match manifest. It uses only the Python standard
library. It never estimates a pose, camera, match, or 3D point. The producer
must convert its final binary model to text before verification.

The ten images are the corrected 768×512 tree photographs. The camera poses
and focal length are estimated by the sparse pipeline without supplied scene
poses. Images, scene background, and physical objects are shared between
training and scoring. The held-out claim concerns **feature observations**:
entire near-identical SIFT keypoint locations must be assigned to one split per
image before any training pair is matched. The manifest stores original
keypoint rows, held-out row IDs, all training match references, compact mapper
row to original row mappings, and all raw held-out descriptor matches. It also
stores each image path and SHA-256. A feature in the held-out set cannot appear
in any training match or mapper feature row, even through another image pair.
The independent database audit checks the actual keypoint blobs and complete
match table against the manifest's compact-to-original row mapping; it also
checks that no held-out feature centre is within 0.25 px of a training feature
centre. This conservative location test catches different SIFT orientations
at one spot. The first run fails it: the producer treated the third field of
COLMAP's six-column affine keypoint row as scale, which did not group these
locations correctly.
Held-out matches are scored as exported, including outliers; scoring does not
filter them using the reconstructed geometry.

The verifier recomputes the camera-relative essential matrix for every pair
whose two images registered, using world-to-camera rotations and translations.
It iteratively removes each camera's radial distortion, then computes square
root Sampson distance in **undistorted pixel units**. No `.5` shift is added
or subtracted: the manifest and COLMAP text coordinates are expected to use
COLMAP's pixel-centre convention (the first centre is `(0.5, 0.5)`). Focal
length and radial distortion therefore matter to the score. A pair without
two registered images is counted as unscored. Results include counts, median,
p90, and thresholds at 1, 2, and 4 pixels per registered pair and overall.
These are conditional prediction errors on raw feature matches. They are not
ground truth pose errors or physical scale measurements.

For an additional conditional three-view check, the verifier finds **closed**
triangles of raw held-out descriptor links. For each triangle whose cameras
registered, it triangulates the first two observations in lexical image order
using closest rays, then predicts the third through its full camera model.
This uses two held-out pixels for point construction, but none to fit cameras.
The minimum ray parallax for a usable prediction is frozen at 1°. Low-parallax,
nonpositive-depth, invalid, and third-view outlier counts stay visible; no
third-view residual is used to select a cycle or revise the reconstruction.

The text-model audit independently checks unit quaternions and proper
rotations, camera centres, camera references, point-track links in both
directions, track lengths, uniqueness of observations, positive depths, and
reprojection distances with the full camera model. Input image and model
hashes are recorded in the JSON report, along with manifest and verifier
hashes. `integrity_status` checks internal data consistency and at least one
scored held-out pair; `quality_accepted` remains false because this diagnostic
has no predeclared reconstruction acceptance gate. An integrity pass would
not mean that geometry is accurate.
COLMAP can hold multiple keypoints from the same image in one track; the audit
reports these tracks and the count with at most one observation per image,
while retaining a passing integrity result if all reciprocal links are valid.
Report medians, coverage, and the per-pair distribution separately. In
particular, low parallax, roughly planar structure, and repeated textures can
yield weak geometry despite a low epipolar residual. Few registered pairs or
many unscored held-out matches must be reported openly.

Run, using the paths from the sparse pipeline:

```sh
python3 scripts/sparse_verify/verify.py \
  --manifest PATH/manifest.json \
  --model PATH/text-model \
  --database PATH/database.db \
  --output PATH/verification.json
python3 scripts/sparse_verify/test_verify.py
```

The small tests include known distorted camera geometry, a shifted withheld
outlier, a leaked held-out row, a point behind a camera, and the `(0.5, 0.5)`
pixel-centre convention, an empty COLMAP observation line, supported multiple
same-image observations in a track, and invalid distortion inverses. They also verify
exact and shifted third-view predictions. The older `tree_epipolar` descriptor-order split is
a pairwise experiment and does not establish this global observation holdout.

## Actual run: `build-opencv/colmap-sparse/run-001`

The independent report is
[`verification.json`](../build-opencv/colmap-sparse/run-001/verification.json).
Its global held-out split is **invalid**: 2,830 held-out feature rows have a
training feature centre within 0.25 px. Accordingly `heldout_valid=false`
and `integrity_status=fail`; the numerical held-out errors below are
contaminated diagnostics, not an independent held-out result. The database
has all ten image feature tables, its 14,208 training matches equal the
exported manifest, and no identical feature row ID appears in both sets.
The model's rotations and reciprocal track links pass, and all 34 sparse
track observations have positive depth. Point #26 has two near-coincident
observations from image #2, a representation COLMAP supports; only 11 of 12
points meet the stricter one-observation-per-image condition. The fitted shared `SIMPLE_RADIAL` camera
has focal length approximately 769.40 pixels and radial coefficient -0.07152.

The geometric result is weak. Only 3/10 images registered, giving 12 3D
points and 34 observed point-track links. Of 1,314 raw held-out matches from
all 45 candidate pairs, only 79 have two registered cameras; 1,235 remain
unscored. The contaminated score's square-root Sampson median/p90 are 5.024/68.354 undistorted
pixels, and 36/79 are within 4 pixels. The sparse training-track reprojection
median/p90 are 0.214/0.534 pixels, but that describes just 34 selected
observations and does not offset the poor held-out result. Across all images
there are 173 raw closed three-view held-out cycles; only one lies entirely in
the three registered views, and it has less than 1° ray parallax. Thus zero
cycles provide a usable independent third-view reprojection score. These
coverage, parallax, and split leakage prevent claiming a robust ten-view
sparse model or a valid independent held-out assessment from this run.

For a model with no valid observation holdout, use the model-only audit:

```sh
python3 scripts/sparse_verify/model_only.py \
  --model PATH/text-model \
  --output PATH/model-only-audit.json
```

It checks model consistency, camera centres, depths, and reprojection, records
model hashes before and after the audit plus both verifier source hashes, but
sets `quality_accepted=false` and makes no independent prediction claim.
