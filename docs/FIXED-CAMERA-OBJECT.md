# Fixed-camera foreground object tracks

Supervisor verification: a fresh `fixed-camera-object-supervisor-001` replay
produced byte-identical `points.json` and `report.json` in 2.728 seconds.
The independent audit is saved as
`fixed-camera-object-001/verification-supervisor.json`. It checks accepted
geometry, source graph membership and all count summaries, but does not
independently reclassify every rejected track's reason.

This is a preregistered diagnostic on the ten tree images. Both lanes use the
same nine registered poses and one intrinsic camera from the frozen
`heldout-v2-001` text model. They reconstruct only tracks whose training
feature centres lie inside the reviewed tree envelopes. The baseline lane
uses its verified two-view geometry after applying the ROI to both endpoints;
the foreground lane uses the freshly verified geometry from `foreground-001`.
The foreground model's five cameras and 192 points are not used.

The polygon bytes are pinned to SHA-256
`4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853`.
The runner checks saved and current independent integrity reports, exact heldout
population equality, original PNGs, databases, manifests, model text, ROI,
helper source, runner source, and NumPy implementation hashes before and after.
It requires 9 baseline cameras, 1,773 baseline points, 151 existing baseline
points with all observations inside the ROI, and 1,105 foreground verified
geometry rows. The 151 baseline points are context only; the fair comparison
is the two new lanes computed by the same method.

Each lane maps compact database indices to original feature IDs. ROI-ineligible
edges and edges with an unregistered endpoint are removed before constructing
connected components. Any component with two features from one image is
rejected whole. Remaining components are intersected by least squares in
world coordinates using undistorted rays and frozen camera centres. Screens
are fixed in advance: at least two distinct registered views, maximum ray
parallax at least 1 degree (using the acute angle), finite solver condition
at most 1e8, positive depth in every view, and maximum reprojection error
at most 4 px. There is no point refinement, pose optimization, or parameter
tuning. The report gives rejection reasons, observation and image coverage,
and a separate count of tracks with at least three views.

The protocol was reviewed before one live run. The guarded command was:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.fixed_camera.test_run
.local-tools/colmap-sparse/venv/bin/python -m scripts.fixed_camera.run \
  --output build-opencv/colmap-sparse/fixed-camera-object-001
```

The run requires a fresh output directory. The existing research-job guard
limits it to 300 seconds, 100 MiB combined output and temporary files, a
10 MiB log, and 10 GiB free-space reserve; temporary files stay under
`.local-tools/tmp`. Outputs are `points.json`, `provenance.json`,
`report.json`, `child.log`, and `status.json`. Source data are read-only.

The guarded run succeeded in 2.702 seconds. Its [report](../build-opencv/colmap-sparse/fixed-camera-object-001/report.json)
passed the runner's integrity checks, and an independent candidate review
passed. Both lanes used the same nine frozen cameras and the thresholds above.

| Fixed-camera lane | Accepted tracks | Tracks with ≥3 views | Accepted observations | Images with accepted support |
| --- | ---: | ---: | ---: | ---: |
| Baseline verified edges, both centres inside ROI | 666 | 160 | 1,562 | 9 |
| Foreground verified edges | 707 | 173 | 1,658 | 8 |

The foreground lane gains 41 tracks and 96 observations under this shared
triangulator, but loses support in `The_Tree-46.png`: the baseline lane had
one accepted observation there and the foreground lane had none. This is
therefore a track-support gain with a coverage loss, not a uniform improvement
across images. The existing 151 baseline all-inside COLMAP points are context,
not a comparable denominator for the 707 fixed-ray points. The 4 px, 1 degree,
and solver-condition screens are specific to this experiment, not COLMAP
defaults; no quality claim follows from comparing those point counts.

Heldout pair and three-view errors depend on the fixed camera geometry and
heldout observations. Both are invariant under the new points, so this
experiment cannot claim an improvement in those metrics. The two lanes
measure sparse track support under the chosen ROI and thresholds; neither
provides physical ground truth or production accuracy evidence. Full-frame
poses are usable here only for the moving-camera, static-scene diagnostic.
For a rotating object with a fixed camera, background features do not share
the object's motion; this experiment does not recommend background-based pose
estimation for that workflow. The production path uses board anchors and
object-consistent pose geometry.
