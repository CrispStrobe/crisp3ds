# Original-photo COLMAP point-loss diagnostic

The original-JPEG baseline registered 7 of 10 cameras but finished with zero
3D points. Its saved database still contains 10 images, 42,308 raw feature
matches, and 36,262 inlier links in 45 verified image-pair records. Those
database counts show that correspondence discovery occurred; they do not prove
that the final camera geometry or sparse points are valid.

The mapper-only diagnostic replays a byte copy of that saved database on the
same ten original JPEGs. It uses PyCOLMAP 3.11.1's C++
`IncrementalPipeline`, the baseline's complete mapper defaults, two threads,
and seed 0. The only mapper option additions are one snapshot per new
registered image and a snapshot output path. Verbose logging is level 1.
The [pinned COLMAP implementation](https://github.com/colmap/colmap/blob/3.11.1/src/colmap/controllers/incremental_pipeline.cc)
writes each snapshot after triangulation, local refinement, and any triggered
global refinement; the next-image callback follows it. The initial-pair
callback follows initial global adjustment and filtering. A final refinement
can occur after the last snapshot, so the final model is also recorded.

The first attempt is preserved at `build-opencv/colmap-sparse/originals-mapper-replay-001/`.
It completed mapping but the guard marked it failed because SQLite advanced
its database header counters while reading the copied database. A forensic
comparison found exactly two changed bytes, at zero-based offsets 27 and 95
(the mirrored change counters); the complete SQL dump and all data pages were
identical. The source database stayed byte-identical. The guard was then
narrowed to allow only coordinated changes in these header fields, while
requiring a matching SHA-256 after masking them.

The fresh `originals-mapper-replay-002/` passed the guard in 3.348 seconds.
Its recorded stages were:

| Stage | Registered cameras | 3D points | Focal length |
| --- | ---: | ---: | ---: |
| After initial-pair global adjustment and filtering | 2 | 123 | 2998.67 px |
| After third-image triangulation and refinement | 3 | 16 | 2569.24 px |
| Final model | 3 | 16 | 2569.24 px |

The first and second mapper-only attempts produced the same 3-camera,
16-point trajectory. The verbose log records 247 and 218 filtered
*observations* during local refinement and further filtering during global
refinement. Those counts do not identify whether reprojection error, angle,
or another rule removed each point. The replay brackets most of its point
loss between the initial-pair callback and the snapshot after registering the
third image; it does not isolate a single filter. The final zero-point result
of the original end-to-end baseline remains unexplained by this replay.

The mapper-only result is a new frozen-stage experiment, not an exact
reproduction of the original end-to-end trajectory. The original run set
seed 0 before extraction and matching; replay resets seed 0 at the mapper
start, so the random stream entering mapping can differ. The original run
ended at 7 cameras and 0 points. No thresholds, filters, camera parameters,
or pair geometry were changed in the diagnostic. A camera-prior or model
experiment would require fresh geometric verification because the saved
two-view geometries were formed under the original EXIF focal prior.

Replay artifacts include `events.jsonl`, `snapshot_summary.json`, binary
snapshots, text snapshots, the final text model, `provenance.json`, and the
guard's `status.json` and `child.log`. The source database and JPEGs, the
PyCOLMAP binary, and the diagnostic script were hash checked before and
after; the copied database content hash stayed identical apart from SQLite's
two header counters. These models are training diagnostics, not independent
accuracy measurements.

Root's separate `originals-mapper-supervisor-001/` completed in 3.557 seconds.
All three final text-model files are byte-identical to replay-002, and source
hashes stayed unchanged. An independent model audit passes reciprocal tracks,
rotations and positive depth for all 41 observations; its training reprojection
median/p90 is 0.818/1.550 px. This confirms the small replay model's internal
consistency, not its physical correctness or the original trajectory.
