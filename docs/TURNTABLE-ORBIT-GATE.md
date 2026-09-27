# Optional turntable orbit plausibility gate

`scripts/object_motion/orbit_plausibility.py` checks image-estimated COLMAP cameras against an explicitly declared, uniformly sampled, single-revolution capture order. The declaration lists slot names only; it supplies no camera poses or measured turntable angles. The gate is unsuitable for unordered photos, partial arcs, nonuniform capture, or unknown object motion. It is independent of Berkeley pose metadata, scanner meshes, sensor depth, and held-out images.

The analytic criteria were fixed before evaluating a real model. Camera-center distances are divided by the median projected radius in a PCA plane, so model translation, orientation and positive scale do not change the result. An orbit with fewer than 12 registered slots, below 75% slot coverage, or PCA minor/major axis ratio below 0.25 returns `unavailable`. At least six consecutive-slot pairs and four opposing-slot pairs are required. The gate fails on any consecutive center step above 0.75 radius or full orientation step above 45°; two opposing pairs each closer than 0.5 radius and 30°; total angular winding outside 270–450° or more than 15% significant (>2°) reversed steps; or fewer than 75% of optical axes pointing toward the reconstructed center with cosine at least 0.5. These are broad necessary conditions, not calibrated physical-pose or shape accuracy tests. A `pass` does not authorize metric claims.

The [YCB NP3 profile](../tests/datasets/ycb_np3_full_turn_profile.json) declares 60 slots from `NP3_000.jpg` through `NP3_354.jpg`. A 48-photo training model can be checked against it with missing slots; only consecutive present slots enter the adjacent test, and only present opposite slots enter the collapse test. The ordering suffix is never converted into a pose.

Read-only preflight validates the profile, exact three-file model inventory, bounded regular binary files, SHA-256 hashes, and at least 10 GiB free on both disks. It writes nothing:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.object_motion.orbit_plausibility \
  --model /path/to/model --profile tests/datasets/ycb_np3_full_turn_profile.json \
  --external-root /Volumes/backups
```

Adding `--evaluate` reads those model poses and prints a compact JSON verdict to stdout. The input hashes are checked again after evaluation. Analytic fixtures in `scripts/object_motion/test_orbit_plausibility.py` cover a normal ring, arbitrary Sim(3) scale and orientation, a fold, a reversal, an adjacent outlier, wrong camera facing, incomplete coverage, degenerate centers, and read-only preflight. No gate result changes a camera, model, or prior artifact.

## Frozen-control check

After fixing the criteria and analytic fixtures, read-only evaluation used the same profile on the sealed fresh005 refined 60-view model and the two paired 48-view mustard models. These are control observations, not threshold selection or independent evidence of physical accuracy.

| Image-estimated model | Result | Largest consecutive center/orientation step | Collapsed opposite pairs | Winding | Inward-facing fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| fresh005, 60/60 | pass | 0.136 radius / 6.60° | 0/30 | 360° | 1.000 |
| mustard SAM, 48/60 | fail: jump, collapse, winding | 2.268 radius / 93.37° | 10/24 | 720° | 0.854 |
| mustard v2, 48/60 | fail: jump, collapse, winding, orientation | 2.359 radius / 99.51° | 20/24 | 720° | 0.729 |

The previously evaluated 48-view repaired mustard candidate also failed on jump, collapse, and winding. Its sealed report identifies the same 48 training names as the paired models. The controls show that this gate distinguishes these examples; they do not estimate its false-positive or false-negative rate on other captures. In particular, a passing orbit remains an image-estimated geometry check only.
