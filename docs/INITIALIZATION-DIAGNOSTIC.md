# YCB image-only initialization: retained evidence, not a new SfM trial

The fixed image-only serial re-verification arm in `calibration-ablation-002` registered only 2/60 cracker-box photos, whereas the earlier `verify_matches` image-only replay in `calibration-ablation-001` registered 60/60. This [read-only diagnostic report](../build-opencv/object-motion/initialization-diagnostic-002/report.json) (SHA-256 `962c55e7e75c6f6cdbaebcc5f065075615e6b519113d486e882f704e219fd979`) rehashes and inspects the retained image-derived feature databases, models, and mapper logs. It runs **no** mapping, descriptor matching, Berkeley-pose comparison, depth test, or reference-mesh scoring. Inputs were hashed before and after reading; a 10-GiB free-space floor guarded the small fresh report output.

## What is observed

All three databases start from the same 60 named photos and the same shared `SIMPLE_RADIAL` camera with `f=1536 px`, `cx=640`, `cy=512`, `k=0`. Both replay mappers initialized from image IDs #33 (`NP3_192.jpg`) and #28 (`NP3_162.jpg`) with 126 verified seed-pair inliers, but the two verification methods did **not** produce identical geometry. The original foreground database and successful `001` wrapper replay have the *same 126 seed correspondences byte-for-byte*. The failed `002` serial replay shares 125/126 of those correspondences; its seed fundamental/homography matrices also differ. The seed pair is classified `UNCALIBRATED` (pinned PyCOLMAP 3.11.1 two-view enum value 3), not a direct calibrated relative-pose constraint. All three have 79 exact seed-correspondence cycles through a third image in their respective verified graphs, so simple pair count or triplet-connectivity count alone does not explain the divergence.

| Retained model | Registered | 3-D points | Final shared focal | Final radial `k` |
| --- | ---: | ---: | ---: | ---: |
| Original foreground, image-only | 60 | 3,939 | 1077.58 px | −0.00723 |
| `001` wrapper-reverified, image-only | 60 | 3,937 | 1086.83 px | −0.03681 |
| `002` serial-reverified, image-only | **2** | **126** | **2995.00 px** | **−6.99465** |

The `002` log records 11 distinct attempted third views, each tried twice after retriangulation/global bundle adjustment. They see 44–97 seed points each, yet all 22 attempts are rejected. The `001` log accepts its first third view (#29, 96 visible seed points) and continues to all 60. The extremely different two-view-only focal/distortion in `002` is concrete evidence of an unstable image-only initialization/self-calibration result that coincides with failure to register a third view. It does **not** prove that the one differing inlier caused the failure, that focal drift alone is the cause, or that PyCOLMAP's threaded wrapper is generally better than the serial estimator. The initial two-view poses, planar scene content, different geometric-verification behavior, RANSAC state, and bundle-adjustment conditioning remain possible contributors; this audit did not isolate them experimentally.

## Image-only next-seed policy, not a quality claim

The report ranks *alternative* seed pairs from the **original foreground verified geometry only**. It admits pairs with at least 80 verified inliers and PyCOLMAP's `UNCALIBRATED` two-view configuration, then ranks by the number of seed inliers whose feature indices agree through an independently verified third-image correspondence; ties use the number of third views with at least 40 such exact cycles, then pair inliers. The failed fixed seed is excluded from the alternative list. Filename acquisition-angle gap is recorded for context only: it is neither a ranking input nor validated parallax/camera pose. No Berkeley metadata, supplied calibration, depth, or model is used. The rule is a dataset-specific *support screen*; exact cycles can still occur on nearly planar or repeated texture, and do not guarantee a well-conditioned essential matrix, feasible third-view PnP, physical intrinsics, or mesh accuracy.

| Rank | Alternative seed pair | Pair inliers | Best exact triplet support | Third views with ≥40 cycles | Acquisition-label gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `NP3_018.jpg` / `NP3_030.jpg` | 361 | 255 via `NP3_024.jpg` | 7 | 12° |
| 2 | `NP3_180.jpg` / `NP3_192.jpg` | 343 | 233 via `NP3_186.jpg` | 9 | 12° |
| 3 | `NP3_168.jpg` / `NP3_180.jpg` | 340 | 230 via `NP3_174.jpg` | 8 | 12° |
| Failed fixed seed | `NP3_162.jpg` / `NP3_192.jpg` | 126 | 79 | — | 30° |

For a **future predeclared** image-only trial, the first-ranked pair is a reasonable single candidate to test with the same cached photos/features/raw matches and one frozen verifier/intrinsic-refinement protocol. Freeze the seed and an intrinsic-plausibility plus ≥3-camera continuation gate *before* any mapping or surface score; report failure as failure rather than cycling through seeds against downstream results. Given the 12° acquisition-label gap, parallax/planarity should be diagnosed independently before treating even a registered model as reliable. No such next trial was run in this batch.

Reproduce only this retained-artifact audit and its pure synthetic tests from the repository root:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.object_motion.test_initialization_diagnostic
.local-tools/colmap-sparse/venv/bin/python -m scripts.object_motion.initialization_diagnostic --output build-opencv/object-motion/initialization-diagnostic-REPLAY
```

The output directory must be fresh. See [CALIBRATION-ABLATION.md](CALIBRATION-ABLATION.md) for the separate calibrated-intrinsics arm and why it cannot by itself establish a paired mesh-quality improvement.
