# Recovered-camera masked dense/rough-mesh control

The bounded [015 run](../build-opencv/classical-ycb-recovered-015/result.json) completed 2026-09-27 from the separately sealed [image-only delayed-BA sparse model](../build-opencv/object-motion/initialization-recovery-005-delayed-refinement/ba_trial/report.json). This is **composed-stage, cache-rebound exploration**, not a fresh end-to-end photo-only reconstruction or an accepted shape result. The 005 SfM producer reused previously computed foreground features/matches and copied/rebound its database; no Berkeley intrinsics, supplied poses, reference mesh, or scanner score was used to choose this native dense arm. The [004 parent producer](../build-opencv/object-motion/initialization-recovery-004-fixed-intrinsics/cached_seed_trial/report.json), [cache mutation audit](../build-opencv/object-motion/initialization-recovery-cache-audit-002/report.json), exact original-photo hashes, and coarse photo-derived pose-mask hashes were checked before copying any inputs. Producer reports are pinned to their SHA-256 values in `recovered_control.py`; the final model's three binary hashes and camera parameters were checked against 005. The run report records all copied/input and binary hashes and confirms `sources_unchanged=true` after execution.

`recovered_control.py` copied the 60 original 1280×1024 photos and refined `SIMPLE_RADIAL` model into fresh `classical-ycb-recovered-015`, undistorted all registered images to `PINHOLE`, reprojected all 60 coarse masks into the new undistorted camera pixels, then imported the scene into OpenMVS. Every original/undistorted pose matched, and every new image/mask was hash-bound. The dense profile was frozen before execution: requested level 2, `--min-resolution 600 --max-resolution 1600 --geometric-iters 2 --tower-mode 4 --ignore-mask-label 0`, at most two native threads, 600 s total, 1 GiB **entire output-folder** cap and 10 GiB free-space floor. The conservative two-generation DMAP preflight predicted **1,023,915,435 bytes** versus the 1,073,741,824-byte cap; the actual native depth size was **651×517**, corresponding to effective level 1. No resolution/cap tuning or retry occurred after inspection. The full output was **523,457,427 bytes** and free disk after the run was **12,117,086,208 bytes**.

All native stages through the rough mesh completed. Densification took **47.081 s** and produced **40,450** dense PLY points; explicit `ReconstructMesh -p dense.ply` took **1.541 s** and produced **38,440** rough faces. The `dense.mvs` scene stores the original sparse point count, so the explicit dense PLY handoff is essential. All **60/60** base DMAPs matched their undistorted COLMAP camera matrices (maximum absolute K/R/C difference **0 / 2.66e-15 / 2.22e-14**), and **zero positive depth pixels** lay outside their warped native masks. Adjacent-view depth self-consistency found **7,070/7,155 (98.81%)** sampled comparable pixels within 2% relative depth; this is an internal camera/depth check, **not** independent object geometry truth. No mesh refinement or texturing was attempted in 015, and no independent scanner/sensor score is claimed here. The mesh remains research output; metric scale, geometric quality, and shipping are unapproved.

To reproduce the bounded control with the same ignored local artifacts, run:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.classical_backend.recovered_control \
  --producer-report build-opencv/object-motion/initialization-recovery-005-delayed-refinement/ba_trial/report.json \
  --source-report build-opencv/object-motion/initialization-recovery-004-fixed-intrinsics/cached_seed_trial/report.json \
  --model build-opencv/object-motion/initialization-recovery-005-delayed-refinement/ba_trial/refined_model \
  --images .local-tools/test-data/ycb-cracker-box/photos \
  --pose-masks build-opencv/object-motion/prepare-001/masks \
  --manifest build-opencv/object-motion/prepare-001/manifest.json \
  --cache-audit build-opencv/object-motion/initialization-recovery-cache-audit-002/report.json \
  --output build-opencv/classical-ycb-recovered-NEW
```

Use a **fresh output directory** and confirm at least 11.25 GiB free beforehand. The command is experiment-specific: it refuses other producer report bytes rather than silently reclassifying an arbitrary camera model as image-only. The 015 result JSON SHA-256 is `4a856ad150fd2817a99404bd3fe7e49bbb306e593786abe89384ff27923aed77`; its rough `mesh.ply` SHA-256 is `ecb3566a063f85b372492e95d5cc2db92bb14fcaa4213a3bb2bac6c936db17cf`.

## Subsequent fixed-ray sensor comparison: not a quality win

After freezing the mesh, root ran all three rough candidates through the
[unchanged sensor protocol](SENSOR-DEPTH-BENCHMARK.md). Report:
`build-opencv/sensor-depth-004/report.json`, SHA-256
`9027600ccac5488ec422a6adc3736979eb126a7b33d9d8d19d851b107327cc5c`.
All 6,144 selected ray IDs and support metadata exactly match sensor-depth-003;
the repeated 008/014 pooled results are identical. Root independently recomputed
counts, mean absolute residuals and missing-inclusive threshold fractions from
the saved per-ray values. Source/model hashes remained unchanged.

| Rough candidate | Ray hit coverage | Hit-only mean absolute depth disagreement | Within 5 mm / all supported rays |
| --- | ---: | ---: | ---: |
| 008 earlier image-only | 84.08% | 2.881 mm | 70.72% |
| 014 supplied-intrinsics control | 83.66% | 5.959 mm | 38.56% |
| 015 recovered image-only | 80.58% | 2.922 mm | 68.10% |

The new path **does not improve this quality diagnostic over 008**. It recovers
a usable sparse/dense pipeline from the previously failing seed, but loses
3.50 percentage points of supported ray coverage; its hit-only error is nearly
unchanged. The fixed interior panel agrees: 015 has 91.52% coverage, 2.451 mm
hit-only error and 81.38% within 5 mm/all, versus 008's 92.58%, 2.342 mm and
82.73%. Per-view coarse 015 hit coverage at 0/120/240 degrees is
76.90/84.33/80.52%; corresponding errors are 2.189/3.469/3.048 mm. Do not hide
missing predictions by reporting only error among hits.

These are support-conditioned, assumption-qualified sensor diagnostics, not
whole-object completeness, certified metric accuracy, or KIRI parity. Camera
alignment, coarse masks, reference poses and depth rectification remain error
sources. 014's refinement failed, but its rough-mesh stage completed; only rough
meshes are compared. 008 and 015 also differ in recovered cameras and downstream
scene handoff, so this is not a single-parameter causal ablation.

Next quality work should isolate mask/visibility coverage and camera consistency
with fixed evaluation inputs, then test the frozen two-phase alignment policy on
new objects. Do not run a seed/parameter sweep against these three depth views.
