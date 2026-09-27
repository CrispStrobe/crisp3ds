# P1 cracker-box results

2026-09-27, continuing [the frozen plan](CRACKER-STOCK-PLAN.md).
Run implementation/input policy frozen at `e57b669`.

## Protocol deviation: timings are not a clean serial comparison

The masked wrapper was launched before the raw mapper had exited. Masked
feature extraction and raw mapping briefly overlapped (approximately 20 seconds).
Root paused the masked feature-worker process group at 13:17:41 UTC while
leaving its parent resource monitor active; its original ten-minute deadline
was not extended. Consequently neither the raw mapping time nor the masked
feature/wall time can support an isolated speed comparison. The worker resumed
at 13:18:57 UTC (76 seconds paused). The original reports
and logs are retained, and algorithm/input settings were not changed.

## Apple product oracle

The original 60 JPEGs produced a preview USDZ in 30.525 seconds. Sampled direct
child RSS peaked at 859,635,712 bytes; this excludes system/XPC service memory.
All input hashes remained unchanged. The USDZ is 380,651 bytes, SHA-256
`00e6e5bc15a79634ffb7b0bd42492c8df5ebe575d63b228f53f737d4f23e1ae7`.
Installed `usdchecker --arkit` returned Success, with a preceding USD plugin
registration diagnostic; `usdtree` shows one Mesh prim.

A constrained export of that exact Mesh contains 1,054 vertices and 2,092
triangles. Root inspected the neutral three-view surface preview: a roughly
box-shaped region has a conspicuous unwanted thin extension. This is **not a
clean object-quality pass**. Authored meter units are not verified metric scale.
No reference-fitted surface score is claimed and no cleanup was applied.

External evidence remains under `/Volumes/backups/code/crisp3ds-data/`:
`apple-crackerbox60-preview-001/result.json` and `apple-review-001/result.json`,
with the USDZ, exported PLY and neutral preview alongside them.

## Raw stock COLMAP

Execution completed, but the surviving model contains **15/60 cameras and
53 points**, below the predeclared 54-view minimum. All saved track structural
checks pass; structural validity does not make the recovered cameras correct.
Its post-hoc Berkeley diagnostic has center RMS disagreement 0.94579 times the
reference radius and p95 orientation disagreement 168.70 degrees on the paired
15 views. This is gross disagreement, not a usable object trajectory. No dense
continuation is authorized from this arm. Recorded wall time is 278.314 seconds,
subject to the overlap caveat above; no settings or pair graph were repaired.

Exact compact reports are retained in `tests/evidence/` under
`cracker-stock-raw-001.json`, `cracker-stock-raw-camera-001.json`,
`apple-crackerbox60-preview-001.json` and `apple-crackerbox60-review-001.json`.

## Feature-masked stock COLMAP

Execution completed with **2/60 cameras and 339 points**, not a usable orbit.
Only NP3_192 and NP3_198 survive. The 54-view eligibility gate fails and fewer
than four poses preclude the named-camera similarity diagnostic. No dense stage
or shape score is produced. Exact report: `tests/evidence/cracker-stock-masked-001.json`.
Recorded wall time is 156.770 seconds, including the pause; it is not a speed
comparison with the raw arm or Apple. All source/copied image and mask hashes,
worker configuration and software hashes remain unchanged.

## What this establishes

Neither stock arm produces usable object cameras on this capture; adding coarse
foreground support alone does not fix it. Apple's independent workflow runs but
its unclean output does not establish our target quality either. This narrows
the immediate issue to camera recovery/capture assumptions before dense tuning.
It does not establish a fundamental limit of photogrammetry or of all COLMAP
configurations. The next diagnostic must explain the discrepancy with the earlier
60/60 cracker-box producer, particularly initialization and intrinsics, while
keeping these failed baselines intact.

### First-divergence audit and next diagnostic

The masked database has all 60 images connected and 1,182 verified pairs loaded
by the mapper. The first automatic seed is NP3_192/198, six degrees apart. The
mapper sees 220 and 229 existing 3D points in the adjacent candidate views but
cannot register either. Six automatic seed attempts produce no third view.
The saved two-view SIMPLE_RADIAL camera has `f=5011.34797` and `k=-5.78716`,
starting from the reader heuristic `[1536,640,512,0]`; the distortion magnitude
exceeds the configured `max_extra_param=1.0`. The prior 60-view foreground
producer started from the same heuristic but used a 30-degree seed and finished
near `f=1077.58, k=-0.00723`.

This supports unstable initialization/self-calibration as a hypothesis, not a
proven causal explanation: feature limits, matching graph and verification also
differ from the prior producer. Local verified matches are not simply absent.

The next **single diagnostic**, not executed in this batch, should reuse the
sealed stock masked features/matches and automatic seed selection, changing
only mapping intrinsic refinement to fixed initial `[1536,640,512,0]`. Those are
heuristic values, not calibration or a recovered camera solution. First verify
that the implementation actually freezes both focal length and distortion and
that the saved camera remains exact. A third registered view is an early
discriminator; 54/60 plus structural checks and camera review remains the dense
eligibility requirement. Keep the same 600-second, 4-GiB RSS, 512-MiB output and
dual disk-floor bounds, fresh output and untouched parent evidence. A further
two-view failure would show that preventing intrinsic drift alone is insufficient.
This is not authorization for automatic parameter search or geometry repair.
