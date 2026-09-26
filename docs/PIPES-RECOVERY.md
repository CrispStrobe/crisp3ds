# Pipes additional-view anchor recovery

This completed development experiment re-estimated the frozen 258 Pipes sparse
point IDs under the [predeclared recovery protocol](PIPES-RECOVERY-PROTOCOL.md).
The original four-view point contributed only its lexically first image feature
as an anchor. Ten more prepared photographs, `DSC_0638.JPG`–`DSC_0647.JPG`,
became **training** views. Strict global original-size SIFT matches, fixed-pose
epipolar checks, complete pair agreement among new observations, and the
unchanged fixed-camera triangulation gates determined recovery. Original XYZ
was used only when a point could not be re-estimated. The supplied camera
calibrations and poses were unchanged; this remains a fixed-camera oracle lane.

The [reconstruction report](../build-opencv/pipes-recovery/run-001/report.json)
records **46 re-estimated** and **212 unresolved** points. Of the unresolved
points, 186 had fewer than two directly matched new views and 26 had a conflict
between new-view matches. No otherwise admissible points shared a new-view
feature. The [decisions](../build-opencv/pipes-recovery/run-001/decisions.json)
retain every anchor, support observation, match proof, and fallback reason;
the [candidates](../build-opencv/pipes-recovery/run-001/candidates.json) retain
all 258 IDs. The 46 re-estimates have 150 observations in total: 46 original
anchors and 104 new-view observations. Those new observations came from five
of the ten new images (5 from `0638`, 2 from `0642`, 26 from `0643`, 43 from
`0644`, and 28 from `0645`). The other five images still participated in the
global matching checks.

The separate [sealed score](../build-opencv/pipes-recovery-score/run-001/score.json)
has `status: pass` for integrity validation. It compares each changed point
with all 11,482,717 transformed laser vertices using exhaustive float64
nearest-neighbor search, and reuses the frozen distance for each exact fallback.
Distances below are one-way proximity to measured scan points in millimetres.

| Fixed 258-point population | Frozen baseline | Recovery candidate |
| --- | ---: | ---: |
| Within 10 mm | 119 | 119 |
| Within 20 mm | 151 | 152 |
| Within 50 mm | 188 | 188 |
| Within 100 mm | 238 | 238 |
| Median | 12.798 | 12.798 |
| p90 | 90.280 | 89.721 |
| p95 | 151.540 | 151.540 |
| Maximum | 7,451.330 | 7,451.330 |

At 20 mm, one point moved from farther to nearer and none crossed in the
opposite direction. No point crossed the 10, 50, or 100 mm thresholds. Among
the 46 re-estimates, 24 moved nearer the scan and 22 moved farther. Their
median proximity changed from 6.413 to 5.195 mm; p90 changed from 78.877 to
76.973 mm. Point 229 had the largest adverse move, from 28.943 to 36.923 mm.
This is a modest net change with individual regressions, not a general recovery
of the original long-distance tail. Previously inspected point 104 had only
one new-view match; points 212 and 214 had none, so all three retained their
original XYZ as unresolved fallbacks.

The guarded reconstruction completed in 7.889 s with 349,700,096 bytes peak
process RSS and one effective OpenCV worker. The guarded score completed in
11.691 s with 585,302,016 bytes peak RSS. Five reconstruction and five
evaluator unit tests passed. An independent audit re-extracted the 150 accepted
observation coordinates across seven images and directly projected all
accepted points using NumPy quaternion rotation: every depth was positive and
the largest reprojection error was 1.89911 px. The reconstruction report pins
the prepared manifest, all fourteen source images, baseline points and report,
protocol, implementation, and runtime files before and after its run.
Root also independently compared three distinct changed points (IDs 14, 229,
255) against every scan vertex using separate chunked distance calculations;
all agreed with the scorer within 1e-12 m. These are spot checks, not an
independent replay of all 46 distances. The full targeted regression run
passes 110 Python tests and 11 native CTests.

## Sealed artifacts and reproduction

| Artifact | SHA-256 |
| --- | --- |
| [Protocol](PIPES-RECOVERY-PROTOCOL.md) | `3f6b838747b42f94d9d54dfdc346678f928330be126d42c2df10bf1c775172ed` |
| [Original points](../build-opencv/pipes-sparse/run-003/points.json) | `49f3da8646d7e018cd9db8370d6350f1bdd3e63630058e5806a4daf06d2f6fd0` |
| [Original score](../build-opencv/pipes-score/run-002/score.json) | `9df4234128ca06e1ee10842e34d93d9fc66336ca37f81c598e28af6921523796` |
| [Reconstruction report](../build-opencv/pipes-recovery/run-001/report.json) | `48eab22c7a8ac8ab80f42049eed33abf61c28ba384f26b092b86d420323d75f8` |
| [Candidates](../build-opencv/pipes-recovery/run-001/candidates.json) | `d6e7112d05902f51816e078dca7291dad72339db4e30e84740a8f5d5c5028482` |
| [Decisions](../build-opencv/pipes-recovery/run-001/decisions.json) | `ada969d3e5c22245176202d2b062191b3046bdceca120fe9183345bf353fe207` |
| [Sealed score](../build-opencv/pipes-recovery-score/run-001/score.json) | `bdcc0e05f0fd4d746e08f711c884fef03d4500cfd1f9d8180ce2f62f7e1a5f13` |
| [Reconstruction source](../scripts/pipes_recovery/run.py) | `49f49ce7cc042a52a2bd37c38397552fc4b1d2b2d177e4518e237b6a0d18396b` |
| [Evaluator source](../scripts/pipes_recovery/evaluate.py) | `07a8290558366ee047c91bccbab71333b2a3a8de80ab9036d6b39019eb063bbe` |

With the same local prepared inputs and runtime, use **fresh** output paths:

```sh
/usr/local/bin/python3 -m scripts.pipes_recovery.run \
  --output build-opencv/pipes-recovery/run-002

/usr/local/bin/python3 -m scripts.pipes_recovery.evaluate \
  --prepared build-opencv/pipes-prepare/run-001 \
  --sparse build-opencv/pipes-sparse/run-003 \
  --baseline-score build-opencv/pipes-score/run-002/score.json \
  --candidate build-opencv/pipes-recovery/run-002/candidates.json \
  --candidate-sha256 <new-candidates-sha256> \
  --decisions build-opencv/pipes-recovery/run-002/decisions.json \
  --decisions-sha256 <new-decisions-sha256> \
  --recovery-report build-opencv/pipes-recovery/run-002/report.json \
  --recovery-report-sha256 <new-report-sha256> \
  --output build-opencv/pipes-recovery-score/run-002
```

Compute and seal the three new reconstruction artifact hashes before invoking
the evaluator. Both jobs use a 300-second guard, a 256 MiB generated-output
cap, a 10 GiB free-disk reserve, and one OpenCV worker. The extra photographs
change both the available data and the matching policy, so this is not a
same-four-image algorithm comparison. One-way scan proximity does not establish
surface association, completeness, dimensional certification, or official
ETH3D accuracy. The fixed-camera oracle input and mixed per-point movements
do not support production promotion; an independently measured scene is needed
for acceptance.
