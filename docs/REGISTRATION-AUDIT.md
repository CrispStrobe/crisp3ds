# Registration audit: frozen protocol

This audit tests the already published `align.py` registration algorithm,
without changing its code, seeds, steps or historical transforms. It is a
diagnostic of registration identifiability and convergence, **not** a new
reconstruction score. No real reconstruction mesh or reference F-score is
used to select parameters. The corrected output is written to a fresh
`build-opencv/registration-audit-003/report.json`. A corrected preliminary
report at `build-opencv/registration-audit-002/report.json` (SHA-256
`816357d4304118fde2ab7c87d58c9e7594d9046d9e045be38148a65a4153b207`)
used the frozen cases and remains valid. The final audit adds the unchanged
nearest-sample objective at the *known true* transform and the fitted/true
objective ratio to distinguish search failure from an objective that prefers
wrong poses under missing support. This is analysis of fixed cases, not
parameter selection. The first run, retained at
`build-opencv/registration-audit-001/report.json` (SHA-256
`de473ec9028d3d4aa0872313d41b6e837ca7f92d0d5fc082d410a951bc96f82a`),
is **INVALID FIXTURE EVIDENCE, NOT AN ALIGNER RESULT**: its generator
transformed both reference samples and inversely transformed output samples,
so the true map was the square of the declared truth. It is not used in any
table or interpretation. The corrected generator round-trip is asserted
before each aligner call. The corrected runner also writes a checkpoint after
each case, retaining per-case timeout/error status instead of losing progress.

Use 384 independent area-weighted surface samples per side for synthetic
stress cases (reference seed 100, output seed 200), the existing
24-PCA-seed/four-finalist alignment and
60-second cap per case. A proper Sim(3) with scale 1.7, a fixed composed-axis
rotation, and nonzero translation maps native output points into reference
points. Cases are fixed before inspecting outcomes:

1. An asymmetric irregular tetrahedral surface, complete on both sides.
2. The same body plus a broad separate slab in the reference, with the slab
   missing from output. The slab is analogous to the bunny support plate, not
   a claim that the scanner/photogrammetry shapes are otherwise identical.
3. The complete asymmetric body with 20% distant outlier points appended to
   output, deterministic seed 300.
4. A near-square cuboid footprint (1.05:1) with a small off-centre marker.
   A rotation about the near-symmetric axis may have similar point-set error
   yet disagree with known point identities; report that ambiguity rather
   than automatically calling it a successful pose recovery.
5. The saved Revopoint bunny scanner mesh against a synthetically transformed
   independent sample of **itself**, at both 512 and 1,024 samples per side
   without changing seeds. This tests the aligner on realistic
   geometry with known exact pose, not an MVE/OpenMVS reconstruction.
6. The same scanner as reference and the predeclared Y > -24 scanner-only ROI
   as synthetically transformed output, again at 512 and 1,024 samples. This
   isolates the missing-base case
   on the actual scan without introducing reconstruction geometry.

The exact-correspondence control fits at least four asymmetric landmark
pairs with Umeyama and compares with the known transform. A separate complete
control feeds the **identical point set on both sides** through the full
unknown-correspondence aligner and requires approximately 1e-8 transform
recovery in unit tests. A mirrored landmark input must not pass as a proper
zero-residual Sim(3). These controls distinguish a correspondence solver
defect from the independent-sampling/partial-overlap search problem.
Also report landmark coordinate singular values and a rank-one collinear
landmark control: near-zero point residual on a line does **not** validate
rotation about that line. Rank-two non-collinear planar landmarks are not
categorically rejected; this is an identifiability disclosure, not a change
to `align.umeyama` or camera consumers.
For all cases report
rotation geodesic error, relative scale error, translation error normalized
by the corresponding complete reference diagonal, and RMS error on held-out
*corresponding* points. The latter distinguishes a low sampled-nearest-point
objective from correct pose. Record seeds, mesh hashes, case timing, the
chosen objective, and a near-symmetry warning. No hard pass threshold is
chosen from results. The script and tests operate within 1,500 samples and
write under 50 MiB; keep more than 10 GiB free. Do not tune alignment from
these results to improve the existing bunny F-scores.

## Results

The [final audit report](../build-opencv/registration-audit-003/report.json) is
complete, SHA-256 `309680922736c5f22fea402e7b261e7e18c446bea76aa6bd27729c3a5c620992`.
It binds unchanged `align.py` SHA-256
`24f27a0e2ff8879e5635151f73340b4abebbc04276f115cce627fec515f37b64`
and audit runner SHA-256
`af7deb9d55316e4e713871d3fa90051594fa21daa03c44de576e73e060598029`.
All cases completed within the per-case cap. Numbers below compare the fitted
transform with the *known synthetic truth*, not with any real reconstruction.
The objective is the aligner's symmetric nearest-*sampled-point* RMS; ratio
below 1 means it prefers the wrong fitted pose over the known truth.

| Fixed case | Rotation error | Held-out corresponding RMS / ref diagonal | Objective at truth → fit (fit/truth) |
| --- | ---: | ---: | ---: |
| Asymmetric full, 384 samples | 0.39° | 0.53% | 0.0961 → 0.0947 (0.985) |
| Reference base missing in output, 384 | 110.89° | 22.53% | 0.7060 → 0.4451 (0.631) |
| 20% output outliers, 384 | 178.75° | 26.24% | 5.2494 → 0.4533 (0.086) |
| Near-square marked box, 384 | 1.38° | 1.38% | 0.1151 → 0.1136 (0.987) |
| Scanner self, 512 independent samples | 16.67° | 6.02% | 4.0392 → 4.4331 (1.097) |
| Scanner self, 1,024 independent samples | 25.25° | 9.03% | 2.9526 → 3.8946 (1.319) |
| Full scanner / scanner ROI, 512 | 165.64° | 34.60% | 17.7664 → 12.2354 (0.689) |
| Full scanner / scanner ROI, 1,024 | 168.53° | 33.69% | 17.1542 → 11.7339 (0.684) |

The exact noncoplanar landmark control recovers the transform to
1.9e-15 held-out RMS. The full unknown-correspondence aligner likewise
recovers it to 5.0e-16 RMS when both sides use the *same* 384 points. A
mirrored input cannot be explained by a proper Sim(3): its best proper paired
fit leaves 1.196 RMS. A rank-one collinear landmark set fits to numerical
zero yet leaves 27.60° rotation error and 0.950 held-out body RMS; its
rotation about the line is unconstrained. The marked near-square case happened
to recover the pose under these fixed seeds and therefore does **not** by
itself demonstrate a symmetry failure.

Two distinct failure modes are visible. On the *identical scanner geometry*
with independent samples, the known transform has a lower objective than the
fitted transform, but the PCA-seeded/trimmed-ICP search does not find it;
raising samples from 512 to 1,024 does not cure this fixed-seed case. In the
missing-base and outlier cases, the objective itself gives substantially
lower error to wrong rotations/scales because unmatched surface support is
included in its untrimmed symmetric selection score. A converged flag is not
pose correctness: the synthetic missing-base and outlier cases report
convergence despite 110.89°/178.75° errors. The real scanner self-controls
do not include any reconstructed geometry, so this establishes registration
limitations without diagnosing the actual MVE/OpenMVS surface quality.

Any future v2 registration candidate must be separate from historical
`align.py`, must be specified before candidate scores, and must pass these
known-truth full/partial/outlier controls. It cannot be selected by improving
the existing bunny reference F-score. Independently checked landmarks or a
held-out registered object remain necessary for quality attribution.
