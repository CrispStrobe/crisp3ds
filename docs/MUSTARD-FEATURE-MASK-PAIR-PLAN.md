# Frozen mustard feature-mask camera ablation: awaiting live approval

This is a two-arm **training-only sparse SfM** comparison. The accepted SAM
coarse support is the control; the root-reviewed photo-refined v2 is approved
only as a bounded coarse feature-support ablation. The 12 held-out photos,
scanner mesh, sensor depth, prior camera models, native dense pipeline and
reference scores are outside both arms. Earlier mustard SfM and mask outputs
are immutable. No live arm has run under this plan.

The read-only [wrapper](../scripts/classical_backend/mustard_mask_pair.py)
is frozen at SHA-256
`47d5f5db014ee3fb249967f8ffbd5fd8d333c25b728d4af32488fe930cf8b584`.
Its
preflight checks the 48-name sealed train list, exact staged JPEG/SAM PNG bytes,
the v2 report and root visual decision, binary 1280×1024 masks, no added v2
pixels, preservation of all prior pixels at `y < 435`, and the exact per-view
removal counts. It rejects symlinks, missing/extra names, held-out names and
occupied output paths. It verifies the sealed `run.py`, interpreter, PyCOLMAP
extension and all OpenMVS binary hashes. The OpenMVS executables are checked
because `run.py` validates its toolchain; `--stop-after-sfm` prevents invoking
them. The existing producer copies each arm's selected inputs to its own fresh
external directory and rehashes those sources afterward. No source database
or prior experiment is opened for mutation.

Both commands use the exact same 48 original JPEG bytes and ordering, CPU
SIFT with maximum 1,800 features and 1,200-pixel image size, exhaustive
matching, seed `20260927`, shared `SIMPLE_RADIAL` camera, automatic initial
pair, fixed-initial intrinsics policy, up to five mapper models, minimum model
size 10, and 70% registration gate. The runner's ImageReader sets initial
focal factor 1.2; the previous same-profile run saved
`[f=1536,cx=640,cy=512,k=0]` on 1280×1024 images. Its
`fixed-initial` policy disables focal, principal-point and extra-parameter
refinement during bundle adjustment and focal/extra refinement during
absolute-pose fitting. This is a heuristic initial camera, **not measured
calibration**. Neither arm supplies poses, camera intrinsics, a manual seed
pair, or a sparse model. The only algorithmic input difference is
`--pose-mask-dir`: staged accepted SAM masks versus the 48 v2 masks. The
separate `--output` paths distinguish artifacts; all other flags, including
4.5-minute per-arm timeout, are identical. One wrapper invocation launches
arms serially within a 10-minute combined wall limit, even if the first arm
fails its registration gate; no threshold, seed or option retry follows.

The fresh outputs are
`/Volumes/backups/code/crisp3ds-data/mustard-feature-mask-pair-001-sam` and
`...-v2`, with a separate
`/Volumes/backups/code/crisp3ds-data/mustard-feature-mask-pair-001-receipt.json`.
Each arm is capped at 0.45 GiB (483,183,820 bytes), so the two producer
outputs plus a ≤1 MiB receipt stay below 1 GiB. Each stage has a 16 MiB log,
4 GiB sampled child RSS and two CPU threads. The wrapper requires at least
10 GiB free on the internal volume and **10 GiB plus the entire 1 GiB output
allowance** on external storage before launch, checks again before each arm,
and records both floors after the second arm. The producer itself monitors both
volumes and the per-arm limit during stages. Temporary files stay on the
external volume. The wrapper's fresh receipt saves the preflight, exact
commands, each arm's exit/producer status, effective PyCOLMAP options,
SHA-256 of every output file and all source photo/mask hashes before and after.
A failed/partial arm remains retained, not silently replaced or promoted. The
wrapper marks the pair `completed` only if **both** producer processes exit
zero **and** both producer reports say `sparse_complete`, plus unchanged sources,
bounded total output and the final disk floors. A zero exit with any other
producer status is `partial_or_failed`.

The read-only preflight passed with 48 verified training views, 23,711,059,968
bytes internal and 17,118,318,592 bytes external free. Four focused wrapper
tests passed, including a simulated failure after receipt creation that seals
`partial_or_failed` and a zero-exit/non-`sparse_complete` rejection. After approved
execution, compare registered names/count, verified-pair graph, tracks with
distinct-view counts, reciprocal track links, reprojection residual
denominators, saved intrinsics and acquisition-order orbit fold diagnostics.
No mesh-quality claim follows from registration count. Do not run native dense
from this plan; any dense mask ablation requires a separately selected, frozen,
plausible common camera model and its own plan.

To repeat the read-only preflight from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .local-tools/colmap-sparse/venv/bin/python \
  -m scripts.classical_backend.mustard_mask_pair
```

The `--run` switch is reserved for root's explicit live approval.
