# Proposed mustard sparse-SfM mask comparison

Status: **plan only**. The original 48-view SAM mask inventory was generated
but root rejected it for checkerboard leakage in five views. Do not stage
masks or run either SfM arm until a new complete 48-view candidate has its
own independent root visual-QA acceptance. A successful inference manifest
or five-view correction review alone remains unaccepted for SfM. This
experiment compares image-estimated sparse reconstruction, not mesh quality.

## Immutable input boundary

The [VPS-validated mustard evaluation package](../build-opencv/ycb-evaluation-vps-001/mustard-package.json)
(SHA-256 `45b6ed1430627a747eaeb8d4dff7d2c49502083cca0a7914e25aacd45b594425`)
binds 48 training originals and 12 held-out originals. Its acquisition-manifest
SHA-256 is `c48998dffc7798e7d21f275383b1c67dc2e5e9313a8bfe699f72303f61583b52`.
The 48 JPEGs total **48,683,099 bytes**. Their exact ordered basenames are:

```text
NP3_000.jpg NP3_006.jpg NP3_012.jpg NP3_018.jpg NP3_030.jpg NP3_036.jpg
NP3_042.jpg NP3_048.jpg NP3_060.jpg NP3_066.jpg NP3_072.jpg NP3_078.jpg
NP3_090.jpg NP3_096.jpg NP3_102.jpg NP3_108.jpg NP3_120.jpg NP3_126.jpg
NP3_132.jpg NP3_138.jpg NP3_150.jpg NP3_156.jpg NP3_162.jpg NP3_168.jpg
NP3_180.jpg NP3_186.jpg NP3_192.jpg NP3_198.jpg NP3_210.jpg NP3_216.jpg
NP3_222.jpg NP3_228.jpg NP3_240.jpg NP3_246.jpg NP3_252.jpg NP3_258.jpg
NP3_270.jpg NP3_276.jpg NP3_282.jpg NP3_288.jpg NP3_300.jpg NP3_306.jpg
NP3_312.jpg NP3_318.jpg NP3_330.jpg NP3_336.jpg NP3_342.jpg NP3_348.jpg
```

The [frozen split rule](YCB-EVALUATION-PROTOCOL.md) excludes angles
24°, 54°, 84°, 114°, 144°, 174°, 204°, 234°, 264°, 294°, 324°, and 354°.
Neither those JPEGs nor Berkeley depth/poses, supplied masks, or Google scan
geometry may enter either arm. Angle labels select the split only; they do not
seed camera estimation. Stage exactly the 48 JPEG bytes and, after acceptance,
the 48 cleaned SAM PNGs in a **fresh ≤150 MiB** external-volume directory.
After full-view QA acceptance, read the complete per-frame inventory's
`frames/IMAGE_NAME/clean.png` files and stage them in one `masks/` directory
with exact names `IMAGE_NAME.png` (for example `NP3_000.jpg.png`), as required
by the existing PyCOLMAP reader. Rehash each staged photo against the package
and each cleaned mask against the sealed complete inventory and accepted QA
hashes; reject missing, duplicate, linked, extra, or changed files. Record
the accepted QA decision and complete-inventory SHA.
Make a basename-only `train-names.txt` in the order above and bind its hash.
The [train-only staging helper](../scripts/classical_backend/mustard_stage.py)
now implements these checks, including a fresh exact output, decoded binary
mask/photo dimensions, hashes and a ≤150 MiB copy cap; its tests are in
[`test_mustard_stage.py`](../scripts/classical_backend/test_mustard_stage.py).
It deliberately requires a separately sealed 48-mask inventory and explicit
root visual-QA acceptance, neither of which this plan substitutes. No staging
of accepted masks has been run under this protocol yet. A separate photo-only
48-TRAIN stage completed at
`/Volumes/backups/code/crisp3ds-data/sam21-m1-train48-001` (report SHA-256
`d6e72143fc02ee3d18dd8961ae508417bf122cbdbc24ebc4465e207a30997861`);
it contains no masks or held-out images and does not satisfy the mask/QA gate.

## Five-view correction and new full-set decision

The sealed original inventory (SHA-256
`b47e8caec7d05fae7bb84ae9b3eec7d4a35b1457ca9df67a3b984cbaaeed8a98`)
remains rejected by the [tracked root QA](../tests/datasets/sam21_mustard_m1_full_review.json)
(SHA-256 `4084e8e458b55c566dd138f9d4b8b853ab2c68bfc20f5f831447a0cc99e9528a`).
Only `NP3_318.jpg`, `NP3_330.jpg`, `NP3_336.jpg`, `NP3_342.jpg`, and
`NP3_348.jpg` may be replaced. The [separate correction prompts](../tests/datasets/sam21_mustard_m1_board5_prompts.json)
(SHA-256 `8cc0b6fa49d977194870ef91f10b83a793294d6b3903cb1e08643016bdc9b4f8`)
retain the original three points and add a negative board point `(620,330,0)`
for those five. They do **not** change the original 43 prompts or make the
original global prompt SHA a valid label for corrected masks.

The [candidate composer](../scripts/classical_backend/mustard_candidate48.py)
requires a completed five-view worker manifest **and** successful resource
supervisor, fixed SAM checkpoint/source/runner seals, an independent root
review accepting exactly those five corrected masks, and exact original
photo and per-frame hashes. It checks all 48 original raw/clean masks, all
five replacement raw/clean masks, and metadata before and after copying.
Its fresh `sam21_mustard_m1_composed_candidate_v1` inventory records an
ordered 43 `base43` / 5 `board5` origin and per-image prompt SHA; it remains
`generated_unreviewed`. It cannot inherit the rejection's acceptance.
The [stager](../scripts/classical_backend/mustard_stage.py) recognizes this
new lane only with a *second*, independent
`mustard_sam_composed_candidate_qa_v1` decision accepting and hashing all
48 composed cleaned masks. The old inventory lane and its QA contract remain
separate. Neither corrected inference, composition, staging, nor SfM has
been run under this correction protocol yet. Tests use tiny synthetic masks
and do not substitute for the root's image review.

There is no existing mustard SfM baseline. Prior cracker-box raw/masked trials
are settings precedent, not a cross-object baseline. The control must be a
fresh **unmasked run on these same 48 mustard photos**.

## Paired frozen run

Use the existing [`scripts.classical_backend.run`](../scripts/classical_backend/run.py)
with `--stop-after-sfm`, once without masks and once with the accepted mask
directory. Its source explicitly sets a single unknown shared camera,
`SIMPLE_RADIAL`, initial focal factor 1.2, CPU feature/matching, and the
chosen options below. Use the same pinned PyCOLMAP 3.11.1 executable and
OpenMVS binary-directory hashes in both runs; the CLI currently checks all
five OpenMVS tools even when stopping after SfM, although none is executed.
No supplied poses, calibrated intrinsics, manual initialization pair, or
reference-fitted transform are allowed. Run serially in fresh output paths.

Both commands share these explicit flags (not merely defaults):

```text
--images <staged-train-photos> --image-list <sealed-train-names.txt>
--max-views 48 --stop-after-sfm --camera-model SIMPLE_RADIAL
--matching sequential --sequential-overlap 8
--sift-max-features 1800 --sift-max-image-size 1200 --seed 20260927
--sfm-max-models 5 --sfm-min-model-size 10 --min-registered-fraction 0.7
--max-threads 2 --max-gib 2 --max-rss-gib 4 --max-log-mib 32
--timeout-minutes 10 --python <pinned-pycolmap-venv-python>
--binary-dir <verified-openmvs-bin-directory>
```

Only the masked command adds `--pose-mask-dir <staged-cleaned-masks>`; each
uses its own `--output <fresh-external-path>`. These matching, SIFT, and seed
values reproduce the earlier image-only object-motion **option profile**;
the five-model retry is the existing classical runner's generic policy and
is held equal in both arms. Do not select an initial pair based on results.
The runner captures effective `pycolmap-options.json`, input hashes, stages,
and `sfm.json`, including registered and missing names and candidate models.
An arm below 70% registration is a failed *preset gate*, but retain its
diagnostic counts; do not erase or relabel it as a completed result.

Each run is capped at **2 GiB output**, **10 minutes**, **32 MiB per stage
log**, **4 GiB sampled child RSS**, and two CPU threads. The runner already
enforces a 10 GiB free-space floor on its output filesystem. Before launch,
verify ≥10 GiB plus the ≤150 MiB stage and both 2 GiB run allowances on the
external volume, and ≥10 GiB on the Mac internal volume. The current
classical runner now checks the distinct internal device at preflight,
copy-time, during each child stage and post-stage, in addition to its output
device. The [device-selection test](../scripts/classical_backend/test_mustard_stage.py)
and [stage-stop test](../scripts/brush_backend/test_masked_smoke.py) must
remain passing before launch. Set process temporary files to
the external run directory. The VPS system Python currently lacks PyCOLMAP;
the existing Mac venv is the shorter runnable path after a train-only staged
transfer. No transfer or installation is authorized by this document.

## Read-only comparison after sealing

Report, for **both** arms: registered names/count out of 48; candidate-model
counts; verified two-view-match graph connected components including isolated
images; sparse point count; per-image triangulated observations; track-length
median, p95, and fraction with ≥3 observations; and mean/median/p95 of
per-observation L2 reprojection error in pixels. The graph uses one vertex
per training image and an edge for a nonempty verified `two_view_geometries` row, with that
definition fixed before reading outputs. Project every retained 3D track
point into each registered observation's camera and compare to its measured
2D coordinate; report the number of finite observation residuals. Read
tracks and reprojection from the saved binary COLMAP models, never from
held-out or scan data. The runner does **not** itself emit all component,
track, or reprojection statistics. The separate
[`sparse_sfm_metrics.py`](../scripts/object_dataset/sparse_sfm_metrics.py)
now reads only a sealed run's training package, producer report, SQLite
feature/match database and binary sparse model, hashes them before/after,
and reports these metrics with finite/invalid reprojection denominators.
Its analytic and tiny native PyCOLMAP tests are in
[`test_sparse_sfm_metrics.py`](../scripts/object_dataset/test_sparse_sfm_metrics.py).
It has **not** been run on a mustard reconstruction because neither paired
SfM arm exists yet.
Report all denominators and any missing models. Lower reprojection error with
fewer tracks or cameras is not automatically better. Camera gauge is arbitrary
Sim(3), and a sparse success is neither an object mesh nor physical accuracy.

Do not inspect held-out photos to choose masks or SfM options. A later
held-out image evaluation needs separately estimated test cameras and its
own leakage disclosure; this paired training-only result is not that test.
