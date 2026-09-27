# Proposed mustard sparse-SfM mask comparison

Status: **plan only**. Do not stage inputs or run either arm until the frozen
48-view SAM point-mask result is complete **and** root records a visual QA
decision accepting its cleaned masks for coarse pose support. A successful
inference manifest alone remains `complete_unreviewed`, not acceptance. This
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
Flatten the six batch mask folders to one `masks/` directory with exact names
`IMAGE_NAME.png` (for example `NP3_000.jpg.png`), as required by the existing
PyCOLMAP reader. Rehash each staged photo against the package and each mask
against the sealed batch manifest; reject missing, duplicate, linked, extra,
or changed files. Record the accepted QA decision and full-run manifest SHA.
Make a basename-only `train-names.txt` in the order above and bind its hash.
The [train-only staging helper](../scripts/classical_backend/mustard_stage.py)
now implements these checks, including a fresh exact output, decoded binary
mask/photo dimensions, hashes and a ≤150 MiB copy cap; its tests are in
[`test_mustard_stage.py`](../scripts/classical_backend/test_mustard_stage.py).
It deliberately requires a separately sealed 48-mask inventory and explicit
root visual-QA acceptance, neither of which this plan substitutes. No staging
has been run under this protocol yet.

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
