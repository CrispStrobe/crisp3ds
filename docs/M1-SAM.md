# M1 SAM2 three-view parity gate

The new [Mac adapter](../scripts/object_motion/sam_m1_parity.py) is a separate,
CPU-only diagnostic for the same mustard SAM2.1 tiny candidate. It never
changes the earlier VPS reports or claims that generated masks are suitable
for reconstruction. The external-only setup `002` completed before the
one approved parity run; setup report SHA-256 is
`2c520f3e07729ea881707b13fff234fd5400e851acbe1390b743abcec07fc874`.

## Frozen three-view contract

The inputs are the exact mustard TRAIN package, the [full-48 common-point
prompt manifest](../tests/datasets/sam21_mustard_point48_prompts.json) with
SHA-256 `28440795d5f38590df8c5568969970d6623fbe22ea6b3e8781cba10608149d32`,
SAM2 source inventory SHA-256
`c7eb4585a22dadd4f54ffd9134e3103c1951745a3b4631ceb0684b55742069e3`,
and tiny checkpoint SHA-256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
It selects `NP3_000.jpg`, `NP3_066.jpg`, and `NP3_108.jpg` only. These are
not the older three-view smoke prompts: all three use the full-48 candidate's
positive `(600,525)`, negatives `(720,620),(520,630)`, and original-pixel box
`[480,300,760,650]`. `000` is from sealed VPS batch 0; `066` and `108` are
from sealed batch 1. The adapter verifies **all 16** prior raw and cleaned
masks and both batch manifests, not just the three selected views, before
inference and again after it. The six unsealed batch-2 masks are excluded.

For each selected image, the adapter reports candidate and sealed-reference
raw/clean SHA-256, foreground pixel counts, exact-pixel equality,
intersection/union and foreground IoU, and point memberships. The engineering
parity gate is frozen before use: **raw and cleaned IoU each ≥0.99**, with
candidate and reference memberships `[1,0,0]`. Cross-platform bit identity is
reported but not required. A passing numeric result is only
`parity_observed_pending_visual_qa`; root must inspect the three outputs
before any attempt at the remaining 32 TRAIN views. No held-out image,
reference mesh/depth, supplied pose, or reconstruction metric enters this
check. A failed gate does not authorize changing prompts, threshold or model.

## Host and storage bounds

The supervisor requires native arm64 macOS, a mounted `/Volumes/backups`
device distinct from the workspace's internal APFS Data volume, and output
under `/Volumes/backups/code/crisp3ds-data`. The isolated Python environment,
pip cache/temp and checkpoint must reside under
`/Volumes/backups/ai/crisp3ds-sam21-m1-002`; pinned source, photos/reference,
and outputs reside on the mounted external code/data volume. Transfer/install steps remain
separately gated to ≤2 GiB setup growth and ≥10 GiB free on both volumes.
The completed setup's data staging locations are
`/Volumes/backups/code/crisp3ds-data/sam21-m1-source-002` (contains `sam2/` and `LICENSE`) and
`/Volumes/backups/code/crisp3ds-data/sam21-m1-three-train-002` (`photos/`,
`prior/`, `package.json`, `prompts.json`) for the three photos plus all 16
sealed reference mask pairs/manifests. Setup `001` stopped before pip
installation because its source-inventory helper sorted relative path strings
while the reviewed source digest sorted path components; all 78 local/remote
file hashes remained unchanged. Fresh `002` is the approved setup attempt,
and completed. The run used the exact staged paths above plus checkpoint
`/Volumes/backups/ai/crisp3ds-sam21-m1-002/checkpoints/sam2.1_hiera_tiny.pt`
and output `/Volumes/backups/code/crisp3ds-data/sam21-m1-parity-001`.

The run uses SAM2's CPU predictor with two Torch/OMP/BLAS threads, ≤180 s
worker wall time, ≤1.5 GiB worker RSS, ≤20 MiB output/log cap (log ≤1 MiB),
and ≥10 GiB free on both volumes. The Mac RAM gate is at least 2.5 GiB from
`vm_stat` **free + inactive** pages. This is a conservative operational
estimate, **not equivalent to Linux `MemAvailable`**; `ps -o rss=` provides
worker RSS. The supervisor checks limits while running and after exit, and
the worker checks essential host/resource conditions itself. Failures before
worker output creation get an exclusive sibling supervisor report. The
separate [tests](../scripts/object_motion/test_sam_m1_parity.py) cover exact
name-to-batch mapping, Darwin parser failures, external-mount/device rejection,
binary IoU, and all terminal cap boundaries; the focused pinned-Python run
passed 7/7. A rejected raw mask is saved and hash-reported before point/area
checks.

## Actual parity result

Immediately before the approved CPU-only run, read-only validation confirmed
the exact three staged training JPEGs, all 16 sealed prior mask pairs and
batch manifests, checkpoint/source pins, fresh output, 3,700,688 KiB of
free+inactive RAM estimate, and 11,112,300,544 bytes as the smaller of the
two disk-free readings. The frozen adapter source SHA-256 was
`01b159a3cf876923524f9dd8e8b0283fcbe4828d8c20a4e68a918829c92a0047`.
The single trial completed in 15.61 s with peak worker RSS 1,222,672 KiB;
the output manifest SHA-256 is
`bc99129f36654888b9b9fdbd7f9ac606b8527518d91424ca48e7283ac9f42741`
and supervisor SHA-256 is
`dc82764bb2363aa41597af7280ec3462d85e14e20dae4dbdd7e667b8f4b2cc86`.
Both volumes remained above the 10 GiB floor. No retry, MPS variant or
remaining-32 inference was run.

For `NP3_000`, `NP3_066` and `NP3_108`, **all six decoded raw/cleaned mask
arrays were pixel-identical** to the corresponding sealed VPS arrays (IoU
1.0; point membership `[1,0,0]` in each). The PNG **file hashes differ**,
so this is pixel parity, not byte/encoding parity; the reason for different
encoded bytes was not established. Cleanup removed 972 detached raw pixels
in `000`, and zero in the other two. The [hash-bound RGB/raw/clean sheet](</Volumes/backups/code/crisp3ds-data/sam21-m1-parity-review-001/review_sheet.jpg>)
is 93,795 bytes, SHA-256
`f5d84f9c09b6e7a51925572b400c549ff21264e5c69d2ee0824f2eb5dfaf7e4d`.
The machine status remains `parity_observed_pending_visual_qa`. Root
independently viewed the sheet and decoded all six candidate/reference PNG
pairs, confirming pixel equality; the [tracked narrow QA decision](../tests/datasets/sam21_mustard_m1_parity_review.json)
accepts **only these three** as M1 CPU parity/coarse-support evidence. It
does not certify exact silhouettes or a complete 48-view mask package.
`full_training_package_ready=false`; the missing 32 views were not inferred
and no SfM/dense pipeline consumed these masks.

Meta's [SAM2 requirements](https://github.com/facebookresearch/sam2/blob/main/setup.py)
include Python ≥3.10, PyTorch ≥2.5.1 and matching TorchVision. The pinned
VPS candidate used CPU PyTorch 2.7.0; [PyPI](https://pypi.org/project/torch/2.7.0/)
lists a 68.6 MB CPython 3.11 macOS arm64 wheel and PyTorch's
[version guidance](https://docs.pytorch.org/get-started/previous-versions/)
pairs it with TorchVision 0.22.0. Wheel availability does not prove this
model/source runs correctly on M1, which is why parity precedes larger work.
