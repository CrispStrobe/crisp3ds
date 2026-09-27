# SAM 2.1 tiny mustard mask smoke

Status: **three training-view smoke completed, visually unapproved**. This is a
photo-only, prompted segmentation candidate for coarse pose support, not a
validated silhouette or a ready reconstruction mask set. No held-out photo,
reference mesh, supplied mask, depth image, calibration, or pose was read by the
runner. No SfM/dense run consumed these masks. The fixed box prompt was the
previously reviewed original-pixel ROI `[480,300,760,650]` in 1280×1024 RGB.
No prompt or threshold was selected by reconstruction or reference score.

## Pin and execution boundary

The [Meta SAM 2 repository](https://github.com/facebookresearch/sam2) was
cloned on the VPS only at Git commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4` (clean tracked checkout).
Its actual Python/config/license file-inventory digest, as defined by
`source_digest()` in the runner, was
`c7eb4585a22dadd4f54ffd9134e3103c1951745a3b4631ceb0684b55742069e3`.
The official Meta Hugging Face checkpoint
[`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny)
was fetched at repository revision `de431c4043854a71d8101e17995dfe596bf101a5`:
`sam2.1_hiera_tiny.pt`, 156,008,466 bytes, SHA-256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
The observed digest equals the publisher's LFS SHA-256 metadata. Meta's
[license statement](https://github.com/facebookresearch/sam2/blob/main/README.md#license)
explicitly covers SAM 2 checkpoints and code under Apache-2.0; the HF model
card also labels this checkpoint Apache-2.0. That permissive license has no
noncommercial restriction, subject to its notice and other terms. This is a
license inventory, not legal certification.

The CPU VPS had Python 3.12.3, PyTorch 2.7.0+cpu, torchvision 0.22.0+cpu,
Pillow, and NumPy. Missing SAM2 runtime dependencies `hydra-core==1.3.2` and
`iopath==0.1.10` plus their dependencies were installed with `pip --target`
under `/mnt/storage/crisp3ds-data/sam21-tiny-prereq-001/deps`; the attempted
venv on that storage mount failed because it could not create a `lib64`
symlink. No global package install, Mac model download, fast-volume scratch,
model conversion, or GPU was used. The source checkout occupied 65 MiB and
the complete prerequisite directory approximately 279 MiB. Before inference,
available RAM was ~3.45 GiB, above the runner's 2.5 GiB gate. The runner
sets PyTorch/OMP/MKL/OpenBLAS to two threads, caps the child at 1.5 GiB RSS,
300 s wall time, 1 MiB log and 20 MiB output, and requires at least 10 GiB
free on the output filesystem. Its monitor measures the worker process, not
arbitrary descendants; this model invocation created no observed extra worker.

`scripts/object_motion/sam_mask_trial.py` validates the pinned 48-train package,
but opens only the fixed smoke subset `NP3_000`, `NP3_006`, `NP3_012`. It
rehashes each source photo before and after decode and hashes the masks,
checkpoint, source inventory, package, runner, and shared verifier. The
checkpoint/source expected hashes were supplied as reviewed exact CLI values;
the code is not a general trusted-model allowlist. Outputs are full-resolution
binary `IMAGE_NAME.png` masks plus an original-ROI overlay/masked contact sheet.
The machine status is `complete_unreviewed`, never QA acceptance.

## Actual smoke and visual limitation

The first fresh run, `/mnt/storage/crisp3ds-data/sam21-mustard-smoke-001`,
failed before writing a mask because the official predictor returns binary
float32 values while the initial validator accepted only bool/uint8. Its
failure manifest and supervisor report remain intact; elapsed 22.2 s and peak
RSS 925,028 KiB. A narrow interface correction accepted only finite exact
float32 0/1 and copied the RGB NumPy array before Torch conversion. The
model, checkpoint, prompt, and three photos did not change. Three unit tests
pass with an injected fake predictor, including exact binary output and fresh
output rejection.

The fresh run `/mnt/storage/crisp3ds-data/sam21-mustard-smoke-002` completed
all three masks in 28.96 s, peak worker RSS 1,014,004 KiB (~0.97 GiB). Mask
support areas were 20,805, 20,694, and 20,579 pixels; inference took about
5.0–5.6 s per image. The [local copied review sheet](../.local-tools/sam21-mustard-review-002/smoke_overlay.jpg)
has SHA-256 `4aa05de85b75a8b2d2153af94236a76dd1be334e863aa8f60687396e0320e047`;
its matching manifest/supervisor were also copied there. The three full-size
masks remain on the VPS under `sam21-mustard-smoke-002/masks/`.

On inspection of the original-ROI sheet, the blue/red bottle labels are
retained far better than the rejected warm-color masks. However, a few
non-object pixels remain near the cap and checkerboard, especially in
`NP3_012`. This **does not pass full-view visual QA** and does not authorize
the 48-view run. Any future 48-mask candidate needs a separately approved,
frozen extension and all-48 visual review before pose matching; an exact
dense silhouette would require a separate contract.

## Optional connected-object cleanup smoke

The follow-on [cleanup helper](../scripts/object_motion/sam_mask_cleanup.py)
does **not** rerun SAM or alter the prompt. It takes the sealed raw-smoke
manifest (SHA-256
`6804b82de786e2488bf6dd2e5f776b59025fef019a4bb8d3101fefbef5cddb94`)
and its three exact-hash raw PNGs, then keeps only the sole largest
**8-connected** foreground component. It does not fill holes, erode, dilate,
smooth, or move surviving pixels. Equal-sized largest components, empty or
nonbinary masks, and removal over 10% are rejected as ambiguous. This is an
explicit single-connected-object assumption, **not** a general segmentation
guarantee: a truly detached visible object part would be removed. Both raw
byte-identical copies and cleaned masks are retained in separate directories.

One fresh VPS cleanup run completed at
`/mnt/storage/crisp3ds-data/sam21-mustard-cleanup-smoke-001/`; manifest SHA-256
`789fe5a942d21b32a0967d2e8479a67873240ec6f5382a69651a5c7347aa3c07`.
It removed 5 pixels from `NP3_000`, 3 from `NP3_006`, and 31 from `NP3_012`
(the latter is one detached component); the retained fractions all exceed
99.84%. The [three-view raw-versus-cleaned crop](../.local-tools/sam21-mustard-cleanup-review-001/raw_vs_cleaned_crop.jpg)
has SHA-256 `ba3fa08e56a389055f6fa6eed9baccac0c40c4f55aaa4e8182e977e0c428e07a`.
It shows the detached `NP3_012` checkerboard speck removed while the bottle
and label remain visible. Six focused SAM/cleanup tests passed. This is only a
three-view QA candidate; the output remains `complete_unreviewed`, and no
48-view inference was part of that smoke. The later separately bounded 48-view
candidate below supersedes any suggestion that these three views establish
general mask quality.

## Frozen 48-training-view extension: rejected

After reviewing the cleaned three-view smoke, root approved one extension
using the **same box, SAM2 source/checkpoint, and largest-8-component rule**,
strictly on the 48 training photos. The [full runner](../scripts/object_motion/sam_mask_full.py)
fixes three sequential 16-view batches with no per-view prompt change,
two CPU threads, a 120 s/1.5 GiB worker RSS bound per batch, 420 s overall,
60 MiB output ceiling, and a new 2.5 GiB available-RAM check before each
batch. A failed or ambiguous cleaned mask is recorded `failed_unusable` while
its raw mask remains; no forced component is published. Source/model/helper
hashes are rebound before and after each batch. Review sheets are hash-checked
against batch manifests and show failed/missing tiles explicitly. Eight
focused SAM tests passed before this VPS run.

The fresh output is `/mnt/storage/crisp3ds-data/sam21-mustard-full-001/`.
Batch 0 attempted 16 views in 113.97 s, peak RSS 1,120,960 KiB;
batch 1 attempted 16 in 116.50 s, peak RSS 1,122,640 KiB. Both respected
their bounds. The 48-view supervisor stopped **before launching batch 2**
because available RAM fell below 2.5 GiB at that pre-batch check. It retained
the 32 attempted views and wrote a `failed_or_partial_unusable` report, SHA-256
`0441b19a257dd424c3c77728bdee405efddf62ace9563d105d6d76ffcc67c936`.
The batch manifests have SHA-256
`aa37ec1e6dbed0c7141ab1637f086ac19a5ffec09634faa5a4d8dd2d2f9656a1`
and `fdb1f738609a247bb2bfdfe3a41e946e8aa3d26c8f24528bd2573a8b2d99ec6b`.
In those 32 views, 19 numerically cleaned and 13 were flagged unusable
because removing disconnected foreground would exceed the frozen 10% guard.
The unused batch 2 was **not** later resumed: root inspected the first 32 and
rejected the method outright, so more inference would not change this
candidate's decision.

The [raw](../.local-tools/sam21-mustard-full-review-001/training_raw_sheet.jpg),
[cleaned](../.local-tools/sam21-mustard-full-review-001/training_cleaned_sheet.jpg),
and [full-context](../.local-tools/sam21-mustard-full-review-001/training_full_context_sheet.jpg)
review sheets were copied locally with exact SHA-256 values
`5ae4f78ff36ed60586352081877e7053f84af23adc90caabc4c2316c937d8033`,
`61dc600c720e55a5a83f601222fd129f3a53c68c05fc21d5a40998c7542eee43`,
and `07db3aa1ecf26db3f460ec80fd7520ecbdd63c80fa1692ae7063b94d4c2fb33c`.
Root and the implementation agent independently found the decisive visual
failure: from about `NP3_042` and across much of the mid-orbit, SAM switches
from the bottle to the bright checkerboard/background; the bottle becomes a
hole or tiny fragment. Thus even the 19 numerical cleanups are **not usable
object masks**. Largest-component filtering cannot repair a wrong semantic
target. The tracked [visual decision](../tests/datasets/sam21_mustard_full_review.json)
binds report/sheet hashes and rejects this candidate for reconstruction.
No SfM or dense stage used any of these masks; no held-out photo or reference
mesh/depth informed selection or QA.

A separate future candidate could use an explicit positive object point and
negative background points on frozen training views 0°, 66°, and 180° to test
whether prompting prevents target switching. That is a proposed **new trial**,
not an adjustment or rerun here; prompts and acceptance rules must be frozen
before inference, with no held-out/reference use and no reconstruction before
full visual QA.
