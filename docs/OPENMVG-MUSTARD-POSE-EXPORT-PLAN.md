# Read-only audit: OpenMVG mustard camera-pose JSON export

Status: **plan only; no sixth-target build or conversion has run**. The sealed
mustard photo-only SfM attempt remains a failed sparse gate: its receipt at
`/Volumes/backups/code/crisp3ds-data/openmvg-mustard-photo-sfm-001/receipt.json`
has SHA-256
`e45c400dcf71a3ba86fa22f7af2138418d6dc511fddcd6d8470ef33599804744`,
status `failed_registration_or_sparse_gate`, **46/48 poses**, 222 tracks and
1,497 residual observations. Its `sparse/sfm_data.bin` SHA-256 is
`dc11f9b3a74809ebc080260b360ff1dd0e6f9f9526ea151aac002ace75b2df8b`.
Exporting that model may explain camera layout; it cannot turn the failed
48-camera requirement into a pass. The original model, receipt, logs and
Sceaux outputs remain unchanged.

## Exact pinned converter interface

At the LiGT-off OpenMVG v2.1 source pin, the target is
`openMVG_main_ConvertSfM_DataFormat`. Its main source SHA-256 is
`f58a28392fe160059625c20727396e58f7b26111be6227f064e082fb1f3d7622`.
`main_ConvertSfM_DataFormat.cpp` defines required `-i|--input_file` and
`-o|--output_file`. Optional switches `-V`, `-I`, `-E`, `-S`, `-C` select
views, intrinsics, extrinsics, structure and control points respectively.
No switches exports all fields. It loads the input with `ESfM_Data(ALL)` and
saves according to the switches and `.json` output suffix. The minimal
named-pose export command is:

```text
openMVG_main_ConvertSfM_DataFormat \
  -i /Volumes/backups/code/crisp3ds-data/openmvg-mustard-photo-sfm-001/sparse/sfm_data.bin \
  -o /Volumes/backups/code/crisp3ds-data/openmvg-mustard-photo-pose-export-001/sfm_camera_poses.json \
  -V -I -E
```

The JSON serializer writes `views`, `intrinsics` and `extrinsics`; for this
version it also writes empty `structure` and `control_points` containers
when those fields are omitted. `View::save` records `filename`, `id_view`,
`id_intrinsic` and `id_pose`; `Pose3::save` records a 3×3 `rotation` and a
three-element `center`. Join a named view to a pose by `id_pose` (not by
array position). No reference camera, Berkeley calibration, board corner,
scanner depth, mesh or held-out photo is an input to the converter or orbit
diagnostic. The saved `root_path` is provenance text, not an instruction to
open images. This is an export of the already estimated failed model, with
no remapping, refinement or new SfM attempt.

## Sixth-target build and license closure gate

The generated CMake/Ninja graph already contains this target. Both generated
and frozen `build.ninja` SHA-256 equal
`4dc33b80221bdf9f9933d0fba33e207b12bfaa0009d0e7560d70263a4a8d808b`.
A read-only `ninja -f frozen-build.ninja -n
openMVG_main_ConvertSfM_DataFormat` showed **exactly two pending steps**:
compile `main_ConvertSfM_DataFormat.cpp`, then link the converter. The
target command closure contains 363 commands (SHA-256
`6dc06bbba6d3a285ce0dc3dfe0909da34408cdd3d2e3cc8cbd30baa7753121d0`).
Comparison to the already built `GeometricFilter` closure finds only those
two converter-specific commands; the sorted delta SHA-256 is
`5ed2caa1a42eb70991d008381855f4055a6cc057eae95f887e5a7a3a93a2afd9`.
The generated command closure contains no LiGT source/object or
`USE_PATENTED_LIGT` definition and no fetch/install command.

The converter main names MPL-2.0 and directly includes the already known
LGPLv3+ `third_party/cmdLine/cmdLine.h`. Its generated link line uses only
libraries already present across the five-target union: OpenMVG static
archives, CoinUtils/Clp/Osi, vendored Ceres, existing Homebrew image
libraries, Accelerate and system libraries. The source/link graph therefore
shows **no new unresolved dependency class**. It does not clear the existing
LGPL/EPL/Ceres-Eigen/VLFeat questions, nor establish shipping rights. After
any separately approved build, hash the exact binary and source, inspect
archive members for LiGT, and compare `otool -L` with the five-target receipts;
stop if a new dynamic dependency appears. Do not state a binary-level
closure result before that check.

The proposed one-shot build extension should require the existing five-target
build manifests, receipts, source and generated/frozen graph hashes to match;
an absent converter binary, build log and sixth-target receipt; exactly the
two dry-run steps; and no other active external-disk producer. Use only
`ninja -C BUILD -f frozen-build.ninja -j 2
openMVG_main_ConvertSfM_DataFormat`, with a 900-second wall cap, 4 GiB
process-tree RSS cap, 16 MiB log cap, the existing 1 GiB fork-tree cap and
11 GiB free-space floor on both devices. Before starting, reserve the unused
fork-tree allocation above the 11 GiB external floor. Scratch/cache remains
inside the fork's external `tmp`; no install, configure, network fetch,
all-target build or cleanup is included. Save a separate one-shot build
manifest and license closure receipt, retaining partial outputs on failure.

The [sixth-target supervisor](../scripts/classical_backend/openmvg_converter_build.py)
now implements that build gate. Its default invocation is read-only:

```text
python -m scripts.classical_backend.openmvg_converter_build
```

It pins the five-target receipt/log/source/graph hashes, requires the exact
two-command delta and no new link library or framework, and rejects any
existing sixth-target attempt. The separate `--build` path would create only
`logs/07-converter-build.log`, `converter-build-manifest.json`, the converter
binary and `converter-license-receipt.json` in the fork. **That path has not
been run.** The manifest and receipt preserve the evaluation-only status.

## Separate one-shot conversion and neutral check

Only after the sixth binary and its closure receipt pass review, a distinct
conversion preflight should verify the exact failed mustard receipt and
`sfm_data.bin` hashes above, sixth binary/receipt hashes, all expected
source and output paths, and a fresh external directory
`openmvg-mustard-photo-pose-export-001`. The conversion worker receives
the single sealed `.bin` path and output `.json` path, with no image path or
reference dataset argument. Bound the new output tree to 64 MiB, log to
4 MiB, process-tree RSS to 1 GiB, wall time to 120 seconds, CPU time to
120 seconds, and both disks to the 10 GiB floor plus 1 GiB margin. Before
starting, reserve the full 64 MiB allocation above the external 11 GiB
floor. Stop the converter process group on a cap breach and preserve the
failed receipt/output. No retry or overwrite of the failed SfM attempt.

The conversion receipt should store the input/bin, converter binary, build
receipt, command, log and JSON hashes, byte sizes, wall time and peak RSS.
Before any reference comparison, freeze and hash the JSON and require exactly
the sealed 48 view filenames, 46 named poses matching the failed SfM report,
finite centers and rotations, valid 3×3 orthonormal rotations, and no
unexpected pose IDs. Then measure adjacent and opposing TRAIN-label camera
geometry with the previously frozen fold thresholds. Missing two poses and
any folded orbit remain failures; this export is diagnostic only. Reference
poses/mesh may be consulted later for a separately sealed posthoc score, never
fed back into conversion or threshold selection.
