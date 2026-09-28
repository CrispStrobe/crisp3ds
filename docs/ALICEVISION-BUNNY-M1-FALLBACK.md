# AliceVision bunny on M1: stopped before photo processing

On 2026-09-27/28, the existing SeedeXR `Meshroom-0.1.0-arm64.dmg` was mounted
read-only and its native AliceVision CLI was probed for the requested 73-photo
bunny comparison. **No bunny photographs were processed by AliceVision. No
AliceVision depth map or mesh was produced.** The first command,
`aliceVision_cameraInit --help`, did not start reliably, so there is no
three-photo smoke or complete pipeline result to report.

The DMG at
`/Volumes/backups/code/alicevision-for-mac-audit/dmg-oracle-audit-001/Meshroom-0.1.0-arm64.dmg`
is the already downloaded 1,247,000,900-byte release asset (SHA-256
`41f397e834ab862d9792cd2220f104a6340efb9194eb6c0b951f405ebe0ab5b2`).
It mounts a 3.0 GiB app without copying it to the SSD. The packaged
`Contents/Resources/PACKAGING_README.txt` (SHA-256
`4666e3e150508e1cde276bc0b112a38b534b129654c8c54b450fcd67cccf1fce`)
calls the bundle a development build and says it is not self-contained;
dylib relocation, code signing and notarization remain incomplete. The
`meshroom-venv/bin/python3.13` symlink points to an absent framework path,
although Homebrew Python 3.13 exists elsewhere on this host.

The packaged `aliceVision_cameraInit` executable (SHA-256
`95e3f3b80eaeed1488ecec5e2df4f4da117aac236ee42f4be8f79fdd15409768`)
was invoked directly from the read-only mount with `--help` and
`DYLD_LIBRARY_PATH` set to the app's bundled `Resources/lib`. It produced
no stdout or stderr before a 15-second timeout. A second 8-second probe with
`DYLD_PRINT_LIBRARIES=1` also produced no loader trace before timeout. The
earlier [Mac oracle audit](ALICEVISION-MAC-ORACLE.md) saw a similar roughly
50-second stall and identified missing transitive library names. The exact
stall mechanism remains unresolved; replacing the Python symlink would not
establish that the native command or complete graph works.

At the check, `/Volumes/backups` had about 13 GiB available. Preserving the
11 GiB free-space floor leaves about 2 GiB for new files; copying the 3.0 GiB
app or rebuilding its dependency closure does not fit that bound. No app copy,
dependency install, source build, three-photo smoke, or 73-photo job was
attempted. The read-only DMG was detached after the probes.

## Reviewed Kaggle GPU fallback, not launched

The [official AliceVision v3.3.0 Linux release](https://github.com/alicevision/AliceVision/releases/tag/v3.3.0)
has a 1,505,191,867-byte `AliceVision-3.3.0-Linux.tar.gz` asset with publisher
SHA-256 `f43f498312859af627f2f7f65a6d33c2a3411b37989b8b680c04c8c690dcb640`.
Its release notes include a turntable-object pipeline. AliceVision's
[installation guide](https://github.com/alicevision/AliceVision/blob/develop/INSTALL.md)
also documents CUDA-based Linux Docker builds. Neither the tarball's dynamic
library closure nor CUDA compatibility with a particular Kaggle notebook
image has been checked; a working Kaggle AliceVision runtime is **not
established**. Do not assume a Docker daemon or permission to launch a
container inside Kaggle. The tarball is the smaller runtime candidate to
test first in an isolated notebook.

The sealed contrast-prepared bunny input is 73 PNGs totaling 121,809,661
bytes in the HIGH OpenMVG receipt. A transport archive plus manifest would
therefore carry roughly 116.2 MiB of photo payload before compression;
the full input hashes must match
`build-opencv/bunny-gamma05-clahe2/prepare-manifest.json` and the source
receipt. The separate Revopoint scanner and supplied PRO pose/calibration
files must stay out of the reconstruction dataset. The 3DLF-Scan photographs
are a research asset with the [bunny rights caveat](OBJECT-DATASETS.md):
the physical print derives from the Stanford bunny, whose commercial rights
are not cleared here. [Kaggle's dataset guide](https://www.kaggle.com/docs/datasets)
supports private datasets, but private still means uploading the files to an
external service. Review that transfer and keep both dataset and notebook
private before any upload. No dataset or notebook was created in this audit.

A runnable plan requires these gates in order:

1. Select the intended Kaggle account explicitly without putting credentials
   in code or output. The local access-token file exists, but the Kaggle Python
   module was absent in the tested interpreter. Confirm account, current GPU
   allowance, GPU type, CUDA driver, free disk, and notebook time limit in a
   read-only session. The historical local guide's 30-hour weekly quota is
   not a verified current allowance. Kaggle documents that GPU availability
   and limits can vary in its [notebook guide](https://www.kaggle.com/docs/notebooks).
2. Pin the official Linux asset by SHA-256, download it inside a disposable
   GPU notebook, and run only `aliceVision_cameraInit --help` and a CUDA/depth
   capability probe. Record extracted bytes, dynamic loader errors, GPU
   details, commands, and runtime. Stop if the binary cannot load or exceeds
   the notebook's disk/time budget.
3. Stage a private, hash-verified photo dataset and run a three-photo command
   smoke, then a 73-photo graph with bounded per-stage time, disk, and output.
   Use AliceVision's turntable configuration, record the exact graph/template
   revision, and validate actual depth maps, finite mesh payload and texture
   files. A 73-camera count or zero exit status alone is insufficient.
4. Download only a bounded mesh, texture, logs and provenance receipt for
   independent visual/shape evaluation. Keep the scanner out of generation,
   and label the run a separate cloud GPU arm, not an M1 timing result.

The existing `scripts/remote_quality/prepare_kaggle.py` stages private
kernel metadata but does not push or run it. Its associated package helper
permits at most 250 MiB of explicit input files and checks for credential
patterns; the 121.8 MB photo payload fits that *nominal* cap, subject to a
fresh archive size/hash check. No Kaggle upload or job launch was authorized
or performed here.
