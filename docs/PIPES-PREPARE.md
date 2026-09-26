# ETH3D pipes preparation

`scripts/eth3d_multiview/prepare.py` prepares the two previously downloaded
ETH3D pipes archives for local evaluation. It makes no network requests. The
input is `.local-tools/test-data/eth3d-multiview-pipes-v2` and the fresh output
is `build-opencv/pipes-prepare/run-001`. The two source archives, original
download manifest, and reviewed inventory are preserved.

Preparation completed on 2026-09-26. The [validated metadata](../build-opencv/pipes-prepare/run-001/prepare-metadata.json)
records 19 extracted files totaling 286,754,145 bytes and their measured
SHA-256 values. All 14 JPEGs are 6220×4141 and match the supplied PINHOLE
camera (camera ID 0). The released scan has 11,482,717 vertices with 12 bytes
per vertex, a 630-byte header, and one separate 84-byte camera record. The
MeshLab alignment is a finite proper rigid transform; it includes a small
translation and is not exactly the identity. The first four images were
visually inspected: the first three overlap on the pipes, while the fourth
includes a less overlapping corner. This preparation establishes readable,
internally consistent inputs; no reconstruction-to-laser accuracy or
completeness has yet been measured.

Run the read-only preflight first, using the installed Python 3.11 (the local
Homebrew Python 3.14 currently cannot load its XML parser):

```sh
.local-tools/colmap-sparse/venv/bin/python3.11 scripts/eth3d_multiview/prepare.py --preflight-only
```

The live run used the same script without `--preflight-only`. The output
directory must be fresh for any new run. If a run fails, its partial output
remains for inspection; choose a new fresh destination in the script after
resolving the failure. Do not rerun into the partial directory.

The preflight checks the reviewed inventory's pinned SHA-256, archive sizes
and SHA-256 values, and fresh raw `7z l -slt` listings. It rejects unsafe
paths, links, special entries, encryption, file collisions, unexpected files,
and expanded sizes beyond 500 MiB. The extraction has a 300-second limit,
checks file paths and sizes while 7z runs, and maintains a 10 GiB free-space
floor. `TMPDIR` is `.local-tools/tmp`. The two archives share the `pipes`
directory; the second extraction skips existing entries only after confirming
that no file paths collide. All 19 extracted files must match listed paths,
sizes, and archive CRCs. Their SHA-256 values are measured and recorded.
The source archive and inventory hashes are checked again afterward.

`prepare-metadata.json` in the output root records all 14 JPEGs in name order.
Each `images` record has `id`, `name`, `path` relative to the output root,
`size_bytes`, `sha256`, `camera_id`, embedded `camera` (`model`, `width`,
`height`, `params`), `qvec`, `tvec`, and `pose_convention: world_to_camera`.
The file also records the PLY vertex count, stride, header and payload sizes,
all declared element properties, and the scan alignment matrix. The original
JPEGs remain at `pipes/images/dslr_images_undistorted/`; the scan is at
`pipes/dslr_scan_eval/scan1.ply`.

Validation reads JPEG headers and terminal markers for dimensions without
decoding all pixels, requires finite positive PINHOLE focal lengths, camera
dimensions matching each JPEG, 14 unit-quaternion world-to-camera poses,
and contained image paths. The PLY must have a binary little-endian header,
vertex positions, and a payload exactly matching declared counts and strides.
The MeshLab alignment must name the contained scan and have a finite proper
rigid 4×4 matrix. This stage does not read laser point coordinates, score a
reconstruction, or treat the SfM `points3D.txt` as measured surface truth.

Run focused tests with:

```sh
.local-tools/colmap-sparse/venv/bin/python3.11 -m unittest scripts.eth3d_multiview.test_prepare -v
```
