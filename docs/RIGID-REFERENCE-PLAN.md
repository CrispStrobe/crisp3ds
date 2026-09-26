# Real rigid multi-view reference: ETH3D pipes

The next measured surface reference is ETH3D's **pipes** high-resolution
multi-view training scene. It has 14 real indoor views and a released laser
reference. Download exactly these two files, totaling **199,173,411 bytes**
according to the source HTTP headers checked on 2026-09-26:

| Artifact | Bytes | Role |
| --- | ---: | --- |
| [pipes_dslr_undistorted.7z](https://www.eth3d.net/data/pipes_dslr_undistorted.7z) | 145,321,540 | Images and supplied PINHOLE calibration |
| [pipes_dslr_scan_eval.7z](https://www.eth3d.net/data/pipes_dslr_scan_eval.7z) | 53,851,871 | Cleaned laser points filtered for multi-view observability |

The [official dataset index](https://www.eth3d.net/datasets) rounds each
archive to 0.1 GB. The [format documentation](https://www.eth3d.net/documentation)
describes `cameras.txt`, `images.txt`, the world-to-camera pose convention,
and the evaluation point cloud. Its `points3D.txt` contains SfM points, **not**
measured surface truth. The [benchmark paper](https://www.eth3d.net/data/schoeps2017cvpr.pdf)
explains that supplied camera parameters were estimated and refined against
the laser scans. Agreement with those cameras tests consistency with a
published calibration; it does not independently establish physical pose
accuracy. Laser comparison remains a separate surface-quality measure.

The [ETH3D homepage](https://www.eth3d.net/) licenses the data CC BY-NC-SA
4.0 and requests citation of the benchmark paper. Use this reference for
research; review the non-commercial and share-alike terms before any
commercial use or redistribution. Download and local evaluation require no
account. The site requires an account only for result submission.

The reviewed `scripts/eth3d_multiview/fetch.py` creates a fresh ignored
directory, limits each file to
200 MiB and both to 250 MiB, checks the exact HTTP byte counts, preserves
a 10 GiB free-space floor with 500 MiB reserve, and records measured SHA-256
values. ETH3D does not publish hashes alongside these links, so these
hashes establish local provenance and repeatability, not upstream identity.
The script lists 7z members and declared expanded sizes, rejects unsafe
paths and links, and **does not extract**.

Acquisition on 2026-09-26 completed in
`.local-tools/test-data/eth3d-multiview-pipes-v2`. The first attempt left an
empty `eth3d-multiview-pipes-v1` directory because local Python 3.11 could
not find a TLS certificate authority bundle; the successful attempt used
`SSL_CERT_FILE=/etc/ssl/cert.pem` and kept HTTPS certificate verification
enabled. The [local manifest](../.local-tools/test-data/eth3d-multiview-pipes-v2/manifest.json)
records the download time, exact source URLs, measured hashes, script and 7z
tool identities, and complete member inventories. The undistorted archive
is 145,321,540 bytes (SHA-256
`718981351c14e84759fcc73215e7251fce93d6e9ea1fe24f9e15f1028232c12c`),
and the evaluation scan is 53,851,871 bytes (SHA-256
`d900ecf9ab3bfda3dddbd78174f1ef90103aa8060d1f403253ad7c9e3dc54f4c`).
Together they occupy 199,173,411 compressed bytes. 7z declares 286,754,145
expanded bytes across 25 entries. The image archive contains 14 JPEGs and
`cameras.txt`, `images.txt`, and `points3D.txt`; the scan archive contains
`scan1.ply` and `scan_alignment.mlp`. These were inventory observations at
acquisition time; extraction and metadata validation have since completed.

Read-only review of both raw `7z l -slt` listings found 19 regular files and
six directories, all unencrypted, with no links or special entries. This 7z
version reports directories through `Attributes = D_`; the original manifest's
`directory` flag is therefore not reliable. The original manifest is frozen
with SHA-256 `84f09c185f0333b8747c078c2b605b6bda9db6268d5b0c86464b205afc2d4149`.
Run `python3 scripts/eth3d_multiview/fetch.py --inventory-only
.local-tools/test-data/eth3d-multiview-pipes-v2` to verify that manifest and
both archive hashes, classify member types from the actual 7z attributes,
reject encrypted or special entries, and write a new `inventory-reviewed.json`.
This mode makes no network request and does not extract. The listed paths were
checked for containment before acquisition completed. The offline review
completed successfully on 2026-09-26: the [reviewed inventory](../.local-tools/test-data/eth3d-multiview-pipes-v2/inventory-reviewed.json)
records 19 regular files, six directories, and 286,754,145 declared expanded
bytes. Its SHA-256 is
`142c62c1c623ae581854f41b4f41d9ef6b85a0b13d08ff5b2029f6ddfdc500c6`.
The original manifest and archives were unchanged. The bounded offline
preparation then extracted to the fresh
[`build-opencv/pipes-prepare/run-001`](../build-opencv/pipes-prepare/run-001)
directory, preserving the source files and a 10 GiB free-space floor. Its
[metadata report](../build-opencv/pipes-prepare/run-001/prepare-metadata.json)
records measured SHA-256 values for all 19 regular files (286,754,145 bytes).
Preparation verified their exact paths, sizes, and archive CRCs. All 14 JPEGs parsed at
6220×4141 and matched the supplied finite PINHOLE camera, camera ID 0. The
14 world-to-camera poses have unit quaternions and contained image paths.
The binary little-endian scan header declares 11,482,717 XYZ vertices at
12 bytes each, followed by one 84-byte camera record; its declared payload
exactly matches the file size. The MeshLab file names the contained scan and
holds a finite proper rigid alignment with a small nonzero translation. The
first four images were visually inspected: the first three overlap on the
pipes, while the fourth shows a less overlapping corner. See
[preparation details](PIPES-PREPARE.md). The laser coordinates have not yet
been evaluated against a reconstruction.

The 14 images and matching published camera have passed parsing. For initial
reconstruction, use a deterministic three-or-more-view subset with overlap
verified from the images, and reserve separate views for validation.
Compare reconstructed, unfilled points to the released eval laser cloud in
the common scene frame, applying its visibility scope. Report accuracy and
completeness at stated metric thresholds (ETH3D uses centimetre-scale
thresholds) and coverage. Do not present pose agreement or sparse reprojection
error as surface accuracy. The reference is an indoor scene, so it does not
by itself validate small turntable-object millimetre accuracy.

Alternative: [Middlebury's TempleSparseRing/DinoSparseRing](https://vision.middlebury.edu/mview/data/)
each has 16 views at 640×480 and a roughly 4 MB download with camera matrices,
but the [laser truth is not distributed](https://vision.middlebury.edu/mview/).
It is a small multi-view geometry diagnostic, not a local measured-surface
benchmark. [DTU's sample set](https://roboimagedata.compute.dtu.dk/?page_id=36)
starts at 6.3 GB, with another 6.3 GB for reference points.
