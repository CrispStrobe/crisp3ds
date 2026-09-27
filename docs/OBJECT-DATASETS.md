# Real object photos with an independent scan

## VPS expansion: mustard bottle and power drill

[Two further YCB objects](YCB-EXPANSION.md) are acquired and hash-recorded on
the VPS: 60 original NP3 images each and separate Google-scanner meshes.
These are independent shape references, not yet registered benchmark goldens.
One elevation misses undersides; the objects occupy a relatively small part of
each 1280×1024 frame. The existing cracker-box mask must not be reused unchanged.
No reconstruction result on these two objects is claimed yet.

## YCB cracker box: second real-object subset

The [official YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html) explicitly licenses its **data** CC BY 4.0 (and its code separately MIT). Attribute Berk C. Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M. Dollar, the dataset title, source URL, license, and changes. CC BY's copyright grant does not itself clear Cheez-It branding or packaging trademarks for asset redistribution; review those rights before shipping photos or model textures in a product.

This local `003_cracker_box` subset uses 60 *real* 1280×1024 JPEGs from one Berkeley RGB-D camera, `NP3`, at turntable angles 0° through 354° in 6° steps. The reference is the same object scanned on YCB's separate Google scanner, not a reconstruction from the selected RGB photos. It is a **separate-sensor shape oracle**, not certified metrology ground truth. The datasets do not supply a verified registration between the selected photos' reconstruction and the Google mesh. A reference-fitted Sim(3) score would describe shape only, not independently recovered scale or metric accuracy. Never feed the Google mesh, depth maps, masks, or supplied camera poses into an image-only reconstruction or training run.

Run from the repository root, with 10 GiB free-space reserve:

```sh
mkdir -p .local-tools/tmp
TMPDIR="$PWD/.local-tools/tmp" python3 scripts/object_dataset/prepare_ycb.py --fetch
python3 scripts/object_dataset/evaluate.py --reference .local-tools/test-data/ycb-cracker-box/reference/google_64k_geometry_f64.ply
```

The fetcher pins exact official archive sizes, ETags, and locally measured SHA-256 hashes. If the two archives are already present and verified, omit `--fetch`. It refuses changed or unverified existing archives and a prepared destination; no silent overwrite or re-download occurs. Source transfer is bounded to 750 million bytes and selected expansion to 200 million bytes, with a 10 GiB free-space floor and 15-minute overall deadline. The two full source `.tgz` files total 680,498,737 bytes; the selected 60 JPEGs total 61,113,267 bytes. The RGB-D tar has mixed, non-angle-ordered depth entries, so a short gzip prefix cannot guarantee 60 photos around one camera's full rotation. The extractor validates tar member types and paths, selects only named JPEGs and the original Google PLY, and records per-file SHA-256, source member, camera, and angle in the ignored local `manifest.json`; tracked [anchor hashes](../tests/datasets/ycb_cracker_box.json) identify this exact subset. The S3 ETags are multipart identifiers, **not** substitute MD5 or SHA-256 checksums.

| Role | Local path | Use |
| --- | --- | --- |
| Real RGB input | `.local-tools/test-data/ycb-cracker-box/photos/` | Exactly 60 original NP3 JPEGs; only this folder is reconstruction input |
| Original reference | `.local-tools/test-data/ycb-cracker-box/reference/google_64k_original_ascii.ply` | Separate Google scanner ASCII PLY, excluded from reconstruction |
| Converted reference | `.local-tools/test-data/ycb-cracker-box/reference/google_64k_geometry_f64.ply` | Binary little-endian double-XYZ PLY for bounded evaluator; triangle indices unchanged, no scale/cleanup |

The original Google PLY contains 32,770 vertices and 65,536 triangular faces. The geometry-only conversion preserves each decimal XYZ value as a binary float64 and all triangle vertex indices, while omitting normals and texture coordinates; it does not alter topology or scale. The converted mesh passes finite-coordinate/index and zero-area checks. The JPEGs show a checkerboard on the turntable that rotates with the object; image-only SfM may reconstruct that background as well as the box. Inspect object isolation before treating any output as object geometry. The generic alignment/evaluation cautions below apply equally to this subset.

## 3DLF-Scan bunny: research-only rights caveat

The local **3DLF-Scan bunny** subset has 73 original PNG photographs from the PhotonicSense apiCAM PRO turntable sequence and a fused Revopoint Miraco structured-light mesh of the same physical resin print. The source is [Vodianyk, Nava-Baro, and Popov, 3DLF-Scan, version 1](https://data.mendeley.com/datasets/ngvgpsvd8b/1), DOI [10.17632/ngvgpsvd8b.1](https://doi.org/10.17632/ngvgpsvd8b.1), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the authors, dataset title, DOI, license, and any changes when sharing derived material. The [data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969299/) explains the turntable and scanner acquisition.

**Commercial-rights review remains required.** The physical bunny print is based on a Stanford model. The [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) permits research use and free redistribution of its own models but says commercial use or inclusion in a product for sale requires permission. The canonical Stanford STL was excluded here, yet the rights of a physical-print derivative and its rescans are not independently cleared by this project. Treat this subset as an isolated local research/test asset; do not ship it or assume commercial redistribution is approved solely from the 3DLF-Scan CC BY 4.0 label.

Run from the repository root:

```sh
mkdir -p .local-tools/tmp
TMPDIR="$PWD/.local-tools/tmp" python3 scripts/object_dataset/fetch_3dlf.py
python3 scripts/object_dataset/evaluate.py --reference .local-tools/test-data/3dlf-scan-bunny/revopoint/bunny/fuse_mesh_rgb.ply
```

The fetcher extracts only these members from the public 9,829,049,614-byte ZIP using HTTP byte ranges. It never downloads the full ZIP. It bounds each range, each decompressed member, aggregate compressed and expanded bytes (2 GiB), runtime, and free space (10 GiB floor). It rejects an existing destination; review or move that folder before a deliberate refetch. It validates ZIP member CRCs and writes SHA-256 and exact ZIP offsets for every selected file to the ignored `.local-tools/test-data/3dlf-scan-bunny/manifest.json`. The upstream full-ZIP SHA-256 is *published metadata*, not locally verified because the full ZIP was not downloaded. The tracked [subset manifest](../tests/datasets/3dlf_scan_bunny.json) records the source URL and anchor hashes.

| Role | Local path | Use |
| --- | --- | --- |
| Original real photos | `.local-tools/test-data/3dlf-scan-bunny/pro/bunny/rgb/` | 73 PNG input frames, including the full 360° turntable sweep |
| Camera metadata | `.local-tools/test-data/3dlf-scan-bunny/pro/bunny/pcd/poses_metric.json` and `calib_pro/intrinsics/rgb_optic.json` | Supplied poses and calibration; inspect conventions before use |
| Independent reference | `.local-tools/test-data/3dlf-scan-bunny/revopoint/bunny/fuse_mesh_rgb.ply` | Excluded from reconstruction; post hoc comparison or explicitly reference-fitted diagnostic only |

The original PNGs are dark matte objects on a bright backdrop. Any brightness adjustment should be derived only from the input photos, recorded as a separate variant, and never overwrite the originals. The `rgb_bright`, depth, masks, and Revopoint RGB/depth folders were not included in this input split.

The Revopoint mesh is genuinely independent of the PRO photo/depth pipeline, but **the two sensor coordinate frames are not registered by this subset**. The supplied PRO poses are turntable/ICP estimates, not metrology-grade camera truth. Do not compare raw coordinates or report a metric reconstruction error until a transform with clear provenance has been independently established and checked. The geometry evaluator reports finite vertices, face validity, bounds, and degeneracies; its quality score is intentionally null by default. The source Stanford bunny STL represents the canonical design, not the measured shape of this printed physical instance, and was not fetched as ground truth.

Once an independently justified alignment exists, `evaluate.py` can make a bounded, descriptive surface comparison. Supply `--output <reconstruction.ply> --transform <sim3.json> --threshold <positive-distance>`. The transform JSON must contain a finite 4×4 matrix mapping reconstructed coordinates into Revopoint coordinates, `object_id`, `source_frame`, `target_frame`, `reference_units`, `provenance`, `scale_provenance`, `registration_basis` (`external-calibration`, `manual-landmarks`, or `reference-fit`), and SHA-256 hashes named `reference_sha256` and `output_sha256` for the exact two meshes. Reflections and unequal axis scales are rejected; no registration or ICP is performed automatically. The threshold is expressed in the declared reference units. Optional `--samples` is capped at 4,096 per mesh, and `--save-report` writes only to a fresh path.

The comparison samples triangles with probability proportional to area using a recorded random seed. It reports bidirectional nearest-*sampled-point* RMS, median, p90, precision, recall, and F-score. These are finite-sample estimates, not exact point-to-triangle distances. A transform fitted to the reference itself is labeled `reference-fit` and cannot establish independently recovered metric scale or held-out accuracy. Even with externally supplied alignment, the tool records its provenance but does not verify that calibration, so it does not assert a metric accuracy claim on its own. Geometry used for scoring or preview is capped at 512 MiB per PLY, 1 million vertices, and 2 million faces per mesh. Both meshes and temporary NumPy triangle arrays are resident during sampling, so a run near these caps can require roughly 1–2 GiB of RAM; use a machine with headroom beyond the input file sizes. The sample cap remains 4,096 per mesh for scoring and 20,000 for preview.

For a reconstruction with no separate cross-sensor calibration, the [alignment diagnostic](../scripts/object_dataset/align.py) can fit a Sim(3) *to the independent scan* for a shape-only description. Run `python3 scripts/object_dataset/align.py --reference <revopoint.ply> --output <reconstruction.ply> --object-id 3dlf-scan:bunny --reference-units 'reference-coordinate units (physical units unverified)' --save-transform <fresh-sim3.json>` with the Python 3.11 environment that has NumPy. Then pass that JSON to `evaluate.py` with an explicit threshold. A reproducible relative threshold is 1% of the reference bounding-box diagonal: `0.01 × sqrt(dx² + dy² + dz²)`. For this local bunny scan, the measured bbox extents give approximately `2.08588` reference-coordinate units, so `--threshold 2.08588` is an example; it is **not 2.08588 mm**. This is a reference-fit result, not fully held-out validation: the same scan supplies alignment and comparison. It does not demonstrate that the reconstruction independently recovered physical scale or alignment. The alignment script never changes either mesh or promotes an output.

The alignment algorithm was fixed before inspecting the bunny reconstruction: 1,024 deterministic area-weighted samples per mesh (at most 1,500), all 24 proper PCA axis/sign initializations, the best four candidates refined for eight iterations each, then the best for twenty iterations. Each iteration pairs nearest sampled points in both directions, trims the worst 20% of each direction, and fits a proper Sim(3) with Umeyama. Candidate ranking and best-transform retention use symmetric bidirectional RMS; there is no automatic ICP beyond this specified local fit. It has a 60-second deadline. The transform records the script hash, mesh hashes, parameters, objective history, and fit provenance. A local minimum, sampling noise, partial geometry, or background surface can still bias this diagnostic.

For the final sampled-surface report, use `--samples 4096 --seed 2027 --reference-self-control` with the same 1%-of-reference-bbox threshold chosen before examining the score. The self-control draws the *reference twice* with seeds 2027 and 2028, then measures nearest sampled-point distances between those draws. It estimates a sampling-distance floor; it is neither a reconstruction score nor a correction to the primary score. The main reconstruction/reference comparison remains unchanged. The report also counts faces excluded from area sampling because they are nontriangular or zero-area.

For visual inspection after a transform has been saved, [preview.py](../scripts/object_dataset/preview.py) accepts the same exact-mesh-bound transform and writes a fresh PNG under `.local-tools/`. Its three XY/XZ/YZ rows show the independent reference and transformed reconstruction side by side with identical projection bounds, using at most 20,000 sampled surface points per mesh. The preview is a diagnostic image, not evidence of metric alignment or geometric accuracy.

An OmniObject3D battery pair was also investigated because the [official dataset](https://github.com/omniobject3d/OmniObject3D) is CC BY 4.0 and offers real videos with raw scanner meshes. The official public Google Drive lists `raw_scans/battery.tar.gz` (78,246,728 bytes, file ID `1gAW6iciDOHiU_rBCRHMbwbL2OUrZOsvC`) and `videos_processed/battery.tar.gz` (448,166,011 bytes, ID `1-4ekYqedvqfHkmCISubXUPw6QeE9fG2l`), but anonymous download currently returns a Google quota/access error. No OmniObject3D bytes were accepted locally. Rendered-image mirrors are not a substitute for the original video frames.

[ObjectFolder-Real](https://objectfolder.stanford.edu/objectfolder-real-download)
is another technically relevant candidate: 100 real household objects, HD
turntable videos and separately scanned meshes. The official Real download page
does not state a license. The CC BY 4.0 statement on the separate
[ObjectFolder 2.0 page](https://objectfolder.stanford.edu/objectfolder2-0-download)
must not be silently transferred to Real. No Real files were downloaded or
accepted as commercially cleared assets; explicit maintainer terms are needed.

The [3DLF-Scan release](https://data.mendeley.com/datasets/ngvgpsvd8b/1) contains only seven printed Stanford figures, so another object from this ZIP would retain the same third-party rights concern. Other possible sources need separate checks: [ObjectFolder-Real](https://objectfolder.stanford.edu/objectfolder-real-download) lists real rotating videos and independently acquired meshes of 100 household objects, but its exact Real-data license scope and selective archive sizes are not yet verified; [OpenSubstance](https://opensubstance.github.io/) provides real multi-view images and scanner shapes but presents a registration/request workflow and no clear commercial data license on its project page. No alternative was downloaded or marked commercially cleared.
