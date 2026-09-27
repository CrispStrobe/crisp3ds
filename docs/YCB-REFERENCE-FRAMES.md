# YCB cracker-box reference-frame audit

**Finding:** The selected `NP3_000`–`NP3_354` photos do have usable *Berkeley rig* calibration and turntable poses in the already-downloaded official RGB-D archive. They do **not** have a supplied transform to the exact Google 64k mesh frame. The current Google-mesh surface comparison must therefore remain labeled **reference-fitted**; the provided camera/turntable metadata alone cannot turn it into an independently aligned metric accuracy result.

## What is present

The existing [local YCB source manifest](../.local-tools/test-data/ycb-cracker-box/manifest.json) binds the official `003_cracker_box_berkeley_rgbd.tgz` (SHA-256 `15185a1e9da0f5da5264eef8dfad129437f157ea993a05ef75f80134aa86adc5`) and `003_cracker_box_google_64k.tgz` (SHA-256 `a4d7173ef4a43c51af5df166deed85ca2b7014626ec5f6f2895561cdbf7c2cd5`). Both local archives were rehashed against that manifest before inspection. No new archive or large dataset was downloaded, and none of this metadata entered reconstruction.

The Berkeley tar index contains all 60 selected `003_cracker_box/NP3_<angle>.jpg` members and matching `003_cracker_box/poses/NP5_<angle>_pose.h5` members at 0°, 6°, …, 354°. It has 120 NP5 pose files total and **no separate NP3 pose files**. It also contains `003_cracker_box/calibration.h5` and `poses/turntable.h5`. The pose files are 275,520 uncompressed bytes in total. Five small members (calibration, turntable, and NP5 poses at 0°, 6°, 354°) were extracted only into the ignored [metadata audit folder](../.local-tools/test-data/ycb-cracker-box/metadata-reference-frames-001/), not into any photo or reconstruction folder. Their SHA-256 hashes are:

| Member | SHA-256 |
| --- | --- |
| `calibration.h5` | `15f724c10131129183e97cb3ae6fdb3f13ff7ad8c5a50812313f06652a7f6c07` |
| `poses/turntable.h5` | `b4cf00e0e463585dd76c7671730f2fc2d6ba455c42a44c4f2bf65534cdf4e152` |
| `poses/NP5_0_pose.h5` | `f303e9b8f8a601a0fe7633278c1acf2f2b81fc84204da9ae2eae47ed9c968aeb` |
| `poses/NP5_6_pose.h5` | `8a336cd50d3cf4d9fa5d4337412fa561cb5b66abe8f28ede585fe2788b7e48cc` |
| `poses/NP5_354_pose.h5` | `96f2044a442f69be67027882d2ea3a5b518901a15a1922951890d6e122cb2793` |

`h5dump` shows `calibration.h5` datasets `/NP3_rgb_K` (3×3), `/NP3_rgb_d`, and `/H_NP3_from_NP5` (4×4). The sampled pose files each contain `/H_table_from_reference_camera` (4×4) and `/board_frame_offset` (3-vector). `turntable.h5` contains `/center` and `/normal`. This agrees with the [BigBIRD calibration documentation](https://rll.berkeley.edu/bigbird/access.html): NP5 is the reference camera; `calibration.h5` holds camera-to-reference relative transforms, while `NP5_<angle>_pose.h5` maps reference camera to turntable frame. The [YCB data paper](https://personalrobotics.cs.washington.edu/publications/calli2017ycb-data.pdf) likewise lists Berkeley RGB-D images, camera poses and intrinsics as one product, and the Google scanner meshes as another.

For a point in the Berkeley turntable frame, the documented matrix names imply the per-angle NP3 camera mapping

```text
H_NP3_from_table(angle)
  = H_NP3_from_NP5 · inverse(H_table_from_reference_camera(angle))
```

This is a frame-composition statement, **not** a validated reprojection result. Exact interpretation of `board_frame_offset`, distortion, image coordinates, and pose convention should be tested against chessboard/depth observations before treating the derived matrices as camera ground truth. In particular, the nominal filename angle is not itself a calibrated pose.

## What is missing for the Google 64k reference

The verified Google 64k tar index contains only `nontextured.ply`, `nontextured.stl`, textured OBJ/DAE/MTL and texture, plus `kinbody.xml`. Its kinbody file names only the mesh assets and contains no explicit cross-scanner transform. The [official YCB catalog](https://ycb-benchmarks.s3.amazonaws.com/index.html) and [YCB data paper](https://personalrobotics.cs.washington.edu/publications/calli2017ycb-data.pdf) describe the Berkeley rig and Google structured-light scanner as separate acquisition systems; neither gives an exact Google-mesh-to-Berkeley-turntable transform for this object. Absence from the inspected archives and cited documentation is not proof that no external unpublished registration exists, but **none is established by the available evidence**.

Consequently, one may use the supplied metadata in a separately labeled *camera/turntable oracle* evaluation of Berkeley-frame geometry, while keeping it out of the image-only reconstruction lane. One may also fit the Google mesh to a Berkeley scan or to the reconstructed surface, but that is a model-based alignment and must report the fitting source and uncertainty. It is not independent calibration for the exact Google 64k mesh. Do not silently equate the Berkeley turntable frame with the Google mesh frame, and do not relabel the existing reference-fitted F-score as independently calibrated.

## Post hoc Berkeley camera-oracle result

The [camera-oracle script](../scripts/object_motion/ycb_camera_reference.py) has since extracted all 60 matching NP5 pose files plus calibration into a fresh, bounded output folder (61 HDF5 members, 201,304 uncompressed bytes; no depth, masks or Google mesh). It verifies the existing archive SHA-256, the official photo hashes against the frozen SfM producer's image hashes, and all three binary model hashes before evaluation. It reads the HDF5 numeric datasets through the local `h5dump` executable with 17-significant-digit floating output, validates proper rigid transforms and the exact image-name set, and composes the matrices above. The [precision-preserving report](../build-opencv/object-motion/ycb-camera-reference-003/report.json) (SHA-256 `744b50dfcbf66c0801fa9f36c3a7eac124ba09f3053909c10e7c647d585a723e`) retains per-member hashes, per-camera residuals, alignment matrix, intrinsics, and the full comparison. The earlier default-decimal report `002` is retained for provenance; increasing dump precision changed center RMS by only 1.5×10⁻⁸ native units and rotation p95 by 0.00024°.

The image-only foreground SfM model registered 60/60 selected photos. A proper named-center Sim(3) fit from its arbitrary SfM frame into the Berkeley turntable frame gives center RMS 0.00872, p95 0.01287, leave-one-out RMS 0.00903 and leave-one-out p95 0.01336 in the supplied HDF5 translation units. The supplied virtual camera centers form a roughly 0.488-radius orbit in those units. The fit scale is 0.131115 Berkeley units per SfM unit. After the same world rotation is applied, per-view orientation error is median 2.48° and p95 2.99°. This is strong *trajectory agreement* in an evaluation-only, fitted frame; it does not test the Google mesh, nor remove possible systematic error in the supplied rig metadata or in the frame interpretation. The 60-view orbit is nearly planar, so its full 3-D similarity constraint is less informative than a diverse camera path. The `board_frame_offset` dataset has not yet been independently verified against a chessboard/depth observation and is not inserted into the above frame composition.

The recovered shared `SIMPLE_RADIAL` focal length is 1077.58 px; supplied `/NP3_rgb_K` gives fx/fy 1081.1501/1081.1438 px (differences −3.5663/−3.5601 px). The recovered principal point was fixed at the image center (640, 512) px, whereas the supplied calibration is (621.3445, 480.5832) px; those differences are +18.6555 and +31.4168 px. The recovered one-parameter distortion and supplied five-parameter distortion are **not directly comparable**. The camera diagnostic must not be described as exact calibration recovery or physical mesh accuracy.

Run only after the image-only producer has been sealed, using the already downloaded archive and existing local Python environment:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.object_motion.test_ycb_camera_reference -v
.local-tools/colmap-sparse/venv/bin/python -m scripts.object_motion.ycb_camera_reference \
  --output build-opencv/object-motion/ycb-camera-reference-NEW
```

The output path must be fresh. This post hoc camera-oracle lane is deliberately separate from SfM/MVS inputs and from the Google 64k surface reference. Native HDF5 translation units are reported as given; the current audit has not separately proved a unit declaration for these exact matrices.
