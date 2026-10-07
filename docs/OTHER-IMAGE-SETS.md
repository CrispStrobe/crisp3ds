# Other image sets

The seven 3DLF objects share one light-field camera with soft photos, one
light backdrop, one turntable and one matte grey print. This page records how
the native pipeline (`crates/dense`) does on sets that differ in camera, rig,
material and capture path, scored against each set's ground truth. Ground
truth is used for scoring only. Evaluators: `scripts/turntable_mesh/scan_evaluate.py`
(shape-only similarity fit, F1 at 0.5 / 1 / 2 % of the reference diagonal;
`--no-platform` for complete object meshes) and
`scripts/turntable_mesh/dtu_evaluate.py` (the DTU protocol: accuracy and
completeness in millimetres inside the observability mask, above the table
plane, distances under 20 mm).

## Candidate sets

| Set | License (as stated by its authors) | Ground truth | Capture | Size | Our camera providers |
| --- | --- | --- | --- | --- | --- |
| DTU MVS, masks and cameras of the IDR preprocessing | research use; not redistributable | structured-light points, observability masks | robot arm, 49 or 64 views on part of a sphere, 1600 x 1200, rectified | IDR subset 2.1 GB, points 7 GB, masks 6.9 GB | `colmap`; `import` of supplied poses (dense stages only); not `turntable` |
| YCB / BigBIRD Berkeley RGB-D | YCB CC BY 4.0 | Google scanner meshes (16k, 64k) | five cameras over a turntable, 3 degree steps, 1280 x 1024, light booth, glass turntable carrying a chessboard | 0.6 to 1 GB per object | `turntable`, `colmap`; lens with tangential terms |
| Google Scanned Objects | CC BY 4.0 | the meshes themselves | none: photos would be rendered by us | about 1000 objects | any, on rendered photos |
| BlendedMVS | CC BY 4.0 | textured meshes, rendered depth | aerial and object orbits, 768 x 576 renders blended with photos | low-resolution set 27.5 GB | `colmap`, `import` |
| OmniObject3D | CC BY-NC-SA 4.0 | scanned meshes | phone video orbits with masks and COLMAP poses; Blender renders | large, behind a login | `colmap`, `import` |
| Tanks and Temples (training) | CC BY 4.0 | laser scans | video around large objects and scenes, no masks | 1 to 5 GB per scene | `colmap`, `import`; masks missing |
| ETH3D | CC BY-NC-SA 4.0 | laser scans | scenes, not single objects (`.local-tools/test-data/eth3d-*`) | small | not object-centric |
| CO3D | CC BY-NC 4.0 | none (COLMAP point clouds) | phone orbits with masks | 1.4 TB in all | no geometric score possible |
| MVImgNet | research use | none | phone orbits | 6.5 M frames | no geometric score possible |

Run here: DTU (four scans) and YCB (three objects). Downloads 16.5 GB on the
VPS (`/mnt/storage/datasets/eval-e/`), of which about 1.6 GB came to the Mac.

## Results

Times are wall clock for the whole command on the M1, while the machine was
shared with other jobs (load average 4 to 80), so they are rough.

| Set | Object | Photos | Masks | Cameras | Score | Time | What the sheet shows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DTU | scan 65, skull | 49 | IDR (import) | DTU poses (dense only) | overall 1.48 mm (acc 1.94, comp 1.02); F1 1 mm 0.69 | 153 s | face and teeth right; the unseen back of the hull stays as a slab; ripples on the white dome |
| DTU | scan 65 | 49 | IDR (import) | `colmap` (PyCOLMAP stand-in), ring gates loosened | overall 1.30 mm (acc 1.60, comp 1.00); F1 1 mm 0.67 | 193 s | as above; aligned to DTU by camera centres (residual 0.73 mm) |
| DTU | scan 65 | 17 (every 3rd) | IDR (import) | DTU poses | overall 2.81 mm (acc 3.97, comp 1.66) | 46 s | coarser, more hull left |
| DTU | scan 63, fruit (coloured, glossy) | 49 | IDR (import) | DTU poses | overall 1.19 mm (acc 1.89, comp 0.48); F1 1 mm 0.78 | 106 s | apples, pear and pumpkin clean; base debris |
| DTU | scan 63 | 49 | IDR (import) | `colmap`, ring gates loosened | overall 1.48 mm (acc 2.04, comp 0.92); F1 1 mm 0.57 | 269 s | as above |
| DTU | scan 110, golden rabbit (shiny metal) | 64 | IDR (import) | DTU poses | overall 1.83 mm (acc 2.75, comp 0.91); F1 1 mm 0.60 | 150 s | specular parts noisy |
| DTU | scan 24, house on a table | 49 | IDR (import) | DTU poses | overall 3.97 mm | 193 s | fails: the IDR masks cover 55 to 83 % of the frame, the hull fills the box |
| YCB | cracker box (printed, flat faces) | 60 (6 deg) | dataset (import), undistorted | `turntable` | F1 0.24 / 0.48 / 0.69 | 57 s | box right, a hull cone under it |
| YCB | cracker box | 60 | dataset (import) | `turntable`, `--set support_evidence=false` | F1 0.45 / 0.67 / 0.81 | 52 s | flat base; faces bulge where the print is plain |
| YCB | power drill (glossy red and black) | 60 | dataset (import) | `turntable`, support flattened | F1 0.33 / 0.52 / 0.76 | 56 s | blobby surface on glossy plastic; small in the frame |
| YCB | power drill | 60 | dataset, photos not undistorted, p1 p2 dropped | `turntable` | F1 0.18 / 0.37 / 0.64 (cone kept) | 62 s | about the same as undistorted with the cone (0.19 / 0.38 / 0.64): the tangential term is not what limits it |
| YCB | power drill | 60 | `threshold` | `turntable` | refused by the gates | 9 s | threshold takes the chessboard on the turntable as object |
| YCB | mustard bottle (smooth, nearly symmetric) | 60 | dataset (import) | `turntable`; `colmap` | refused by the gates, both | 7 s, 53 s | too few features on the object (median 241) for either solver |

All runs where the gates passed registered every photo. Where a camera solution
was wrong, the ring and audit gates refused it; no wrong solution reached the
dense stages.

## What breaks, and what would fix it

- **Support under cameras that look down** (YCB, all objects). With every view
  above the object, the hull continues below the support as a cone as deep as
  `tan(elevation)` times the footprint radius; `support_evidence` takes a cone
  deeper than 15 % of the object's height for a floating object and keeps it.
  `--set support_evidence=false` flattens the base and doubles the cracker
  box's F1 at 0.5 %. Fix (dense stages, `fusion.rs`): compare the depth below
  the support with the cone the camera elevation explains, instead of with a
  fixed share of the height.
- **Captures that do not surround the object** (DTU). The unseen back of the
  hull remains as a slab inside the observability mask, which costs accuracy
  (accuracy is 1.6 to 2.8 mm against completeness 0.5 to 1.0 mm). Fix (dense
  stages): leave hull surface that no view measures out of the mesh when the
  views cover less than a full turn, or close it at the measured surface.
- **Non-ring camera paths**: `--capture orbit` keeps the camera audit and
  drops the ring gates (`turntable`, the default, keeps them; the turntable
  provider refuses `orbit`). DTU 65 and 63 with `colmap` and no other flag:
  49 of 49 registered, overall 1.31 and 1.43 mm.
- **Masks that are not the object** (DTU scan 24, where the masks cover most of
  the frame; YCB with `threshold`, which takes the turntable's chessboard).
  Imported masks are needed there.
- **Few features on the object** (YCB mustard). Both solvers fail when
  features are taken inside the mask only. The turntable's chessboard turns
  with the object and would carry the motion, but features over the whole
  photo add the static background, and the order check then rightly refuses
  (neighbours and photos four apart share equally many matches). Tried and
  not kept: a lower feature contrast threshold, and steps fixed to 360 degrees
  over the photos (the turn closed, but the adjustment did not hold the ring).
  Tried next and not kept: features also in a band beside and below the
  mask, keeping off-mask matches only if they move between photos (the
  background stands still), and locating the scene by points inside the
  masks. The bottle then registers (60 of 60, 1 114 features per photo, the
  order check and every gate pass), but the cameras are wrong: the steps
  came out 19 % too large before closing the turn, and the mesh is a hollow
  shell (F1 0.21 / 0.40 / 0.59). The 0.6 to 1.4 turn window for closing a
  turn let that through; it is now 0.75 to 1.15 and refuses such a solution
  (1.19 turns). The axis search (tilts up to 50 degrees) is at its edge for
  cameras looking down this steeply; a wider search, checked against YCB's
  supplied poses, is open.
- **Lens with tangential terms** (YCB, p1 = 0.006). Photos undistorted with the
  full model beforehand gave the same score as dropping p1 and p2, so the
  missing tangential term is not what limits these runs; the calibration format
  (`radialk3`) cannot carry it.
- **Glossy, dark or plain surfaces** (drill, rabbit, skull dome) give noisy
  depth, as expected for photo-consistency matching.
