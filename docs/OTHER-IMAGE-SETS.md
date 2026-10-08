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
| Google Scanned Objects | CC BY 4.0 | the meshes themselves | none: photos rendered by `crisp3ds-dense render` | about 1000 objects, 5 to 15 MB each | any, on rendered photos |
| BlendedMVS | CC BY 4.0 | textured meshes, rendered depth | aerial and object orbits, 768 x 576 renders blended with photos | low-resolution set 27.5 GB | `colmap`, `import` |
| OmniObject3D | CC BY-NC-SA 4.0 | scanned meshes | phone video orbits with masks and COLMAP poses; Blender renders | large, behind a login | `colmap`, `import` |
| Tanks and Temples (training) | CC BY 4.0 | laser scans | video around large objects and scenes, no masks | 1 to 5 GB per scene | `colmap`, `import`; masks missing |
| ETH3D | CC BY-NC-SA 4.0 | laser scans | scenes, not single objects (`.local-tools/test-data/eth3d-*`) | small | not object-centric |
| CO3D | CC BY-NC 4.0 | none (COLMAP point clouds) | phone orbits with masks | 1.4 TB in all | no geometric score possible |
| MVImgNet | research use | none | phone orbits | 6.5 M frames | no geometric score possible |

Run here: DTU (four scans), YCB (three objects) and six Google Scanned
Objects rendered as turntable captures (next section). Downloads 16.5 GB on the
VPS (`/mnt/storage/datasets/eval-e/`), of which about 1.6 GB came to the Mac;
the six GSO models are 70 MB.

## Results

Times are wall clock for the whole command on the M1, while the machine was
shared with other jobs (load average 4 to 80), so they are rough.

| Set | Object | Photos | Masks | Cameras | Score | Time | What the sheet shows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DTU | scan 65, skull | 49 | IDR (import) | DTU poses (dense only) | overall 1.48 mm (acc 1.94, comp 1.02); F1 1 mm 0.69 | 153 s | face and teeth right; the unseen back of the hull stays as a slab (before `mesh_open_unseen`, which gives overall 1.24 mm); ripples on the white dome |
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

## Google Scanned Objects, rendered

Six GSO models (Copyright 2020 Google LLC, CC BY 4.0, as each model's
`model.config` and `metadata.pbtxt` state; Downs et al., ICRA 2022,
<https://arxiv.org/abs/2204.11918>; downloaded from Gazebo Fuel, owner
GoogleResearch) rendered by `crisp3ds-dense render` as a 3DLF-like capture: 72
views on one ring at 20 degrees elevation (5 degree steps), 1749 x 1155, the
3DLF lens (`calibration/3dlf-pro.json`) with its distortion, 4 samples per
pixel, one light that turns with the camera and casts a hard shadow, 1 px
Gaussian blur, noise sigma 0.01, light disc and backdrop. Then the default
command from the photos (`threshold` masks, `turntable` cameras), scored with
`scan_evaluate --no-platform` against the mesh as placed (`reference.ply`).
The renderer also knows the true cameras: solved camera centres are fitted to
them by a proper similarity and compared (centre error in % of the orbit
radius, orientation error after the fit). Every scored run chose the proper
handedness, and orientation errors of 0.06 to 0.21 degrees under a proper fit
confirm no solution was a mirror image. "Exact masks" replaces the threshold
masks with the renderer's silhouettes (`--masks import:DIR`) to separate the
masks from the rest. Times: M1 shared with other jobs (load 6 to 19), four
threads. Scripts: `scripts/gso_eval/` (`pipeline.sh` per case;
`compare_sheet.py` draws four views of photo, exact mesh and reconstruction
coloured by its distance to the mesh, green 0, yellow 1 %, red 2 % of the
diagonal; `camera_error.py`; `summarize.py`).

| Object | Kind | Masks | Registered | Cameras: centre % / orientation deg | F1 0.5 / 1 / 2 % | Run | What the sheet shows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Schleich African Black Rhino | dark figurine, thin legs and horns | threshold | 72 / 72 | 0.13 / 0.08 | **0.867 / 0.959 / 0.990** | 161 s | clean; horns, ears and legs right; a little shadow debris between the legs |
| Vans Honey Nut Crunch box | printed box, flat faces | threshold | 72 / 72 | 0.10 / 0.06 | 0.833 / 0.890 / 0.969 | 242 s | faces flat and placed right; grooves along horizontal print lines, dents in the plain yellow band at the top |
| Nintendo Mario figure | colourful toy, white gloves | threshold | 72 / 72 | 0.25 / 0.21 | 0.710 / 0.832 / 0.907 | 160 s | body right; the white gloves and the lit nose are cut away |
| Nintendo Mario figure | | exact | 72 / 72 | 0.19 / 0.15 | 0.830 / 0.947 / 0.979 | 147 s | gloves and face whole; small errors in folds (eyes, straps) |
| Reebok Zig Cooperstown shoe | black and white shoe | threshold | 72 / 72 | 0.22 / 0.15 | 0.431 / 0.645 / 0.814 | 143 s | black upper right; the white logo stripes and white sole parts cut through; ankle opening capped |
| Reebok Zig Cooperstown shoe | | exact | 72 / 72 | 0.22 / 0.13 | 0.597 / 0.735 / 0.843 | 161 s | no holes; the inside of the shoe and the cleated sole, never seen, stay closed or flat |
| ACE coffee mug | patterned mug with a handle | threshold | 72 / 72 | 0.19 / 0.12 | 0.448 / 0.595 / 0.717 | 219 s | outside and handle (with its hole) right; the opening is capped at the rim, the inside not carved |
| Threshold bead cereal bowl | plain white, rotationally symmetric | threshold | refused | | | 31 s | refused by the closure gate (steps add up to 180 degrees); the threshold mask also takes the shadowed disc and drops the lit inside |

Render times were 93 to 161 s per capture of 72 views.

Where it breaks, with a diagnosis for each:

- **Light parts of an object with `threshold` masks** (Mario's gloves and
  nose, the shoe's white stripes and sole, the white bowl). The threshold
  provider assumes a dark object on a light backdrop; object pixels brighter
  than its level are left out of the mask and the hull carves them away,
  while the hard cast shadow on the disc is taken in (it is carved later,
  since it moves over the disc from photo to photo). With exact masks, Mario
  goes from 0.710 to 0.830 at 0.5 % and the shoe from 0.431 to 0.597: the
  masks cost most of the loss. A mask provider for light or mixed objects
  (colour against the backdrop, or SAM) is the fix; the cameras were right in
  both cases.
- **Concavities the ring never sees into** (the mug's inside, the shoe's
  inside, the shoe's cleated sole). From 20 degrees above, the inner far wall
  of the mug shows only near the rim and the bottom never shows; the result
  is capped at the opening, as the hull is. About half of the mug's reference
  surface is inside, which bounds completeness and so F1. A view from above
  would be needed; this is the capture, not the solver.
- **Print with edges along the ring's epipolar lines** (the box's horizontal
  text lines and panel edges). On one horizontal ring the epipolar lines are
  nearly horizontal, so a horizontal edge gives no disparity and the depth
  wanders along it (grooves up to about 2 % of the diagonal); the plain yellow
  band at the top has no texture at all and dents. A second ring at another
  elevation, or a smoothness prior on plain regions, would address it.
- **Symmetric and plain** (the bowl). A rotationally symmetric white object
  shows no rotation between photos; the solver found half a turn and the
  closure gate refused it, as it should. No wrong camera solution reached the
  dense stages in any of the seven runs.
- **Thin parts** (the rhino's horns, ears and legs, the mug's handle) came
  out right at this resolution when the masks are right; the rhino's thin
  tail is partly lost.

Robustness on the rhino (same render otherwise; default command):

| Change | Registered | Cameras: centre % / orientation deg | F1 0.5 / 1 / 2 % | Accuracy median / p90 % | What the sheet shows |
| --- | --- | --- | --- | --- | --- |
| sharp (blur 0) | 72 / 72 | 0.11 / 0.07 | **0.901 / 0.966 / 0.993** | 0.07 / 0.61 | as the baseline, smoother; skin folds still not resolved |
| baseline: 72 views, blur 1 px | 72 / 72 | 0.13 / 0.08 | 0.867 / 0.959 / 0.990 | 0.11 / 0.71 | clean |
| blur 1.5 px | 72 / 72 | 0.22 / 0.16 | 0.821 / 0.942 / 0.986 | 0.15 / 0.89 | softer surface |
| blur 3 px | 72 / 72 | 0.86 / 0.49 | 0.665 / 0.867 / 0.972 | 0.32 / 1.43 | shape right, surface lumpy; cameras four times worse |
| 36 views (10 degree steps) | 36 / 36 | 0.12 / 0.07 | 0.848 / 0.945 / 0.989 | 0.11 / 0.81 | as the baseline |
| 24 views (15 degree steps) | 24 / 24 | 0.17 / 0.09 | 0.708 / 0.815 / 0.879 | 0.22 / 4.44 | body right; slabs of cast shadow between and beside the legs |
| dark backdrop (grey 0.3) | 72 / 72 | 0.16 / 0.08 | 0.127 / 0.242 / 0.434 | 2.54 / 7.72 | a block: the threshold masks take the backdrop as object (83 % of the frame) |

What this answers:

- **Sharper photos.** Blur moves F1 at 0.5 % by about 0.03 to 0.05 per half
  pixel near 1 px and costs 0.2 at 3 px; within 1 % of the diagonal the loss
  stays small up to 1.5 px. If the 3DLF photos are about 1.5 px soft, a sharp
  camera would gain about 0.08 at 0.5 % on an object like this and little at
  1 % and 2 %. At 3 px the cameras themselves degrade (0.86 % of the radius).
- **Fewer views.** 36 views lose 0.02 at 0.5 %; at 24 views the cameras are
  still right (0.17 %, steps 15.01 degrees, raw turn 359.6 degrees), and the
  loss is the cast shadow: the threshold masks take it in, and 24 views do not
  carve the moving shadow away the way 72 do. With masks that leave the shadow
  out, 24 views would likely hold (not run).
- **Dark backdrop.** The cameras are unaffected; the threshold masks are not
  usable (they assume a light backdrop). The run went through with only the
  warning that the object region touches the frame in all 72 photos; a mask
  covering most of every frame should be refused, as the camera gates refuse a
  wrong solution. Import or SAM masks are needed for dark backdrops.

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
  hull remained as a slab inside the observability mask, which cost accuracy
  (accuracy 1.6 to 2.8 mm against completeness 0.5 to 1.0 mm). Fixed
  (`mesh_open_unseen`, on): when the views do not go around the object, hull
  surface without measured or extrapolated evidence is left out of the mesh,
  which is then open. Overall, before / after: scan 65 on DTU poses 1.43 /
  1.24 mm, scan 63 1.07 / 1.03 mm; with `--cameras colmap --capture orbit`
  1.34 / 1.30 and 1.39 / 1.31 mm. Closing the unseen part with a membrane
  instead scored between the two (numbers in `crates/dense/README.md`).
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
