# Photos to dense inputs

`python -m scripts.turntable_mesh.photos_to_inputs` takes a folder of turntable
photos and a lens file and produces everything the dense stage
(`dense_pipeline.py --inputs`) starts from: object masks, recovered cameras,
undistorted photos and the `inputs/` directory. It replaces the per-object
drivers that were written by hand for the Dragon, the Armadillo and the Bunny.

```sh
export CRISP3DS_ALICEVISION=/path/to/av.py            # or an install prefix
export CRISP3DS_ALICEVISION_DENSE=/path/to/av-dense.py
export CRISP3DS_SAM_PYTHON=/path/to/sam-venv/bin/python
export CRISP3DS_SAM_SOURCE=/path/to/sam2-checkout
export CRISP3DS_SAM_CHECKPOINT=/path/to/sam2.1_hiera_tiny.pt

python -m scripts.turntable_mesh.photos_to_inputs \
  --photos data/bunny/rgb \
  --calibration scripts/turntable_mesh/calibrations/3dlf-pro.json \
  --output runs/bunny-front
python -m scripts.turntable_mesh.dense_pipeline --inputs runs/bunny-front/inputs --output runs/bunny-dense
```

Exit code 0 means the cameras passed every gate and `inputs/` is ready. Exit
code 2 means cameras were recovered but rejected by a gate; nothing is handed
on. Exit code 1 means a step failed (tool error, deadline, disk, cancel).
`frontend.json` has the reasons in every case.

## What it assumes about the capture

- One turn of a turntable (or of the camera around the object), photographed
  by **one camera with a fixed lens and fixed focus**, 3 to 96 photos, all the
  same size. The file order (natural order: `x_2` before `x_10`) is the capture
  order.
- A **dark object on a light backdrop**. The coarse masks are a grey-level
  threshold and the hole cleanup only fills dark holes; a light object on a
  dark backdrop is not supported.
- A **known lens**: a radial k1, k2, k3 model. The lens is locked during camera
  recovery; it is never refined. Earlier free-lens solutions on these sets
  folded the radial model while still showing low residuals.
- The object stays inside the frame and is the largest dark region in it (or
  inside `--envelope`).
- Nothing else is used: no scanner mesh, no supplied poses, no depth.

## What must be installed

Nothing is downloaded or installed by the command.

| Needed | Used for | License |
| --- | --- | --- |
| AliceVision build with `cameraInit`, `featureExtraction`, `imageMatching`, `featureMatching`, `globalSfM` and `prepareDenseScene` (see `docs/ALICEVISION-LOCAL-INSTALL.md`) | camera recovery, undistortion | MPL-2.0 (parts derived from libmv are MIT); its dependencies carry their own licenses |
| SAM 2 source checkout and a SAM 2.1 checkpoint (tested: `sam2.1_hiera_tiny.pt`) | segmentation | code and checkpoints Apache-2.0 |
| An interpreter with Torch and the SAM 2 dependencies (hydra-core, omegaconf with `antlr4`, `iopath`), plus NumPy, SciPy, Pillow | runs `scripts/turntable_mesh/segment.py` | |
| An interpreter with NumPy, SciPy, OpenCV and Pillow | every other step | |

AliceVision may be given as a wrapper script (called as `wrapper TOOL args`; a
`.py` wrapper is run with `--python`) or as an install prefix containing
`bin/aliceVision_TOOL`, in which case `ALICEVISION_ROOT` and the library path
are set for it. If the sparse tools and `prepareDenseScene` live in two builds,
give the second one with `--alicevision-dense`. If the SAM interpreter lacks
some pure-Python dependency, add directories with `--sam-pythonpath`.

## Calibration file

```json
{
  "schema": "crisp3ds_lens_calibration_v1",
  "model": "radialk3",
  "calibration_width": 583,
  "calibration_height": 385,
  "fx": 776.09, "fy": 776.62, "cx": 291.05, "cy": 184.91,
  "k1": -0.1718, "k2": 0.4123, "k3": -3.7170,
  "principal_point_convention": "pixel_centre",
  "sensor_width_mm": 36.0,
  "provenance": {"free": "form"}
}
```

- `fx`, `fy`, `cx`, `cy` are pixels **at the calibration resolution**. The
  centre of the top-left pixel is (0, 0) (the OpenCV convention).
- The distortion is `x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6)` on normalised
  coordinates. A Brown model with zero tangential terms is the same thing.
- If the photos have another size, the model is scaled: focal lengths by the
  size ratio, the principal point as `(c + 0.5) * s - 0.5`, coefficients
  unchanged. The photos must be a resize of the calibration frame; a different
  aspect ratio (a crop) is refused.
- `sensor_width_mm` is optional and nominal. It only fixes the unit of the
  focal length written into the AliceVision scene.
- `provenance` is not read. Say where the numbers come from.
- The raw 3DLF `calib_pro/intrinsics/rgb_optic.json` (`"model": "brown5"` with
  a nested `distortion`, `p1 = p2 = 0`) is accepted as it is.

`scripts/turntable_mesh/calibrations/3dlf-pro.json` is the 3DLF lens in this
format. It was stated for 583x385 while the photos are 1749x1155; scaling it by
three is an assumption that nobody has verified independently.

## Steps

| Stage | Step | Tool |
| --- | --- | --- |
| `masks` | copy photos as `capture_NNNN.png`, coarse dark-object masks | this module |
| | segmentation prompted by the coarse masks | `segment.py` (SAM 2.1) |
| | fill small dark holes | `silhouette_cleanup.py` |
| | masks, statistics, contact sheet | this module |
| `cameras` | gamma + CLAHE contrast images (features only) | this module |
| | list views, SIFT features inside the masks | `cameraInit`, `featureExtraction` |
| | apply the declared lens, locked | `cameraInit` |
| | pairs, matching | `imageMatching`, `featureMatching` |
| | cameras | `globalSfM --lockAllIntrinsics true` |
| | audit and ring gates | `alicevision_cameras.audit_scene`, this module |
| | undistorted photos | `prepareDenseScene` |
| | camera table and undistorted masks | `dense_all_views_inputs.py` |
| | sparse points drawn on photos | this module |

Every step is its own process group with a deadline; the free-space floor is
checked before each one. Creating a file named `cancel` next to the event log
stops the run. Photo copies, contrast images, features and matches are deleted
at the end unless `--keep-intermediates` is given. After that `sfm/final.sfm`
still names the deleted contrast images as its view paths; later stages read
`native-prepared/` and `inputs/` and only use the base names.

The undistorted photos in `native-prepared/` are the contrast images, as in the
hand-driven runs. The dense stage normalises grey levels inside the mask.

## Output

```
<output>/
  inputs/                cameras.json, masks/, sparse_points.npy
  masks/                 capture_NNNN.png, 0/255, original (distorted) frame
  sfm/final.sfm          cameras and sparse points
  sfm/camera-audit.json  reprojection per view, lens check
  sfm/ring-sanity.json   ring numbers
  sfm/gates.json         limits, numbers, pass/fail, reasons
  native-prepared/       <viewId>.png undistorted photos
  frontend.json          configuration, every command with exit code and time, gates, warnings, status
  frontend-config.json   the resolved options
  photo-map.json         capture name <- original file name, coarse mask numbers
  masks-report.json      mask area per photo, pixels SAM added to or dropped from the coarse region
  mask-contact-sheet.png sparse-overlay.png  events.jsonl  logs/NN-step.log
  work/                  coarse masks, SAM output, cleanup report (small)
```

## Events

With `--events PATH` the stage appends to that log (default
`<output>/events.jsonl`). It writes two stages, `masks` then `cameras`, each
with `stage_started`, `progress` and `stage_finished`. It does not write
`run_started` or `run_finished`; those belong to whatever drives the run.

| Event | Stage | Content |
| --- | --- | --- |
| `metric` | `masks` | `mask_area_median_pixels`, `mask_area_median_fraction` |
| `artifact` `mask_sheet` | `masks` | `mask-contact-sheet.png`: twelve photos, mask tinted green, dark pixels outside the mask in red |
| `metric` | `cameras` | `registered_views`, `input_photos`, `reprojection_median_pixels`, `reprojection_p95_pixels`, `ring_radius_spread_percent`, `ring_largest_gap_deg` |
| `artifact` `sparse_overlay` | `cameras` | `sparse-overlay.png`: sparse points on four undistorted photos with their masks |
| `artifact` `report` | `cameras` | `frontend.json` |
| `error` | either | a failed step, or `cameras rejected: <reasons>` |

`mask_sheet` and `sparse_overlay` are new artifact kinds and `masks` and
`cameras` are new stage names. They are not yet in `dense_events.ARTIFACT_KINDS`
/ `STAGES` or in `docs/ENGINE-CONTRACT.md`; the events are emitted with
`EventLog.emit` directly until they are.

## Gates

| Gate | Option | Default |
| --- | --- | --- |
| registered share of the photos | `--minimum-registered-fraction` | 0.8 |
| sparse observations in every registered view | `--minimum-observations-per-view` | 20 |
| reprojection p95 in every view, pixels | `--maximum-view-reprojection-p95` | 4 |
| lens unchanged and still locked, radial model monotonic over the frame | (always) | |
| share of sparse observations in front of their camera | `--minimum-positive-depth-fraction` | 0.999 (1.0 would let a single stray point reject the cameras) |
| camera distance from the ring axis, max minus min, % of radius | `--maximum-radius-spread-percent` | 5 |
| camera distance from the ring plane, % of radius | `--maximum-out-of-plane-percent` | 5 |
| largest angle between neighbouring cameras | `--maximum-angular-gap-deg` | 30 (use 360 to accept a partial turn) |
| capture steps running against the turning direction | `--maximum-reversed-steps` | 0 |
| optical axis passes the ring axis within, % of radius, and points towards it | `--maximum-optical-axis-miss-percent` | 25 |

The Dragon, Armadillo and Bunny rigs measure about 0.5% radius spread, under 1%
out of plane and gaps of about 6 degrees, so the ring limits are loose: they
catch a broken solution, not a slightly inaccurate one. The gates certify that
the cameras are usable for photo reconstruction; they say nothing about the
accuracy of the shape.

## Options

Tool locations (flag, then environment variable): `--python` /
`CRISP3DS_PYTHON`, `--alicevision` / `CRISP3DS_ALICEVISION`,
`--alicevision-dense` / `CRISP3DS_ALICEVISION_DENSE`, `--sensor-database` /
`CRISP3DS_ALICEVISION_SENSOR_DB`, `--sam-python` / `CRISP3DS_SAM_PYTHON`,
`--sam-source` / `CRISP3DS_SAM_SOURCE`, `--sam-checkpoint` /
`CRISP3DS_SAM_CHECKPOINT`, `--sam-config` / `CRISP3DS_SAM_CONFIG`,
`--sam-pythonpath` / `CRISP3DS_SAM_PYTHONPATH`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `mps` | SAM device. `segment.py` currently accepts `mps` and `cpu` only; `cuda` is refused until it does |
| `--threads` | 2 | CPU threads for every step |
| `--minimum-free-gib` | 10 | free disk required before every step |
| `--alicevision-memory-gib` | 4 | memory hint passed to the AliceVision tools |
| `--keep-intermediates` | off | keep photo copies, contrast images, features, matches |
| `--envelope` | `auto` | where the object can be: whole frame, or `x0,y0,x1,y1` in pixels (fractions if all are at most 1) |
| `--dark-threshold` | 70 | grey level below which a pixel counts as object, or `otsu` per photo |
| `--sam-multimask`, `--sam-preserve-holes`, `--sam-automatic-cues` | on | the `segment.py` flags; turn off with `--no-sam-...` |
| `--hole-cleanup-budget` | 0.02 | largest share of the foreground the dark-hole fill may add |
| `--contrast-gamma` | 0.5 | gamma of the feature images; 1 turns it off |
| `--clahe-clip`, `--clahe-grid` | 2.0, 8 | CLAHE on lightness; clip 0 turns it off |
| `--initial-field-of-view` | 45 | placeholder lens before the declared one is applied |
| `--describer-types`, `--describer-preset` | `sift`, `normal` | feature extraction |
| `--matching-method` | `Exhaustive` | `imageMatching` method |
| `--random-seed` | 0 | matching and SfM |
| `--sfm-option ARG` | none | extra token appended to `globalSfM`; repeat per token |
| `--sam-timeout`, `--features-timeout`, `--matching-timeout` | 1800 s | deadlines |
| `--sfm-timeout`, `--prepare-timeout`, `--small-step-timeout` | 900, 600, 600 s | deadlines |

The deadlines are sized for a 16 GB M1 that is also running a GPU job. On an
idle machine the 73-photo sets take about 1 min for SAM, 2 to 3 min for
features, 3 to 7 min for matching and 3 to 5 min for global SfM.

## Known failure modes

- **Camera recovery is not repeatable.** Three Bunny runs on identical photos
  and bit-identical masks gave three different camera sets: 8,463 to 9,087
  sparse points, per-view rotations differing by 0.3 to 0.9 degrees in the
  median (up to 1.8) after aligning the frames, and camera centres by 0.4 to
  2% of the ring radius. Running `globalSfM` alone a second time on
  byte-identical features and matches, same seed and thread count, moved the
  cameras just as much (median 0.7, up to 1.5 degrees), so the variation comes
  from the SfM step itself. The difference is a smooth once-per-turn drift in
  the viewing direction, not noise between neighbours (step sizes agree to
  about 0.03 degrees). Every solution has a 0.41 to 0.42 px median
  reprojection error, so the gates cannot tell them apart. Whether a single
  thread makes the step repeatable has not been tried.
- **One stray sparse point could reject good cameras.** In one of those three
  runs a single observation out of 42,382 lay behind its camera, and a limit of
  1.0 for `--minimum-positive-depth-fraction` (the policy the Dragon rig was
  accepted under) rejected the run. The default is therefore 0.999; pass 1.0
  for the strict policy.
- **Thin parts dropped by SAM.** The Armadillo's hands, the Bunny's far ear and
  the Dragon's horns are missing from the masks in runs of consecutive photos
  (up to about 7% of the figure). `frontend.json` warns when more than 3% of
  the coarse dark region is dropped, and the contact sheet shows it in red. The
  dense stage has a multi-view mask repair for this; the front stage does not
  merge the coarse mask back in, because the coarse mask also contains the
  contact shadow under the object.
- **Duplicate frames.** If the turntable did not move between two photos they
  get the same pose (Bunny photos 13 and 14). This is reported as a warning
  with the photo pairs; it is not a failure, but the dense stage then has a
  zero-baseline neighbour.
- **Uneven steps.** Recovered steps scatter by about half a degree around the
  nominal five on the 3DLF sets. Whether that is the turntable or pose error is
  not known; no gate tests uniformity.
- **Sense of rotation against the dataset's own poses.** On the Dragon, the
  Armadillo and the Bunny the recovered orbit turns the opposite way to the
  3DLF `poses_metric.json` (depth-derived) poses, with equal magnitudes. That
  points at a handedness difference between the photos and the dataset's depth
  and scan frame. Nothing here depends on those poses, but a comparison against
  the scanner mesh must allow for a mirrored frame.
- **Dark things other than the object.** With `--envelope auto` the largest
  dark region of the whole frame is taken as the object. A dark frame edge, a
  dark turntable or a deep shadow larger than the object will win; give an
  explicit `--envelope` or a lower `--dark-threshold`. A warning is recorded
  when the region touches the frame or the envelope.
- **More than 96 photos** are refused: the camera audit is bounded there.
- **No physical scale.** The reconstruction frame has an arbitrary scale and up
  direction.
