# Photo-only regression checks

Run cases serially through masks, recovered turntable cameras, stereo and mesh.
With `--texture`, export GLB from each resulting STL and verify every triangle
coordinate and winding byte. Reference meshes, supplied poses, dataset masks
and dataset depths must not be reconstruction inputs. The caller must verify
photo provenance; the runner cannot establish it from a directory name.

```sh
python scripts/photo_regression/run.py --manifest cases.json \
  --binary /path/to/crisp3ds-dense --output /path/to/new-results --texture
```

Example local manifest (paths resolve beside the manifest):

```json
{"cases": [
  {"id": "light-object", "photos": "capture/photos", "calibration": "capture/lens.json",
   "masks": "background", "require_closed": true,
   "minimum_textured_area_fraction": 0.6,
   "attribution": "The capture's required attribution"}
]}
```

`masks` may be `threshold`, experimental `background`, or native `sam` (requires
an appropriate build/model/runtime). Optional `photo_options` contains CLI
arguments such as `--turntable-region=surface`; data/provider overrides and
imported match caches are refused. Surface recovery is experimental, requires
one roughly uniform full turn with moving planar support features, and has
not yet produced acceptable mustard geometry.

Results retain logs, reports, masks and meshes. The free-space threshold defaults
to 9 GiB; reconstruction stops if crossed. An open mesh fails unless the case
explicitly sets `require_closed: false`, for example for unobserved undersides.
A PASS means execution, closure (when requested), texture coverage and geometry
preservation passed. It does **not** mean the shape is accurate. Render and
review untextured meshes and score against an independent reference separately.
Texture coverage also does not establish colour accuracy.

Run the geometry-check test with:

```sh
python -m unittest discover -s scripts/photo_regression -p 'test_*.py'
```

## Camera observations reserved before recovery

`camera_holdout.py partition --matches photo-matches.json --output split`
removes entire connected match components, including ambiguous ones. Recover
cameras using `split/training-matches.json`, then run:

```sh
python scripts/photo_regression/camera_holdout.py evaluate \
  --matches photo-matches.json --reserved split/reserved-tracks.json \
  --cameras recovered/cameras.json --calibration lens.json --output scores
```

The evaluator triangulates alternating observations and scores the others.
It reports all reserved observations and a subset selected using only the
fitting observations (at least three, each below 1 px). Bad correspondences
remain visible in the full report. This is a camera diagnostic, not a dense
accuracy gate. Matches must use the original calibrated photo coordinates;
calibration is the engine's `radialk3` model. Do not fit with the original cache.

Add `--foreground-masks` to `evaluate` to report errors separately inside and
outside the recovered cameras' photo-derived masks. The masks must match each
camera's image dimensions; normalized feature coordinates are mapped using
that camera's intrinsics. These groups do not alter triangulation, track
selection or the overall scores. Report group counts: a small number of good
object observations cannot establish accuracy everywhere, and incorrect masks
can misclassify observations. Points beyond the mask bounds form a separate
group rather than being counted as background.

## Local relief through the stages

`detail_profiles.py --spec regions.json --output new-folder` compares raw,
filtered and merged depth maps and meshes with an independently aligned scanner.
Scanner geometry is read only by this evaluation script. It never writes
reconstruction inputs. NumPy/SciPy are required; plots also need Matplotlib
(or pass `--no-plots`). A specification contains:

```json
{"cameras":"own-inputs/cameras.json", "evaluation_reference_stl":"aligned-scan.stl",
 "regions":[{"name":"eye","view":"view_10","crop":[1048,505,90,90]}],
 "depths":{"raw":{"archive":"stereo/depths-pass-0-raw.npz",
                     "metadata":"stereo/depths-stage-cameras.json"}},
 "meshes":{"final":"mesh/mesh.stl"}}
```

Select and inspect boxes on the source photographs first. Crops are original
photo coordinates, mapped through each stage's actual intrinsics. Reports
include coverage, ray-depth error, inverse-depth relief after plane detrending,
and Gaussian high-pass relief at 2/4/8 px. Each stage has its own common-ray
statistics and a second comparison restricted to the same rays across **all**
stages. Filtered depths equal raw values where retained, so on the fixed
intersection they can have identical errors while coverage falls. Report both.
Alignment error, crop boundaries and missing data can affect these measures;
a gain or correlation is not proof that an eye or nose has been recovered.
