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
