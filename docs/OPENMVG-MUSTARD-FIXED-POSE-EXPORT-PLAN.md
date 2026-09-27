# Fixed-intrinsic mustard pose JSON export — review gate

This is a separate, one-shot diagnostic conversion of the sealed
`openmvg-mustard-photo-sfm-fixed-002/sparse/sfm_data.bin`. It does not rerun or
repair SfM. The fixed-intrinsic ablation retained 46/48 camera poses and failed
the full-registration gate (162 tracks, 1292 residuals); exporting named poses
does not change that assessment.

The worker pins the -002 receipt SHA-256
`5445b512ac6d171e8097134d9c86d84e6428b1d958a3088593d6234e5bb1fa24`,
model SHA-256
`4147cd6ea9e0848dd97210701535371c100425125dc83df07a67482ebd8eca55`,
and SfM log SHA-256
`7eaab977dddc6f1dc748fffc1d1c0affebf150bd762f56e333fdf9308898513d`.
It checks the -002 receipt's full output inventory and fixed `-f NONE` stage,
then verifies the existing sixth-target converter build manifest, license
receipt, log, executable and dependency closure through the prior sealed gate.
The converter receives only the fixed model (`-i`), a new JSON output (`-o`),
and `-V -I -E`; no reference poses/calibration/mesh, scanner, board corners,
masks or held-out photos are inputs. Export validation reuses the prior
48-name/46-finite-pose schema gate.

Default `python3 -m scripts.classical_backend.openmvg_mustard_fixed_pose_export`
is a read-only preflight. The explicit `--convert` entry point is not approved
by this plan; it requires separate review. It writes only under fresh external
`openmvg-mustard-photo-sfm-fixed-pose-export-003`, never overwriting -001 or
-002. The 11 GiB external reserve plus full 64 MiB output reservation and
11 GiB internal reserve are checked before and during conversion. Limits are
1 GiB observed process-tree RSS, 120 s wall/CPU, 4 MiB log and 64 MiB total
output; one thread is requested. The receipt retains command, input/build
seals, exit/status, log/JSON hashes, size, pose-name integrity and partial
output on failure. It is evaluation-only; no GPL/AGPL shipping clearance or
new geometry-quality claim follows from a successful export.

An ID-bound orbit check using `openmvg_mustard_json_pose_validator.parse_export`
is deferred until a real -003 receipt and JSON exist and their exact hashes are
reviewed and pinned separately; this worker's 46-pose structural check is not
that validation.
