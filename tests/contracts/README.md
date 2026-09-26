# Shared project fixtures

All three JSON files are valid v1 manifests. `calibrated-project.json` contains invented calibration values solely for validating the serialization contract; no images or reconstruction artifacts accompany it. Do not use it as a real camera calibration or measured scan.

`pose-ready-project.json` adds the optional R01 metadata: OpenCV radial-tangential distortion, ArUco DICT_4X4_50, and four synthetic marker squares on z=0. It records pose inputs only; no pose estimate has run. Decoded marker corners are TL, TR, BR, BL. In the fixture's printed layout, board x points right and y down, and the origin is at the grid center.

The web and native validators should agree on these fixtures and reject mutations including unsupported versions, duplicate image IDs, non-relative/traversing paths, nonpositive focal lengths, noninteger image dimensions, a stationary marker board, and unknown stage names/states. Validation checks manifest structure, not image existence or reconstruction accuracy.

Do not use duplicate JSON object keys. The native parser rejects them; the browser currently uses standard `JSON.parse`, which retains the last value. Browser import therefore is not an authoritative native validation step. A future worker must always validate the serialized project at its own boundary.
