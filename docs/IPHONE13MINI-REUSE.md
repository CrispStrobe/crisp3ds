# iPhone 13 mini: RGB-first capture and repository reuse

Reviewed 2026-09-27. Our available phone is an iPhone 13 mini, not a Pro.
Its [Apple specifications](https://support.apple.com/en-gb/111873) list rear
Main/Ultra Wide cameras and front TrueDepth, not rear LiDAR. Therefore do not
require rear RGB-D, RoomPlan or LiDAR scene reconstruction for ordinary capture.
Front TrueDepth is a separate optional experiment, not aligned rear-camera
depth and not a requirement for our product.

[Apple's capture guidance](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture)
explicitly allows ordinary digital-camera photographs without depth, and both
camera orbits and rotating objects. Depth can supply scale; absent an independent
scale constraint, photo-only geometry must not be labelled dimensionally verified.
The phone can provide photographs to the cross-platform classical backend or
an optional Apple-only reconstruction adapter on the Mac. Framework availability
must be queried at runtime; capture support and reconstruction support are
separate capabilities. No physical iPhone test has been performed here.

## ObjectScanner: useful capture implementation, not a portable engine

Audited revision: `db426e6da1e3ecd92a7e54bcf775113b459390cb`.
The [repository](https://github.com/burakSahinkaya/ObjectScanner/tree/db426e6da1e3ecd92a7e54bcf775113b459390cb)
separates capture modes from a shared output model and exports source images for
Mac processing. Its reconstruction depends on Apple's frameworks, not a new
cross-platform photogrammetry algorithm. We have not built or run that app.

The [photo coordinator](https://github.com/burakSahinkaya/ObjectScanner/blob/db426e6da1e3ecd92a7e54bcf775113b459390cb/ObjectScanner/Engines/Turntable/PhotoCaptureCoordinator.swift)
falls back from a LiDAR capture device to the rear wide camera and enables depth
only when supported. Focus/exposure/white-balance locking and recording actual
capture capability are useful integration patterns. Its blur rejection deletes
captured files: our import/quality assessment must instead preserve originals
and record selection decisions. Capture settings and blur scores are not camera
calibration or shape-accuracy measurements.

The [turntable engine](https://github.com/burakSahinkaya/ObjectScanner/blob/db426e6da1e3ecd92a7e54bcf775113b459390cb/ObjectScanner/Engines/Turntable/TurntableCaptureEngine.swift)
still requires on-device photogrammetry support before capture can start.
A fork for our phone should decouple RGB capture/export from that requirement,
leaving local reconstruction optional. Its one-rectangle object region assumes
a fixed phone; that is not a sufficient segmentation solution for our moving
objects or board-overlap datasets. Also, a fixed phone's ARKit world pose does
not describe the changing object-relative camera pose. Use object feature SfM
or a separately validated object-motion estimate in that case.

The pinned [LICENSE](https://github.com/burakSahinkaya/ObjectScanner/blob/db426e6da1e3ecd92a7e54bcf775113b459390cb/LICENSE)
is Apache-2.0 and has a [NOTICE](https://github.com/burakSahinkaya/ObjectScanner/blob/db426e6da1e3ecd92a7e54bcf775113b459390cb/NOTICE).
Retain required attribution if code is incorporated. This review introduces
no upstream code into our tree and is not an App Store compliance certification.

## Scoped implementation

1. Sol capture task: hash-bound RGB folder manifest, optional depth/pose metadata,
   explicit motion/coordinate policy, unchanged originals and honest scale status.
   This is an import contract, not a shipped iOS camera application.
2. Sol Apple task: original optional Mac photogrammetry CLI and bounded runner;
   compile and query runtime support first. No heavy scan under current disk limits.
3. Sol neural audit: [msplat-ios](MSPLAT-IOS-AUDIT.md), including actual trainer,
   input-camera requirements, Metal portability, third-party code and output type.
   Treat splat appearance and extracted surface quality as different benchmarks.
4. Root: review tests and live probes, then decide on a small ObjectScanner
   capture-only fork or native Tauri plugin. Do not build multiple camera shells
   before validating photo export with the real phone.

All future photo capture should retain one consistent physical rear lens and
original metadata, record orientation and settings when available, and distinguish
stationary-object camera motion from object motion. Learned or captured depth is
optional and must carry its provenance; absence is a supported condition.
