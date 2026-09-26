# Commercial-compatible CPU end-to-end milestone

This plan supersedes the earlier board-first and correspondence-first priority.
The user requires ordinary overlapping photographs of rigid objects with unknown
poses, including rotation and translation between exposures. A marker board is
optional, not a prerequisite. Initial target: desktop CPU, including Apple Silicon;
Linux/Windows are required portability targets, not yet verified deliverables.

## Acceptance and boundaries

- A single documented invocation must estimate cameras from photos, reconstruct
  dense geometry, and produce a nonempty triangle mesh. Exit status alone is not
  sufficient: validate stage artifacts and record registered/missing images.
- No supplied reference poses, scan vertices, or reference-derived sparse seeds
  enter the unknown-camera reconstruction lane. Reference alignment is a separate,
  explicitly reported evaluation step; scale-fitted comparisons cannot establish
  recovered physical scale.
- CPU operation must not require CUDA. Desktop worker execution is acceptable;
  iOS needs separate library/sandbox integration and is not implied by a Mac build.
- Only commercially usable dependencies may become application requirements.
  Existing GPL research exceptions do not authorize shipping. CeCILL-B and other
  custom terms need explicit review; no root-license badge clears bundled tools.
- New acceptance datasets must permit commercial use. Real photographs and an
  independent scan must be identified separately from rendered training images.
- Preserve at least 10 GiB free disk; temporary files remain in `.local-tools/tmp`.
  Initial bounded acquisition/build budget: 5 GiB across concurrent tasks.

## Scoped work and ownership

1. **CPU backend integration (Sol):** evaluate pinned headless MicMac, inspect
   selected code/helpers, compile on arm64, add a safe staged runner and tests.
   Stop release promotion if a required dependency violates policy. Reuse existing
   selected MVE work as an alternative if its complete pipeline has a cleaner path.
2. **Object reference acquisition (Sol):** obtain a small commercially licensed
   real-photo/scan pair, preferably 50–100 usable images; store source URLs, hashes,
   license, attribution, dimensions, and camera/coordinate provenance. Implement
   bounded acquisition and preparation tests without exposing credentials.
3. **Independent supervision:** review patches and license findings, run native
   and runner tests, reproduce a live reconstruction, inspect actual mesh contents,
   and document resource use, failures and cross-platform gaps.
4. **Quality evaluation:** report bidirectional surface proximity, completeness at
   declared thresholds and registration coverage; disclose visibility scope,
   alignment, reference uncertainty, and peak-memory measurement limitations.
5. **Product integration:** promote only a verified backend into capabilities;
   wire the same structured worker contract into the desktop app after the CLI
   path works. Do not claim the existing C++ `reconstruct` command works merely
   because a separate experimental runner succeeds.

## Completion report

Report what actually ran: image count and resolution, estimated camera count,
point/face counts, elapsed stages, machine, peak memory where measured, output
paths, reference score (or the exact missing prerequisite), license status and
tested platforms. Unit tests, a live mesh, measured quality, and release approval
are four distinct gates.

## Source audit update

The ordinary MicMac build failed the commercial-use gate: NEC noncommercial
code and GPL graph-cut code enter its core build, with further Windows GPL and
LGPL dependencies. See [exact pinned evidence](MICMAC-BACKEND.md). No executable
was built or added. The implementation task now targets a selected MVE
`makescene → sfmrecon → dmrecon → scene2pset → fssrecon → meshclean` path,
excluding GPL `sceneupgrade` and the unresolved SURF algorithm path. Its SIFT
implementation is BSD-covered; upstream OpenCV documents expiration of the
SIFT patent in 2020. This engineering selection still requires review of exact
build artifacts and packaged dependencies before release.

For unknown-pose photographs, a reference-fitted similarity alignment may be
reported as a **normalized shape diagnostic** using a declared fixed alignment
protocol. It cannot establish independently recovered physical scale or camera
accuracy. Retain the original unaligned reconstruction, report the fitted scale,
and do not feed aligned scan geometry back into reconstruction.

## Current milestone outcome

The selected CPU photo-to-mesh runner is operational on M1, with a repeatable
real tree mesh, separate source/binary provenance, resource guards and final
geometry validation. A 73-photo object run reached a mesh through a preserved
continuation after fixing cleanup integration. Its [reference-fitted shape
diagnostic](BUNNY-EVALUATION.md) is poor; it is not production-quality acceptance.

Two photo/scan objects are local: 3DLF bunny (third-party shape-rights caveat)
and YCB cracker box (explicit data CC BY 4.0, separate Google scan). The latter
is prepared but not reconstructed. No scanned vertices, depths or supplied
poses have entered the image-only reconstruction lane. Cross-platform compile
verification and App Store packaging remain separate gates, not consequences
of a successful local CLI run.
