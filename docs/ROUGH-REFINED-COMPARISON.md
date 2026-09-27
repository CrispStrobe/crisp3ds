# Frozen rough-versus-refined YCB sensor-depth comparison

This check uses the **existing** 008 native `mesh.ply` and `refined.ply` from
the same masked, image-only camera gauge. It reruns the already frozen Berkeley
sensor-depth protocol: angles 0°/120°/240°, up to 2,048 identical measured
support rays per view, original photo-derived support and separately the fixed
16-pixel-eroded interior, first-hit ray/triangle depth, 300-second total cap,
20 MB report cap and 10 GiB disk reserve. The camera-only Sim(3) is the same
sealed original-008 report for both meshes. No surface registration, new
masks, new reconstruction, parameter search or altered rays are permitted.

The stage gate in `sensor_depth.py` accepts only native `mesh.ply` as the
producer's completed `mesh` or `rough_mesh` stage, and only `refined.ply` as
its completed `refine` stage. It checks the producer's reported face count
against both the lossless geometry export and the parsed mesh before ray
scoring, and records the actual stage and verified faces. A producer that later
failed refinement can still contribute its already-completed `mesh.ply`,
but a failed `refine` cannot authorize `refined.ply`. Arbitrary filenames,
unknown producer schemas and duplicate stage records are rejected.

Pre-score read-only checks found 008 rough `mesh.ply`: 38,902 faces (`mesh`
stage complete) and 008 `refined.ply`: 5,680 faces (`refine` complete). The
same rule also matched existing 014 rough mesh (38,550 faces, producer later
failed at refinement) and 015 rough mesh (38,440 faces). These face counts do
not imply quality. The comparison asks whether **this existing refinement**
changes sensor-supported ray hits or residuals under one fixed camera gauge;
it does not attribute any change to a unique internal refinement mechanism.

The approved single score is
`build-opencv/sensor-depth-005-rough-refined/report.json` (SHA-256
`d63ac4b83a0b78ce672bbfcf49da69cf9bd129b4fe16a98b499b6c15e4ff1f32`,
1,106,824 bytes). Root confirmed the selected rays and observations exactly
match the prior sensor-depth004 run; the 008 rough-mesh pooled result is
identical. Both rows record the completed native stage and exact verified
face count.

| Same fixed coarse support | 008 rough | 008 refined |
| --- | ---: | ---: |
| First-hit rays / 6,144 | 5,166 (84.08%) | 5,055 (82.28%) |
| Hit-only mean absolute depth residual | 2.881 mm | 2.641 mm |
| All-supported rays within 5 mm | 70.72% | 71.16% |
| Interior all-supported rays within 5 mm | 82.73% | 83.47% |

An independent paired recount of the report's raw per-ray arrays found 5,051
rays hit by both meshes, 115 hit only by rough (lost after refinement), four
hit only by refined (gained), and 974 missed by both. On **exactly those 5,051
shared-hit rays**, mean absolute residual is 2.807 mm rough versus 2.636 mm
refined (paired mean change −0.171 mm). This separates the modest residual
improvement from the reduction in first-hit coverage; the lower hit-only
mean alone would have a selection bias. The slightly higher missing-inclusive
5 mm rate coexists with lower hit coverage because some remaining hits have
smaller residuals. It is a mixed coverage/accuracy tradeoff, not an
unqualified refinement win.

Sensor-depth residuals remain assumption-qualified diagnostics, not physical
ground-truth shape accuracy; the RGB/depth projection, Berkeley pose metadata,
mask and depth scale limitations documented in `SENSOR-DEPTH-BENCHMARK.md`
still apply. This comparison does not identify which refinement operation
caused any individual ray transition and should not be generalized to other
objects or settings.
