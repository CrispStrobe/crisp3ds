# Mustard TRAIN mask coverage and label QA plan

This is a read-only, photo-only diagnostic for the sealed 48 TRAIN inputs. It
does not generate masks, run SfM, use held-out images, or read Berkeley poses,
depth, or mesh. The staged 0/255 PNGs are **accepted coarse pose support**, not
object silhouettes or semantic ground truth. The rejected warm-mask candidates
in `YCB-OBJECT-MASKS.md` show why area/shape checks alone cannot certify label
coverage or board exclusion.

The harness binds the exact stage-report SHA-256 and its 48 TRAIN names, image
and mask hashes. It checks decoded RGB/JPEG and L/PNG dimensions, binary mask
values, staged pixel counts, and exact directory membership. It reports per-view
support area, bounding box, connected-component count/largest share, and border
contact. A per-view flag only identifies a review target; it is not a bottle,
board, or background classification. No files are written by the harness.

Semantic counts remain `unknown` unless a separate, independently reviewed
photo-only manifest is supplied. Its schema is `mustard_photo_semantic_labels_v1`:
`status=validated_photo_only`, nonempty `reviewer` and `reviewed_at`, exact 48
TRAIN-name `labels`, and one SHA-256-bound 1280×1024 L/PNG per name. Pixel
codes are 0=unknown, 1=bottle/object, 2=board/support, 3=background. The
manifest must bind each label to the staged photo SHA-256 and declare
`source=manual_photo_review`; no reference-derived or model-generated labels
are accepted. Missing, partial, linked, changed, or invalid annotations fail
closed. The CLI currently has **no approved semantic-manifest hash pinned** and
therefore refuses every semantic input. The pure audit function's synthetic
label path tests the future contract, not an accepted real annotation set.
Even with labels, unknown pixels stay unknown; class fractions are
computed only over independently labeled pixels and no claim is made about
unlabeled portions. This contract does not imply that such annotations exist.

Synthetic tests cover abstention, supported class intersections, invalid
provenance, tampered hashes, and wrong mask/label geometry. The real diagnostic
below uses no semantic manifest, so all three semantic coverage fields abstain.

Run: `python -m scripts.classical_backend.openmvg_mustard_mask_qa` (JSON to
stdout only). On the sealed local stage, all 48 photo/mask hashes and binary
1280×1024 decodes passed. Total support is 846,453 pixels (1.345% of image
pixels); per-view range is 13,847 (`NP3_276`) to 20,496 (`NP3_006`). All 48
supports are one 8-connected component and none touches the image border.
These geometry checks do **not** validate bottle coverage, label-strip
preservation, or board/background exclusion. The semantic result is
`unknown_abstain` for every view. The low-area views `NP3_276`, `NP3_096`,
`NP3_270`, `NP3_102`, and `NP3_090` are reasonable photo-overlay review
priorities, but low area alone is not evidence of an omission.
