# Gated OpenMVG HIGH bunny rough-to-texture continuation

Status: prepared, synthetic-tested, **not run**. This is a separate composed continuation of the 73-photo OpenMVG HIGH sparse reconstruction and the cache-resumed OpenMVS rough reconstruction. It neither reruns the original pipeline end-to-end nor establishes metric scale or mesh quality.

The runner is [`openmvg_bunny_high_finish.py`](../scripts/classical_backend/openmvg_bunny_high_finish.py). Its default mode is read-only preflight. It requires the exact SHA-256 of the rough `result.json`, whose schema must be `openmvg_bunny_high_cache_rough_v1` and status `rough_complete_pending_quality_review`, and a **separate** positive review receipt. No positive rough review has been assumed or generated here. Review must inspect native PLY validity **and** recognizable bunny shape. A native parse alone, a successful exit code, or the earlier camera approval does not satisfy this gate.

The review receipt is a JSON file with these required fields:

```json
{
  "schema": "openmvg_bunny_high_rough_visual_review_v1",
  "decision": "approved_for_refine_texture",
  "native_mesh_valid": true,
  "visual_object_shape_reviewed": true,
  "reviewer": "named reviewer",
  "rough_receipt_sha256": "64 lowercase hex characters",
  "dense_mvs_sha256": "64 lowercase hex characters",
  "mesh_ply_sha256": "64 lowercase hex characters"
}
```

Both receipt **paths and caller-provided hashes** must match before any output path is created. The runner also checks the rough receipt's two complete native stages, artifact and output-inventory hashes, all 73 staged photos, positive native dense/mesh counts, executable native binaries, and a fresh output root. It copies only the hash-verified 73 photos plus `dense.mvs` and `mesh.ply`; no scanner, ground truth, or external camera poses enter this continuation.

Planned root: `/Volumes/backups/code/crisp3ds-data/openmvg-bunny-high-openmvs-refine-003`. Native stages are OpenMVS `RefineMesh` (`dense.mvs` plus `mesh.ply`, resolution level 2, one scale) and then `TextureMesh` (`dense.mvs` plus `refined.ply`, OBJ export). Both request two threads and run under macOS `caffeinate`. The whole continuation, including copying, has a 20-minute **host wall-clock** deadline; native supervision also checks process elapsed time so a sleeping Mac cannot silently turn a 20-minute trial into an hours-long trial. Output is capped at 256 MiB with 16 MiB per-stage logs, sampled 4 GiB process-tree RSS, and an 11 GiB free-space floor on source, workspace and output disks. The preflight totals the exact 73 PNG, `dense.mvs`, and `mesh.ply` copy bytes, rejects them if they leave less than 96 MiB for refinement/texture/logs, and requires another 256 MiB of available disk beyond the 11 GiB reserve. A failure leaves its separate output and receipt for diagnosis; it does not mutate or remove the rough source.

Once the native and visual review has actually been issued, the read-only check is:

```sh
python3 -m scripts.classical_backend.openmvg_bunny_high_finish \
  --source-receipt-sha256 ROUGH_RESULT_SHA256 \
  --review-receipt /absolute/path/to/review.json \
  --review-receipt-sha256 REVIEW_SHA256
```

Only after independent review of that preflight should `--execute` be added. A successful receipt would say `complete_conditional_continuation_only`, with `quality_accepted`, `metric_scale_verified`, and `shipping_approved` all false. The texture needs separate UV/atlas, geometry, and foreground/background inspection; completion is not a quality ranking against Apple, COLMAP, MVE, KIRI, or the scanner.

The synthetic preflight suite is `python3 -m unittest scripts.classical_backend.test_openmvg_bunny_high_finish -v`.
