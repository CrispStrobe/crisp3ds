# Brush Apple Silicon preflight (2026-09-27)

Status: **CLI and COLMAP inputs audited; bounded 20-step Apple Silicon smoke executed; no mesh result or quality claim.** Brush is a Gaussian-splat renderer/trainer, not a photo-to-mesh backend. The existing image-derived COLMAP poses are a prerequisite, not something Brush estimates from a raw photo folder.

## Two distinct pins

| Lane | Identity | Evidence |
|---|---|---|
| Source audit | [`6378a76add3b93501abb55c2dc08d71688537679`](https://github.com/ArthurBrussee/brush/commit/6378a76add3b93501abb55c2dc08d71688537679), 2026-09-26 | [COLMAP loader](https://github.com/ArthurBrussee/brush/blob/6378a76add3b93501abb55c2dc08d71688537679/crates/brush-dataset/src/formats/colmap.rs), [dataset split and mask lookup](https://github.com/ArthurBrussee/brush/blob/6378a76add3b93501abb55c2dc08d71688537679/crates/brush-dataset/src/formats/mod.rs), [CLI settings](https://github.com/ArthurBrussee/brush/blob/6378a76add3b93501abb55c2dc08d71688537679/crates/brush-process/src/config.rs) |
| Tested prebuilt CLI | [v0.3.0 release](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0), commit `3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486` | Apple Silicon archive 41,447,012 bytes, SHA-256 `65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c`; unpacked binary SHA-256 `8380ed40cce870025393e1ea0257e0752351c67a409d827b76dc75ec3a999a71` |

The release binary is **not** a build of the source-audit commit. Use the release's actual `--help` contract for the proposed command. Do not transfer newer-source mask or CLI behavior to v0.3.0 without testing it.

## Live local result

`python3 -m scripts.brush_backend.preflight --dataset build-opencv/classical-ycb-native-masked-008/dense --output build-opencv/brush-ycb-001 --producer-result build-opencv/classical-ycb-native-masked-008/result.json` passed structural checks: 60 registered images, a single PINHOLE camera at 1282×1024 (`fx=fy=1077.5837619820666`, `cx=641`, `cy=512`), and nonempty binary COLMAP model files. Model SHA-256 values are `3c4636b59afcf57c4fc4b1f4e85780a21bfbaeceaa9f52926aafa82ec3ce391d` (cameras), `251cb13882aa9f536498dd9521c4586c3aa8dd7cb8aa8ab5fd9fcb48fd801641` (images), and `c11fd9cc7510a8cf0782774d3569f15ce7d7b052f2debe08ca0b9a5eb80e5ada` (points). Producer result hash: `8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9`. These hashes inventory what was read; the producer result alone does **not** cryptographically bind this undistorted dataset to the producer. The preflight therefore reports `training_authorized=false` and `producer_association=unverified`.

Host: Apple M1/arm64; local Rust 1.98.1; Apple Metal compiler available. The verified release archive was unpacked into `.local-tools/brush-v030` under a 50 MiB compressed, 150 MiB unpacked, 220 MiB total allowance with a 10 GiB free-space floor. It contains an arm64 Mach-O `brush_app`, README, CHANGELOG, and Apache-2.0 LICENSE. `brush_app --version` returned `brush-cli 0.3.0`; `--help` exited successfully. The executable is ad-hoc signed and dynamically links Apple's Metal and system frameworks. No trainer was launched. Free space after extraction was 11,289,018,368 bytes (~10.5 GiB), leaving little room for training outputs.

## Input and camera contract

Brush discovers the largest COLMAP model by `cameras.bin` / `images.bin` and reads `points3D.bin` in that same directory. The tested input has `dense/sparse/{cameras.bin,images.bin,points3D.bin}` and `dense/images/<registered-name>`; all 60 names resolve, with no `dense/masks` directory. It uses the undistorted PINHOLE camera, so no distortion translation is needed. In the audited source, COLMAP poses are interpreted as world-to-camera and inverted to Brush camera-to-world; principal point is normalized by image width/height. [The v0.3 loader](https://github.com/ArthurBrussee/brush/blob/3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486/crates/brush-dataset/src/formats/colmap.rs) has the same inversion and normalization. Pixel-center/renderer equivalence and visual pose orientation remain untested without a real render.

The audited newer source looks for optional `masks/` and its default `eval_split_every=None` puts every loaded view in training. The proposed v0.3 command omits `--eval-split-every` and does not supply masks. This avoids an *implicit* in-dataset holdout, but does not establish that no external held-out/reference files were added elsewhere. The preflight optionally verifies a `brush_train_split_v1` JSON manifest with `train_image_sha256` (exact name→SHA map for **every** registered image) and a disjoint `heldout_image_sha256` map. Even with that manifest, `training_authorized` stays false until the producer/dataset lineage and external-input boundary are reviewed. Do not use the YCB Google/reference mesh, sensor-depth or fitted registration to create or tune Brush training inputs.

## Sealed YCB-008 smoke lane

`scripts.brush_backend.lineage.validate_008` now verifies the exact existing evidence chain: the SHA-pinned 008 result links the SHA-pinned mask report and 002 result; 002's undistortion stage completed despite its later densification failure; the mask report binds the 60 current undistorted image hashes and three binary camera/model hashes; 002 links the foreground image-only SfM producer summary/provenance and the original 60 photo hashes. This is the older 60-photo YCB package, **not** a 48-train/12-heldout split. The source SfM used photo-derived pose masks; the proposed Brush input uses no masks, so a later object-centric training run needs a separately audited mask adapter. No ground-truth scan, sensor depth, or fitted transform enters the input.

The read-only command `python3 -m scripts.brush_backend.smoke --preflight` passed locally and printed the exact command and budgets. Root approved one bounded live smoke. It used:

```sh
.local-tools/brush-v030/brush-app-aarch64-apple-darwin/brush_app \
  build-opencv/classical-ycb-native-masked-008/dense \
  --total-steps 20 --max-resolution 640 --max-splats 50000 --seed 42 \
  --export-every 20 --export-path /Volumes/backups/code/crisp3ds-data/brush-smoke-001 \
  --export-name 'export_{iter}.ply'
```

The bounded launcher (`python3 -m scripts.brush_backend.smoke`) requires the exact fresh external output path on a separately mounted volume and the exact verified release binary. It enforces 300 seconds, 512 MiB for output plus project-local temporary files, 16 MiB log, 4 GiB sampled process RSS, and 10 GiB free-space floors on internal and external volumes. It terminates the process group on cap/timeout failure, rechecks input and binary hashes, and validates a finite, bounded Gaussian-splat PLY payload before declaring only **execution** complete. Process RSS does not measure all GPU unified memory or OS shader cache writes.

Actual result: [`brush-smoke-001/report.json`](/Volumes/backups/code/crisp3ds-data/brush-smoke-001/report.json) reports `complete`, exit 0 in 6.482 seconds, peak sampled process RSS 290,930,688 bytes, and 985,843 output bytes. `export_20.ply` is 931,153 bytes, SHA-256 `4346b17bcf0408d24171115951c51e43ea230d7a34a4efa5741a3afd611095db`, with 3,939 finite Gaussian vertices and the expected position/color/opacity/scale/rotation properties. This is exactly the initial sparse point count; a component-sorted coordinate comparison differed from initializer coordinates (median absolute component difference ~6.04e-5 arbitrary SfM units), consistent with but not proof of optimizer updates. `train.log` was empty. The host reports Apple M1/Metal 4 and the binary links Metal, but this run did **not** independently log its selected GPU backend. The output is splat PLY, not a triangle mesh. No geometry or quality ranking against MVE/OpenMVS follows from this smoke.

## License status

Brush's [root LICENSE](https://github.com/ArthurBrussee/brush/blob/6378a76add3b93501abb55c2dc08d71688537679/LICENSE) and the release's included LICENSE are Apache-2.0. The archive has no separate dependency-license inventory or `NOTICE`; the executable statically includes Rust dependencies that `otool -L` cannot enumerate. Dependency/release redistribution terms require a complete SBOM/license review. This release is a research tool here, not shipping-approved; neither its Apache root license nor a local Metal test clears all dependency or product-distribution questions.

Run pure tests: `python3 -m unittest scripts.brush_backend.test_preflight scripts.brush_backend.test_smoke`. Fetch helper is explicit/opt-in (`python3 -m scripts.brush_backend.fetch_release --target <fresh-path>`), fixed to the v0.3 Apple Silicon URL and SHA; it never builds Rust dependencies.
