# Bounded real-image subset

The local tree subset comes from Matthew Guertin's [Single Tree Photogrammetry Dataset](https://huggingface.co/datasets/Matt1up/tree-minnetonka-photogrammetry), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: “Single Tree Photogrammetry Dataset — Matthew Guertin, 2020.” The exact Hugging Face revision is `5f9de5e4a1be429b192a928cf1359c066dadb4b3`. The source README and license are retained beside the fetched data.

Run from the repository root:

```sh
mkdir -p .local-tools/tmp
TMPDIR="$PWD/.local-tools/tmp" python3 scripts/real_data/fetch_tree_subset.py
TMPDIR="$PWD/.local-tools/tmp" python3 -m unittest discover -s scripts/real_data -p 'test_*.py'
```

This obtains 10 original JPEGs and pinned `colmap/cameras.txt` and `colmap/images.txt` in ignored `.local-tools/test-data/tree-subset/`. The manifest records source hashes, file sizes, poses, intrinsics, excluded suspect cameras, and the selection method. Re-running verifies existing files against the pinned source; it refuses changed files or a changed manifest. It checks source-declared sizes, downloaded bytes, hashes, path containment, and a 10 GiB free-space floor. Its limits are 250 MB of photos and 1 GiB total source bytes. Generated images and dense outputs need their own space allowance within the batch budget.

The upstream COLMAP export has empty image-observation lines, and its sparse points have no observation tracks. Consequently the selected cameras are chosen by nearby camera centers and similar optical axes, a geometric overlap *heuristic*. The distance scale (`2.0`) is in arbitrary reconstruction coordinates. Actual correspondence and whether enough views overlap must be established from the photos. Upstream poses and sparse points are useful diagnostics, not measured depth ground truth. Four cameras listed by the upstream author in [suspect_cameras.txt](https://github.com/Matt1Up/tree-photogrammetry-dataset/blob/942f050f7bb7fed530e61c80b3c72e5674986b48/poses/suspect_cameras.txt) as having implausible focal lengths are excluded.

The fetcher intentionally leaves out the 58 MB `points3D.txt`: its absent tracks cannot drive observed-view selection, and the dense proof uses image-derived feature matches. No full-dataset archive or sample bundle is fetched.

## Measured-seed patch diagnostic

After constructing the local MVE scene, run `python3 scripts/real_data/diagnose_seeds.py`. It reads only the frozen `build-opencv/tree-dense/scene/conversion.json`, `measured-seeds.txt`, and ten resized scene PNGs. Its JSON output records SHA-256 hashes of those inputs and the diagnostic script, checked again at the end. It uses bilinear 7×7 grayscale patches, rejects clipped or constant patches, and computes zero-mean normalized cross correlation (ZNCC). The 0.3, 0.6, and 0.8 bins are descriptive, not acceptance criteria.

For the current 575 pair-derived seeds, [MVE's logs](../build-opencv/tree-dense/dmrecon-1.log) selected neighbors 0/4/8 for reference 1 and 0/1/3 for reference 4. The table compares recorded match locations with projections of the same 3D seeds using the supplied poses. “≥2 at 0.6” counts seeds with at least two of the three selected neighbors meeting that diagnostic ZNCC level.

| Reference | Pair seeds involving ref | Recorded match ZNCC median, selected pair only | Same seed and pair at pose projection, median | Pose-projected ≥2 at 0.6 | Pose-projected all 3 at 0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 275 | 0.815 (242 pairs) | 0.720 (242 pairs) | 20/275 | 8/275 |
| 4 | 283 | 0.806 (259 pairs) | 0.578 (259 pairs) | 18/283 | 20/283 |

The measured match pair is often photometrically plausible, but support across the **selected three-view set** is scarce under this simple patch test. For reference 1, 33 recorded pairs involve a view outside its selected neighbors; for reference 4, 24 do. These are excluded from the selected-pair comparison, rather than counted as invalid patches. On the full projection test, reference 1 has 232/275 seeds with valid patches in all three neighbors; reference 4 has 278/283, so low support is not mainly an image-boundary artifact.

The local MVE source uses `minNCC = 0.3`, `acceptNCC = 0.6`, `minParallax = 10°`, and local view selection succeeds only when it fills the configured number of neighbors. This run configured `--neighbors=3 --local-neighbors=3` in [dense-status.json](../build-opencv/tree-dense/dense-status.json). MVE uses warped patches, surface normals, scale handling, optimization, and a 5×5 default patch, while this diagnostic compares unwarped 7×7 patches. Viewpoint and scale changes can therefore lower these ZNCC values even with correct poses. These correlated pair seeds are not independent ground truth, and the diagnostic does not isolate a definitive cause of MVE's 0 and 2 filled pixels.
