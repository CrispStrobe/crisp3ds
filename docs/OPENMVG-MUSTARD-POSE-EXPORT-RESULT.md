# OpenMVG mustard camera-pose export result

Read-only verification on 2026-09-27 confirms that the separate, one-shot
OpenMVG v2.1 converter build completed in **3.009 s** and the diagnostic
conversion completed in **0.971 s** (exit 0). The converter's current
`otool -L` output matches its sealed license receipt and reports no new
dependency class. It remains evaluation-only; no shipping-license clearance
is claimed.

| Sealed artifact | SHA-256 |
| --- | --- |
| Failed mustard SfM receipt | `e45c400dcf71a3ba86fa22f7af2138418d6dc511fddcd6d8470ef33599804744` |
| Failed mustard `sparse/sfm_data.bin` | `dc11f9b3a74809ebc080260b360ff1dd0e6f9f9526ea151aac002ace75b2df8b` |
| Converter build manifest / license receipt | `e7f0b3cce52612466756aad53231eb6e1b2fc83b884bec829e548f9f6b4a22fa` / `51f8e5855bc7158f27cfdcbd6a84c4bee594e9ee24d42155d7d7837f43290bf4` |
| Converter build log / executable | `c036acd2e1162bd7afbcff48d187900d5dc3465c1752e2a2360ea195760d6b35` / `95bebb65afd1374aadae21aecc5f4cf66f4257432477d54ceae22c4cc35636b0` |
| Export receipt / 66,004-byte camera JSON | `ab73c2b729adb8abd1f84d97f1908b26635cd8caa6d2d1756185d9a31a34dac3` / `c1e0ce4431245ca9392dd82b64cc8007d69e0c41a2e87cf78cec10f33ecc1124` |
| Empty conversion log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The export command read only the sealed failed model and wrote a fresh
`sfm_camera_poses.json` with `-V -I -E`. Independent read-only inspection
found **48 views, 46 poses, one intrinsic, and no exported structure or
control points**. `NP3_042.jpg` and `NP3_048.jpg` have no estimated pose.
These counts agree with the failed SfM receipt's 46/48 camera gate; the
conversion did not recover cameras, rerun SfM, or change that failure.

The exported single `pinhole_radial_k1` intrinsic has focal length
**552.94 px**, principal point **[552.37, 1019.12] px** on 1280×1024
images, and radial `k1` **8.16144**. The run began from the disclosed
photo-width heuristic of 1536 px and allowed intrinsic refinement. This
large drift and near-bottom principal point are suspicious self-calibration
outcomes, not a new acceptance threshold or a comparison with reference
calibration. No Berkeley pose, calibration, scanner or mesh data was accessed
for this verification.

The [export plan](OPENMVG-MUSTARD-POSE-EXPORT-PLAN.md) preserves the input and
resource contract. Any orbit analysis must use this now-sealed JSON and its
view-to-pose IDs; it cannot relabel this incomplete reconstruction as a
48-view success.
