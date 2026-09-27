# Frozen mustard common-crop dense continuation

This is a single predeclared image-only continuation after the failed, [sealed full-frame dense run](MUSTARD-DENSE-RESULTS.md). It does not change SfM, repair, masks, camera poses, 3D points, tracks, native dense settings, or the [geometry benchmark protocol](MUSTARD-DENSE-BENCHMARK-PLAN.md). It does not use the Google shape reference, Berkeley depths/poses, or held-out photos to construct the crop. No native run or candidate score preceded this protocol freeze.

## Fixed crop and measured training-only inputs

The exact 48 accepted training masks from the repaired-model input have a union of nonzero pixels with inclusive bounds `x=525..682, y=361..598` in the original 1280×1024 images. Convert to half-open bounds and add exactly 32 pixels on each side, clipped to the image: **`[x0,y0,x1,y1]=[493,329,715,631]`**, common to all 48 images, width 222 and height 302. No resizing, interpolation, per-view crop, mask dilation, or mask-derived exclusion of sparse tracks is allowed. The accepted mask support ranges from 13,847 to 20,496 pixels per view; that is 20.65–30.57% of the crop, 26.30% pooled over 48 crops. The original full-frame support was only about 1.06–1.56%. This change increases the object *fraction of image*, but cannot by itself validate camera geometry or guarantee OpenMVS will pair views.

All **4,734** registered sparse 2D observations in the repaired 48-view model lie strictly inside the proposed common crop (observed original-coordinate range `x=528.071..678.535, y=364.898..597.517`). This was checked before the run; the producer must independently reject any missing/out-of-bounds point, changed track, or changed 3D XYZ. In crop coordinates, subtract `(493,329)` from every stored 2D observation and from the principal point. The source's exact zero-radial-distortion intrinsics `SIMPLE_RADIAL [1536,640,512,0]` become **PINHOLE `[fx,fy,cx,cy]=[1536,1536,147,183]`** for a 222×302 image. Camera poses and 3D points stay numerically identical; independent projection of the same physical ray before/after must agree to floating precision. The 48 RGB crops and native mask crops must be pixel-for-pixel equal to their source rectangles, and all originals stay byte-identical. No image from the 12 held-out angles enters this continuation.

## One-run producer contract

Store decoded RGB crops as lossless PNGs, with an explicit `.jpg` to `.png`
name map in the copied reconstruction; never recompress crops as JPEG.
Untriangulated 2D features outside the crop may have negative translated
coordinates, but their indices and point associations must remain unchanged.
All tracked observations must be in bounds. This preserves the original
track-index layout without inventing measurements or deleting difficult tracks.

Use a fresh external-SSD output, leaving failed 001/002 and the sealed source untouched. Bind before and after hashes for the accepted TRAIN photo/mask manifest, repaired sparse model and QA, source runner, OpenMVS binaries, cropped images/masks/model, and native logs/artifacts. The source camera/pose/observation/track/XYZ inventory must equal the original except for the exact 2D and principal-point translation. A crop or projection equivalence failure blocks native execution. The native image loader may scale the smaller crop under its existing resolution rules; record the actual depth-map dimensions rather than assuming 222×302 is the depth-map size.

Keep the prior native settings: CPU, two threads, resolution level 0, min resolution 600, max resolution 1280, two geometric iterations, tower mode 4, mask label 0; at most 4 GiB child RSS, 3 GiB output, 10 minutes, and at least 10 GiB free on both internal and external volumes. No retry or threshold change follows a failed result automatically. Require a complete, nonempty native dense/rough-mesh artifact and sealed producer report before evaluation. A native exit code of zero alone is insufficient.

If and only if an eligible mesh is sealed, normalize geometry in a fresh evaluation directory; run the *unchanged* proper-Sim(3) Google-reference fit with 1,024 samples/seed 2026, then both paired and overlay previews for root visual registration review. Do not select a new fit from score results. Only after visual approval, score the frozen whole-mesh 4,096 samples/seed 2027 and 0.5/1/2% reference-bbox thresholds, recording accuracy, completeness, F, distance distributions, normalized Chamfer, normals, topology, alignment uncertainty, and all producer costs. That reference is a separate shape oracle, not physical ground truth; neither crop success nor a high registration score proves metrology.
