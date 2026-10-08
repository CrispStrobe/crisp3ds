# Photo-textured meshes

The native pipeline can export a self-contained GLB with UV coordinates and an
embedded photo atlas. This is an opt-in first implementation; it preserves the
input geometry and does not resolve the outstanding loss of facial relief.

```sh
crisp3ds-dense run --photos PHOTOS --calibration LENS.json --output RUN --texture
# Or texture an existing result using its own recovered cameras:
crisp3ds-dense texture --inputs RUN/frontend/inputs --mesh RUN/mesh/mesh.stl \
  --output TEXTURE --views 12 --tile-size 1024
```

`mesh.glb`, `atlas.png` and a coverage report are written under `RUN/texture`
(or `TEXTURE`). Studio offers **Export textured GLB (experimental)** for built-in
native/browser engines; completed runs offer GLB download/share. The existing
viewer continues to display untextured geometry. JSON run options accept
`"texture": true`; the default is false. The browser package builds, and the
pipeline with textures passes the native memory-tree test, but real browser
execution/memory and real-device sharing have not yet been measured for this
option.

The exporter selects up to twelve evenly spaced recovered views. It rejects
back-facing, masked-out and occluded candidates using the reconstructed mesh's
perspective-correct depth buffer at 2048 pixels on the longest side. Triangle
corners and centres are checked. A smoothed normal field makes photo selection
more coherent; it does not move mesh vertices. Each selected photo is cropped
to its own foreground bounds and packed in a padded rectangular atlas tile.
Triangles receive projected UVs, with vertex splits at photo seams. Unobserved
faces point to a neutral-grey tile. Reports include untextured surface area.
`ATTRIBUTION.txt` beside the input cameras, or at a pipeline run root when
writing its `texture` subfolder, is copied and embedded in GLB copyright metadata.
Source licensing must be supplied by the caller; the exporter does not infer it.
No scanner mesh, supplied dataset poses/depth/masks, or generated appearance is
used. No new third-party library or algorithm source was incorporated.

The atlas contains captured colour **and lighting**, not recovered intrinsic
albedo. Discrete view selection still leaves seams, exposure/shadow differences
and projection errors where geometry is wrong. It has no seam levelling,
multiband blending, photometric calibration, efficient chart packing, normal or
roughness estimation. Long triangles can distort projective UV interpolation.
Texture resolution does not establish geometric resolution. Mesh coordinates
and winding are preserved; physical scale and orientation remain those of the
pipeline and are not established by GLB export.

## What KIRI publicly describes

KIRI's [Photo Scan](https://www.kiriengine.app/features/photo-scan) describes
cloud photogrammetry with mesh and texture export. Its
[API](https://docs.kiriengine.app/photo-scan/image-upload/) exposes separate model
quality and texture quality (1K–8K), masking, and texture smoothing, which its
documentation says can trade sharpness for coherent textures. Its separate
[Featureless Object Scan](https://www.kiriengine.app/features/featureless-object-scan)
is described as NeRF-derived neural surface reconstruction producing meshes.
These are vendor descriptions; they do not disclose the exact solver, training
recipe or meshing/texturing implementation. We have not compared identical
inputs in KIRI, and have not uploaded these photos to that service.

KIRI also recommends multiple capture heights and detail photographs. Our
one-ring captures leave unseen surfaces and weak stereo directions; extra
texture pixels cannot fix that. A fair comparison needs the same input photos,
fixed geometric evaluation and matched views of both textured and untextured
meshes. A texture or normal map can convey detail in a rendered view without
that detail being present in the printable mesh.

## Further work

- Texture: correct exposure/seams, improve visibility confidence, select sharper
  photos, and pack connected charts more efficiently. MIT-licensed
  [xatlas](https://github.com/jpcy/xatlas) is one possible UV-library reference,
  not currently a dependency and not a patent clearance.
- Geometry: local matching/camera accuracy on eye and nose regions remains the
  priority; the independent plane experiments have only modest gains. Evaluate
  untextured geometry separately from appearance.
- Robustness: photo-derived masks and valid camera recovery must precede
  texturing. Mustard, light-coloured objects and printed/glossy objects expose
  different front-end failures; do not substitute dataset truth to obtain a
  plausible-looking textured result.
