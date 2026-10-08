/** Inspect the envelope before a GLTF loader can fetch any external resource. */
export function checkPhotoGlb(bytes: ArrayBuffer): void {
  if (bytes.byteLength < 20) throw Error("The textured model is truncated.");
  const view = new DataView(bytes);
  const jsonSize = view.getUint32(12, true);
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2 ||
      view.getUint32(8, true) !== bytes.byteLength || view.getUint32(16, true) !== 0x4e4f534a ||
      jsonSize % 4 !== 0 || jsonSize > bytes.byteLength - 20) {
    throw Error("The textured model is not a valid GLB 2 container.");
  }
  const doc: unknown = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, jsonSize)));
  if (!doc || typeof doc !== "object") throw Error("The textured model has no glTF document.");
  const record = doc as Record<string, unknown>;
  // Our exporter embeds buffers and the atlas as GLB chunks/buffer views.
  for (const key of ["buffers", "images"]) {
    const rows = record[key];
    if (!Array.isArray(rows) || rows.some((row: unknown) => !row || typeof row !== "object" || "uri" in row)) {
      throw Error("The preview requires a self-contained photo-textured GLB.");
    }
  }
}
