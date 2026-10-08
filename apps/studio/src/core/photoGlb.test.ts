import { describe, expect, it } from "vitest";
import { checkPhotoGlb } from "./photoGlb";
function envelope(doc: unknown): ArrayBuffer {
  const text = new TextEncoder().encode(JSON.stringify(doc));
  const size = Math.ceil(text.length / 4) * 4;
  const bytes = new ArrayBuffer(20 + size);
  const view = new DataView(bytes);
  [0x46546c67, 2, bytes.byteLength, size, 0x4e4f534a].forEach((v, i) => view.setUint32(i * 4, v, true));
  new Uint8Array(bytes, 20).fill(32); new Uint8Array(bytes, 20, text.length).set(text);
  return bytes;
}
describe("photo GLB envelope", () => {
  it("accepts embedded buffer and image declarations", () => {
    expect(() => checkPhotoGlb(envelope({ buffers: [{ byteLength: 16 }], images: [{ bufferView: 0, mimeType: "image/png" }] }))).not.toThrow();
  });
  it("rejects external buffer and image URIs before loading", () => {
    for (const key of ["buffers", "images"]) {
      expect(() => checkPhotoGlb(envelope({ buffers: [{}], images: [{}], [key]: [{ uri: "https://invalid.example/resource" }] }))).toThrow("self-contained");
    }
  });
  it("rejects short, truncated and inconsistent containers", () => {
    expect(() => checkPhotoGlb(new ArrayBuffer(10))).toThrow("truncated");
    const bytes = envelope({ buffers: [], images: [] });
    expect(() => checkPhotoGlb(bytes.slice(0, -4))).toThrow("container");
    new DataView(bytes).setUint32(12, 0xfffffff0, true);
    expect(() => checkPhotoGlb(bytes)).toThrow("container");
  });
});
