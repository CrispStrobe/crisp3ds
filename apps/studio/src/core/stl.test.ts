import { describe, expect, it } from "vitest";
import { fixtureBytes, fixtureEvents } from "../testing/fixture";
import { meshBytes, parseBinaryStl, parseBinaryStlChunked, StlError, stlTriangleCount, type ParsedMesh } from "./stl";

/** Builds a binary STL from triangles given as nine numbers each. */
function stl(triangles: number[][]): ArrayBuffer {
  const buffer = new ArrayBuffer(84 + 50 * triangles.length);
  const view = new DataView(buffer);
  view.setUint32(80, triangles.length, true);
  triangles.forEach((triangle, index) => {
    triangle.forEach((value, i) => view.setFloat32(84 + index * 50 + 12 + i * 4, value, true));
  });
  return buffer;
}

const TETRAHEDRON = [
  [0, 0, 0, 0, 1, 0, 1, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 1, 0, 1, 0],
  [1, 0, 0, 0, 1, 0, 0, 0, 1],
];

/** Edges used by exactly two triangles in opposite directions: a closed, consistently oriented surface. */
function closedAndOriented(mesh: ParsedMesh): boolean {
  const edges = new Map<number, number>();
  const vertices = mesh.positions.length / 3;
  for (let i = 0; i < mesh.indices.length; i += 3) {
    for (let e = 0; e < 3; e++) {
      const a = mesh.indices[i + e]!;
      const b = mesh.indices[i + ((e + 1) % 3)]!;
      edges.set(a * vertices + b, (edges.get(a * vertices + b) ?? 0) + 1);
    }
  }
  for (const [key, count] of edges) {
    const a = Math.floor(key / vertices);
    const b = key % vertices;
    if (count !== 1 || edges.get(b * vertices + a) !== 1) return false;
  }
  return true;
}

describe("parseBinaryStl", () => {
  it("welds shared vertices", () => {
    const mesh = parseBinaryStl(stl(TETRAHEDRON));
    expect(mesh.triangles).toBe(4);
    expect(mesh.positions.length).toBe(4 * 3);
    expect(mesh.indices.length).toBe(12);
    expect(mesh.dropped).toBe(0);
    expect(mesh.min).toEqual([0, 0, 0]);
    expect(mesh.max).toEqual([1, 1, 1]);
    expect(closedAndOriented(mesh)).toBe(true);
  });

  it("computes unit normals pointing outwards", () => {
    const mesh = parseBinaryStl(stl(TETRAHEDRON));
    for (let v = 0; v < 4; v++) {
      const n = [mesh.normals[v * 3]!, mesh.normals[v * 3 + 1]!, mesh.normals[v * 3 + 2]!];
      expect(Math.hypot(...n)).toBeCloseTo(1, 5);
      // The centroid is at (0.25, 0.25, 0.25): an outward normal points away from it.
      const p = [mesh.positions[v * 3]! - 0.25, mesh.positions[v * 3 + 1]! - 0.25, mesh.positions[v * 3 + 2]! - 0.25];
      expect(n[0]! * p[0]! + n[1]! * p[1]! + n[2]! * p[2]!).toBeGreaterThan(0);
    }
  });

  it("treats -0 and +0 as the same coordinate", () => {
    const mesh = parseBinaryStl(stl([[0, 0, 0, 1, 0, 0, 0, 1, 0], [-0, -0, -0, 0, 1, 0, 0, 0, 1]]));
    expect(mesh.positions.length / 3).toBe(4);
  });

  it("drops degenerate and non-finite triangles instead of drawing them", () => {
    const mesh = parseBinaryStl(
      stl([
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, NaN, 0, 0, 0, 1, 0],
        [0, 0, 0, Infinity, 0, 0, 0, 1, 0],
      ]),
    );
    expect(mesh.triangles).toBe(4);
    expect(mesh.dropped).toBe(3);
    expect(mesh.indices.length).toBe(3);
    expect(mesh.max.every(Number.isFinite)).toBe(true);
  });

  it("handles an empty mesh", () => {
    const mesh = parseBinaryStl(stl([]));
    expect(mesh.triangles).toBe(0);
    expect(mesh.min).toEqual([0, 0, 0]);
    expect(mesh.max).toEqual([0, 0, 0]);
  });

  it("copes with a triangle soup that has no shared vertices at all", () => {
    const soup: number[][] = [];
    for (let i = 0; i < 3000; i++) soup.push([i, 0, 0, i + 0.5, 1, 0, i + 0.25, 0, 1]);
    const mesh = parseBinaryStl(stl(soup));
    expect(mesh.positions.length / 3).toBe(9000);
    expect(mesh.indices.length).toBe(9000);
    expect(Math.max(...mesh.indices)).toBe(8999);
  });

  it("rejects truncated files, text STL and non-STL data with a clear message", () => {
    const whole = stl(TETRAHEDRON);
    expect(() => parseBinaryStl(whole.slice(0, 150))).toThrow(/incomplete.*announces 4 triangles but holds only 1/);
    expect(() => parseBinaryStl(new ArrayBuffer(10))).toThrow(StlError);
    const ascii = new TextEncoder().encode("solid sphere\n facet normal 0 0 1\n  outer loop\n" + "   vertex 0 0 0\n".repeat(20));
    expect(() => parseBinaryStl(ascii.buffer as ArrayBuffer)).toThrow(/text \(ASCII\) STL/);
  });

  it("ignores trailing bytes after the announced triangles", () => {
    const whole = new Uint8Array(stl(TETRAHEDRON));
    const padded = new Uint8Array(whole.length + 7);
    padded.set(whole);
    expect(stlTriangleCount(padded.buffer)).toBe(4);
  });
});

describe("the meshes of the recorded sphere run", () => {
  const meshes = fixtureEvents().filter((event) => event.type === "artifact" && String(event.kind).endsWith("_mesh"));

  it("covers all four surfaces", () => {
    expect(meshes.map((event) => event.path)).toEqual([
      "stereo/preview/00-hull/mesh.stl",
      "stereo/preview/01-hull-repaired/mesh.stl",
      "stereo/preview/10-level-0/mesh.stl",
      "mesh/mesh.stl",
    ]);
  });

  it.each(meshes.map((event) => [event.path as string, event.triangles as number] as const))(
    "%s parses to the %d triangles its event announces, closed and consistently oriented",
    (path, triangles) => {
      const buffer = fixtureBytes(path);
      expect(buffer.byteLength).toBe(84 + 50 * triangles);
      const mesh = parseBinaryStl(buffer);
      expect(mesh.triangles).toBe(triangles);
      expect(mesh.dropped).toBe(0);
      expect(mesh.indices.length).toBe(triangles * 3);
      // Euler characteristic of a sphere: V - E + F = 2, and E = 3F/2 for a closed triangle mesh.
      expect(mesh.positions.length / 3 - (triangles * 3) / 2 + triangles).toBe(2);
      expect(closedAndOriented(mesh)).toBe(true);
      for (let axis = 0; axis < 3; axis++) expect(mesh.max[axis]! - mesh.min[axis]!).toBeGreaterThan(0.5);
      expect(meshBytes(mesh)).toBeLessThan(buffer.byteLength);
    },
  );

  it("agrees with the mesh report on the vertex count of the final surface", () => {
    expect(parseBinaryStl(fixtureBytes("mesh/mesh.stl")).positions.length / 3).toBe(16902);
  });

  it("shares one coordinate frame: every surface overlaps the final one", () => {
    const final = parseBinaryStl(fixtureBytes("mesh/mesh.stl"));
    for (const event of meshes) {
      const mesh = parseBinaryStl(fixtureBytes(event.path as string));
      for (let axis = 0; axis < 3; axis++) {
        // Same place (centres agree) and same size (the hull is a little larger; the final has its base cut flat).
        const centre = (mesh.min[axis]! + mesh.max[axis]!) / 2 - (final.min[axis]! + final.max[axis]!) / 2;
        const ratio = (mesh.max[axis]! - mesh.min[axis]!) / (final.max[axis]! - final.min[axis]!);
        expect(Math.abs(centre)).toBeLessThan(0.25);
        expect(ratio).toBeGreaterThan(0.75);
        expect(ratio).toBeLessThan(1.4);
      }
    }
  });

  it("gives the same result in slices as in one go, yielding in between", async () => {
    const whole = parseBinaryStl(fixtureBytes("mesh/mesh.stl"));
    let pauses = 0;
    const fractions: number[] = [];
    const sliced = await parseBinaryStlChunked(
      fixtureBytes("mesh/mesh.stl"),
      async () => {
        pauses += 1;
      },
      (fraction) => fractions.push(fraction),
      5000,
    );
    expect(pauses).toBe(6);
    expect(fractions.at(-1)).toBeLessThan(1);
    expect(sliced.positions).toEqual(whole.positions);
    expect(sliced.indices).toEqual(whole.indices);
    expect(sliced.normals).toEqual(whole.normals);
  });
});
