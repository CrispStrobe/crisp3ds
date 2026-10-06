import { describe, expect, it } from "vitest";
import { chooseMesh, LatestOnly } from "./meshChoice";
import type { MeshStep } from "./reducer";

const step = (seq: number, kind: MeshStep["kind"], path: string, triangles?: number): MeshStep =>
  ({ seq, kind, path, label: path, triangles, stage: kind === "final_mesh" ? "mesh" : "stereo" }) as MeshStep;

const hull = step(4, "preview_mesh", "stereo/preview/00-hull/mesh.stl", 9568);
const level = step(14, "preview_mesh", "stereo/preview/10-level-0/mesh.stl", 8388);
const final = step(25, "final_mesh", "mesh/mesh.stl", 33800);
const small = () => false;

describe("which surface the view shows", () => {
  it("is the same however the steps arrived: fast replays deliver them in one batch, slow ones one by one", () => {
    const oneByOne = [hull, level, final];
    const outOfOrder = [final, hull, level];
    for (const meshes of [oneByOne, outOfOrder]) {
      const choice = chooseMesh(meshes, null, small);
      expect(choice.target?.path).toBe(final.path);
      expect(choice.following).toBe(true);
    }
    // Along the way, the newest by line number so far.
    expect(chooseMesh([hull], null, small).target).toBe(hull);
    expect(chooseMesh([level, hull], null, small).target).toBe(level);
  });

  it("keeps the newest preview up while a big final surface waits to be asked for", () => {
    const big = (s: MeshStep) => s.kind === "final_mesh";
    const choice = chooseMesh([hull, final, level], null, big);
    expect(choice.selected).toBe(final);
    expect(choice.awaitingConsent).toBe(true);
    expect(choice.target).toBe(level);
    // A surface picked by hand stays, and a big one picked by hand shows nothing until asked for.
    expect(chooseMesh([hull, level, final], hull.path, big).target).toBe(hull);
    expect(chooseMesh([hull, level, final], final.path, big)).toMatchObject({ target: undefined, following: false });
    // A picked surface that is gone (a replay scrubbed back) means following again.
    expect(chooseMesh([hull], final.path, small)).toMatchObject({ target: hull, following: true });
  });
});

describe("loads that finish out of order", () => {
  it("only the newest load may put its surface up", async () => {
    const latest = new LatestOnly();
    const shown: string[] = [];
    const load = (path: string, delay: number) => {
      const ticket = latest.begin();
      return new Promise<void>((resolve) =>
        setTimeout(() => {
          if (latest.isCurrent(ticket)) shown.push(path);
          resolve();
        }, delay),
      );
    };
    // The level preview is slow to parse; the final surface was asked for after it and is quick.
    await Promise.all([load(level.path, 30), load(final.path, 5)]);
    expect(shown).toEqual([final.path]);
  });
});
