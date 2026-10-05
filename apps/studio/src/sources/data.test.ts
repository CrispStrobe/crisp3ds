import { describe, expect, it } from "vitest";
import { cleanPath, crumbs, joinPath, listable, parentPath, splitTyped } from "../core/paths";
import { fakeFetch, json } from "../testing/fixture";
import { HttpEngine } from "./httpEngine";
import { ReplaySource } from "./replaySource";

describe("paths typed into a field", () => {
  it("cleans what people type", () => {
    expect(cleanPath(" /dragon//inputs/ ")).toBe("dragon/inputs");
    expect(cleanPath("dragon\\inputs")).toBe("dragon/inputs");
    expect(cleanPath("./a/./b")).toBe("a/b");
    expect(cleanPath("")).toBe("");
  });

  it("splits the typed text into the folder to list and the name being typed", () => {
    expect(splitTyped("")).toEqual({ parent: "", leaf: "" });
    expect(splitTyped("dra")).toEqual({ parent: "", leaf: "dra" });
    expect(splitTyped("dragon/")).toEqual({ parent: "dragon", leaf: "" });
    expect(splitTyped("dragon/in")).toEqual({ parent: "dragon", leaf: "in" });
    expect(splitTyped("/a//b/c")).toEqual({ parent: "a/b", leaf: "c" });
  });

  it("joins, climbs and makes breadcrumbs", () => {
    expect(joinPath("", "sphere")).toBe("sphere");
    expect(joinPath("a/b", "c")).toBe("a/b/c");
    expect(parentPath("a/b/c")).toBe("a/b");
    expect(parentPath("a")).toBe("");
    expect(crumbs("a/b/c")).toEqual([
      { name: "a", path: "a" },
      { name: "b", path: "a/b" },
      { name: "c", path: "a/b/c" },
    ]);
    expect(crumbs("")).toEqual([]);
  });

  it("never lists a path that climbs out", () => {
    expect(listable("a/../b")).toBe(false);
    expect(listable("..")).toBe(false);
    expect(listable("a/b..c")).toBe(true);
  });
});

describe("GET /api/data", () => {
  it("lists a folder, tolerating rows and fields it does not know", async () => {
    const { fetcher, requests } = fakeFetch(() =>
      json({
        path: "bunny",
        entries: [
          { name: "inputs", directory: true, inputs: true, views: 36 },
          { name: "masks", directory: true, inputs: false },
          { name: "scan.ply", directory: false, inputs: false },
          { directory: true },
          "junk",
        ],
      }),
    );
    const listing = await new HttpEngine("http://e", { fetcher, token: "t" }).listData("bunny");
    expect(listing).toEqual({
      path: "bunny",
      entries: [
        { name: "inputs", directory: true, inputs: true },
        { name: "masks", directory: true, inputs: false },
        { name: "scan.ply", directory: false, inputs: false },
      ],
    });
    expect(requests[0]).toMatchObject({ url: "http://e/api/data?path=bunny", headers: { Authorization: "Bearer t" } });
  });

  it("encodes the path and passes a refusal on", async () => {
    const { fetcher, requests } = fakeFetch(() => json({ error: "no such folder under the data directory" }, 400));
    await expect(new HttpEngine("http://e", { fetcher }).listData("my scans/a&b")).rejects.toThrow(/no such folder/);
    expect(requests[0]?.url).toBe("http://e/api/data?path=my%20scans%2Fa%26b");
  });
});

describe("file size by HEAD", () => {
  it("reads Content-Length without downloading", async () => {
    const { fetcher, requests } = fakeFetch(() => new Response(null, { headers: { "Content-Length": "50700084" } }));
    const source = new HttpEngine("http://e", { fetcher, token: "t" }).openRun("r1");
    expect(await source.fileSize!("mesh/mesh.stl")).toBe(50700084);
    expect(requests[0]).toMatchObject({ method: "HEAD", url: "http://e/api/runs/r1/files/mesh/mesh.stl", headers: { Authorization: "Bearer t" } });
  });

  it("is undefined when the server cannot say", async () => {
    const answers = [
      () => new Response(null, { status: 501 }),
      () => new Response(null),
      () => new Response(null, { headers: { "Content-Length": "100", "Content-Encoding": "gzip" } }),
      () => Promise.reject(new TypeError("Failed to fetch")),
    ];
    for (const answer of answers) {
      const source = new ReplaySource("https://host/b/", { fetcher: answer as never });
      expect(await source.fileSize("mesh/mesh.stl")).toBeUndefined();
    }
  });
});
