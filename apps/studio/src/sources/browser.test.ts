import { describe, expect, it } from "vitest";
import type { FromWorker, ToWorker, WorkerLike } from "../browser/protocol";
import { optionErrors, optionsFor, optionTokens } from "../core/providerOptions";
import { initialState, reduce } from "../core/reducer";
import { chosenProviders, importFields, missingFields, parseStartPoints, startBody } from "../core/startPoints";
import { event, fixtureEvents, settle } from "../testing/fixture";
import { downloadName } from "../ui/download";
import { BrowserEngine, describeInputs, PREVIEW_LIMIT_MEGAPIXELS, previewAdvice, relativeFiles } from "./browserEngine";
import { RunStore } from "./runStore";

/** A stand-in for the worker: records what it is sent and lets the test answer. */
class FakeWorker implements WorkerLike {
  onmessage: ((event: { data: FromWorker }) => void) | null = null;
  sent: ToWorker[] = [];
  terminated = false;
  constructor(private readonly webgpu = true) {}
  postMessage(message: ToWorker): void {
    this.sent.push(message);
    if (message.type === "load") {
      queueMicrotask(() =>
        this.answer(
          this.webgpu
            ? { type: "loaded", ok: true, settings: { settings: [{ name: "grid", group: "Hull", meaning: "m", kind: "integer", default: 400 }] }, adapter: "apple metal-3" }
            : { type: "loaded", ok: false, message: "This browser has no WebGPU, which the engine computes with." },
        ),
      );
    }
    if (message.type === "file") {
      const bytes = message.path === "mesh/mesh.stl" ? new Uint8Array([1, 2, 3]).buffer : message.path === "mesh/result.json" ? new TextEncoder().encode('{"closed":true}').buffer : null;
      queueMicrotask(() => this.answer({ type: "file", request: message.request, bytes: bytes as ArrayBuffer | null }));
    }
  }
  answer(message: FromWorker): void {
    this.onmessage?.({ data: message });
  }
  terminate(): void {
    this.terminated = true;
  }
}

function engineWith(worker: FakeWorker) {
  const revoked: string[] = [];
  let made = 0;
  const engine = new BrowserEngine({
    module: "https://host/app/engine/crisp3ds-dense.js",
    createWorker: () => worker,
    clock: () => 1000,
    objectUrls: { create: () => `blob:${++made}`, revoke: (url) => revoked.push(url) },
  });
  return { engine, revoked };
}

const file = (name: string) => ({ name }) as unknown as File;
const picked = (megapixels = 2 * 24) => ({ name: "sphere", files: [["cameras.json", file("cameras.json")]] as [string, File][], views: 24, megapixels });

describe("the engine in this browser", () => {
  it("loads the package from next to the app and reports the adapter and the settings", async () => {
    const worker = new FakeWorker();
    const { engine } = engineWith(worker);
    const health = await engine.health();
    expect(worker.sent[0]).toEqual({ type: "load", module: "https://host/app/engine/crisp3ds-dense.js" });
    expect(health).toMatchObject({ device: "WebGPU: apple metal-3", canStartRuns: true, choosesDevice: false, scoresReference: false });
    expect(health.startPoints?.map((point) => point.id)).toEqual(["inputs"]);
    expect((await engine.settings())[0]?.name).toBe("grid");
    expect(worker.sent).toHaveLength(1); // loaded once
    expect(engine.kind).toBe("browser");
  });

  it("says so plainly when the browser has no WebGPU", async () => {
    const { engine } = engineWith(new FakeWorker(false));
    await expect(engine.health()).rejects.toThrow(/no WebGPU/);
    await expect(engine.startRun({})).rejects.toThrow(/no WebGPU/);
  });

  it("runs on the picked files, feeds the reducer, serves files from memory and revokes its object URLs", async () => {
    const worker = new FakeWorker();
    const { engine, revoked } = engineWith(worker);
    await expect(engine.startRun({})).rejects.toMatchObject({ status: 400, message: expect.stringContaining("choose the inputs folder") });
    engine.setInputs(picked());
    const id = await engine.startRun({ settings: { grid: 96 }, name: "my sphere!" });
    expect(id).toBe("browser-01-my-sphere-");
    const sent = worker.sent.at(-1) as Extract<ToWorker, { type: "run" }>;
    expect(sent).toMatchObject({ type: "run", id, options: { live_previews: true, settings: { grid: 96 } } });
    expect(sent.files.map(([path]) => path)).toEqual(["cameras.json"]);
    await expect(engine.startRun({})).rejects.toThrow(/already in progress/);

    const source = engine.openRun(id);
    const store = new RunStore(source);
    source.start();
    const events = fixtureEvents();
    for (const recorded of events.slice(0, 6)) worker.answer({ type: "event", id, event: recorded, wasmBytes: 40_000_000 });
    expect(store.get().run.status).toBe("running");
    expect(store.get().run.meshes.map((mesh) => mesh.label)).toEqual(["Silhouette hull"]);
    expect((await engine.listRuns())[0]).toMatchObject({ id, status: "running", stage: "stereo", events: 6 });

    for (const recorded of events.slice(6)) worker.answer({ type: "event", id, event: recorded, wasmBytes: 47_000_000 });
    worker.answer({ type: "finished", id, ok: true, seconds: 12, peakWasmBytes: 49_000_000 });
    expect(store.get().run.status).toBe("complete");
    expect(store.get().run.eventCount).toBe(events.length);
    expect(store.get().link.state).toBe("ended");
    expect(engine.peakMemory(id)).toBe(49_000_000);
    expect((await engine.listRuns())[0]).toMatchObject({ status: "complete", stage: null });

    expect(new Uint8Array(await source.fetchBytes("mesh/mesh.stl"))).toEqual(new Uint8Array([1, 2, 3]));
    expect(await source.fetchJson("mesh/result.json")).toEqual({ closed: true });
    await expect(source.fetchBytes("nope.png")).rejects.toMatchObject({ status: 404 });
    expect(await source.imageUrl("mesh/mesh.stl")).toBe("blob:1");
    expect(await source.imageUrl("mesh/mesh.stl")).toBe("blob:1");
    source.stop();
    expect(revoked).toEqual(["blob:1"]);

    // A second view of the same run gets every event again, from memory.
    const again = engine.openRun(id);
    const second = new RunStore(again);
    again.start();
    expect(second.get().run.eventCount).toBe(events.length);

    // Removing a finished run frees it in the worker; a new run may then start.
    engine.remove(id);
    expect(worker.sent.at(-1)).toEqual({ type: "dispose", id });
    expect(await engine.listRuns()).toEqual([]);
    const gone = engine.openRun(id);
    const links: string[] = [];
    gone.subscribe((update) => update.type === "link" && links.push(update.link.state));
    gone.start();
    expect(links.at(-1)).toBe("failed");
  });

  it("cancels through the worker and closes a run the engine could not even start", async () => {
    const worker = new FakeWorker();
    const { engine } = engineWith(worker);
    engine.setInputs(picked(), false);
    const id = await engine.startRun({});
    expect((worker.sent.at(-1) as Extract<ToWorker, { type: "run" }>).options).toEqual({ live_previews: false });
    const source = engine.openRun(id);
    const store = new RunStore(source);
    source.start();
    await source.cancel!();
    expect(worker.sent.at(-1)).toEqual({ type: "cancel", id });
    worker.answer({ type: "finished", id, ok: false, message: "out of memory", seconds: 3, peakWasmBytes: 1 });
    await settle();
    expect(store.get().run.status).toBe("failed");
    expect(store.get().run.errors.map((error) => error.message)).toEqual(["out of memory"]);
    // The next run may start now.
    expect(await engine.startRun({})).toBe("browser-02-sphere");
  });

  it("turns intermediate surfaces off above the memory threshold and says why", () => {
    expect(previewAdvice(24, 0.4)).toMatchObject({ on: true });
    expect(previewAdvice(30, PREVIEW_LIMIT_MEGAPIXELS).on).toBe(true);
    const bunny = previewAdvice(73, 146);
    expect(bunny.on).toBe(false);
    expect(bunny.reason).toContain("73 photos, 146 megapixels");
    expect(bunny.reason).toContain("4 GiB");
  });

  it("reads a picked folder: paths inside it, hidden files dropped, size from cameras.json", () => {
    const entries = ["bunny/cameras.json", "bunny/images/b.png", "bunny/images/a.png", "bunny/.DS_Store", "bunny/masks/.hidden/x.png", "bunny"].map((path) => ({ path, file: file(path) }));
    const files = relativeFiles(entries);
    expect(files.map(([path]) => path)).toEqual(["cameras.json", "images/a.png", "images/b.png"]);
    const cameras = JSON.stringify({ views: [{ width: 1600, height: 1250 }, { width: 1600, height: 1250 }, { name: "no size" }] });
    expect(describeInputs("bunny", files, cameras)).toMatchObject({ name: "bunny", views: 3, megapixels: 4 });
    // Photos and masks the cameras name must be in the folder: a browser reads nothing else.
    const named = (image: string) => JSON.stringify({ views: [{ width: 10, height: 10, image, mask: "images/b.png" }] });
    expect(describeInputs("bunny", files, named("./images/a.png")).views).toBe(1);
    expect(() => describeInputs("bunny", files, named("images/missing.png"))).toThrow(/names 1 file that is not in this folder \(the first: images\/missing\.png\)/);
    expect(() => describeInputs("bunny", files, named("/disk/prepared/1.png"))).toThrow(/full paths elsewhere on a disk/);
    expect(() => describeInputs("x", files, "{}")).toThrow(/not an inputs folder/);
    expect(() => describeInputs("x", files, "not json")).toThrow(/not an inputs folder/);
  });

  it("names downloads after the run and what the file is", () => {
    expect(downloadName("browser-01-bunny", "mesh/mesh.stl")).toBe("browser-01-bunny-mesh.stl");
    expect(downloadName("r 1", "stereo/depth-merged.png")).toBe("r-1-stereo-depth-merged.png");
    expect(downloadName("", "input-sheet.png")).toBe("input-sheet.png");
  });
});

describe("the photos start", () => {
  const described = [
    {
      id: "photos",
      label: "Turntable photos",
      fields: [{ key: "photos", label: "Photos", kind: "folder", required: true }, { key: "calibration", label: "Lens calibration (.json)", kind: "file", required: true }],
      providers: [
        {
          module: "masks",
          label: "Masks",
          default: "external-sam",
          options: [
            { id: "threshold", label: "threshold", available: true, reason: null, external: [], license: "this crate (AGPL-3.0-only)" },
            { id: "import", label: "import", available: true, external: [] },
            { id: "external-sam", label: "external-sam", available: false, reason: "SAM is not set up: the Python interpreter missing. See Tools.", external: ["Python interpreter with PyTorch"] },
          ],
        },
        {
          module: "cameras",
          label: "Cameras",
          default: "alicevision",
          options: [{ id: "alicevision", external: ["aliceVision_cameraInit"], license: "AliceVision MPL-2.0" }, { id: "colmap", available: false, reason: "COLMAP is not set up." }, { id: "import" }],
        },
      ],
    },
  ];
  const point = parseStartPoints(described)[0]!;

  it("keeps providers that cannot run, with the reason, and defaults to one that can", () => {
    const [masks, cameras] = point.providers;
    expect(masks!.options.map((option) => [option.id, option.available])).toEqual([["threshold", true], ["import", true], ["external-sam", false]]);
    expect(masks!.options[2]!.reason).toContain("SAM is not set up");
    expect(masks!.default).toBe("threshold");
    expect(cameras!.default).toBe("alicevision");
    expect(cameras!.options[0]).toMatchObject({ external: ["aliceVision_cameraInit"], license: "AliceVision MPL-2.0", available: true });
    expect(chosenProviders(point, {})).toEqual({ masks: "threshold", cameras: "alicevision" });
    expect(chosenProviders(point, { cameras: "import", masks: "nonsense" })).toEqual({ masks: "threshold", cameras: "import" });
  });

  it("builds the request the engine expects: fields, providers, only the options that differ", () => {
    const form = {
      name: "bunny",
      device: "",
      reference: "",
      values: { photos: "rgb", calibration: "/repo/scripts/turntable_mesh/calibrations/3dlf-pro.json", masks_import: "left over" },
      providers: { masks: "threshold", cameras: "alicevision" },
      options: { "threshold-level": "otsu", "alicevision-describer-preset": "high", "alicevision-sfm-option": "--maxIter  50", "colmap-matching": "ring", "random-seed": " 0 " },
    };
    expect(startBody(point, form, { grid: 320 })).toEqual({
      photos: "rgb",
      calibration: "/repo/scripts/turntable_mesh/calibrations/3dlf-pro.json",
      name: "bunny",
      providers: { masks: "threshold", cameras: "alicevision" },
      photo_options: ["--alicevision-describer-preset", "high", "--alicevision-sfm-option", "--maxIter", "--alicevision-sfm-option", "50"],
      settings: { grid: 320 },
    });
    // Everything at its default: no option words at all.
    expect(startBody(point, { ...form, options: {} }, {})).not.toHaveProperty("photo_options");
  });

  it("asks for the place of imported masks and cameras", () => {
    const providers = { masks: "import", cameras: "import" };
    expect(importFields(point, providers).map((entry) => entry.spec.key)).toEqual(["masks_import", "cameras_import"]);
    expect(missingFields(point, { photos: "rgb", calibration: "lens.json" }, providers)).toEqual(["masks_import", "cameras_import"]);
    const body = startBody(point, { name: "", device: "", reference: "", values: { photos: "rgb", calibration: "lens.json", masks_import: "bunny/masks", cameras_import: "bunny/final.sfm" }, providers }, {});
    expect(body).toMatchObject({ masks_import: "bunny/masks", cameras_import: "bunny/final.sfm", providers });
    expect(missingFields(point, { photos: "rgb", calibration: "lens.json" }, {})).toEqual([]);
  });

  it("writes booleans and repeated options the way the photos stage reads them", () => {
    expect(optionTokens({ masks: "external-sam", cameras: "colmap" }, { "sam-multimask": false, "sam-preserve-holes": true, "colmap-masks": false, "colmap-overlap": "5", "sam-device": "cpu" })).toEqual([
      "--colmap-overlap",
      "5",
      "--colmap-masks",
      "off",
      "--sam-device",
      "cpu",
      "--no-sam-multimask",
    ]);
    // The same flag has another default under another provider.
    expect(optionTokens({ masks: "external-sam" }, { "threshold-level": "otsu" })).toEqual(["--threshold-level", "otsu"]);
    expect(optionTokens({ masks: "threshold" }, { "threshold-level": "70" })).toEqual(["--threshold-level", "70"]);
    expect(optionsFor("masks", "import").map((spec) => spec.flag)).toEqual(["hole-cleanup-budget"]);
  });

  it("catches typing slips before sending and never names a program or a place", () => {
    expect(optionErrors({ cameras: "colmap" }, { "colmap-overlap": "many", "colmap-max-features": "4096", "clahe-clip": "x", "colmap-matching": "spiral" })).toEqual({
      "colmap-overlap": "Needs a whole number.",
      "clahe-clip": "Needs a number.",
      "colmap-matching": "Must be one of: exhaustive, ring, sequential.",
    });
    expect(optionTokens({ cameras: "colmap" }, { "colmap-overlap": "many" })).toEqual([]);
    const reserved = ["photos", "output", "events", "calibration", "masks", "cameras", "python", "alicevision", "alicevision-library-path", "colmap", "sam-python", "sam-source", "sam-checkpoint", "sam-repository", "stop-after"];
    for (const module of ["masks", "cameras"]) {
      for (const provider of ["threshold", "import", "external-sam", "alicevision", "colmap"]) {
        for (const spec of optionsFor(module, provider)) expect(reserved).not.toContain(spec.flag);
      }
    }
  });
});

describe("a run that started from photos, in the reducer", () => {
  it("puts masks and cameras first, in that order, whenever they turn up", () => {
    let state = reduce(initialState(), event("run_started"));
    state = reduce(state, event("stage_started", { stage: "cameras", provider: "alicevision" }, 1));
    state = reduce(state, event("stage_started", { stage: "masks", provider: "threshold" }, 2));
    state = reduce(state, event("stage_started", { stage: "texture" }, 3));
    expect(state.stages.map((stage) => stage.name)).toEqual(["masks", "cameras", "inputs", "stereo", "mesh", "check", "evaluate", "texture"]);
    // A run from an inputs folder has neither.
    expect(initialState().stages.map((stage) => stage.name)).toEqual(["inputs", "stereo", "mesh", "check", "evaluate"]);
  });

  it("shows the mask sheet, the sparse overlay and the stage's metrics", () => {
    let state = reduce(initialState(), event("run_started"));
    state = reduce(state, event("stage_started", { stage: "masks", provider: "threshold" }, 1));
    state = reduce(state, event("metric", { stage: "masks", name: "mask_area_median_fraction", value: 0.15893 }, 2));
    state = reduce(state, event("artifact", { stage: "masks", kind: "mask_sheet", path: "frontend/mask-contact-sheet.png", label: "Masks on photos (red: dark pixels left out)" }, 3));
    state = reduce(state, event("stage_finished", { stage: "masks", seconds: 21.4 }, 4));
    state = reduce(state, event("stage_started", { stage: "cameras", provider: "alicevision" }, 5));
    state = reduce(state, event("artifact", { stage: "cameras", kind: "sparse_overlay", path: "frontend/sparse-overlay.png", label: "Sparse points on undistorted photos and masks" }, 6));
    expect(state.sheets.map((sheet) => [sheet.kind, sheet.stage])).toEqual([["mask_sheet", "masks"], ["sparse_overlay", "cameras"]]);
    expect(state.metrics).toEqual([{ name: "mask_area_median_fraction", value: 0.15893, stage: "masks", seq: 2 }]);
    expect(state.stages.slice(0, 2).map((stage) => [stage.name, stage.status])).toEqual([["masks", "done"], ["cameras", "running"]]);
    expect(state.ignored).toBe(0);
  });
});
