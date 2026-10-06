/**
 * "This browser": the dense pipeline running in a worker of this page (WebAssembly and
 * WebGPU, the package of crates/dense/web). The fourth way to get events and files,
 * behind the same interfaces as the others. Runs live in memory and are gone when the
 * page is closed.
 *
 * No DOM here: the worker is made by an injected factory, object URLs by injected
 * functions, so this is tested with a stand-in worker.
 */

import { normaliseEvent, type RunEvent } from "../core/events";
import { parseSettingsSchema, type SettingSpec } from "../core/settings";
import type { StartPoint } from "../core/startPoints";
import type { FromWorker, WorkerLike } from "../browser/protocol";
import {
  EngineError,
  type Engine,
  type EngineHealth,
  type FetchOptions,
  type LinkStatus,
  type RunSource,
  type RunSummary,
  type SourceUpdate,
} from "./types";

/**
 * Live previews cost memory in a browser: each preview volume is written and meshed
 * inside the 4 GiB that WebAssembly can address. Measured on 73 photos of 2 megapixels
 * (146 megapixels in all): 1.9 GiB at the peak without previews, 2.5 GiB with them.
 * Up to this many megapixels in all, previews stay within about 1 GiB and are on.
 */
export const PREVIEW_LIMIT_MEGAPIXELS = 60;

export interface PickedInputs {
  /** Name of the picked folder. */
  name: string;
  /** Path inside the inputs folder, and the file. */
  files: [string, File][];
  views: number;
  /** Sum over all photos, from cameras.json. */
  megapixels: number;
}

/** 0.4, 3.9, 146: megapixels as a person would say them. */
export function megapixelText(megapixels: number): string {
  return megapixels < 10 ? megapixels.toFixed(1) : String(Math.round(megapixels));
}

/** Whether live previews are on by default for these inputs, and why. */
export function previewAdvice(views: number, megapixels: number): { on: boolean; reason: string } {
  const size = `${views} photos, ${megapixelText(megapixels)} megapixels in all`;
  if (megapixels <= PREVIEW_LIMIT_MEGAPIXELS) {
    return { on: true, reason: `${size}: small enough to show the surface while it is computed.` };
  }
  return {
    on: false,
    reason: `${size}: intermediate surfaces are off, because they need about a third more memory and a browser gives the engine at most 4 GiB. Only the final surface is shown.`,
  };
}

/**
 * Files of a picked folder as paths inside it. `entries` carry each file's path including
 * the folder itself ("bunny/cameras.json"), as browsers report it; hidden files are dropped.
 */
export function relativeFiles(entries: { path: string; file: File }[]): [string, File][] {
  const files: [string, File][] = [];
  for (const { path, file } of entries) {
    const parts = path.split("/").filter((part) => part !== "");
    const inside = parts.slice(1);
    if (inside.length === 0 || inside.some((part) => part.startsWith("."))) continue;
    files.push([inside.join("/"), file]);
  }
  return files.sort((a, b) => a[0].localeCompare(b[0]));
}

/** Reads what the form needs from the picked folder's cameras.json. Throws a sentence if it is not an inputs folder. */
export function describeInputs(name: string, files: [string, File][], camerasJson: string): PickedInputs {
  let views: { width?: unknown; height?: unknown; image?: unknown; mask?: unknown }[];
  try {
    const parsed = JSON.parse(camerasJson) as { views?: unknown };
    if (!Array.isArray(parsed.views)) throw new Error();
    views = parsed.views as typeof views;
  } catch {
    throw new Error("cameras.json in this folder cannot be read; it is not an inputs folder.");
  }
  let pixels = 0;
  // A browser reads only the chosen folder: every photo and mask the cameras name has to be in it.
  const present = new Set(files.map(([path]) => path));
  const outside: string[] = [];
  for (const view of views) {
    if (typeof view.width === "number" && typeof view.height === "number") pixels += view.width * view.height;
    for (const named of [view.image, view.mask]) {
      if (typeof named === "string" && !present.has(named.replace(/^\.\//, ""))) outside.push(named);
    }
  }
  if (outside.length > 0) {
    const absolute = outside.some((path) => path.startsWith("/") || /^[A-Za-z]:[\\/]/.test(path));
    throw new Error(
      `cameras.json names ${outside.length} ${outside.length === 1 ? "file that is" : "files that are"} not in this folder (the first: ${outside[0]}). ` +
        (absolute ? "They are given by full paths elsewhere on a disk. " : "") +
        "A browser can only read the folder you chose: put the photos and masks inside it and name them relative to it.",
    );
  }
  return { name, files, views: views.length, megapixels: pixels / 1e6 };
}

const BROWSER_START: StartPoint[] = [
  {
    id: "inputs",
    label: "Inputs folder",
    meaning: "A folder on this device with cameras.json, the undistorted photos and their masks. It is read here; nothing is uploaded.",
    fields: [{ key: "inputs", label: "Inputs folder", kind: "inputs", required: true, help: "A folder with cameras.json, the photos and their masks." }],
    providers: [],
    optionGroups: [],
  },
];

interface Held {
  id: string;
  events: RunEvent[];
  status: string;
  started: number;
  stage: string | null;
  fraction: number;
  /** Peak WebAssembly memory seen for this run, bytes. */
  peak: number;
  ended: boolean;
  listeners: Set<() => void>;
}

export interface BrowserOptions {
  /** Absolute URL of `crisp3ds-dense.js` of the engine package. */
  module: string;
  createWorker(): WorkerLike;
  /** Wall clock in Unix seconds. */
  clock?: () => number;
  objectUrls?: { create(blob: Blob): string; revoke(url: string): void };
}

const IMAGE_TYPES: Record<string, string> = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg" };

export class BrowserEngine implements Engine {
  readonly kind = "browser" as const;
  readonly label = "this browser";
  private worker: WorkerLike | null = null;
  private loaded: Promise<{ settings: unknown; adapter: string | null }> | null = null;
  private resolveLoaded: ((value: { settings: unknown; adapter: string | null }) => void) | null = null;
  private rejectLoaded: ((problem: Error) => void) | null = null;
  private runs = new Map<string, Held>();
  private requests = new Map<number, (bytes: ArrayBuffer | null) => void>();
  private nextRequest = 1;
  private counter = 0;
  private picked: PickedInputs | null = null;
  private previews: boolean | null = null;
  private readonly clock: () => number;

  constructor(private readonly options: BrowserOptions) {
    this.clock = options.clock ?? (() => Date.now() / 1000);
  }

  /** The folder a run will start from, and whether intermediate surfaces are wanted (null: by the advice). */
  setInputs(picked: PickedInputs | null, livePreviews: boolean | null = null): void {
    this.picked = picked;
    this.previews = livePreviews;
  }

  inputs(): PickedInputs | null {
    return this.picked;
  }

  /** Peak WebAssembly memory of a run in bytes, as far as seen. */
  peakMemory(id: string): number {
    return this.runs.get(id)?.peak ?? 0;
  }

  private load(): Promise<{ settings: unknown; adapter: string | null }> {
    if (this.loaded === null) {
      this.loaded = new Promise((resolve, reject) => {
        this.resolveLoaded = resolve;
        this.rejectLoaded = reject;
      });
      try {
        const worker = this.options.createWorker();
        worker.onmessage = ({ data }) => this.receive(data);
        worker.onerror = () => this.rejectLoaded?.(new EngineError("The engine could not be started in this browser.", 0));
        this.worker = worker;
        worker.postMessage({ type: "load", module: this.options.module });
      } catch {
        this.rejectLoaded?.(new EngineError("This browser cannot start a worker for the engine.", 0));
      }
    }
    return this.loaded;
  }

  private receive(message: FromWorker): void {
    if (message.type === "loaded") {
      if (message.ok) this.resolveLoaded?.({ settings: message.settings, adapter: message.adapter });
      else this.rejectLoaded?.(new EngineError(message.message, 0));
    } else if (message.type === "event") {
      const run = this.runs.get(message.id);
      if (run === undefined) return;
      const event = normaliseEvent(message.event, run.events.length);
      if (event === null) return;
      // Line numbers as in a log on disk: the position in the list.
      run.events.push({ ...event, seq: run.events.length });
      run.peak = Math.max(run.peak, message.wasmBytes);
      if (event.type === "run_started") run.status = "running";
      else if (event.type === "stage_started") {
        run.stage = event.stage;
        run.fraction = 0;
      } else if (event.type === "progress" && typeof event.fraction === "number") run.fraction = event.fraction;
      else if (event.type === "run_finished") {
        run.status = typeof event.status === "string" ? event.status : "failed";
        if (run.status === "complete") run.stage = null;
      }
      for (const listener of [...run.listeners]) listener();
    } else if (message.type === "finished") {
      const run = this.runs.get(message.id);
      if (run === undefined) return;
      run.peak = Math.max(run.peak, message.peakWasmBytes);
      // The engine closes its own log. If it could not even start one, say why here.
      if (!run.events.some((event) => event.type === "run_finished")) {
        const time = this.clock();
        const add = (type: string, fields: Record<string, unknown>) => run.events.push({ type, time, stage: null, seq: run.events.length, ...fields });
        if (run.events.length === 0) add("run_started", { schema: "crisp3ds_dense_events_v1", device: "WebGPU" });
        add("error", { message: message.message ?? "The engine stopped without saying why." });
        add("run_finished", { status: "failed", seconds: message.seconds });
        run.status = "failed";
      }
      run.ended = true;
      for (const listener of [...run.listeners]) listener();
    } else if (message.type === "file") {
      this.requests.get(message.request)?.(message.bytes);
      this.requests.delete(message.request);
    }
  }

  async health(): Promise<EngineHealth> {
    const { adapter } = await this.load();
    return {
      schema: "crisp3ds_dense_events_v1",
      device: adapter === null ? "WebGPU" : `WebGPU: ${adapter}`,
      canStartRuns: true,
      startPoints: BROWSER_START,
      choosesDevice: false,
      scoresReference: false,
    };
  }

  async settings(): Promise<SettingSpec[]> {
    return parseSettingsSchema((await this.load()).settings);
  }

  listRuns(): Promise<RunSummary[]> {
    const runs = [...this.runs.values()].map((run) => ({
      id: run.id,
      status: run.status,
      started: run.started,
      stage: run.stage,
      stageFraction: run.fraction,
      events: run.events.length,
    }));
    return Promise.resolve(runs.sort((a, b) => b.started - a.started || b.id.localeCompare(a.id)));
  }

  async startRun(body: Record<string, unknown>): Promise<string> {
    await this.load();
    if (this.picked === null) throw new EngineError("inputs: choose the inputs folder first", 400);
    if ([...this.runs.values()].some((run) => !run.ended)) {
      throw new EngineError("A run is already in progress in this browser. Wait for it, or cancel it.", 400);
    }
    const name = typeof body.name === "string" && body.name.trim() !== "" ? body.name.trim() : this.picked.name;
    const id = `browser-${String(++this.counter).padStart(2, "0")}-${name.replace(/[^A-Za-z0-9._-]+/g, "-").slice(0, 40) || "run"}`;
    const previews = this.previews ?? previewAdvice(this.picked.views, this.picked.megapixels).on;
    const options: Record<string, unknown> = { live_previews: previews };
    if (typeof body.settings === "object" && body.settings !== null) options.settings = body.settings;
    this.runs.set(id, { id, events: [], status: "running", started: this.clock(), stage: null, fraction: 0, peak: 0, ended: false, listeners: new Set() });
    this.worker?.postMessage({ type: "run", id, files: this.picked.files, options });
    return id;
  }

  openRun(id: string): RunSource {
    return new BrowserRunSource(this, id, this.options.objectUrls, this.clock);
  }

  /** For the run source. */
  held(id: string): Held | undefined {
    return this.runs.get(id);
  }

  cancel(id: string): void {
    this.worker?.postMessage({ type: "cancel", id });
  }

  file(id: string, path: string): Promise<ArrayBuffer | null> {
    if (this.worker === null || !this.runs.has(id)) return Promise.resolve(null);
    const request = this.nextRequest++;
    return new Promise((resolve) => {
      this.requests.set(request, resolve);
      this.worker?.postMessage({ type: "file", request, id, path });
    });
  }

  /** Forgets a finished run and frees what it holds in the engine's memory. */
  remove(id: string): void {
    if (this.runs.get(id)?.ended !== true) return;
    this.worker?.postMessage({ type: "dispose", id });
    this.runs.delete(id);
  }
}

export class BrowserRunSource implements RunSource {
  readonly kind = "browser" as const;
  private listeners = new Set<(update: SourceUpdate) => void>();
  private link: LinkStatus = { state: "connecting" };
  private delivered = 0;
  private detach: (() => void) | null = null;
  private images = new Map<string, Promise<string>>();
  private created: string[] = [];
  private readonly objectUrls: NonNullable<BrowserOptions["objectUrls"]>;

  constructor(
    private readonly engine: BrowserEngine,
    readonly title: string,
    objectUrls: BrowserOptions["objectUrls"],
    private readonly clock: () => number,
  ) {
    this.objectUrls = objectUrls ?? { create: (blob) => URL.createObjectURL(blob), revoke: (url) => URL.revokeObjectURL(url) };
  }

  subscribe(listener: (update: SourceUpdate) => void): () => void {
    this.listeners.add(listener);
    listener({ type: "link", link: this.link });
    return () => this.listeners.delete(listener);
  }

  start(): void {
    const run = this.engine.held(this.title);
    if (run === undefined) {
      this.setLink({ state: "failed", message: "This run is no longer in this browser's memory. Runs made here are gone when the page is reloaded." });
      return;
    }
    const deliver = () => {
      if (run.events.length > this.delivered) {
        const events = run.events.slice(this.delivered);
        this.delivered = run.events.length;
        this.emit({ type: "events", events });
      }
      this.setLink({ state: run.events.some((event) => event.type === "run_finished") ? "ended" : "live" });
    };
    run.listeners.add(deliver);
    this.detach = () => run.listeners.delete(deliver);
    deliver();
  }

  stop(): void {
    this.detach?.();
    this.detach = null;
    this.delivered = 0;
    for (const url of this.created) this.objectUrls.revoke(url);
    this.created = [];
    this.images.clear();
  }

  now(): number {
    return this.clock();
  }

  imageUrl(path: string): Promise<string> {
    let pending = this.images.get(path);
    if (pending === undefined) {
      pending = this.fetchBytes(path).then((bytes) => {
        const extension = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
        const url = this.objectUrls.create(new Blob([bytes], { type: IMAGE_TYPES[extension] ?? "application/octet-stream" }));
        this.created.push(url);
        return url;
      });
      pending.catch(() => this.images.delete(path));
      this.images.set(path, pending);
    }
    return pending;
  }

  async fetchBytes(path: string, options: FetchOptions = {}): Promise<ArrayBuffer> {
    const bytes = await this.engine.file(this.title, path);
    options.signal?.throwIfAborted();
    if (bytes === null) throw new EngineError("Not found.", 404);
    options.onProgress?.(bytes.byteLength, bytes.byteLength);
    return bytes;
  }

  async fetchJson(path: string, options: FetchOptions = {}): Promise<unknown> {
    return JSON.parse(new TextDecoder().decode(await this.fetchBytes(path, options)));
  }

  /** Peak memory of the engine for this run, for the status line. */
  memoryNote(): string | undefined {
    const peak = this.engine.peakMemory(this.title);
    return peak > 0 ? `engine memory at the peak ${(peak / 2 ** 30).toFixed(peak >= 2 ** 30 ? 1 : 2)} GiB` : undefined;
  }

  cancel(): Promise<void> {
    this.engine.cancel(this.title);
    return Promise.resolve();
  }

  private setLink(link: LinkStatus): void {
    if (link.state === this.link.state && link.message === this.link.message) return;
    this.link = link;
    this.emit({ type: "link", link });
  }

  private emit(update: SourceUpdate): void {
    for (const listener of [...this.listeners]) listener(update);
  }
}
