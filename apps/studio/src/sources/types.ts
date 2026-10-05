/**
 * The one interface the UI talks to, whatever is behind it: a recorded bundle (replay),
 * an engine over HTTP, or -- later -- the local engine of the Tauri shell.
 *
 * A `RunSource` delivers the events of one run and its files. An `Engine` lists and
 * starts runs and hands out `RunSource`s. Nothing here touches the DOM.
 */

import type { RunEvent } from "../core/events";
import type { ReplaySnapshot, ReplaySpeed } from "../core/replay";
import type { SettingSpec } from "../core/settings";

export type SourceKind = "replay" | "http" | "local";

export type LinkState =
  /** First request not answered yet. */
  | "connecting"
  /** Events are flowing (or the recording is loaded). */
  | "live"
  /** A request failed; trying again with backoff. `message` says why. */
  | "retrying"
  /** The run has finished; nothing more will come. */
  | "ended"
  /** Gave up: unknown run, refused token, unreadable bundle. `message` says why. */
  | "failed";

export interface LinkStatus {
  state: LinkState;
  message?: string;
}

export type SourceUpdate =
  /** New events, in order, to append. */
  | { type: "events"; events: RunEvent[] }
  /** Forget everything and start again from these events (replay scrubbing backwards). */
  | { type: "reset"; events: RunEvent[] }
  | { type: "link"; link: LinkStatus };

export interface FetchOptions {
  signal?: AbortSignal;
  /** Bytes received so far and the total when the server said it. */
  onProgress?: (received: number, total: number | undefined) => void;
}

export interface ReplayControls {
  snapshot(): ReplaySnapshot;
  subscribe(listener: (snapshot: ReplaySnapshot) => void): () => void;
  play(): void;
  pause(): void;
  setSpeed(speed: ReplaySpeed): void;
  seek(position: number): void;
  step(delta: number): void;
}

export interface RunSource {
  readonly kind: SourceKind;
  /** Short human description: the bundle URL or the run id. */
  readonly title: string;
  /** Updates are delivered only between `start()` and `stop()`. */
  subscribe(listener: (update: SourceUpdate) => void): () => void;
  start(): void;
  /** Stops polling/playing and releases everything the source holds (object URLs, timers). */
  stop(): void;
  /** "Now" on the run's own time axis (Unix seconds), for elapsed-time displays. */
  now(): number;
  /** A URL an `<img>` can show for a run file. May be an object URL; owned by the source. */
  imageUrl(path: string): Promise<string>;
  fetchBytes(path: string, options?: FetchOptions): Promise<ArrayBuffer>;
  fetchJson(path: string, options?: FetchOptions): Promise<unknown>;
  /** Size of a run file in bytes without downloading it (HTTP HEAD), or undefined when it cannot be told. */
  fileSize?(path: string, signal?: AbortSignal): Promise<number | undefined>;
  /** Present when the run can be cancelled from here. */
  cancel?(): Promise<void>;
  /** Present for recordings. */
  readonly replay?: ReplayControls;
}

export interface EngineHealth {
  schema?: string;
  device?: string;
  canStartRuns: boolean;
}

export interface RunSummary {
  id: string;
  status: string;
  /** Unix seconds, or null when the run has not written its first event. */
  started: number | null;
  stage: string | null;
  stageFraction: number;
  events: number;
}

/** A request the engine answered with an error status. `status` 0 means it could not be reached. */
export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "EngineError";
  }
}

export interface DataEntry {
  name: string;
  directory: boolean;
  /** A run can start from this folder. */
  inputs: boolean;
}

export interface DataListing {
  /** The listed folder, relative to the data directory, "" for the top. */
  path: string;
  entries: DataEntry[];
}

export interface Engine {
  readonly kind: SourceKind;
  /** Where it is, without credentials. */
  readonly label: string;
  health(signal?: AbortSignal): Promise<EngineHealth>;
  settings(signal?: AbortSignal): Promise<SettingSpec[]>;
  listRuns(signal?: AbortSignal): Promise<RunSummary[]>;
  /** Returns the id of the new run. Throws `EngineError` with status 400 and the engine's sentence on invalid input. */
  startRun(body: Record<string, unknown>): Promise<string>;
  openRun(id: string): RunSource;
  /** Folders and files under the engine's data directory; `path` is relative to it ("" is the top). */
  listData?(path: string, signal?: AbortSignal): Promise<DataListing>;
}
