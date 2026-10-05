/**
 * STUB. The local source for the future Tauri shell: same shape as the HTTP engine, not
 * implemented. It exists so that wrapping the app later means filling in this file, not
 * touching the UI.
 *
 * Intended implementation (docs/ENGINE-CONTRACT.md, "Local (desktop shell)"):
 *   - events:  the shell tails `<run>/events.jsonl`, remembers the line count, and pushes
 *              complete lines to the web view (Tauri event or channel) -> `SourceUpdate`.
 *   - files:   read from the run directory through a scoped asset protocol
 *              (`convertFileSrc`) for images, and as bytes for STL and JSON.
 *   - engine:  list run directories, start `dense_pipeline` as a sidecar process,
 *              cancel by creating `<run>/cancel`.
 * The simplest first step is not to implement this at all: let the shell start
 * `engine_server.py` on localhost and point `HttpEngine` at it.
 */

import type { SettingSpec } from "../core/settings";
import type { Engine, EngineHealth, FetchOptions, RunSource, RunSummary, SourceUpdate } from "./types";

export class NotImplementedError extends Error {
  constructor(what: string) {
    super(`${what} is not available: the local engine is only present in the desktop app, which is not built yet.`);
    this.name = "NotImplementedError";
  }
}

/** True inside a Tauri web view. The UI uses it only to decide whether to offer the local engine. */
export function localEngineAvailable(): boolean {
  return false;
}

export class LocalEngine implements Engine {
  readonly kind = "local" as const;
  readonly label = "This computer";

  health(): Promise<EngineHealth> {
    return Promise.reject(new NotImplementedError("The local engine"));
  }

  settings(): Promise<SettingSpec[]> {
    return Promise.reject(new NotImplementedError("Reading settings"));
  }

  listRuns(): Promise<RunSummary[]> {
    return Promise.reject(new NotImplementedError("Listing runs"));
  }

  startRun(): Promise<string> {
    return Promise.reject(new NotImplementedError("Starting a run"));
  }

  openRun(id: string): RunSource {
    return new LocalRunSource(id);
  }
}

export class LocalRunSource implements RunSource {
  readonly kind = "local" as const;
  private listeners = new Set<(update: SourceUpdate) => void>();

  constructor(readonly title: string) {}

  subscribe(listener: (update: SourceUpdate) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    const message = new NotImplementedError("Opening a local run").message;
    for (const listener of [...this.listeners]) listener({ type: "link", link: { state: "failed", message } });
  }

  stop(): void {
    this.listeners.clear();
  }

  now(): number {
    return Date.now() / 1000;
  }

  imageUrl(): Promise<string> {
    return Promise.reject(new NotImplementedError("Reading run files"));
  }

  fetchBytes(_path: string, _options?: FetchOptions): Promise<ArrayBuffer> {
    return Promise.reject(new NotImplementedError("Reading run files"));
  }

  fetchJson(_path: string, _options?: FetchOptions): Promise<unknown> {
    return Promise.reject(new NotImplementedError("Reading run files"));
  }
}
