/** Folds what a source delivers into one `RunState` with the reducer. The views subscribe here. No DOM. */

import { initialState, reduceAll, type RunState } from "../core/reducer";
import type { LinkStatus, RunSource, SourceUpdate } from "./types";

export interface RunSnapshot {
  run: RunState;
  link: LinkStatus;
}

export class RunStore {
  private snapshot: RunSnapshot = { run: initialState(), link: { state: "connecting" } };
  private listeners = new Set<(snapshot: RunSnapshot) => void>();
  private readonly detach: () => void;

  constructor(readonly source: RunSource) {
    this.detach = source.subscribe((update) => this.apply(update));
  }

  get(): RunSnapshot {
    return this.snapshot;
  }

  subscribe(listener: (snapshot: RunSnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  dispose(): void {
    this.detach();
    this.listeners.clear();
  }

  private apply(update: SourceUpdate): void {
    const { run, link } = this.snapshot;
    if (update.type === "events") this.snapshot = { run: reduceAll(update.events, run), link };
    else if (update.type === "reset") this.snapshot = { run: reduceAll(update.events), link };
    else this.snapshot = { run, link: update.link };
    for (const listener of [...this.listeners]) listener(this.snapshot);
  }
}
