/** Test helpers: the recorded sphere run in tests/fixtures/dense-run-sphere, and fakes for fetch and timers. */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseEventLines, type RunEvent } from "../core/events";
import type { FetchLike } from "../sources/transport";

export const FIXTURE = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../tests/fixtures/dense-run-sphere");

export function fixtureText(path: string): string {
  return readFileSync(resolve(FIXTURE, path), "utf8");
}

export function fixtureBytes(path: string): ArrayBuffer {
  const data = readFileSync(resolve(FIXTURE, path));
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
}

export function fixtureEvents(): RunEvent[] {
  return parseEventLines(fixtureText("events.jsonl")).events;
}

export function event(type: string, fields: Record<string, unknown> = {}, seq = 0): RunEvent {
  return { type, seq, time: 1000 + seq, stage: null, ...fields };
}

/** A clock and timer queue advanced by hand. */
export class FakeTime {
  ms = 0;
  private queue: { at: number; id: number; callback: () => void }[] = [];
  private nextId = 1;

  readonly clock = {
    now: () => this.ms,
    setTimer: (callback: () => void, delayMs: number) => this.set(callback, delayMs),
    clearTimer: (handle: unknown) => this.clear(handle),
  };

  readonly timer = {
    set: (callback: () => void, delayMs: number) => this.set(callback, delayMs),
    clear: (handle: unknown) => this.clear(handle),
  };

  /** Delays of the timers that are waiting, soonest first. */
  pending(): number[] {
    return this.queue.map((entry) => entry.at - this.ms).sort((a, b) => a - b);
  }

  private set(callback: () => void, delayMs: number): number {
    const id = this.nextId++;
    this.queue.push({ at: this.ms + delayMs, id, callback });
    return id;
  }

  private clear(handle: unknown): void {
    this.queue = this.queue.filter((entry) => entry.id !== handle);
  }

  /** Moves time forward, firing timers in order. */
  advance(ms: number): void {
    const end = this.ms + ms;
    for (;;) {
      const due = this.queue.filter((entry) => entry.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (due === undefined) break;
      this.queue = this.queue.filter((entry) => entry !== due);
      this.ms = Math.max(this.ms, due.at);
      due.callback();
    }
    this.ms = end;
  }
}

/** Lets pending promise callbacks run. */
export async function settle(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

export interface RecordedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
}

export type Responder = (request: RecordedRequest) => Response | Promise<Response>;

/** A fetch that records what was asked and answers from `responder`. */
export function fakeFetch(responder: Responder): { fetcher: FetchLike; requests: RecordedRequest[] } {
  const requests: RecordedRequest[] = [];
  const fetcher: FetchLike = async (url, init) => {
    const request: RecordedRequest = {
      url,
      method: init?.method ?? "GET",
      headers: { ...(init?.headers as Record<string, string> | undefined) },
      body: typeof init?.body === "string" ? init.body : undefined,
    };
    requests.push(request);
    return responder(request);
  };
  return { fetcher, requests };
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
