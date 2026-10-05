/**
 * Plays a recorded event list back with its recorded timing. Pure timing logic:
 * the wall clock and the timer are injected, so tests drive it deterministically.
 */

import type { RunEvent } from "./events";

export const REPLAY_SPEEDS = [1, 4, 16, Infinity] as const;
export type ReplaySpeed = (typeof REPLAY_SPEEDS)[number];

export interface ReplayClock {
  /** Wall clock in milliseconds. */
  now(): number;
  setTimer(callback: () => void, delayMs: number): unknown;
  clearTimer(handle: unknown): void;
}

export const systemClock: ReplayClock = {
  now: () => performance.now(),
  setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export interface ReplaySnapshot {
  /** Number of events played: events[0 .. position) are visible. */
  position: number;
  total: number;
  speed: ReplaySpeed;
  playing: boolean;
  /** True once every event has been played. */
  ended: boolean;
}

export class ReplayPlayer {
  private position = 0;
  private speed: ReplaySpeed;
  private playing = false;
  /** Recorded time (Unix seconds) at `anchorWall`. */
  private anchorRecorded: number;
  private anchorWall = 0;
  private timer: unknown = null;
  private listeners = new Set<(snapshot: ReplaySnapshot) => void>();

  constructor(
    private readonly events: readonly RunEvent[],
    private readonly clock: ReplayClock = systemClock,
    speed: ReplaySpeed = 1,
  ) {
    this.speed = speed;
    this.anchorRecorded = events[0]?.time ?? 0;
  }

  snapshot(): ReplaySnapshot {
    return {
      position: this.position,
      total: this.events.length,
      speed: this.speed,
      playing: this.playing,
      ended: this.position >= this.events.length,
    };
  }

  subscribe(listener: (snapshot: ReplaySnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** The recorded time (Unix seconds) the playback head is at now. */
  recordedTime(): number {
    if (!this.playing || this.speed === Infinity) return this.anchorRecorded;
    const head = this.anchorRecorded + ((this.clock.now() - this.anchorWall) / 1000) * this.speed;
    const next = this.events[this.position];
    return next === undefined ? this.anchorRecorded : Math.min(head, next.time);
  }

  play(): void {
    if (this.playing) return;
    if (this.position >= this.events.length) this.moveTo(0);
    this.playing = true;
    this.anchorWall = this.clock.now();
    this.advance();
  }

  pause(): void {
    if (!this.playing) return;
    this.anchorRecorded = this.recordedTime();
    this.playing = false;
    this.cancelTimer();
    this.emit();
  }

  setSpeed(speed: ReplaySpeed): void {
    if (speed === this.speed) return;
    this.anchorRecorded = this.recordedTime();
    this.anchorWall = this.clock.now();
    this.speed = speed;
    if (this.playing) this.advance();
    else this.emit();
  }

  /** Scrub: show exactly the first `position` events. Keeps playing from there if it was playing. */
  seek(position: number): void {
    this.moveTo(Math.min(this.events.length, Math.max(0, Math.round(position))));
    if (this.playing) this.advance();
    else this.emit();
  }

  /** Step one event forwards or backwards and pause. */
  step(delta: number): void {
    this.playing = false;
    this.cancelTimer();
    this.seek(this.position + delta);
  }

  dispose(): void {
    this.cancelTimer();
    this.listeners.clear();
    this.playing = false;
  }

  private moveTo(position: number): void {
    this.cancelTimer();
    this.position = position;
    const last = this.events[position - 1] ?? this.events[0];
    this.anchorRecorded = last?.time ?? 0;
    this.anchorWall = this.clock.now();
  }

  /** Plays every event that is due, then sleeps until the next one. */
  private advance(): void {
    this.cancelTimer();
    const now = this.clock.now();
    const head =
      this.speed === Infinity ? Infinity : this.anchorRecorded + ((now - this.anchorWall) / 1000) * this.speed;
    while (this.position < this.events.length && this.events[this.position]!.time <= head) this.position += 1;
    if (this.position >= this.events.length) {
      this.anchorRecorded = this.events[this.events.length - 1]?.time ?? 0;
      this.playing = false;
    } else {
      // Re-anchor on what was reached, so rounding never accumulates.
      this.anchorRecorded = head;
      this.anchorWall = now;
      const wait = ((this.events[this.position]!.time - head) / this.speed) * 1000;
      this.timer = this.clock.setTimer(() => {
        this.timer = null;
        if (this.playing) this.advance();
      }, Math.max(0, wait));
    }
    this.emit();
  }

  private cancelTimer(): void {
    if (this.timer !== null) this.clock.clearTimer(this.timer);
    this.timer = null;
  }

  private emit(): void {
    const snapshot = this.snapshot();
    for (const listener of [...this.listeners]) listener(snapshot);
  }
}
