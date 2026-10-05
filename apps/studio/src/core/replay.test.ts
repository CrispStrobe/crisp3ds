import { describe, expect, it } from "vitest";
import { FakeTime, fixtureEvents } from "../testing/fixture";
import type { RunEvent } from "./events";
import { ReplayPlayer, type ReplaySnapshot } from "./replay";

function at(times: number[]): RunEvent[] {
  return times.map((time, seq) => ({ type: "progress", seq, time: 5000 + time, stage: "stereo" }));
}

function player(times: number[], speed: 1 | 4 | 16 | typeof Infinity = 1) {
  const time = new FakeTime();
  const instance = new ReplayPlayer(at(times), time.clock, speed);
  const seen: ReplaySnapshot[] = [];
  instance.subscribe((snapshot) => seen.push(snapshot));
  return { time, instance, seen };
}

describe("ReplayPlayer", () => {
  it("plays events at their recorded offsets", () => {
    const { time, instance } = player([0, 1, 1.5, 4]);
    expect(instance.snapshot()).toMatchObject({ position: 0, total: 4, playing: false, ended: false });
    instance.play();
    expect(instance.snapshot().position).toBe(1);
    time.advance(999);
    expect(instance.snapshot().position).toBe(1);
    time.advance(1);
    expect(instance.snapshot().position).toBe(2);
    time.advance(500);
    expect(instance.snapshot().position).toBe(3);
    time.advance(2499);
    expect(instance.snapshot().position).toBe(3);
    time.advance(1);
    expect(instance.snapshot()).toMatchObject({ position: 4, ended: true, playing: false });
    expect(time.pending()).toEqual([]);
  });

  it("delivers events with the same time together", () => {
    const { time, instance } = player([0, 2, 2, 2, 3]);
    instance.play();
    time.advance(2000);
    expect(instance.snapshot().position).toBe(4);
  });

  it.each([
    [4, 1000],
    [16, 250],
  ] as const)("at %d× a four second gap takes %d ms", (speed, wall) => {
    const { time, instance } = player([0, 4], speed);
    instance.play();
    time.advance(wall - 1);
    expect(instance.snapshot().position).toBe(1);
    time.advance(1);
    expect(instance.snapshot().position).toBe(2);
  });

  it("plays everything at once when instant", () => {
    const { time, instance, seen } = player([0, 10, 500, 9000], Infinity);
    instance.play();
    expect(instance.snapshot()).toMatchObject({ position: 4, ended: true, playing: false });
    expect(seen).toHaveLength(1);
    expect(time.pending()).toEqual([]);
  });

  it("pauses without losing its place and resumes with the remaining wait", () => {
    const { time, instance } = player([0, 10]);
    instance.play();
    time.advance(4000);
    instance.pause();
    expect(instance.snapshot().playing).toBe(false);
    expect(instance.recordedTime()).toBeCloseTo(5004, 6);
    time.advance(60_000);
    expect(instance.snapshot().position).toBe(1);
    instance.play();
    time.advance(5999);
    expect(instance.snapshot().position).toBe(1);
    time.advance(1);
    expect(instance.snapshot().position).toBe(2);
  });

  it("changes speed in the middle of a gap", () => {
    const { time, instance } = player([0, 10]);
    instance.play();
    time.advance(2000); // 2 s of 10 played
    instance.setSpeed(4); // 8 s left -> 2 s of wall time
    time.advance(1999);
    expect(instance.snapshot().position).toBe(1);
    time.advance(1);
    expect(instance.snapshot().position).toBe(2);
  });

  it("switching to instant while playing finishes at once", () => {
    const { instance } = player([0, 10, 20]);
    instance.play();
    instance.setSpeed(Infinity);
    expect(instance.snapshot()).toMatchObject({ position: 3, ended: true });
  });

  it("scrubs to any event, backwards and forwards, and continues from there", () => {
    const { time, instance } = player([0, 1, 2, 3, 4]);
    instance.play();
    time.advance(3000);
    expect(instance.snapshot().position).toBe(4);
    instance.seek(1);
    expect(instance.snapshot()).toMatchObject({ position: 1, playing: true });
    // Position 1 means event 0 is shown; event 1 is one recorded second later.
    time.advance(999);
    expect(instance.snapshot().position).toBe(1);
    time.advance(1);
    expect(instance.snapshot().position).toBe(2);
    instance.seek(99);
    expect(instance.snapshot()).toMatchObject({ position: 5, ended: true, playing: false });
    instance.seek(-5);
    expect(instance.snapshot()).toMatchObject({ position: 0, ended: false });
  });

  it("scrubbing while paused stays paused", () => {
    const { time, instance } = player([0, 1, 2]);
    instance.seek(2);
    expect(instance.snapshot()).toMatchObject({ position: 2, playing: false });
    time.advance(10_000);
    expect(instance.snapshot().position).toBe(2);
  });

  it("steps one event at a time and pauses", () => {
    const { time, instance } = player([0, 5, 10]);
    instance.play();
    instance.step(1);
    expect(instance.snapshot()).toMatchObject({ position: 2, playing: false });
    instance.step(-1);
    expect(instance.snapshot().position).toBe(1);
    time.advance(60_000);
    expect(instance.snapshot().position).toBe(1);
  });

  it("starts again from the beginning when played after the end", () => {
    const { time, instance } = player([0, 1]);
    instance.play();
    time.advance(1000);
    expect(instance.snapshot().ended).toBe(true);
    instance.play();
    expect(instance.snapshot()).toMatchObject({ position: 1, playing: true, ended: false });
  });

  it("handles an empty recording", () => {
    const { instance } = player([]);
    instance.play();
    expect(instance.snapshot()).toMatchObject({ position: 0, total: 0, playing: false, ended: true });
  });

  it("stops its timer when disposed", () => {
    const { time, instance } = player([0, 5]);
    instance.play();
    expect(time.pending()).toHaveLength(1);
    instance.dispose();
    expect(time.pending()).toEqual([]);
  });

  it("replays the recorded sphere run in its recorded duration", () => {
    const events = fixtureEvents();
    const time = new FakeTime();
    const instance = new ReplayPlayer(events, time.clock, 1);
    instance.play();
    const duration = (events[events.length - 1]!.time - events[0]!.time) * 1000;
    time.advance(duration - 50);
    expect(instance.snapshot().ended).toBe(false);
    time.advance(100);
    expect(instance.snapshot()).toMatchObject({ position: 35, ended: true });
  });
});
