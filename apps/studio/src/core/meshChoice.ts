/**
 * Which surface the 3D view shows, as a pure function of the run's mesh steps, so the
 * answer depends only on the data and never on how fast events arrived. No DOM.
 */

import type { MeshStep } from "./reducer";

export interface MeshChoice {
  /** The step the person chose, or the newest one when following. */
  selected: MeshStep | undefined;
  /** The step to put on screen: `selected`, or while a big final surface waits for consent, the newest step that may load. */
  target: MeshStep | undefined;
  /** `selected` is a big final surface that has not been asked for yet. */
  awaitingConsent: boolean;
  following: boolean;
}

/**
 * `meshes` in run order (by line number, as the reducer keeps them). `chosen`: a path picked
 * by hand, or null to follow the newest. `needsConsent(step)`: the step is too big to load
 * without being asked.
 */
export function chooseMesh(meshes: readonly MeshStep[], chosen: string | null, needsConsent: (step: MeshStep) => boolean): MeshChoice {
  // Newest by line number, whatever order they were appended in.
  const ordered = [...meshes].sort((a, b) => a.seq - b.seq);
  const newest = ordered[ordered.length - 1];
  const picked = chosen !== null ? ordered.find((step) => step.path === chosen) : undefined;
  const selected = picked ?? newest;
  const following = picked === undefined;
  const awaitingConsent = selected !== undefined && needsConsent(selected);
  let target = selected;
  if (awaitingConsent) target = following ? [...ordered].reverse().find((step) => !needsConsent(step)) : undefined;
  return { selected, target, awaitingConsent, following };
}

/**
 * Hands out a ticket per load and says whether a ticket is still the newest, so a load that
 * finishes after a later one has started is dropped instead of replacing the newer surface.
 */
export class LatestOnly {
  private latest = 0;

  begin(): number {
    this.latest += 1;
    return this.latest;
  }

  isCurrent(ticket: number): boolean {
    return ticket === this.latest;
  }
}
