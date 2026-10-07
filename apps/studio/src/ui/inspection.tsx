import { useEffect, useState } from "preact/hooks";
import type { Inspection } from "../core/reducer";
import type { RunSource } from "../sources/types";
import { useImageUrl } from "./gallery";

interface Props {
  source: RunSource;
  steps: Inspection[];
}

/**
 * The inspection sheets, one per step, as a flip-through: the same views and the same detail
 * crops after every surface, so a change from one step to the next shows by flipping.
 */
export function InspectionPanel({ source, steps }: Props) {
  const [index, setIndex] = useState(0);
  const [following, setFollowing] = useState(true);
  // While a run is going, show the newest step until the person picks one.
  const steps_ = steps.filter((step) => step.step !== "steps");
  useEffect(() => {
    if (following) setIndex(Math.max(0, steps_.length - 1));
  }, [steps_.length, following]);
  if (steps.length === 0) return null;
  const current = steps[Math.min(index, steps.length - 1)]!;
  const go = (next: number) => {
    setFollowing(false);
    setIndex(Math.max(0, Math.min(steps.length - 1, next)));
  };
  return (
    <section class="panel inspection" aria-labelledby="inspection-heading">
      <div class="panel-head">
        <h2 id="inspection-heading">Step by step</h2>
        <span class="sub">
          {index + 1} of {steps.length}: {current.label}
        </span>
      </div>
      <div class="inspection-steps" role="tablist" aria-label="Steps">
        {steps.map((step, position) => (
          <button
            key={step.path}
            type="button"
            role="tab"
            class={`button small${position === index ? " primary" : ""}`}
            aria-selected={position === index}
            onClick={() => go(position)}
          >
            {step.label}
          </button>
        ))}
      </div>
      <InspectionImage source={source} step={current} />
      <div class="actions">
        <button type="button" class="button" onClick={() => go(index - 1)} disabled={index === 0} aria-label="Previous step">
          Previous
        </button>
        <button type="button" class="button" onClick={() => go(index + 1)} disabled={index >= steps.length - 1} aria-label="Next step">
          Next
        </button>
      </div>
    </section>
  );
}

function InspectionImage({ source, step }: { source: RunSource; step: Inspection }) {
  const { url, failed } = useImageUrl(source, step.path);
  return (
    <div class="inspection-frame">
      {url !== undefined && !failed ? <img src={url} alt={`Inspection after: ${step.label}`} /> : <span class="sheet-placeholder">{failed ? "Image not available" : "Loading..."}</span>}
    </div>
  );
}
