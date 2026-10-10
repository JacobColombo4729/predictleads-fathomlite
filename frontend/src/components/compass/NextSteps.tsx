import { useState } from "react";
import type { CompassStep } from "../../types";

const CYCLE = ["Not started", "Current", "Done", "Skipped"] as const;
type StepStatus = (typeof CYCLE)[number];

/** Frequent, least-supported requirements first. Tapping a status cycles Not started, Current, Done, Skipped. */
export default function NextSteps({ steps, picks }: { steps: CompassStep[]; picks: number }) {
  const [status, setStatus] = useState<Record<string, StepStatus>>({});
  if (steps.length === 0) return <p className="cp-sub">Nothing stands out: your profile shows evidence for what these jobs ask for most.</p>;
  return (
    <ol className="cp-path">
      {steps.map((s) => {
        const current = status[s.requirementId] ?? "Not started";
        return (
          <li key={s.requirementId}>
            <div>
              <div className="cp-path__t">{s.text}</div>
              <div className="cp-path__d">{s.action} <span className="cp-muted">(in {s.n} of {picks} picks)</span></div>
            </div>
            <button
              type="button"
              className="cp-stbtn"
              data-s={current}
              aria-label={`Status for ${s.text}: ${current}. Press to change.`}
              onClick={() => setStatus((m) => ({ ...m, [s.requirementId]: CYCLE[(CYCLE.indexOf(current) + 1) % CYCLE.length] }))}
            >
              {current}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
