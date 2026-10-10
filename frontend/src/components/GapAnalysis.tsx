import { useState } from "react";
import type { GapAssessment, QualificationStatus } from "../types";
import { api } from "../lib/api";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "error"; message: string }
  | { status: "done"; assessment: GapAssessment };

const STATUS_LABEL: Record<QualificationStatus, string> = {
  supported: "Supported by your profile",
  unknown: "Not shown in your profile",
  explicit_gap: "Conflicts with your profile",
};
const STATUS_ICON: Record<QualificationStatus, string> = { supported: "✓", unknown: "?", explicit_gap: "✕" };
const KIND_LABEL = { required: "Required", preferred: "Preferred", unclear: "Mentioned" } as const;

/** Holds the analysis for one job so it survives switching tabs. The details panel is keyed per job, so each job starts clean. */
export function useGapAnalysis(jobId: string, snapshotId: string | null) {
  const [consent, setConsent] = useState(false);
  const [state, setState] = useState<State>({ status: "idle" });
  const run = async () => {
    if (!snapshotId || !consent) return;
    setState({ status: "loading" });
    try {
      const result = await api.gapAnalysis(jobId, snapshotId);
      setState(result.state === "unavailable" ? { status: "unavailable" } : { status: "done", assessment: result.assessment });
    } catch (e) {
      setState({ status: "error", message: e instanceof Error ? e.message : "The analysis could not be completed." });
    }
  };
  return { consent, setConsent, state, setState, run, ready: Boolean(snapshotId) };
}

/** AI gap analysis of one job against the saved profile. Runs only on click, after a per-use consent. */
export default function GapAnalysis({ gap }: { gap: ReturnType<typeof useGapAnalysis> }) {
  const { consent, setConsent, state, setState, run, ready } = gap;

  return (
    <section className="section section--first ga">
      <h3>AI gap analysis</h3>
      <p className="section__note">
        An AI model compares this job's requirements with your profile and shows, for each one, the exact lines it relied on. A requirement your profile does not
        mention is shown as “not shown in your profile”, which is different from “you don't have it”. It is not a hiring or eligibility judgment.
      </p>

      {state.status !== "done" && (
        <>
          <p>
            <label>
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />{" "}
              I agree to send my profile's experience and skills (not my name, email, phone or links) and this job's description to OpenAI so they can be compared.
              The result is kept in your demo session so running it again is free.
            </label>
          </p>
          <p>
            <button type="button" className="btn btn--sm" disabled={!consent || state.status === "loading" || !ready} onClick={() => void run()}>
              {state.status === "loading" ? "Analysing…" : "Analyse this job"}
            </button>
          </p>
        </>
      )}

      {state.status === "unavailable" && (
        <p className="notice" role="alert">The AI gap analysis is turned off on this server, so nothing was sent. Ask whoever runs the demo to start it with the -Assess option.</p>
      )}
      {state.status === "error" && <p className="field__error" role="alert">{state.message}</p>}
      {state.status === "done" && <Result assessment={state.assessment} onAgain={() => setState({ status: "idle" })} />}
    </section>
  );
}

function Result({ assessment: a, onAgain }: { assessment: GapAssessment; onAgain: () => void }) {
  const text = (id: string) => a.evidence_catalog.find((e) => e.evidence_id === id)?.text ?? null;
  const quotes = (ids: string[], who: string) =>
    ids.map(text).filter((t): t is string => Boolean(t)).map((t, i) => <blockquote key={`${who}${i}`} className="ga__quote"><span>{who}</span>{t}</blockquote>);

  if (a.evidence_sufficiency === "insufficient") {
    return (
      <>
        <p className="notice">There is not enough to assess this job: {a.insufficiency_reason ?? "the profile or the listing is too sparse."}</p>
        <p><button type="button" className="btn btn--sm" onClick={onAgain}>Back</button></p>
      </>
    );
  }

  const counts = { supported: 0, unknown: 0, explicit_gap: 0 } as Record<QualificationStatus, number>;
  a.qualifications.forEach((q) => { counts[q.status] += 1; });

  return (
    <div className="ga__result">
      <p className="ga__summary" role="status">
        {a.qualifications.length} requirements read: <strong>{counts.supported}</strong> supported by your profile,{" "}
        <strong>{counts.unknown}</strong> not shown in your profile, <strong>{counts.explicit_gap}</strong> conflicting.
      </p>

      <h4>Requirements</h4>
      <ul className="ga__list">
        {a.qualifications.map((q) => (
          <li key={q.qualification_id} className={`ga__item ga__item--${q.status}`}>
            <div className="ga__head">
              <span className={`ga__status ga__status--${q.status}`}><span aria-hidden="true">{STATUS_ICON[q.status]}</span> {STATUS_LABEL[q.status]}</span>
              <span className="ga__kind">{KIND_LABEL[q.classification]}</span>
            </div>
            <p className="ga__req">{q.qualification}</p>
            <p className="ga__why">{q.rationale}</p>
            {quotes(q.candidate_evidence_ids, "Your profile")}
            {quotes(q.job_evidence_ids, "The listing")}
          </li>
        ))}
      </ul>

      {a.transferable_experience.length > 0 && (
        <>
          <h4>Experience that may carry over</h4>
          <ul className="ga__list">
            {a.transferable_experience.map((t, i) => (
              <li key={i} className="ga__item">
                <p className="ga__req">{t.job_duty}</p>
                <p className="ga__why">{t.rationale}</p>
                {quotes(t.candidate_evidence_ids, "Your profile")}
                {quotes(t.job_evidence_ids, "The listing")}
              </li>
            ))}
          </ul>
        </>
      )}

      {a.next_steps.length > 0 && (
        <>
          <h4>Next steps</h4>
          <ul className="ga__steps">{a.next_steps.map((s, i) => <li key={i}>{s.action}</li>)}</ul>
        </>
      )}

      <p className="cp-sub">Generated by an AI model from the lines shown. Check each point against your own history and the full listing.</p>
      <p><button type="button" className="btn btn--sm" onClick={onAgain}>Back</button></p>
    </div>
  );
}
