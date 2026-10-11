import { useEffect, useState } from "react";
import type { SignalCompany } from "../../types";
import { api, ApiError } from "../../lib/api";
import { FAMILY_LABEL, PATTERN_HELP } from "../../lib/signals";
import PatternBadge from "./PatternBadge";
import Sparkline from "./Sparkline";
import SignalDetail from "./SignalDetail";

type State = { status: "loading" } | { status: "none" } | { status: "error" } | { status: "ready"; data: SignalCompany };

/** Hiring activity for the job's company, shown at the top of the Company tab. */
export default function CompanyHiring({ companyId }: { companyId: string | null | undefined }) {
  const [state, setState] = useState<State>({ status: companyId ? "loading" : "none" });

  useEffect(() => {
    if (!companyId) { setState({ status: "none" }); return; }
    let cancelled = false;
    setState({ status: "loading" });
    api.signalsCompany(companyId)
      .then((data) => { if (!cancelled) setState({ status: "ready", data }); })
      .catch((e) => { if (!cancelled) setState(e instanceof ApiError && (e.status === 404 || e.status === 503) ? { status: "none" } : { status: "error" }); });
    return () => { cancelled = true; };
  }, [companyId]);

  if (state.status === "loading") return <section className="section section--first"><h3>Hiring activity</h3><p className="section__note">Loading…</p></section>;
  if (state.status === "error") return <section className="section section--first"><h3>Hiring activity</h3><p className="section__note">Hiring activity could not be loaded.</p></section>;
  if (state.status === "none") {
    return (
      <section className="section section--first">
        <h3>Hiring activity</h3>
        <p className="section__note">No hiring activity is available for this company: it is not in the AI/ML, data and technical-sales pool the signals were built from.</p>
      </section>
    );
  }

  const c = state.data;
  const mix = (Object.keys(FAMILY_LABEL) as (keyof typeof FAMILY_LABEL)[]).filter((f) => c.families[f].new_30 > 0);
  return (
    <section className="section section--first sg-company">
      <h3>Hiring activity</h3>
      <p><PatternBadge pattern={c.pattern} /></p>
      <p className="section__note">{PATTERN_HELP[c.pattern]}</p>
      <div className="sg-company__nums">
        <div><strong>{c.new_30}</strong><span>new openings in 30 days</span></div>
        <div><strong>{c.active}</strong><span>still observed</span></div>
        <div><strong>{c.enough_history && c.baseline_monthly !== null ? c.baseline_monthly.toFixed(c.baseline_monthly < 10 ? 1 : 0) : "n/a"}</strong><span>per month before</span></div>
      </div>
      <Sparkline values={c.weekly} weekEnds={c.week_ends ?? []} tone={c.pattern} width={260} />
      <p className="cp-sub">New openings per week, last 26 weeks.</p>
      {mix.length > 0 && <div className="cp-chips">{mix.map((f) => <span key={f} className="cp-chip">{FAMILY_LABEL[f]} {c.families[f].new_30}</span>)}</div>}
      <SignalDetail detail={c} />
      <p className="cp-sub">Dates are when the crawler saw a listing, not when it was posted.</p>
    </section>
  );
}
