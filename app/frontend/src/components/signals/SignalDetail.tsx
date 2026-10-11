import type { SignalCompany } from "../../types";
import { shortDate } from "../../lib/signals";

/** The evidence behind a company's hiring signal: the surge by role family, example openings, technologies and counts. */
export default function SignalDetail({ detail }: { detail: SignalCompany }) {
  return (
    <>
      {detail.intent.map((s) => (
        <p key={s.family} className="sg-detail__intent"><strong>{s.signal}:</strong> {s.meaning}.</p>
      ))}
      {detail.evidence && detail.evidence.length > 0 && (
        <div>
          <h4>Example new openings</h4>
          <ul className="sg-detail__list">{detail.evidence.slice(0, 5).map((e) => <li key={e.job_id}>{e.title} <span className="cp-muted">· first seen {shortDate(e.first_seen)}</span></li>)}</ul>
        </div>
      )}
      {detail.technologies && detail.technologies.length > 0 && (
        <div>
          <h4>Technologies named in new openings</h4>
          <div className="cp-chips">{detail.technologies.slice(0, 10).map((t) => <span key={t.technology} className="cp-chip">{t.technology} · {t.new_30}</span>)}</div>
        </div>
      )}
      <p className="cp-sub">
        {detail.active} openings still observed, {detail.gone_30} no longer seen in the last 30 days.
        {detail.surge_z !== null && detail.pattern === "burst" ? ` The jump is ${detail.surge_z.toFixed(1)} standard deviations above this company's own pace.` : ""}
      </p>
    </>
  );
}
