import { useEffect, useState } from "react";
import type { SignalCompany } from "../../types";
import type { CompanyRow, MarketResult } from "../../lib/market";
import { money } from "../../lib/market";
import { api } from "../../lib/api";
import { shortDate } from "../../lib/signals";
import PatternBadge from "./PatternBadge";
import Sparkline from "./Sparkline";
import SignalDetail from "./SignalDetail";
import ResearchCard from "./ResearchCard";

interface Props {
  result: MarketResult;
  researchEnabled: boolean;
}

const n = (v: unknown): number => (typeof v === "number" ? v : 0);
const s = (v: unknown): string => (typeof v === "string" ? v : "");

/** Renders one answer from the market engine. Every figure on screen is a count or statistic of real listings. */
export default function MarketResultView({ result, researchEnabled }: Props) {
  const weekEnds = result.week_labels ?? [];
  return (
    <div className="mk-result">
      {result.notes.length > 0 && <ul className="mk-notes">{result.notes.map((t) => <li key={t}>{t}</li>)}</ul>}
      {result.task === "companies" && <Companies rows={result.rows as unknown as CompanyRow[]} summary={result.summary} weekEnds={weekEnds} researchEnabled={researchEnabled} />}
      {result.task === "compare" && <Compare rows={result.rows as unknown as CompanyRow[]} weekEnds={weekEnds} researchEnabled={researchEnabled} />}
      {result.task === "technologies" && <Technologies rows={result.rows} summary={result.summary} />}
      {result.task === "listings" && <Listings rows={result.rows} summary={result.summary} />}
      {result.task === "trend" && <Trend rows={result.rows} summary={result.summary} />}
      {result.task === "pay" && <Pay rows={result.rows} summary={result.summary} />}
      <p className="cp-sub mk-asof">Data as of {shortDate(result.as_of)}. Dates are when our crawler first saw a listing, not when it was posted.</p>
    </div>
  );
}

function Companies({ rows, summary, weekEnds, researchEnabled }: { rows: CompanyRow[]; summary: Record<string, number>; weekEnds: string[]; researchEnabled: boolean }) {
  if (rows.length === 0) return <p className="cp-empty">No companies match.</p>;
  return (
    <>
      <p className="cp-sub">{n(summary.companies).toLocaleString()} {n(summary.companies) === 1 ? "company" : "companies"} match, with {n(summary.new_openings).toLocaleString()} new openings. Showing the top {rows.length}.</p>
      <ol className="mk-list">{rows.map((r) => <CompanyItem key={r.company_id} row={r} weekEnds={weekEnds} researchEnabled={researchEnabled} />)}</ol>
    </>
  );
}

function CompanyItem({ row, weekEnds, researchEnabled }: { row: CompanyRow; weekEnds: string[]; researchEnabled: boolean }) {
  const [open, setOpen] = useState<"none" | "details" | "research">("none");
  const [detail, setDetail] = useState<SignalCompany | null>(null);
  useEffect(() => {
    if (open !== "details" || detail) return;
    api.signalsCompany(row.company_id).then(setDetail).catch(() => undefined);
  }, [open, detail, row.company_id]);
  const name = row.name ?? row.domain ?? "Unnamed company";
  const techs = Object.entries(row.technologies);
  return (
    <li className="mk-item">
      <div className="mk-item__main">
        <div className="mk-item__who">
          <span className="mk-item__name">{name}</span>
          <span className="mk-item__sub">{row.name ? row.domain ?? "domain not provided" : "name not provided"}</span>
        </div>
        {row.pattern && row.pattern !== "quiet" ? <PatternBadge pattern={row.pattern} /> : <span className="cp-sub">No notable pattern</span>}
        <div className="mk-item__num"><strong>{row.new}</strong><span>new · {row.active} still observed</span></div>
        <Sparkline values={row.weekly} weekEnds={weekEnds} tone={row.pattern ?? "quiet"} width={132} height={32} />
      </div>
      <p className="mk-item__meta">
        {row.example_titles.length > 0 && <span>{row.example_titles.join(" · ")}</span>}
        {row.places.length > 0 && <span className="cp-muted"> — {row.places.join(", ")}</span>}
      </p>
      {techs.length > 0 && <div className="cp-chips">{techs.map(([t, c]) => <span key={t} className="cp-chip">{t} · {c}</span>)}</div>}
      <div className="mk-item__actions">
        <button type="button" className="btn btn--sm" aria-expanded={open === "details"} onClick={() => setOpen(open === "details" ? "none" : "details")}>Details</button>
        {researchEnabled && <button type="button" className="btn btn--sm" aria-expanded={open === "research"} onClick={() => setOpen(open === "research" ? "none" : "research")}>Research this company</button>}
      </div>
      {open === "details" && <div className="sg-detail">{detail ? <SignalDetail detail={detail} /> : <p className="cp-sub">Loading…</p>}</div>}
      {open === "research" && <div className="sg-detail"><ResearchCard companyId={row.company_id} companyName={name} /></div>}
    </li>
  );
}

function Compare({ rows, weekEnds, researchEnabled }: { rows: CompanyRow[]; weekEnds: string[]; researchEnabled: boolean }) {
  if (rows.length === 0) return <p className="cp-empty">No companies to compare.</p>;
  return (
    <div className="mk-compare">
      {rows.map((r) => (
        <div key={r.company_id} className="mk-compare__col">
          <h3>{r.name ?? r.domain}</h3>
          {r.pattern && r.pattern !== "quiet" ? <PatternBadge pattern={r.pattern} /> : <span className="cp-sub">No notable pattern</span>}
          <dl className="kv">
            <div><dt>New openings</dt><dd>{r.new}</dd></div>
            <div><dt>Still observed</dt><dd>{r.active}</dd></div>
            <div><dt>Listings seen</dt><dd>{r.listings}</dd></div>
          </dl>
          <Sparkline values={r.weekly} weekEnds={weekEnds} tone={r.pattern ?? "quiet"} width={180} height={36} />
          {r.families && Object.keys(r.families).length > 0 && <div className="cp-chips">{Object.entries(r.families).map(([f, c]) => <span key={f} className="cp-chip">{f} {c}</span>)}</div>}
          {r.top_technologies && r.top_technologies.length > 0 && (
            <>
              <h4>Technologies in listings</h4>
              <div className="cp-chips">{r.top_technologies.map((t) => <span key={t.name} className="cp-chip">{t.name} · {t.listings}</span>)}</div>
            </>
          )}
          {r.places.length > 0 && <p className="cp-sub">{r.places.join(", ")}</p>}
          {researchEnabled && <ResearchCard companyId={r.company_id} companyName={r.name ?? "this company"} />}
        </div>
      ))}
    </div>
  );
}

function Technologies({ rows, summary }: { rows: Record<string, unknown>[]; summary: Record<string, number> }) {
  if (rows.length === 0) return <p className="cp-empty">No technologies found.</p>;
  const max = Math.max(1, ...rows.map((r) => n(r.listings)));
  return (
    <>
      <p className="cp-sub">Across {n(summary.listings).toLocaleString()} listings at {n(summary.companies).toLocaleString()} companies.</p>
      <table className="table mk-table">
        <thead><tr><th scope="col">Technology</th><th scope="col">Listings</th><th scope="col">Companies</th><th scope="col">New in window</th><th scope="col">Window before</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={s(r.technology)}>
              <td>{s(r.technology)}</td>
              <td><span className="mk-bar"><i style={{ width: `${(100 * n(r.listings)) / max}%` }} /></span> {n(r.listings).toLocaleString()}</td>
              <td>{n(r.companies).toLocaleString()}</td>
              <td>{n(r.new).toLocaleString()}</td>
              <td className="cp-muted">{n(r.earlier).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Listings({ rows, summary }: { rows: Record<string, unknown>[]; summary: Record<string, number> }) {
  if (rows.length === 0) return <p className="cp-empty">No openings match in this window.</p>;
  return (
    <>
      <p className="cp-sub">{n(summary.openings).toLocaleString()} new openings match; the newest {rows.length} are shown.</p>
      <ul className="mk-list">
        {rows.map((r) => (
          <li key={s(r.job_id)} className="mk-item">
            <div className="mk-item__name">{s(r.title)} <span className="cp-muted">· {s(r.company)}</span></div>
            <p className="mk-item__meta">
              {[s(r.place), s(r.level), s(r.pay), `first seen ${shortDate(s(r.first_seen))}`].filter(Boolean).join(" · ")}
            </p>
            {Array.isArray(r.technologies) && r.technologies.length > 0 && <div className="cp-chips">{(r.technologies as string[]).map((t) => <span key={t} className="cp-chip">{t}</span>)}</div>}
          </li>
        ))}
      </ul>
    </>
  );
}

function Trend({ rows, summary }: { rows: Record<string, unknown>[]; summary: Record<string, number> }) {
  const values = rows.map((r) => n(r.new));
  const last = n(summary.last_4_weeks), prev = n(summary.previous_4_weeks);
  return (
    <>
      <div className="sg-company__nums">
        <div><strong>{last.toLocaleString()}</strong><span>new openings, last 4 weeks</span></div>
        <div><strong>{prev.toLocaleString()}</strong><span>the 4 weeks before</span></div>
        <div><strong>{n(summary.openings_in_range).toLocaleString()}</strong><span>over {values.length} weeks</span></div>
      </div>
      <Sparkline values={values} weekEnds={rows.map((r) => s(r.week_ending))} tone="sustained" width={560} height={90} />
      <p className="cp-sub">Each bar is one week of new openings, oldest on the left.</p>
    </>
  );
}

function Pay({ rows, summary }: { rows: Record<string, unknown>[]; summary: Record<string, number> }) {
  if (!("median" in summary)) return <p className="cp-empty">Too few matching listings state a yearly pay to show a range.</p>;
  return (
    <>
      <div className="sg-company__nums">
        <div><strong>{money(summary.median)}</strong><span>median of {n(summary.with_yearly_pay).toLocaleString()} listings with yearly pay</span></div>
        <div><strong>{money(summary.p25)} to {money(summary.p75)}</strong><span>middle half (25th to 75th percentile)</span></div>
        <div><strong>{money(summary.low)} to {money(summary.high)}</strong><span>lowest to highest</span></div>
      </div>
      {rows.length > 0 && (
        <table className="table mk-table">
          <thead><tr><th scope="col">Level</th><th scope="col">Listings</th><th scope="col">Median</th><th scope="col">Middle half</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={s(r.level)}><td>{s(r.level)}</td><td>{n(r.listings)}</td><td>{money(n(r.median))}</td><td>{money(n(r.p25))} to {money(n(r.p75))}</td></tr>)}</tbody>
        </table>
      )}
    </>
  );
}
