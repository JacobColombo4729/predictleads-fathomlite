import { useCallback, useEffect, useRef, useState } from "react";
import type { SignalCompany, SignalFamily, SignalPattern, SignalsOverview, SignalsQuery } from "../types";
import { api } from "../lib/api";
import { FAMILY_LABEL, PATTERN_HELP, PATTERN_LABEL, pct, shortDate } from "../lib/signals";
import PatternBadge from "../components/signals/PatternBadge";
import Sparkline from "../components/signals/Sparkline";
import SignalDetail from "../components/signals/SignalDetail";
import AskMarket from "../components/signals/AskMarket";

const PAGE = 25;
const PATTERNS: SignalPattern[] = ["burst", "sustained", "cooling"];
const SORTS: { value: SignalsQuery["sort"]; label: string }[] = [
  { value: "new_30", label: "New openings (30 days)" },
  { value: "active", label: "Still-observed openings" },
  { value: "surge_z", label: "Size of the jump" },
];

type Load = { status: "loading" } | { status: "ready" } | { status: "error"; message: string };

export default function SignalsPage() {
  const [overview, setOverview] = useState<SignalsOverview | null>(null);
  const [pattern, setPattern] = useState<SignalPattern | null>("burst");
  const [family, setFamily] = useState<SignalFamily | null>(null);
  const [sort, setSort] = useState<SignalsQuery["sort"]>("new_30");
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState<SignalCompany[]>([]);
  const [total, setTotal] = useState(0);
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [openId, setOpenId] = useState<string | null>(null);
  const request = useRef(0);

  useEffect(() => {
    api.signalsOverview().then(setOverview).catch((e) => setLoad({ status: "error", message: e instanceof Error ? e.message : "Signals could not be loaded." }));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => setQuery(text.trim()), 250);
    return () => clearTimeout(t);
  }, [text]);

  const fetchRows = useCallback(async (offset: number) => {
    const mine = ++request.current;
    if (offset === 0) setLoad({ status: "loading" });
    try {
      const r = await api.signalsCompanies({ pattern, family, q: query, sort, limit: PAGE, offset });
      if (mine !== request.current) return;
      setRows((cur) => (offset === 0 ? r.companies : [...cur, ...r.companies]));
      setTotal(r.total);
      setLoad({ status: "ready" });
    } catch (e) {
      if (mine === request.current) setLoad({ status: "error", message: e instanceof Error ? e.message : "Signals could not be loaded." });
    }
  }, [pattern, family, query, sort]);

  useEffect(() => { setOpenId(null); void fetchRows(0); }, [fetchRows]);

  const carry = (p: SignalPattern) => overview?.carry_over.find((c) => c.pattern === p);
  const weekEnds = overview?.week_ends ?? [];
  const asOf = overview?.snapshot ?? weekEnds[weekEnds.length - 1] ?? null;

  return (
    <main className="cp sg" aria-label="Hiring signals">
      <header className="cp-head">
        <div>
          <div className="cp-eyebrow">Signals</div>
          <h1>Who is hiring right now</h1>
          <p className="cp-sub">
            Companies ranked by how many roles they opened in the 30 days before {asOf ? shortDate(asOf) : "the snapshot"}, in AI/ML, data and technical sales.
          </p>
        </div>
      </header>

      <AskMarket />

      <section className="cp-card sg-explain" aria-label="What the patterns mean">
        {PATTERNS.slice(0, 2).map((p) => {
          const c = carry(p);
          return (
            <div key={p} className="sg-explain__item">
              <PatternBadge pattern={p} />
              <p>{PATTERN_HELP[p]}</p>
              {c && (
                <p className="sg-explain__stat">
                  Next month in our backtest: <strong>{pct(c.next_as_share_of_this_month)}</strong> of this month's volume, and{" "}
                  <strong>{pct(c.share_still_hiring_next_month)}</strong> still hiring.
                </p>
              )}
            </div>
          );
        })}
        <p className="cp-sub sg-explain__foot">
          These dates are when the crawler saw a listing, not when it was posted, so read them as "noticed hiring", not "decided to hire".
          Counts exclude reposts and bulk-import days. Backtest on four months of data: a burst is a timing trigger, not a volume forecast.
        </p>
      </section>

      <div className="sg-filters" role="group" aria-label="Filter companies">
        <div className="sg-seg" role="group" aria-label="Pattern">
          {PATTERNS.map((p) => (
            <button key={p} type="button" aria-pressed={pattern === p} className={`sg-seg__btn${pattern === p ? " is-on" : ""}`} onClick={() => setPattern(pattern === p ? null : p)}>
              {PATTERN_LABEL[p]}{overview ? ` (${overview.pattern_counts[p].toLocaleString()})` : ""}
            </button>
          ))}
        </div>
        <label className="sg-field">
          <span>Role family</span>
          <select value={family ?? ""} onChange={(e) => setFamily((e.target.value || null) as SignalFamily | null)}>
            <option value="">All families</option>
            {(Object.keys(FAMILY_LABEL) as SignalFamily[]).map((f) => <option key={f} value={f}>{FAMILY_LABEL[f]}</option>)}
          </select>
        </label>
        <label className="sg-field">
          <span>Sort by</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as SignalsQuery["sort"])}>
            {SORTS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
        <label className="sg-field sg-field--grow">
          <span>Company</span>
          <input type="search" value={text} placeholder="Name or domain" onChange={(e) => setText(e.target.value)} />
        </label>
      </div>

      {load.status === "error" && <div className="cp-card" role="alert"><p>{load.message}</p><button type="button" className="btn" onClick={() => void fetchRows(0)}>Try again</button></div>}
      {load.status === "loading" && <p className="cp-sub" role="status">Loading companies…</p>}
      {load.status === "ready" && rows.length === 0 && <div className="cp-empty">No companies match these filters.</div>}

      <ol className="sg-list">
        {rows.map((c) => (
          <CompanyRow key={c.company_id} company={c} weekEnds={weekEnds} family={family} open={openId === c.company_id} onToggle={() => setOpenId(openId === c.company_id ? null : c.company_id)} />
        ))}
      </ol>
      {load.status === "ready" && rows.length < total && (
        <div className="sg-more"><button type="button" className="btn" onClick={() => void fetchRows(rows.length)}>Show more ({(total - rows.length).toLocaleString()} left)</button></div>
      )}
    </main>
  );
}

function CompanyRow({ company: c, weekEnds, family, open, onToggle }: { company: SignalCompany; weekEnds: string[]; family: SignalFamily | null; open: boolean; onToggle: () => void }) {
  const [detail, setDetail] = useState<SignalCompany | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!open || detail) return;
    api.signalsCompany(c.company_id).then(setDetail).catch(() => setFailed(true));
  }, [open, detail, c.company_id]);

  const shown = family ? c.families[family].new_30 : c.new_30;
  const mix = (Object.keys(FAMILY_LABEL) as SignalFamily[]).filter((f) => c.families[f].new_30 > 0);
  const rate = c.baseline_monthly !== null && c.enough_history ? `${c.baseline_monthly.toFixed(c.baseline_monthly < 10 ? 1 : 0)}/mo before` : "short history";
  return (
    <li className={`sg-row${open ? " is-open" : ""}`}>
      <button type="button" className="sg-row__main" aria-expanded={open} onClick={onToggle}>
        <span className="sg-row__who">
          <span className="sg-row__name">{c.name ?? c.domain ?? "Unnamed company"}</span>
          <span className="sg-row__domain">{c.name ? (c.domain ?? "domain not provided") : "name not provided"}</span>
        </span>
        <PatternBadge pattern={c.pattern} />
        <span className="sg-row__num"><strong>{shown}</strong><span>new in 30 days · {rate}</span></span>
        <Sparkline values={c.weekly} weekEnds={weekEnds} tone={c.pattern} />
        <span className="sg-row__fam">{mix.map((f) => <span key={f} className="cp-chip">{FAMILY_LABEL[f]} {c.families[f].new_30}</span>)}</span>
      </button>
      {open && (
        <div className="sg-detail">
          {failed && <p className="cp-sub">Details could not be loaded.</p>}
          {!detail && !failed && <p className="cp-sub">Loading…</p>}
          {detail && <SignalDetail detail={detail} />}
        </div>
      )}
    </li>
  );
}
