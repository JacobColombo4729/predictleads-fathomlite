import { useState } from "react";
import type { ResearchResponse } from "../../lib/market";
import { api } from "../../lib/api";
import MlNote from "../MlNote";

type State = { status: "idle" } | { status: "loading" } | { status: "error"; message: string } | { status: "done"; data: ResearchResponse };

/**
 * The optional second step: a short, sourced synopsis of one company from the web, written by OpenAI from search results only.
 * It is a separate action the person chooses, and it says plainly that it is not PredictLeads data.
 */
export default function ResearchCard({ companyId, companyName }: { companyId: string; companyName: string }) {
  const [state, setState] = useState<State>({ status: "idle" });

  const run = async () => {
    setState({ status: "loading" });
    try {
      setState({ status: "done", data: await api.companyResearch(companyId) });
    } catch (e) {
      setState({ status: "error", message: e instanceof Error ? e.message : "The research could not be completed." });
    }
  };

  if (state.status === "idle") {
    return (
      <div className="mk-research">
        <p className="cp-sub">Want more than our data shows? We can research {companyName} on the web: a short summary of what the company does, with its sources. Only the company's name and website are sent to OpenAI.</p>
        <button type="button" className="btn btn--sm" onClick={() => void run()}>Research this company</button>
        <MlNote id="company-research" />
      </div>
    );
  }
  if (state.status === "loading") return <p className="cp-sub" role="status">Searching the web for {companyName}… this can take 10 to 30 seconds.</p>;
  if (state.status === "error") {
    return (
      <div className="mk-research">
        <p className="field__error" role="alert">{state.message}</p>
        <button type="button" className="btn btn--sm" onClick={() => void run()}>Try again</button>
      </div>
    );
  }
  const d = state.data;
  if (d.state === "unavailable") return <p className="notice" role="alert">Web research is turned off on this server, so nothing was sent. Ask whoever runs the demo to start it without the -NoAI option.</p>;
  if (d.state === "no_sources") return <p className="notice">{d.message}</p>;
  return (
    <div className="mk-research mk-research--done">
      <p className="mk-research__label">{d.label}</p>
      <p>{d.text}</p>
      <div className="mk-research__src">
        <span className="cp-eyebrow">Sources</span>
        <ul>{d.sources.map((s) => <li key={s.url}><a className="link" href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a></li>)}</ul>
      </div>
      <p className="cp-sub">Retrieved {d.retrieved}. Check anything important against the sources.</p>
    </div>
  );
}
