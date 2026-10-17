import { useCallback, useEffect, useState, type FormEvent } from "react";
import type { MarketExample, MarketOverview, MarketQuery, MarketResponse } from "../../lib/market";
import { fullQuery, withoutFields } from "../../lib/market";
import { api } from "../../lib/api";
import { VOICE_MAX_SECONDS, useVoiceRecorder, voiceSupported } from "../../lib/voice";
import MarketResultView from "./MarketResultView";
import MlNote from "../MlNote";

function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

const FALLBACK = "Which companies are hiring people who use Snowflake or dbt?";

/** "Ask the market": a question box over our own hiring data, with a clear read-back of how each question was understood. */
export default function AskMarket() {
  const [overview, setOverview] = useState<MarketOverview | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<MarketResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.marketOverview().then(setOverview).catch(() => setOverview(null));
  }, []);

  const show = useCallback(async (action: () => Promise<MarketResponse>, asked: string) => {
    setBusy(true);
    setError(null);
    setQuestion(asked);
    try {
      setResponse(await action());
    } catch (e) {
      setResponse(null);
      setError(e instanceof Error ? e.message : "That could not be answered. Please try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  // Talk: record a short question, turn it into text, put it in the box and ask it. The person can edit the text and ask again.
  const heard = useCallback(async (audio: Blob) => {
    const result = await api.transcribe(audio);
    if (result.state === "ok") {
      setText(result.text);
      await show(() => api.marketAsk(result.text), result.text);
    } else if (result.state === "empty") {
      setError(result.message);
    } else {
      setError("Voice questions are turned off on this server. Ask whoever runs the demo to start it without the -NoAI option.");
    }
  }, [show]);
  const voice = useVoiceRecorder(heard, setError);

  if (!overview || !overview.available) return null;
  const voiceOn = overview.voice_enabled && voiceSupported();
  const examples = overview.examples.slice(0, 5);
  const placeholder = overview.examples[0]?.question ?? FALLBACK;

  const ask = (e: FormEvent) => {
    e.preventDefault();
    const q = text.trim();
    if (q.length < 3 || busy) return;
    void show(() => api.marketAsk(q), q);
  };

  const useExample = (ex: MarketExample) => {
    setText(ex.question);
    void show(() => api.marketRun(fullQuery(ex.query)), ex.question);
  };

  const loosen = (query: MarketQuery, fields: string[]) => void show(() => api.marketRun(withoutFields(query, fields)), question);

  return (
    <section className="cp-card mk" aria-label="Ask the market">
      <header className="cp-cardhead">
        <div>
          <div className="cp-eyebrow">Ask the market</div>
          <h2>Ask a question about who is hiring</h2>
          <p className="cp-sub">Answers come only from our own listings and hiring signals. Nothing is guessed: if we do not have something, we say so.</p>
        </div>
      </header>

      <form className="mk-form" onSubmit={ask}>
        <input
          type="text"
          className="mk-input"
          value={text}
          maxLength={300}
          placeholder={placeholder}
          aria-label="Ask a question about the hiring market"
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="btn btn--primary" disabled={busy || voice.state !== "idle" || text.trim().length < 3}>{busy ? "Working…" : "Ask"}</button>
        {voiceOn && (
          <button
            type="button"
            className={`btn mk-talk${voice.state === "recording" ? " is-recording" : ""}`}
            aria-pressed={voice.state === "recording"}
            disabled={busy || voice.state === "asking" || voice.state === "sending"}
            title="Speak your question"
            onClick={() => (voice.state === "recording" ? voice.stop() : void voice.start())}
          >
            <MicIcon />
            {voice.state === "recording" ? `Stop (${voice.seconds}s)` : voice.state === "asking" ? "Allow the microphone…" : voice.state === "sending" ? "Turning your voice into text…" : "Talk"}
          </button>
        )}
      </form>
      <p className="sr-only" role="status" aria-live="polite">
        {voice.state === "recording" ? `Listening. Speak your question, then press Stop. It stops by itself after ${VOICE_MAX_SECONDS} seconds.` : ""}
      </p>
      {voice.state === "recording" && <p className="cp-sub mk-listening">Listening… speak your question, then press Stop.</p>}
      <div className="mk-examples" role="group" aria-label="Example questions">
        <span className="cp-sub">Try:</span>
        {examples.map((ex) => <button key={ex.question} type="button" className="chip" disabled={busy} onClick={() => useExample(ex)}>{ex.question}</button>)}
      </div>
      <p className="cp-sub mk-disclosure">
        {overview.ask_enabled
          ? "Your question (nothing from your profile) is sent to OpenAI to be turned into a search. The answer then comes from our data." + (voiceOn ? " Talk records up to 30 seconds from your microphone, sends it to OpenAI to be turned into text, and does not keep it." : "")
          : "Typing your own questions needs the AI switched on for this server. The examples above work without it."}
      </p>
      <MlNote id="ask-market" />

      {error && <p className="field__error" role="alert">{error}</p>}
      {response?.state === "unavailable" && (
        <p className="notice" role="alert">Typed questions are turned off on this server, so nothing was sent. Use one of the examples above, or ask whoever runs the demo to start it without the -NoAI option.</p>
      )}
      {response?.state === "unsupported" && (
        <div className="mk-unsupported" role="status">
          <p><strong>We can't answer that from our data.</strong> {response.reason}</p>
          <p className="cp-sub">Try asking about who is hiring, which technologies appear in listings, pay stated in listings, or how hiring has moved over time.</p>
        </div>
      )}
      {response?.state === "ok" && (
        <div className="mk-answer" aria-live="polite">
          <p className="mk-q">{question}</p>
          <div className="mk-read">
            <span className="cp-eyebrow">How we read your question</span>
            <div className="cp-chips">
              {response.result.interpretation.map((c) => (
                <span key={`${c.label}${c.value}`} className="mk-chip">
                  <b>{c.label}:</b> {c.value}
                  {c.fields.length > 0 && (
                    <button type="button" aria-label={`Remove ${c.label}`} title="Remove this filter and search again" disabled={busy} onClick={() => loosen(response.query, c.fields)}>×</button>
                  )}
                </span>
              ))}
            </div>
          </div>
          <MarketResultView result={response.result} researchEnabled={overview.research_enabled} />
        </div>
      )}
    </section>
  );
}
