import { useEffect, useRef, useState } from "react";
import { prefersReducedMotion } from "./useAnimatedNumber";

interface Props {
  jobCount: number;
  /** True once the analysis has come back. */
  ready: boolean;
  error: string | null;
  onFinished: () => void;
  onRetry: () => void;
  onBack: () => void;
}

const STEP_MS = 850;

/** The "analyzing your picks" sequence: steps tick off while the needle swings, then settles when results arrive. */
export default function Reveal({ jobCount, ready, error, onFinished, onRetry, onBack }: Props) {
  const steps = [
    `Reading your ${jobCount} job descriptions`,
    "Grouping what they ask for into themes",
    "Matching it to your profile and resume",
    "Charting your options",
  ];
  const reduce = prefersReducedMotion();
  const [done, setDone] = useState(0);
  const [angle, setAngle] = useState(0);
  const finished = useRef(false);

  // Steps advance on a timer; the last one waits for the data to arrive.
  useEffect(() => {
    if (error || reduce || done >= steps.length) return;
    if (done === steps.length - 1 && !ready) return;
    const t = setTimeout(() => setDone((d) => d + 1), STEP_MS);
    return () => clearTimeout(t);
  }, [done, ready, error, reduce, steps.length]);

  // The needle wobbles while working.
  useEffect(() => {
    if (error || reduce || done >= steps.length) return;
    const wobble = [70, -40, 25, 0];
    let i = 0;
    const t = setInterval(() => setAngle(wobble[i++ % wobble.length]), 750);
    return () => clearInterval(t);
  }, [done, error, reduce, steps.length]);

  // Settle on the strongest direction, then hand over to the results.
  useEffect(() => {
    if (finished.current || error) return;
    if (reduce ? ready : done >= steps.length && ready) {
      finished.current = true;
      setAngle(40);
      const t = setTimeout(onFinished, reduce ? 0 : 900);
      return () => clearTimeout(t);
    }
  }, [done, ready, error, reduce, steps.length, onFinished]);

  if (error) {
    return (
      <div className="cp-card cp-reveal cp-reveal--error" role="alert">
        <div>
          <h2>The Compass couldn’t be built</h2>
          <p className="cp-sub">{error}</p>
          <div className="cp-row">
            <button type="button" className="btn btn--primary" onClick={onRetry}>Try again</button>
            <button type="button" className="btn" onClick={onBack}>Back to Saved</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="cp-card cp-reveal" role="status" aria-live="polite">
      <svg className="cp-needle" viewBox="0 0 120 120" width="120" height="120" aria-hidden="true">
        <circle cx="60" cy="60" r="54" fill="none" stroke="var(--cp-grid)" strokeWidth="2" />
        <circle cx="60" cy="60" r="40" fill="none" stroke="var(--cp-grid)" strokeWidth="1" />
        <g stroke="var(--cp-axis)" strokeWidth="2"><path d="M60 6v8M60 106v8M6 60h8M106 60h8" /></g>
        <g
          style={{
            transformOrigin: "60px 60px",
            transformBox: "view-box",
            transform: `rotate(${angle}deg)`,
            transition: reduce ? "none" : "transform 0.7s cubic-bezier(.3,.7,.3,1)",
          }}
        >
          <path d="M60 16l8 44H52z" fill="var(--accent)" />
          <path d="M60 104l-8-44h16z" fill="var(--cp-axis)" />
          <circle cx="60" cy="60" r="5" fill="var(--surface)" stroke="var(--text-muted)" strokeWidth="2" />
        </g>
      </svg>
      <div className="cp-reveal__body">
        <div className="cp-eyebrow">Analyzing your picks</div>
        <ul className="cp-steps">
          {steps.map((label, i) => (
            <li key={label} className={i < done ? "is-done" : i === done ? "is-on" : ""}>
              <span className="cp-steps__ico" aria-hidden="true" />
              {label}
            </li>
          ))}
        </ul>
        <div className="cp-progress" aria-hidden="true">
          <i style={{ width: `${(Math.min(done + (done < steps.length ? 0.5 : 0), steps.length) / steps.length) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
