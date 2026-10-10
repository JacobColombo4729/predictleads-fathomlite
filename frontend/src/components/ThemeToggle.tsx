import { useState } from "react";
import { MoonIcon, SunIcon } from "./Icons";

export type Theme = "dark" | "light";
const KEY = "pl.theme";

/** Dark is the default; a saved choice wins. Browser storage may be unavailable, so every access is guarded. */
export function readTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // storage unavailable: fall through to the default
  }
  return "dark";
}

export function applyTheme(theme: Theme, remember: boolean): void {
  document.documentElement.dataset.theme = theme;
  if (!remember) return;
  try {
    window.localStorage.setItem(KEY, theme);
  } catch {
    // not remembered; the choice still applies for this visit
  }
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme === "light" ? "light" : "dark"));
  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="themebtn"
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      onClick={() => { applyTheme(next, true); setTheme(next); }}
    >
      {theme === "dark" ? <SunIcon width={18} height={18} /> : <MoonIcon width={18} height={18} />}
    </button>
  );
}
