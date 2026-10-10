import { useEffect, useState } from "react";
import type { SignalPattern } from "../types";
import { api } from "../lib/api";

/** Loads the burst/sustained map once the service is ready. Failure just means no badges. */
export function useSignalPatterns(enabled: boolean): Record<string, SignalPattern> {
  const [patterns, setPatterns] = useState<Record<string, SignalPattern>>({});
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    api.signalsPatterns().then((p) => { if (!cancelled) setPatterns(p); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [enabled]);
  return patterns;
}
