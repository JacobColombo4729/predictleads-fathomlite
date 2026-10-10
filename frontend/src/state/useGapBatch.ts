import { useCallback, useRef, useState } from "react";
import type { GapBatchResult } from "../types";
import { api } from "../lib/api";

export type GapBatchState =
  | { status: "idle" }
  | { status: "loading"; key: string }
  | { status: "unavailable"; key: string }
  | { status: "error"; key: string; message: string }
  | { status: "done"; key: string; result: Extract<GapBatchResult, { state: "completed" }> };

/** The key identifies the set of jobs a result belongs to, so a result is never shown against different picks. */
export const gapKey = (jobIds: string[]): string => [...jobIds].sort().join("|");

/**
 * AI skills gap analysis for the Compass picks. It lives in the app (not the page) so the result and the consent
 * survive moving between tabs. The server caches each result per session, so asking again for the same picks is free.
 */
export function useGapBatch(snapshotId: string | null) {
  const [consent, setConsent] = useState(false);
  const [state, setState] = useState<GapBatchState>({ status: "idle" });
  const requestId = useRef(0);

  const run = useCallback(async (jobIds: string[]) => {
    if (!snapshotId || !consent || jobIds.length === 0) return;
    const key = gapKey(jobIds);
    const mine = ++requestId.current;
    setState({ status: "loading", key });
    try {
      const result = await api.compassGapAnalysis(snapshotId, jobIds);
      if (mine !== requestId.current) return;
      setState(result.state === "unavailable" ? { status: "unavailable", key } : { status: "done", key, result });
    } catch (e) {
      if (mine === requestId.current) setState({ status: "error", key, message: e instanceof Error ? e.message : "The analysis could not be completed." });
    }
  }, [snapshotId, consent]);

  const reset = useCallback(() => {
    requestId.current++;
    setState({ status: "idle" });
    setConsent(false);
  }, []);

  /** Back to the start for the same picks (keeps the consent tick). */
  const clear = useCallback(() => { requestId.current++; setState({ status: "idle" }); }, []);

  return { consent, setConsent, state, run, reset, clear };
}

export type GapBatch = ReturnType<typeof useGapBatch>;
