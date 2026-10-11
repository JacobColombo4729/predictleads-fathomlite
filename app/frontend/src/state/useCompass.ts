import { useCallback, useRef, useState } from "react";
import type { CompassResult } from "../types";
import { api } from "../lib/api";

export type CompassState =
  | { status: "idle" }
  | { status: "loading"; jobIds: string[]; profileVersion: number }
  | { status: "ready"; jobIds: string[]; profileVersion: number; result: CompassResult; updating?: boolean; notice?: string }
  | { status: "error"; jobIds: string[]; profileVersion: number; message: string };

interface Context {
  profileId: string | null;
  profileVersion: number;
  snapshotId: string | null;
}

/**
 * Runs the Compass analysis for the picked jobs. The result lives in memory only: it is recomputed from the saved
 * profile and the picks, so there is nothing to keep in sync on the server.
 */
export function useCompass({ profileId, profileVersion, snapshotId }: Context) {
  const [state, setState] = useState<CompassState>({ status: "idle" });
  const requestId = useRef(0);
  const latest = useRef(state);
  latest.current = state;

  const run = useCallback(async (jobIds: string[]) => {
    if (!profileId || !snapshotId) return;
    const mine = ++requestId.current;
    setState({ status: "loading", jobIds, profileVersion });
    try {
      const result = await api.compass(profileId, profileVersion, snapshotId, jobIds);
      if (mine === requestId.current) setState({ status: "ready", jobIds, profileVersion, result });
    } catch (e) {
      if (mine === requestId.current) {
        setState({ status: "error", jobIds, profileVersion, message: e instanceof Error ? e.message : "The Compass could not be built." });
      }
    }
  }, [profileId, profileVersion, snapshotId]);

  /** Re-suggest roles for a new stretch setting without replaying the reveal: the picks and everything else stay as they are. */
  const restretch = useCallback(async (stretch: number) => {
    const current = latest.current;
    if (current.status !== "ready" || !profileId || !snapshotId) return;
    const mine = ++requestId.current;
    setState({ ...current, updating: true, notice: undefined });
    try {
      const result = await api.compass(profileId, profileVersion, snapshotId, current.jobIds, stretch);
      if (mine === requestId.current) setState({ status: "ready", jobIds: current.jobIds, profileVersion, result });
    } catch (e) {
      if (mine === requestId.current) setState({ ...current, updating: false, notice: e instanceof Error ? e.message : "The suggestions could not be updated." });
    }
  }, [profileId, profileVersion, snapshotId]);

  const reset = useCallback(() => {
    requestId.current++;
    setState({ status: "idle" });
  }, []);

  return { state, run, restretch, reset };
}
