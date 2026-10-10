import { useCallback, useRef, useState } from "react";
import type { CompassResult } from "../types";
import { api } from "../lib/api";

export type CompassState =
  | { status: "idle" }
  | { status: "loading"; jobIds: string[]; profileVersion: number }
  | { status: "ready"; jobIds: string[]; profileVersion: number; result: CompassResult }
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

  const reset = useCallback(() => {
    requestId.current++;
    setState({ status: "idle" });
  }, []);

  return { state, run, reset };
}
