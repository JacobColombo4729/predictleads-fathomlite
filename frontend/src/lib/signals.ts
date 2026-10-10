import { createContext, useContext } from "react";
import type { SignalFamily, SignalPattern } from "../types";

export const PATTERN_LABEL: Record<SignalPattern, string> = { burst: "Burst", sustained: "Sustained", cooling: "Cooling", quiet: "Quiet" };
export const PATTERN_ICON: Record<SignalPattern, string> = { burst: "▲", sustained: "●", cooling: "▼", quiet: "–" };
export const PATTERN_HELP: Record<SignalPattern, string> = {
  burst: "Opened far more roles than its own recent pace. Time-sensitive: bursts taper, usually to about 60% of their volume the next month.",
  sustained: "Opens roles at a steady pace month after month. Reliable ongoing hiring.",
  cooling: "Hiring has slowed compared with its own recent pace.",
  quiet: "No notable change in hiring.",
};
export const FAMILY_LABEL: Record<SignalFamily, string> = { ai_ml: "AI / ML", data: "Data", technical_sales: "Technical sales" };

/** company id -> pattern, for the companies worth a badge (burst, sustained or cooling). Empty when signals are unavailable. */
export const SignalPatternsContext = createContext<Record<string, SignalPattern>>({});
export const useSignalPattern = (companyId: string | null | undefined): SignalPattern | null =>
  useContext(SignalPatternsContext)[companyId ?? ""] ?? null;

export const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

export const pct = (share: number): string => `${Math.round(share * 100)}%`;
