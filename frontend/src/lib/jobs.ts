import type { HistoricalStatus } from "../types";

export function formatDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatSavedDate(isoTimestamp: string): string {
  return new Date(isoTimestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function safeHttpUrl(url: string): string | null {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

export const historicalStatusText: Record<HistoricalStatus, string> = {
  listed_when_observed: "Listed when observed",
  not_in_latest_snapshot: "Closed in source data",
  unknown_status: "Unknown (not stated in the source)",
};

export function descriptionParagraphs(text: string): { type: "p" | "ul"; items: string[] }[] {
  const blocks: { type: "p" | "ul"; items: string[] }[] = [];
  for (const raw of text.split(/\n\s*\n/)) {
    const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    const bullets = lines.filter((l) => l.startsWith("- "));
    if (bullets.length === lines.length) {
      blocks.push({ type: "ul", items: bullets.map((l) => l.slice(2)) });
    } else {
      blocks.push({ type: "p", items: [lines.join(" ")] });
    }
  }
  return blocks;
}
