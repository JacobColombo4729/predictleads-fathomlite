interface Envelope<T> {
  version: number;
  data: T;
}

export type ReadResult<T> =
  | { status: "missing"; value: null }
  | { status: "ok"; value: T }
  | { status: "corrupt"; value: null };

export function readVersioned<T>(
  key: string,
  version: number,
  validate: (data: unknown) => T | null,
): ReadResult<T> {
  let raw: string | null;
  try {
    raw = localStorage.getItem(key);
  } catch {
    return { status: "missing", value: null };
  }
  if (raw === null) return { status: "missing", value: null };
  try {
    const parsed = JSON.parse(raw) as Partial<Envelope<unknown>> | null;
    if (parsed && typeof parsed === "object" && parsed.version === version) {
      const value = validate(parsed.data);
      if (value !== null) return { status: "ok", value };
    }
  } catch {
    // fall through to corrupt
  }
  return { status: "corrupt", value: null };
}

export function writeVersioned<T>(key: string, version: number, data: T): void {
  try {
    localStorage.setItem(key, JSON.stringify({ version, data } satisfies Envelope<T>));
  } catch {
    // Storage may be full or disabled; state still works for the session.
  }
}
