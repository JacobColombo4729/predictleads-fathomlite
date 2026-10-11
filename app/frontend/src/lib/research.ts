/**
 * An anonymous, random id for this browser. It is sent with every request so the server can tell which person a rating
 * or a shown list belongs to across visits; it carries no name or contact detail. Start over does not clear it.
 */
const KEY = "predictleads.participant";

export function participantId(): string | null {
  try {
    let id = window.localStorage.getItem(KEY);
    if (!id) {
      id = window.crypto.randomUUID();
      window.localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    return null; // storage blocked: the app works the same, nothing is linked to a person
  }
}
