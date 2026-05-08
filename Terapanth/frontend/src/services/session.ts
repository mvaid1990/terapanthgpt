// src/services/session.ts
// Manages the anonymous guest session UUID in localStorage.
// Imported by: src/services/api.ts, src/app/layout.tsx

const SESSION_KEY = "terapanth_session_id";

function generateUUID(): string {
  return crypto.randomUUID();
}

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}
