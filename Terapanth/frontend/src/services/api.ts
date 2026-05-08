// src/services/api.ts
// Centralized API client. Injects X-Session-ID header on every request.
// Imported by: src/hooks/useChat.ts, src/app/page.tsx

import { getSessionId } from "./session";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Citation {
  title: string;
  author: string | null;
  chunk_ref: string;
}

export interface MessageOut {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  follow_ups: string[] | null;
}

export interface ConversationOut {
  id: string;
  title: string;
}

export interface ChatResponse {
  conversation_id: string;
  message: MessageOut;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Session-ID": getSessionId(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = error.detail;
    const message = Array.isArray(detail)
      ? detail.map((e: { msg?: string }) => e.msg ?? JSON.stringify(e)).join(", ")
      : String(detail ?? "Request failed");
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listConversations: (): Promise<ConversationOut[]> =>
    request("/api/v1/conversations"),

  getMessages: (conversationId: string): Promise<MessageOut[]> =>
    request(`/api/v1/conversations/${conversationId}/messages`),

  deleteConversation: (conversationId: string): Promise<void> =>
    request(`/api/v1/conversations/${conversationId}`, { method: "DELETE" }),

  sendMessage: (
    message: string,
    conversationId?: string
  ): Promise<ChatResponse> =>
    request("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ message, conversation_id: conversationId || null }),
    }),
};
