// src/hooks/useChat.ts
// Manages chat state: messages, sending, loading, error.
// Used by: src/app/chat/[id]/page.tsx

"use client";

import { useCallback, useState } from "react";
import { api, MessageOut } from "../services/api";

export function useChat(conversationId: string) {
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      const msgs = await api.getMessages(conversationId);
      setMessages(msgs);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load messages");
    }
  }, [conversationId]);

  const sendMessage = useCallback(
    async (text: string): Promise<string | null> => {
      const userMsg: MessageOut = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        citations: null,
        follow_ups: null,
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      setError(null);

      try {
        const response = await api.sendMessage(text, conversationId);
        setMessages((prev) => [...prev, response.message]);
        return response.conversation_id;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to send message");
        return null;
      } finally {
        setLoading(false);
      }
    },
    [conversationId]
  );

  return { messages, loading, error, loadHistory, sendMessage };
}
