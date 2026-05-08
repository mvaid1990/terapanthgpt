// src/app/chat/[id]/page.tsx
// Active chat view. Loads history and handles message sending.

"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ChatMessage } from "../../../components/ChatMessage";
import { useChat } from "../../../hooks/useChat";

export default function ChatPage() {
  const params = useParams();
  const router = useRouter();
  const rawId = params.id as string;
  const isNew = rawId === "new";

  const [activeId, setActiveId] = useState<string>(isNew ? "" : rawId);
  const { messages, loading, error, loadHistory, sendMessage } = useChat(activeId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeId) loadHistory();
  }, [activeId, loadHistory]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(text: string) {
    if (!text.trim()) return;
    setInput("");
    const returnedId = await sendMessage(text);
    if (returnedId && isNew) {
      setActiveId(returnedId);
      router.replace(`/chat/${returnedId}`, { scroll: false });
    }
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-slate-800 px-4 py-3 text-sm text-slate-400">
        <a href="/" className="hover:text-slate-200">← All conversations</a>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-2xl">
          {messages.length === 0 && !loading && (
            <p className="text-center text-sm text-slate-500">
              Ask anything about Jain philosophy to start.
            </p>
          )}
          {messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              message={msg}
              onFollowUp={(q) => handleSend(q)}
            />
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="rounded-2xl bg-slate-800 border border-slate-700 px-4 py-3 text-sm text-slate-400">
                Thinking...
              </div>
            </div>
          )}
          {error && (
            <p className="text-center text-xs text-red-400">{error}</p>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-slate-800 px-4 py-4">
        <div className="mx-auto flex max-w-2xl gap-2">
          <input
            className="flex-1 rounded-xl border border-slate-700 bg-slate-800 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            placeholder="Ask about Jain philosophy..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend(input)}
            disabled={loading}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
