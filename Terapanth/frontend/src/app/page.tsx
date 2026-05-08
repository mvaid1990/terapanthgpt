// src/app/page.tsx
// Home page: lists past conversations and provides a new chat entry point.

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ConversationList } from "../components/ConversationList";
import { api, ConversationOut } from "../services/api";

export default function HomePage() {
  const router = useRouter();
  const [conversations, setConversations] = useState<ConversationOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listConversations()
      .then(setConversations)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  function handleNewChat() {
    router.push("/chat/new");
  }

  return (
    <main className="mx-auto max-w-xl px-4 py-12">
      <h1 className="mb-2 text-2xl font-semibold text-slate-100">
        Terapanth Learning
      </h1>
      <p className="mb-8 text-sm text-slate-400">
        Ask questions about Jain philosophy, guided by authoritative texts.
      </p>
      {loading ? (
        <p className="text-sm text-slate-500">Loading...</p>
      ) : (
        <ConversationList
          conversations={conversations}
          onNewChat={handleNewChat}
        />
      )}
    </main>
  );
}
