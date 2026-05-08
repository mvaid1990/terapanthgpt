// src/components/ConversationList.tsx
// Sidebar list of past conversations with a new chat button.
// Used by: src/app/page.tsx

"use client";

import Link from "next/link";
import { ConversationOut } from "../services/api";

interface Props {
  conversations: ConversationOut[];
  onNewChat: () => void;
}

export function ConversationList({ conversations, onNewChat }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={onNewChat}
        className="w-full rounded-xl border border-indigo-600 bg-indigo-600/20 px-4 py-2.5 text-sm font-medium text-indigo-300 hover:bg-indigo-600/30 transition-colors"
      >
        + New Conversation
      </button>
      {conversations.length === 0 && (
        <p className="text-center text-xs text-slate-500 mt-4">
          No conversations yet. Ask your first question.
        </p>
      )}
      {conversations.map((conv) => (
        <Link
          key={conv.id}
          href={`/chat/${conv.id}`}
          className="rounded-xl border border-slate-700 bg-slate-800/50 px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 transition-colors truncate"
        >
          {conv.title}
        </Link>
      ))}
    </div>
  );
}
