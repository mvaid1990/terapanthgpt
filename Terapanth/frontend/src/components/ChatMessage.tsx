// src/components/ChatMessage.tsx
// Renders a single chat message bubble with optional citations and follow-ups.
// Used by: src/app/chat/[id]/page.tsx

import { CitationBadge } from "./CitationBadge";
import { FollowUpSuggestions } from "./FollowUpSuggestions";
import { MessageOut } from "../services/api";

interface Props {
  message: MessageOut;
  onFollowUp: (q: string) => void;
}

export function ChatMessage({ message, onFollowUp }: Props) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-slate-800 text-slate-100 border border-slate-700"
        }`}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>

        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1 border-t border-slate-700 pt-2">
            {message.citations.map((c, i) => (
              <CitationBadge key={c.chunk_ref} citation={c} index={i} />
            ))}
          </div>
        )}

        {!isUser && message.follow_ups && (
          <FollowUpSuggestions
            suggestions={message.follow_ups}
            onSelect={onFollowUp}
          />
        )}
      </div>
    </div>
  );
}
