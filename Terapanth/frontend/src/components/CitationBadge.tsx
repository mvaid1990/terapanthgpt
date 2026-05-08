// src/components/CitationBadge.tsx
// Displays a single source citation as a small pill.
// Used by: src/components/ChatMessage.tsx

import { Citation } from "../services/api";

interface Props {
  citation: Citation;
  index: number;
}

export function CitationBadge({ citation, index }: Props) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-indigo-950 border border-indigo-700 px-2 py-0.5 text-xs text-indigo-300">
      [{index + 1}] {citation.title}
      {citation.author && <span className="text-indigo-500">· {citation.author}</span>}
    </span>
  );
}
