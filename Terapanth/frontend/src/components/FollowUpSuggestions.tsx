// src/components/FollowUpSuggestions.tsx
// Row of clickable follow-up question chips shown below assistant messages.
// Used by: src/app/chat/[id]/page.tsx

interface Props {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export function FollowUpSuggestions({ suggestions, onSelect }: Props) {
  if (suggestions.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {suggestions.map((s, i) => (
        <button
          key={i}
          onClick={() => onSelect(s)}
          className="rounded-full border border-indigo-700 bg-indigo-950/50 px-3 py-1 text-xs text-indigo-300 hover:bg-indigo-900 transition-colors"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
