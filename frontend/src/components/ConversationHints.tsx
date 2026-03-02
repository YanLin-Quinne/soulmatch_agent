interface ConversationHintsProps {
  hints: { text: string; type: string }[] | null;
  onSelectHint: (text: string) => void;
  onDismiss: () => void;
}

export default function ConversationHints({ hints, onSelectHint, onDismiss }: ConversationHintsProps) {
  if (!hints || hints.length === 0) return null;

  return (
    <div className="hints-bar">
      <div className="hints-bar-inner">
        {hints.map((hint, idx) => (
          <button
            key={idx}
            className="hint-chip"
            onClick={() => onSelectHint(hint.text)}
            title={hint.text}
          >
            {hint.text}
          </button>
        ))}
      </div>
      <button className="hints-dismiss" onClick={onDismiss} title="Dismiss suggestions">
        &times;
      </button>
    </div>
  );
}
