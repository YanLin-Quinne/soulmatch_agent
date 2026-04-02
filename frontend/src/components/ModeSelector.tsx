interface ModeSelectorProps {
  onSelectMode: (mode: 'personality' | 'playground') => void;
}

export default function ModeSelector({ onSelectMode }: ModeSelectorProps) {
  return (
    <div className="mode-selector-screen">
      <div className="mode-selector-content">
        <h1 className="mode-title">AI YOU</h1>
        <p className="mode-subtitle">Choose your experience mode</p>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('personality')}>
            <div className="mode-icon">🎭</div>
            <h3>Personality Inference Mode</h3>
            <p>Chat with someone for 30 turns and the system will infer your personality profile</p>
            <div className="mode-features">
              <span>✓ 15 characters to choose</span>
              <span>✓ 10 AI + 5 real people</span>
              <span>✓ Big Five personality analysis</span>
            </div>
          </div>

          <div className="mode-card" onClick={() => onSelectMode('playground')}>
            <div className="mode-icon">🧬</div>
            <h3>AI Digital Twin Mode</h3>
            <p>Predict your friend's personality, chat with their AI twin, then compare accuracy</p>
            <div className="mode-features">
              <span>✓ Predict friend's traits</span>
              <span>✓ Chat with AI twin</span>
              <span>✓ Match analysis</span>
            </div>
          </div>
        </div>

        <p className="mode-note">💡 Both modes are powered by multi-agent architecture and Bayesian inference</p>
      </div>
    </div>
  );
}
