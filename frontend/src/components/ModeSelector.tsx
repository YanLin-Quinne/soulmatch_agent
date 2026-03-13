interface ModeSelectorProps {
  onSelectMode: (mode: 'personality' | 'playground') => void;
}

export default function ModeSelector({ onSelectMode }: ModeSelectorProps) {
  return (
    <div className="mode-selector-screen">
      <div className="mode-selector-content">
        <h1 className="mode-title">SoulMatch Agent</h1>
        <p className="mode-subtitle">Choose Your Experience Mode</p>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('personality')}>
            <div className="mode-icon">🎭</div>
            <h3>Personality Inference Mode</h3>
            <p>Chat with a persona for 30 turns, and the system will infer your personality profile</p>
            <div className="mode-features">
              <span>✓ 15 personas available</span>
              <span>✓ 10 AI + 5 real humans</span>
              <span>✓ Big Five personality analysis</span>
            </div>
          </div>

          <div className="mode-card" onClick={() => onSelectMode('playground')}>
            <div className="mode-icon">🧬</div>
            <h3>AI Digital Twin Mode</h3>
            <p>Predict friend's personality, chat with their AI twin, then compare accuracy</p>
            <div className="mode-features">
              <span>✓ Predict friend's traits</span>
              <span>✓ Chat with AI twin</span>
              <span>✓ Match accuracy analysis</span>
            </div>
          </div>
        </div>

        <p className="mode-note">💡 Both modes are based on multi-agent architecture and Bayesian inference</p>
      </div>
    </div>
  );
}
