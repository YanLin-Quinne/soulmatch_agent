interface InferredTraits {
  gender?: string;
  age_range?: string;
  occupation_guess?: string;
  mbti?: string;
  bigFive?: {
    E: number;
    A: number;
  };
  communication_style?: string;
}

interface FriendGuess {
  gender: string;
  age_range: string;
  mbti: string;
  occupation: string;
  EI: number;
  TF: number;
  description: string;
}

interface ComparisonViewProps {
  systemInference: InferredTraits;
  friendGuess: FriendGuess;
  onRestart: () => void;
}

export default function ComparisonView({
  systemInference,
  friendGuess,
  onRestart
}: ComparisonViewProps) {
  // Calculate match score
  let score = 0;
  let total = 0;

  // Gender match
  if (friendGuess.gender && systemInference.gender) {
    total++;
    if (friendGuess.gender === systemInference.gender) score++;
  }

  // MBTI match
  if (friendGuess.mbti && systemInference.mbti && friendGuess.mbti.length === 4) {
    for (let i = 0; i < 4; i++) {
      total++;
      if (friendGuess.mbti[i] === systemInference.mbti[i]) score++;
    }
  }

  // E/I slider match
  total++;
  const sysE = (systemInference.bigFive?.E || 0.5) * 100;
  const friendE = 100 - friendGuess.EI;
  if (Math.abs(sysE - friendE) < 25) score++;

  // T/F slider match
  total++;
  const sysA = (systemInference.bigFive?.A || 0.5) * 100;
  if (Math.abs(sysA - friendGuess.TF) < 25) score++;

  const matchPct = total > 0 ? Math.round((score / total) * 100) : 50;
  const matchClass = matchPct >= 70 ? 'high' : matchPct >= 40 ? 'mid' : 'low';
  const matchText =
    matchPct >= 70
      ? 'High match! You know them well'
      : matchPct >= 40
      ? 'Partial match, you got some traits right'
      : 'Big difference, maybe they have a side you don\'t know';

  return (
    <div className="comparison-screen">
      <div className="comparison-content">
        <h2 className="comparison-title">🔍 Profile Comparison</h2>
        <p className="comparison-subtitle">Your Prediction vs System Inference</p>

        <div className={`match-badge ${matchClass}`}>
          {matchPct}% Match — {matchText}
        </div>

        <div className="compare-grid">
          <div className="compare-card system">
            <h3>🤖 System Inference</h3>
            <div className="trait-row">
              <span className="trait-label">Gender</span>
              <span className="trait-value">{systemInference.gender || '?'}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Age</span>
              <span className="trait-value">{systemInference.age_range || '?'}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Occupation</span>
              <span className="trait-value">{systemInference.occupation_guess || '?'}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">MBTI</span>
              <span className="trait-value">{systemInference.mbti || '?'}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Extraversion</span>
              <span className="trait-value">
                {Math.round((systemInference.bigFive?.E || 0.5) * 100)}%
              </span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Agreeableness</span>
              <span className="trait-value">
                {Math.round((systemInference.bigFive?.A || 0.5) * 100)}%
              </span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Communication Style</span>
              <span className="trait-value">{systemInference.communication_style || '?'}</span>
            </div>
          </div>

          <div className="compare-card friend">
            <h3>👤 Your Prediction</h3>
            <div className="trait-row">
              <span className="trait-label">Gender</span>
              <span className="trait-value">
                {friendGuess.gender}{' '}
                {friendGuess.gender === systemInference.gender ? '✅' : '❌'}
              </span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Age</span>
              <span className="trait-value">{friendGuess.age_range}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Occupation</span>
              <span className="trait-value">{friendGuess.occupation || 'N/A'}</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">MBTI</span>
              <span className="trait-value">
                {friendGuess.mbti || 'N/A'}{' '}
                {friendGuess.mbti === systemInference.mbti ? '✅' : ''}
              </span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Extraversion</span>
              <span className="trait-value">{100 - friendGuess.EI}%</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Emotionality</span>
              <span className="trait-value">{friendGuess.TF}%</span>
            </div>
            <div className="trait-row">
              <span className="trait-label">Your Description</span>
              <span className="trait-value" style={{ fontSize: '11px' }}>
                {friendGuess.description || 'None'}
              </span>
            </div>
          </div>
        </div>

        <button className="restart-btn" onClick={onRestart}>
          🔄 Back to Home
        </button>
      </div>
    </div>
  );
}
