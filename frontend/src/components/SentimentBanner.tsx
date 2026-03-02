interface SentimentBannerProps {
  sentiment: { label: string; score: number; trend: string; hints: string[] } | null;
  turnCount: number;
}

export default function SentimentBanner({ sentiment, turnCount }: SentimentBannerProps) {
  // Only show after turn 5
  if (!sentiment || turnCount < 5) return null;

  const { label, score, trend, hints } = sentiment;

  const trendArrow = trend === 'improving' ? '\u2197' : trend === 'declining' ? '\u2198' : '\u2192';
  const className = `sentiment-banner ${label}`;

  return (
    <div className={className}>
      <span className="sentiment-label">
        {label === 'positive' ? '\u263A' : label === 'negative' ? '\u2639' : '\u25CB'}{' '}
        {label.charAt(0).toUpperCase() + label.slice(1)}
      </span>
      <span className="sentiment-score">{(score * 100).toFixed(0)}%</span>
      <span className="sentiment-trend">{trendArrow} {trend}</span>
      {hints.length > 0 && (
        <span className="sentiment-hint">{hints[0]}</span>
      )}
    </div>
  );
}
