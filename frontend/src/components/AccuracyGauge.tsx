import React from 'react';

interface AccuracyGaugeProps {
  accuracy: number;
  nFeatures: number;
  highConfidenceCount?: number;
  method?: 'ground_truth' | 'proxy' | 'none';
}

const getAccuracyColor = (accuracy: number): string => {
  if (accuracy >= 0.8) return '#22c55e';
  if (accuracy >= 0.6) return '#eab308';
  if (accuracy >= 0.4) return '#f97316';
  return '#ef4444';
};

const getAccuracyLabel = (accuracy: number): string => {
  if (accuracy >= 0.8) return 'High';
  if (accuracy >= 0.6) return 'Moderate';
  if (accuracy >= 0.4) return 'Low';
  return 'Insufficient';
};

const AccuracyGauge: React.FC<AccuracyGaugeProps> = ({
  accuracy,
  nFeatures,
  highConfidenceCount = 0,
  method = 'proxy',
}) => {
  const pct = Math.round(accuracy * 100);
  const color = getAccuracyColor(accuracy);
  const label = getAccuracyLabel(accuracy);

  // SVG arc for the gauge
  const radius = 40;
  const circumference = Math.PI * radius; // half-circle
  const filled = circumference * accuracy;

  return (
    <div className="accuracy-gauge">
      <svg width="100" height="60" viewBox="0 0 100 60">
        {/* Background arc */}
        <path
          d="M 10 55 A 40 40 0 0 1 90 55"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d="M 10 55 A 40 40 0 0 1 90 55"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
      </svg>
      <div className="accuracy-value" style={{ color }}>
        {pct}%
      </div>
      <div className="accuracy-label">Perception: {label}</div>
      <div className="accuracy-meta">
        {nFeatures} features | {highConfidenceCount} confident
        {method === 'ground_truth' && ' | verified'}
      </div>
    </div>
  );
};

export default AccuracyGauge;
