import React from 'react';

interface EmotionDisplayProps {
  emotion: string;
  intensity: number;
  trend?: 'improving' | 'declining' | 'stable';
}

const EMOTION_EMOJIS: Record<string, string> = {
  joy: '😊',
  sadness: '😢',
  anger: '😠',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  neutral: '😐',
  love: '😍',
};

const EmotionDisplay: React.FC<EmotionDisplayProps> = ({
  emotion,
  intensity,
  trend = 'stable',
}) => {
  const emoji = EMOTION_EMOJIS[emotion.toLowerCase()] || '😐';
  const trendSymbol = trend === 'improving' ? '↗' : trend === 'declining' ? '↘' : '→';

  return (
    <div className="emotion-display">
      <div className="emotion-emoji">{emoji}</div>
      <div className="emotion-info">
        <div className="emotion-label">Current Mood</div>
        <div className="emotion-name">{emotion}</div>
        <div className={`emotion-trend ${trend}`}>
          {trendSymbol} {trend} ({Math.round(intensity * 100)}%)
        </div>
      </div>
    </div>
  );
};

export default EmotionDisplay;
