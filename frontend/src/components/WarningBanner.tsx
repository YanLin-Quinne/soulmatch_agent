import React from 'react';

interface WarningBannerProps {
  level: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  onClose: () => void;
}

const WARNING_ICONS: Record<string, string> = {
  low: '⚠️',
  medium: '⚠️',
  high: '🚨',
  critical: '🛑',
};

const WarningBanner: React.FC<WarningBannerProps> = ({
  level,
  message,
  onClose,
}) => {
  const icon = WARNING_ICONS[level] || '⚠️';

  return (
    <div className={`warning-banner ${level}`}>
      <div className="warning-content">
        <span className="warning-icon">{icon}</span>
        <span className="warning-text">{message}</span>
      </div>
      <button className="close-button" onClick={onClose} aria-label="Close warning">
        ✕
      </button>
    </div>
  );
};

export default WarningBanner;
