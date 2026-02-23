import React from 'react';

export interface BotInfo {
  bot_id: string;
  persona_summary: string;
  compatibility_score: number;
  emoji?: string;
}

interface CharacterCardProps {
  bot: BotInfo;
  isActive: boolean;
  onClick: () => void;
}

const CharacterCard: React.FC<CharacterCardProps> = ({
  bot,
  isActive,
  onClick,
}) => {
  const compatibilityPercent = Math.round(bot.compatibility_score * 100);
  const emoji = bot.emoji || '🤖';

  return (
    <div
      className={`character-card ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      <div className="card-header">
        <div className="avatar">{emoji}</div>
        <div className="card-info">
          <h3>{bot.bot_id}</h3>
          <div className="compatibility-score">
            ❤️ {compatibilityPercent}% Match
          </div>
        </div>
      </div>
      <p className="card-summary">{bot.persona_summary}</p>
    </div>
  );
};

export default CharacterCard;
