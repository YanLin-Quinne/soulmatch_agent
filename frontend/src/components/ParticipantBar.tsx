interface Participant {
  id: string;
  name: string;
  avatar: string;
  isHuman: boolean;
  emotion?: string;
  sentiment?: string;
  compatibilityScore?: number;
  inferredPercent?: number;  // 0-100 for human participants
}

const EMOTION_EMOJI: Record<string, string> = {
  joy: '😄', sadness: '😢', anger: '😠', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐', love: '😍',
  excitement: '🤩', anxiety: '😰',
};

const SENTIMENT_BG: Record<string, string> = {
  positive: 'ring-green-500',
  neutral: 'ring-yellow-500',
  negative: 'ring-red-500',
};

function ParticipantCard({ participant }: { participant: Participant }) {
  const { name, avatar, isHuman, emotion, sentiment, compatibilityScore, inferredPercent } = participant;

  const borderStyle = isHuman
    ? 'border-2 border-dashed border-blue-400'
    : 'border-2 border-solid border-green-500';

  const ringStyle = sentiment ? `ring-2 ${SENTIMENT_BG[sentiment] ?? 'ring-gray-500'}` : '';

  return (
    <div className="flex flex-col items-center gap-1 group">
      <div className="relative">
        <img
          src={avatar}
          alt={name}
          className={`w-10 h-10 rounded-full ${borderStyle} ${ringStyle}`}
        />
        {/* Emotion emoji badge */}
        {emotion && !isHuman && (
          <span className="absolute -bottom-1 -right-1 text-sm leading-none">
            {EMOTION_EMOJI[emotion] ?? '😐'}
          </span>
        )}
        {/* Human indicator */}
        {isHuman && (
          <span className="absolute -bottom-1 -right-1 text-sm leading-none">👤</span>
        )}
      </div>

      {/* Name */}
      <span className="text-xs text-gray-300 max-w-[60px] truncate text-center">
        {isHuman ? 'You' : name}
      </span>

      {/* Compatibility or inferred percent */}
      {!isHuman && compatibilityScore !== undefined && (
        <span className="text-[10px] text-pink-400">
          {(compatibilityScore * 100).toFixed(0)}%
        </span>
      )}
      {isHuman && inferredPercent !== undefined && (
        <div className="w-10">
          <div className="h-0.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-400 rounded-full transition-all duration-500"
              style={{ width: `${inferredPercent}%` }}
            />
          </div>
          <span className="text-[10px] text-blue-400 text-center block">
            {inferredPercent}%
          </span>
        </div>
      )}
    </div>
  );
}

interface ParticipantBarProps {
  participants: Participant[];
  activeBotId?: string;
}

export default function ParticipantBar({ participants, activeBotId }: ParticipantBarProps) {
  if (participants.length === 0) return null;

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-800/50 border-b border-gray-700 overflow-x-auto">
      <span className="text-xs text-gray-500 whitespace-nowrap">Participants</span>
      <div className="flex items-center gap-4">
        {participants.map((p) => (
          <div
            key={p.id}
            className={`transition-opacity ${
              activeBotId && !p.isHuman && p.id !== activeBotId ? 'opacity-50' : 'opacity-100'
            }`}
          >
            <ParticipantCard participant={p} />
          </div>
        ))}
      </div>
    </div>
  );
}
