import { useState, useEffect } from 'react';

interface RelationshipData {
  rel_status: string;
  rel_type: string;
  sentiment: string;
  can_advance: boolean;
  advance_prediction_set: string[];
}

interface MilestoneReport {
  turn: number;
  type: string;
  message: string;
  predicted_status_at_turn_30?: string;
  final_status?: string;
  avg_trust_score?: number;
}

interface TrustHistory {
  turn: number;
  trust: number;
}

interface RelationshipTabProps {
  relationshipData: RelationshipData | null;
  milestoneReport: MilestoneReport | null;
  trustHistory: TrustHistory[];
  turnCount: number;
}

const STATUS_ORDER = ['stranger', 'acquaintance', 'crush', 'dating', 'committed'];
const STATUS_LABELS: Record<string, string> = {
  stranger: 'Stranger',
  acquaintance: 'Acquaintance',
  crush: 'Crush',
  dating: 'Dating',
  committed: 'Committed',
};

const SENTIMENT_COLORS: Record<string, string> = {
  positive: 'text-green-400',
  neutral: 'text-yellow-400',
  negative: 'text-red-400',
};

function AdvancePredictionBadge({ predictionSet }: { predictionSet: string[] }) {
  const hasYes = predictionSet.includes('yes');
  const hasUncertain = predictionSet.includes('uncertain');
  const hasNo = predictionSet.includes('no');

  let color = 'bg-gray-700 text-gray-300';
  let label = 'Uncertain';

  if (hasYes && !hasUncertain && !hasNo) {
    color = 'bg-green-900 text-green-300 border border-green-600';
    label = '✓ Can Advance';
  } else if (hasYes && hasUncertain) {
    color = 'bg-yellow-900 text-yellow-300 border border-yellow-600';
    label = '~ Maybe Advance';
  } else if (hasUncertain && hasNo || (hasNo && !hasYes)) {
    color = 'bg-red-900 text-red-300 border border-red-600';
    label = '✗ Not Ready';
  }

  return (
    <div className={`px-3 py-1.5 rounded-full text-xs font-medium ${color}`}>
      {label}
      <span className="ml-2 opacity-70">
        [{predictionSet.join(', ')}] @ 90%
      </span>
    </div>
  );
}

function RelationshipProgressBar({ status }: { status: string }) {
  const currentIdx = STATUS_ORDER.indexOf(status);
  const progress = currentIdx >= 0 ? ((currentIdx + 1) / STATUS_ORDER.length) * 100 : 0;

  return (
    <div className="mb-4">
      <div className="flex justify-between mb-1 text-xs text-gray-400">
        {STATUS_ORDER.map((s) => (
          <span
            key={s}
            className={
              s === status
                ? 'text-pink-400 font-medium'
                : STATUS_ORDER.indexOf(s) < (currentIdx >= 0 ? currentIdx : 0)
                ? 'text-gray-300'
                : 'text-gray-600'
            }
          >
            {STATUS_LABELS[s]}
          </span>
        ))}
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-pink-500 to-purple-500 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

function TrustTrajectoryChart({ history }: { history: TrustHistory[] }) {
  if (history.length < 2) {
    return (
      <div className="h-20 flex items-center justify-center text-gray-500 text-xs">
        Need more turns to show trust trajectory
      </div>
    );
  }

  const maxTrust = Math.max(...history.map((h) => h.trust));
  const minTrust = Math.min(...history.map((h) => h.trust));
  const range = maxTrust - minTrust || 0.1;
  const width = 240;
  const height = 60;
  const padding = 4;

  const points = history.map((h, i) => {
    const x = padding + (i / (history.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((h.trust - minTrust) / range) * (height - 2 * padding);
    return `${x},${y}`;
  });

  const currentTrust = history[history.length - 1]?.trust ?? 0.5;
  const trend = history.length > 1
    ? history[history.length - 1].trust - history[history.length - 2].trust
    : 0;

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="mb-1">
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke="#ec4899"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {history.map((h, i) => {
          const x = padding + (i / (history.length - 1)) * (width - 2 * padding);
          const y = height - padding - ((h.trust - minTrust) / range) * (height - 2 * padding);
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="2"
              fill={i === history.length - 1 ? '#f9a8d4' : '#ec4899'}
            />
          );
        })}
      </svg>
      <div className="flex justify-between text-xs text-gray-400">
        <span>Trust: {(currentTrust * 100).toFixed(0)}%</span>
        <span className={trend > 0 ? 'text-green-400' : trend < 0 ? 'text-red-400' : 'text-gray-400'}>
          {trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} {Math.abs(trend * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

export default function RelationshipTab({
  relationshipData,
  milestoneReport,
  trustHistory,
  turnCount,
}: RelationshipTabProps) {
  const [showMilestone, setShowMilestone] = useState(false);

  useEffect(() => {
    if (milestoneReport) {
      setShowMilestone(true);
      const timer = setTimeout(() => setShowMilestone(false), 8000);
      return () => clearTimeout(timer);
    }
  }, [milestoneReport]);

  if (!relationshipData) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        <div className="text-2xl mb-2">💫</div>
        <p>Relationship analysis will appear</p>
        <p className="text-xs mt-1 text-gray-600">after turn 5</p>
      </div>
    );
  }

  const { rel_status, rel_type, sentiment, advance_prediction_set } = relationshipData;

  return (
    <div className="p-4 space-y-4">
      {/* Milestone Report Modal */}
      {showMilestone && milestoneReport && (
        <div className="bg-purple-900/50 border border-purple-500 rounded-lg p-3 text-sm animate-pulse">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-purple-300 font-medium">
              🎯 Milestone Report (Turn {milestoneReport.turn})
            </span>
            <button
              onClick={() => setShowMilestone(false)}
              className="ml-auto text-gray-500 hover:text-gray-300"
            >
              ×
            </button>
          </div>
          <p className="text-gray-300 text-xs">{milestoneReport.message}</p>
          {milestoneReport.predicted_status_at_turn_30 && (
            <p className="text-purple-300 text-xs mt-1">
              Predicted at Turn 30: {milestoneReport.predicted_status_at_turn_30}
            </p>
          )}
        </div>
      )}

      {/* Relationship Status Progress */}
      <div>
        <div className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Relationship Stage
        </div>
        <RelationshipProgressBar status={rel_status} />
        <div className="flex items-center gap-2 text-xs mt-1">
          <span className="text-gray-500">Type:</span>
          <span className="text-pink-400 capitalize">{rel_type}</span>
          <span className="mx-1 text-gray-600">·</span>
          <span className="text-gray-500">Sentiment:</span>
          <span className={`capitalize ${SENTIMENT_COLORS[sentiment] ?? 'text-gray-400'}`}>
            {sentiment}
          </span>
        </div>
      </div>

      {/* Conformal Prediction: Can Advance? */}
      <div>
        <div className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Can Advance? (Conformal @ 90%)
        </div>
        <AdvancePredictionBadge predictionSet={advance_prediction_set ?? []} />
      </div>

      {/* Trust Trajectory */}
      <div>
        <div className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Trust Trajectory
        </div>
        <TrustTrajectoryChart history={trustHistory} />
      </div>

      {/* Turn counter */}
      <div className="text-xs text-gray-600 text-center border-t border-gray-700 pt-2">
        Next prediction at Turn {Math.ceil(turnCount / 5) * 5 + 5}
      </div>
    </div>
  );
}
