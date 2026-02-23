import { useState, useEffect } from 'react';
import CyberPanel from '../ui/CyberPanel';
import type { RelationshipData, MilestoneReport, TrustPoint } from '../types';
import { STATUS_ORDER, STATUS_LABELS } from '../constants';

interface RelationPanelProps {
  relationshipData: RelationshipData | null;
  milestoneReport: MilestoneReport | null;
  trustHistory: TrustPoint[];
  turnCount: number;
}

export default function RelationPanel({ relationshipData, milestoneReport, trustHistory, turnCount }: RelationPanelProps) {
  const [showMilestone, setShowMilestone] = useState(false);

  useEffect(() => {
    if (milestoneReport) {
      setShowMilestone(true);
      const t = setTimeout(() => setShowMilestone(false), 8000);
      return () => clearTimeout(t);
    }
  }, [milestoneReport]);

  if (!relationshipData) {
    return (
      <CyberPanel title="RELATIONSHIP" accent="fuchsia">
        <div className="text-cyber-dim text-[11px] text-center py-4">
          <div className="text-xl mb-1">💫</div>
          ANALYSIS AFTER T:05
        </div>
      </CyberPanel>
    );
  }

  const { rel_status, rel_type, sentiment, advance_prediction_set, social_votes, vote_distribution } = relationshipData;
  const currentIdx = STATUS_ORDER.indexOf(rel_status);
  const progress = currentIdx >= 0 ? ((currentIdx + 1) / STATUS_ORDER.length) * 100 : 0;

  // Trust SVG
  const renderTrust = () => {
    if (trustHistory.length < 2) return <div className="text-[10px] text-cyber-dim py-2">Need more data...</div>;
    const w = 200, h = 36, pad = 4;
    const vals = trustHistory.map(h => h.trust);
    const min = Math.min(...vals) * 0.9;
    const max = Math.max(...vals) * 1.1 || 1;
    const range = max - min || 0.1;
    const points = vals.map((v, i) => {
      const x = pad + (i / (vals.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - min) / range) * (h - 2 * pad);
      return `${x},${y}`;
    }).join(' ');
    const last = vals[vals.length - 1];
    const trend = vals.length > 1 ? last - vals[vals.length - 2] : 0;

    return (
      <div>
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-9">
          <polyline points={points} fill="none" stroke="#ff00e5" strokeWidth="1.5" strokeLinecap="round" opacity="0.8" />
        </svg>
        <div className="flex justify-between text-[10px]">
          <span className="text-cyber-muted">Trust: {(last * 100).toFixed(0)}%</span>
          <span className={trend > 0 ? 'text-cyber-green' : trend < 0 ? 'text-cyber-red' : 'text-cyber-muted'}>
            {trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} {Math.abs(trend * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    );
  };

  const sentimentColor = sentiment === 'positive' ? 'text-cyber-green' : sentiment === 'negative' ? 'text-cyber-red' : 'text-cyber-amber';

  return (
    <CyberPanel title="RELATIONSHIP" accent="fuchsia">
      {/* Milestone */}
      {showMilestone && milestoneReport && (
        <div className="bg-cyber-fuchsia/10 border border-cyber-fuchsia/30 p-2 mb-2 text-[10px] animate-fade-in">
          <div className="flex justify-between items-center mb-1">
            <span className="text-cyber-fuchsia font-semibold">🎯 MILESTONE T:{milestoneReport.turn}</span>
            <button onClick={() => setShowMilestone(false)} className="text-cyber-dim hover:text-cyber-text">×</button>
          </div>
          <div className="text-cyber-text">{milestoneReport.message}</div>
          {milestoneReport.predicted_status_at_turn_30 && (
            <div className="text-cyber-fuchsia mt-0.5">→ T30: {milestoneReport.predicted_status_at_turn_30}</div>
          )}
        </div>
      )}

      {/* Progress bar */}
      <div className="mb-2">
        <div className="flex justify-between text-[9px] mb-1">
          {STATUS_ORDER.map((s, i) => (
            <span key={s} className={s === rel_status ? 'text-cyber-fuchsia font-semibold' : i < currentIdx ? 'text-cyber-muted' : 'text-cyber-dim'}>
              {STATUS_LABELS[s]}
            </span>
          ))}
        </div>
        <div className="h-1 bg-cyber-border rounded-sm overflow-hidden">
          <div className="h-full bg-gradient-to-r from-cyber-cyan to-cyber-fuchsia transition-all duration-500 rounded-sm"
            style={{ width: `${progress}%` }} />
        </div>
      </div>

      {/* Type + Sentiment */}
      <div className="flex gap-2 text-[10px] mb-2">
        <span className="text-cyber-muted">TYPE:<span className="text-cyber-cyan ml-0.5">{rel_type}</span></span>
        <span className="text-cyber-muted">SENT:<span className={`${sentimentColor} ml-0.5`}>{sentiment}</span></span>
      </div>

      {/* Advance prediction */}
      {advance_prediction_set && (
        <div className="mb-2">
          <div className="text-[9px] text-cyber-dim mb-0.5">CAN ADVANCE? (CP@90%)</div>
          <div className="flex gap-1">
            {advance_prediction_set.map((v) => (
              <span key={v} className={`text-[10px] px-1.5 py-0.5 border ${
                v === 'yes' ? 'bg-cyber-green/10 text-cyber-green border-cyber-green/20' :
                v === 'no' ? 'bg-cyber-red/10 text-cyber-red border-cyber-red/20' :
                'bg-cyber-amber/10 text-cyber-amber border-cyber-amber/20'
              }`}>{v}</span>
            ))}
          </div>
        </div>
      )}

      {/* Trust trajectory */}
      {renderTrust()}

      {/* Social votes */}
      {social_votes && social_votes.length > 0 && (
        <div className="mt-2">
          <div className="text-[9px] text-cyber-dim mb-1">SOCIAL AGENTS ({social_votes.length})</div>
          {vote_distribution && (
            <div className="flex gap-1 mb-1">
              {Object.entries(vote_distribution).map(([vote, count]) => (
                <span key={vote} className={`text-[9px] px-1 py-0.5 ${
                  vote === 'compatible' ? 'bg-cyber-green/10 text-cyber-green' :
                  vote === 'incompatible' ? 'bg-cyber-red/10 text-cyber-red' :
                  'bg-cyber-amber/10 text-cyber-amber'
                }`}>{vote}:{count}</span>
              ))}
            </div>
          )}
          <div className="space-y-1 max-h-24 overflow-y-auto">
            {social_votes.slice(0, 4).map((v, i) => (
              <div key={i} className="bg-cyber-bg p-1 text-[9px]">
                <div className="flex justify-between">
                  <span className="text-cyber-text font-semibold">{v.agent}</span>
                  <span className={v.vote === 'compatible' ? 'text-cyber-green' : v.vote === 'incompatible' ? 'text-cyber-red' : 'text-cyber-amber'}>
                    {v.vote} {Math.round(v.confidence * 100)}%
                  </span>
                </div>
                <div className="text-cyber-muted truncate">{v.reasoning}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next prediction */}
      <div className="text-[9px] text-cyber-dim text-center mt-2 pt-1 border-t border-cyber-border">
        NEXT @ T:{Math.ceil(turnCount / 5) * 5 + 5}
      </div>
    </CyberPanel>
  );
}
