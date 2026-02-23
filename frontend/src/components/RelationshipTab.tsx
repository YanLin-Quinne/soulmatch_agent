import { useState, useEffect } from 'react';

interface RelationshipData {
  rel_status: string;
  rel_type: string;
  sentiment: string;
  can_advance: boolean;
  advance_prediction_set: string[];
  social_votes?: SocialVote[];
  vote_distribution?: Record<string, number>;
}

interface SocialVote {
  agent: string;
  vote: string;
  rel_status: string;
  confidence: number;
  reasoning: string;
  key_factors?: string[];
  demographics?: { age?: number; gender?: string; relationship_status?: string };
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

function AdvancePredictionBadge({ predictionSet }: { predictionSet: string[] }) {
  const hasYes = predictionSet.includes('yes');
  const hasUncertain = predictionSet.includes('uncertain');
  const hasNo = predictionSet.includes('no');

  let bg = 'var(--bg-hover)';
  let textColor = 'var(--text-muted)';
  let borderColor = 'transparent';
  let label = 'Uncertain';

  if (hasYes && !hasUncertain && !hasNo) {
    bg = 'rgba(34,197,94,0.15)'; textColor = '#4ade80'; borderColor = '#22c55e';
    label = '✓ Can Advance';
  } else if (hasYes && hasUncertain) {
    bg = 'rgba(234,179,8,0.15)'; textColor = '#facc15'; borderColor = '#eab308';
    label = '~ Maybe Advance';
  } else if ((hasUncertain && hasNo) || (hasNo && !hasYes)) {
    bg = 'rgba(239,68,68,0.15)'; textColor = '#f87171'; borderColor = '#ef4444';
    label = '✗ Not Ready';
  }

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 500,
      background: bg, color: textColor, border: `1px solid ${borderColor}`,
    }}>
      {label}
      <span style={{ opacity: 0.7 }}>
        [{predictionSet.join(', ')}] @ 90%
      </span>
    </div>
  );
}

function RelationshipProgressBar({ status }: { status: string }) {
  const currentIdx = STATUS_ORDER.indexOf(status);
  const progress = currentIdx >= 0 ? ((currentIdx + 1) / STATUS_ORDER.length) * 100 : 0;

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11 }}>
        {STATUS_ORDER.map((s) => {
          const idx = STATUS_ORDER.indexOf(s);
          let color = 'var(--text-dim)';
          if (s === status) color = 'var(--accent)';
          else if (idx < (currentIdx >= 0 ? currentIdx : 0)) color = 'var(--text-muted)';
          return (
            <span key={s} style={{ color, fontWeight: s === status ? 600 : 400 }}>
              {STATUS_LABELS[s]}
            </span>
          );
        })}
      </div>
      <div style={{
        height: 8, background: 'var(--bg-hover)', borderRadius: 4, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${progress}%`, borderRadius: 4,
          background: 'var(--gradient-1)', transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  );
}

function TrustTrajectoryChart({ history }: { history: TrustHistory[] }) {
  if (history.length < 2) {
    return (
      <div style={{
        height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-dim)', fontSize: 12,
      }}>
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

  const trendColor = trend > 0 ? '#4ade80' : trend < 0 ? '#f87171' : 'var(--text-muted)';

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ marginBottom: 4 }}>
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke="var(--accent)"
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
              r="2.5"
              fill={i === history.length - 1 ? 'var(--accent-glow)' : 'var(--accent)'}
            />
          );
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
        <span>Trust: {(currentTrust * 100).toFixed(0)}%</span>
        <span style={{ color: trendColor }}>
          {trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} {Math.abs(trend * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

const sectionLabel: React.CSSProperties = {
  fontSize: 11, color: 'var(--text-muted)', marginBottom: 8,
  fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
};

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
      <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
        <div style={{ fontSize: 28, marginBottom: 8 }}>💫</div>
        <p>Relationship analysis will appear</p>
        <p style={{ fontSize: 11, marginTop: 4, color: 'var(--text-dim)' }}>after turn 5</p>
      </div>
    );
  }

  const { rel_status, rel_type, sentiment, advance_prediction_set, social_votes, vote_distribution } = relationshipData;
  const sentimentColor = sentiment === 'positive' ? '#4ade80'
    : sentiment === 'negative' ? '#f87171' : '#facc15';

  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Milestone Report */}
      {showMilestone && milestoneReport && (
        <div style={{
          background: 'rgba(147,51,234,0.12)', border: '1px solid rgba(147,51,234,0.4)',
          borderRadius: 8, padding: 12, fontSize: 13,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ color: '#c084fc', fontWeight: 500 }}>
              🎯 Milestone Report (Turn {milestoneReport.turn})
            </span>
            <button
              onClick={() => setShowMilestone(false)}
              style={{
                marginLeft: 'auto', background: 'none', border: 'none',
                color: 'var(--text-dim)', cursor: 'pointer', fontSize: 16,
              }}
            >×</button>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>{milestoneReport.message}</p>
          {milestoneReport.predicted_status_at_turn_30 && (
            <p style={{ color: '#c084fc', fontSize: 12, marginTop: 4 }}>
              Predicted at Turn 30: {milestoneReport.predicted_status_at_turn_30}
            </p>
          )}
        </div>
      )}

      {/* Relationship Stage */}
      <div>
        <div style={sectionLabel}>Relationship Stage</div>
        <RelationshipProgressBar status={rel_status} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, marginTop: 4 }}>
          <span style={{ color: 'var(--text-dim)' }}>Type:</span>
          <span style={{ color: 'var(--accent)', textTransform: 'capitalize' }}>{rel_type}</span>
          <span style={{ color: 'var(--text-dim)', margin: '0 2px' }}>·</span>
          <span style={{ color: 'var(--text-dim)' }}>Sentiment:</span>
          <span style={{ color: sentimentColor, textTransform: 'capitalize' }}>{sentiment}</span>
        </div>
      </div>

      {/* Can Advance? */}
      <div>
        <div style={sectionLabel}>Can Advance? (Conformal @ 90%)</div>
        <AdvancePredictionBadge predictionSet={advance_prediction_set ?? []} />
      </div>

      {/* Trust Trajectory */}
      <div>
        <div style={sectionLabel}>Trust Trajectory</div>
        <TrustTrajectoryChart history={trustHistory} />
      </div>

      {/* Social Agents Voting */}
      {social_votes && social_votes.length > 0 && (
        <div>
          <div style={sectionLabel}>Social Agents Voting ({social_votes.length} agents)</div>
          {vote_distribution && (
            <div style={{ display: 'flex', gap: 8, marginBottom: 10, fontSize: 11 }}>
              {Object.entries(vote_distribution).map(([vote, count]) => {
                const voteColor = vote === 'compatible' ? '#4ade80'
                  : vote === 'incompatible' ? '#f87171' : '#facc15';
                return (
                  <span key={vote} style={{
                    padding: '2px 8px', borderRadius: 10,
                    background: `${voteColor}15`, color: voteColor, border: `1px solid ${voteColor}40`,
                  }}>
                    {vote}: {count}
                  </span>
                );
              })}
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {social_votes.map((v, i) => {
              const voteColor = v.vote === 'compatible' ? '#4ade80'
                : v.vote === 'incompatible' ? '#f87171' : '#facc15';
              return (
                <div key={i} style={{
                  padding: '8px 10px', borderRadius: 6,
                  background: 'var(--bg-hover)', fontSize: 11,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, color: 'var(--text)' }}>{v.agent}</span>
                    <span style={{ color: voteColor, fontWeight: 500 }}>{v.vote} ({Math.round(v.confidence * 100)}%)</span>
                  </div>
                  {v.demographics && (v.demographics.age || v.demographics.gender) && (
                    <div style={{ color: 'var(--text-dim)', fontSize: 10, marginBottom: 3 }}>
                      {[v.demographics.gender, v.demographics.age && `${v.demographics.age}y`, v.demographics.relationship_status].filter(Boolean).join(' · ')}
                    </div>
                  )}
                  <div style={{ color: 'var(--text-muted)', lineHeight: 1.4 }}>{v.reasoning}</div>
                  {v.key_factors && v.key_factors.length > 0 && (
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
                      {v.key_factors.map((f, j) => (
                        <span key={j} style={{
                          padding: '1px 6px', borderRadius: 8, fontSize: 10,
                          background: 'var(--accent-dim)', color: 'var(--accent)',
                        }}>{f}</span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Turn counter */}
      <div style={{
        fontSize: 11, color: 'var(--text-dim)', textAlign: 'center',
        borderTop: '1px solid var(--border)', paddingTop: 8,
      }}>
        Next prediction at Turn {Math.ceil(turnCount / 5) * 5 + 5}
      </div>
    </div>
  );
}