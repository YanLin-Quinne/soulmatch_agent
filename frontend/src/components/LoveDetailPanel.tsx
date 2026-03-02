const LOVE_STAGES = ['attraction', 'exploration', 'bonding', 'commitment', 'deep_attachment'];
const LOVE_STAGE_LABELS: Record<string, string> = {
  attraction: 'Attraction',
  exploration: 'Exploration',
  bonding: 'Bonding',
  commitment: 'Commitment',
  deep_attachment: 'Deep Attachment',
};

const COMPAT_COLORS: Record<string, string> = {
  emotional: '#f472b6',
  intellectual: '#60a5fa',
  lifestyle: '#34d399',
  values: '#fbbf24',
};

interface LoveDetail {
  love_stage: string;
  stage_confidence: number;
  compatibility: { emotional: number; intellectual: number; lifestyle: number; values: number };
  blockers: string[];
  catalysts: string[];
  advice: string;
  can_progress: boolean;
}

interface LoveDetailPanelProps {
  loveDetail: LoveDetail | null;
}

export default function LoveDetailPanel({ loveDetail }: LoveDetailPanelProps) {
  if (!loveDetail) return null;

  const { love_stage, stage_confidence, compatibility, blockers, catalysts, advice, can_progress } = loveDetail;

  const stageIdx = LOVE_STAGES.indexOf(love_stage);
  const stageProgress = stageIdx >= 0 ? ((stageIdx + 1) / LOVE_STAGES.length) * 100 : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Love Stage Progress */}
      <div className="love-stages">
        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
          Love Stage ({Math.round(stage_confidence * 100)}% conf)
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 10 }}>
          {LOVE_STAGES.map((s) => {
            const idx = LOVE_STAGES.indexOf(s);
            let color = 'var(--text-dim)';
            if (s === love_stage) color = '#f472b6';
            else if (idx < (stageIdx >= 0 ? stageIdx : 0)) color = 'var(--text-muted)';
            return (
              <span key={s} style={{ color, fontWeight: s === love_stage ? 600 : 400 }}>
                {LOVE_STAGE_LABELS[s]}
              </span>
            );
          })}
        </div>
        <div style={{ height: 8, background: 'var(--bg-hover)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%', width: `${stageProgress}%`, borderRadius: 4,
            background: 'linear-gradient(90deg, #ec4899, #f472b6)', transition: 'width 0.5s ease',
          }} />
        </div>
      </div>

      {/* Compatibility Dimensions */}
      <div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
          Compatibility
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {Object.entries(compatibility).map(([dim, val]) => (
            <div key={dim} className="love-compat-bar">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                <span style={{ color: 'var(--text-muted)', textTransform: 'capitalize' }}>{dim}</span>
                <span style={{ color: COMPAT_COLORS[dim] || 'var(--accent)' }}>{Math.round(val * 100)}%</span>
              </div>
              <div style={{ height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${val * 100}%`, borderRadius: 3,
                  background: COMPAT_COLORS[dim] || 'var(--accent)', transition: 'width 0.4s ease',
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Blockers & Catalysts */}
      <div style={{ display: 'flex', gap: 12 }}>
        {blockers.length > 0 && (
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
              Blockers
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {blockers.map((b, i) => (
                <span key={i} style={{
                  padding: '2px 8px', borderRadius: 8, fontSize: 10,
                  background: 'rgba(239,68,68,0.15)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)',
                }}>{b}</span>
              ))}
            </div>
          </div>
        )}
        {catalysts.length > 0 && (
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
              Catalysts
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {catalysts.map((c, i) => (
                <span key={i} style={{
                  padding: '2px 8px', borderRadius: 8, fontSize: 10,
                  background: 'rgba(34,197,94,0.15)', color: '#4ade80', border: '1px solid rgba(34,197,94,0.3)',
                }}>{c}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Can Progress Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          padding: '4px 10px', borderRadius: 12, fontSize: 11, fontWeight: 500,
          background: can_progress ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
          color: can_progress ? '#4ade80' : '#f87171',
          border: `1px solid ${can_progress ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
        }}>
          {can_progress ? 'Can Progress' : 'Not Ready'}
        </span>
      </div>

      {/* Advice */}
      <div className="love-advice" style={{
        padding: '8px 12px', borderRadius: 8, fontSize: 12, lineHeight: 1.5,
        background: 'rgba(147,51,234,0.08)', border: '1px solid rgba(147,51,234,0.2)',
        color: 'var(--text-muted)',
      }}>
        <span style={{ color: '#c084fc', fontWeight: 500 }}>Advice: </span>{advice}
      </div>
    </div>
  );
}
