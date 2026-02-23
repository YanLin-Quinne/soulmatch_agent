import CyberPanel from '../ui/CyberPanel';
import type { FeatureData, ContextData } from '../types';

interface ConfidencePanelProps {
  featureData: FeatureData | null;
  contextData: ContextData | null;
  confidenceHistory: number[];
}

export default function ConfidencePanel({ featureData, contextData, confidenceHistory }: ConfidencePanelProps) {
  const avg = featureData ? Math.round(featureData.average_confidence * 100) : 0;

  // SVG trend line
  const renderTrend = () => {
    if (confidenceHistory.length < 2) return null;
    const w = 200, h = 32, pad = 2;
    const pts = confidenceHistory.slice(-20);
    const min = Math.min(...pts) * 0.95;
    const max = Math.max(...pts) * 1.05 || 1;
    const range = max - min || 0.1;
    const points = pts.map((v, i) => {
      const x = pad + (i / (pts.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - min) / range) * (h - 2 * pad);
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-8 mt-1">
        <polyline points={points} fill="none" stroke="#00f0ff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />
        <polyline points={points} fill="none" stroke="#00f0ff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" opacity="0.1" />
      </svg>
    );
  };

  return (
    <CyberPanel title="CONFIDENCE INDEX">
      {featureData ? (
        <>
          <div className="text-center">
            <div className="text-3xl font-bold text-cyber-cyan text-glow-cyan tabular-nums">
              {avg}<span className="text-lg text-cyber-muted">%</span>
            </div>
            <div className="text-[10px] text-cyber-muted mt-0.5">AVG CONFIDENCE</div>
          </div>
          {renderTrend()}
          {/* Context state/risk */}
          {contextData && (
            <div className="mt-2 flex gap-2 text-[10px]">
              {contextData.state && (
                <span className="px-1.5 py-0.5 bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20">
                  {contextData.state}
                </span>
              )}
              {contextData.risk_level && (
                <span className={`px-1.5 py-0.5 border ${
                  contextData.risk_level === 'safe' ? 'bg-cyber-green/10 text-cyber-green border-cyber-green/20' :
                  contextData.risk_level === 'low' ? 'bg-cyber-amber/10 text-cyber-amber border-cyber-amber/20' :
                  'bg-cyber-red/10 text-cyber-red border-cyber-red/20'
                }`}>
                  RISK:{contextData.risk_level.toUpperCase()}
                </span>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="text-cyber-dim text-[11px] text-center py-4">AWAITING DATA...</div>
      )}
    </CyberPanel>
  );
}
