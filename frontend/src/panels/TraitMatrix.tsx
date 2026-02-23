import CyberPanel from '../ui/CyberPanel';
import type { FeatureData } from '../types';

interface TraitMatrixProps {
  featureData: FeatureData | null;
}

export default function TraitMatrix({ featureData }: TraitMatrixProps) {
  if (!featureData) {
    return (
      <CyberPanel title="TRAIT MATRIX">
        <div className="text-cyber-dim text-[11px] text-center py-4">NO FEATURES DETECTED</div>
      </CyberPanel>
    );
  }

  // Sort by confidence, take top 10
  const sorted = Object.entries(featureData.confidences)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  const lowConf = featureData.low_confidence || [];

  return (
    <CyberPanel title="TRAIT MATRIX">
      <div className="space-y-1.5">
        {sorted.map(([key, conf]) => {
          const label = key.replace(/^(big_five_|interest_)/, '').replace(/_/g, ' ');
          const pct = Math.round(conf * 100);
          const isLow = lowConf.includes(key);
          const barColor = isLow ? 'bg-cyber-amber' : pct > 70 ? 'bg-cyber-cyan' : 'bg-cyber-fuchsia';

          return (
            <div key={key} className="group">
              <div className="flex justify-between text-[10px] mb-0.5">
                <span className={`uppercase truncate mr-2 ${isLow ? 'text-cyber-amber' : 'text-cyber-text'}`}>
                  {label}
                </span>
                <span className="text-cyber-muted tabular-nums shrink-0">{pct}%</span>
              </div>
              <div className="h-1 bg-cyber-border rounded-sm overflow-hidden">
                <div
                  className={`h-full ${barColor} transition-all duration-700 rounded-sm`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {/* Low confidence warnings */}
      {lowConf.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {lowConf.slice(0, 5).map((f) => (
            <span key={f} className="text-[9px] px-1.5 py-0.5 bg-cyber-amber/10 text-cyber-amber border border-cyber-amber/20">
              ⚠ {f.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}
    </CyberPanel>
  );
}
