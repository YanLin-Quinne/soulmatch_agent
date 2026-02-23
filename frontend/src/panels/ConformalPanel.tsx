import CyberPanel from '../ui/CyberPanel';
import type { FeatureData } from '../types';

interface ConformalPanelProps {
  featureData: FeatureData | null;
}

export default function ConformalPanel({ featureData }: ConformalPanelProps) {
  const conformal = featureData?.conformal;

  if (!conformal) {
    return (
      <CyberPanel title="CONFORMAL PREDICTION">
        <div className="text-cyber-dim text-[11px] text-center py-4">NO CP DATA</div>
      </CyberPanel>
    );
  }

  return (
    <CyberPanel title="CONFORMAL PREDICTION">
      {/* Summary badges */}
      <div className="flex flex-wrap gap-1 mb-2">
        <span className="text-[9px] px-1.5 py-0.5 bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20">
          COV:{Math.round(conformal.coverage_guarantee * 100)}%
        </span>
        <span className="text-[9px] px-1.5 py-0.5 bg-cyber-amber/10 text-cyber-amber border border-cyber-amber/20">
          AVG:{conformal.avg_set_size.toFixed(1)}
        </span>
        <span className="text-[9px] px-1.5 py-0.5 bg-cyber-fuchsia/10 text-cyber-fuchsia border border-cyber-fuchsia/20">
          SING:{conformal.singletons}/{conformal.total_dims}
        </span>
      </div>

      {/* Prediction sets */}
      <div className="space-y-1.5 max-h-48 overflow-y-auto">
        {Object.entries(conformal.prediction_sets).map(([dim, ps]) => (
          <div key={dim} className="bg-cyber-bg border-l border-cyber-cyan/30 p-1.5">
            <div className="flex justify-between text-[10px] mb-1">
              <span className="text-cyber-text uppercase">{dim.replace(/_/g, ' ')}</span>
              <span className="text-cyber-cyan font-semibold">{ps.point}</span>
            </div>
            {/* Set tags */}
            <div className="flex flex-wrap gap-0.5 mb-1">
              {ps.set.map((val, i) => (
                <span key={i} className={`text-[9px] px-1 py-0.5 ${
                  val === ps.point
                    ? 'bg-cyber-cyan/15 text-cyber-cyan border border-cyber-cyan/30 font-semibold'
                    : 'bg-cyber-border text-cyber-muted'
                }`}>{val}</span>
              ))}
            </div>
            {/* LLM vs Calibrated */}
            <div className="flex justify-between text-[9px]">
              <span className="text-cyber-dim">LLM:{Math.round(ps.llm_conf * 100)}%</span>
              <span className="text-cyber-cyan">CAL:{Math.round(ps.calibrated_conf * 100)}%</span>
            </div>
          </div>
        ))}
      </div>
    </CyberPanel>
  );
}
