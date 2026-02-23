import CyberPanel from '../ui/CyberPanel';
import type { EmotionState, EmotionEntry } from '../types';
import { EMOTION_EMOJI } from '../constants';

interface EmotionPanelProps {
  emotion: EmotionState | null;
  emotionHistory: EmotionEntry[];
}

export default function EmotionPanel({ emotion, emotionHistory }: EmotionPanelProps) {
  return (
    <CyberPanel title="EMOTION STATE" accent="fuchsia">
      {emotion ? (
        <>
          {/* Current emotion badge */}
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">{EMOTION_EMOJI[emotion.emotion] || '😐'}</span>
            <div>
              <div className="text-sm font-semibold text-cyber-fuchsia uppercase text-glow-fuchsia">
                {emotion.emotion}
              </div>
              <div className="text-[10px] text-cyber-muted">
                CONF:{Math.round(emotion.confidence * 100)}%
              </div>
            </div>
            <div className="ml-auto text-right">
              <div className="text-lg font-bold text-cyber-text tabular-nums">
                {Math.round(emotion.intensity * 100)}%
              </div>
              <div className="text-[9px] text-cyber-dim">INTENSITY</div>
            </div>
          </div>
          {/* Intensity bar */}
          <div className="h-1 bg-cyber-border rounded-sm overflow-hidden mb-2">
            <div
              className="h-full bg-cyber-fuchsia transition-all duration-500 rounded-sm"
              style={{ width: `${emotion.intensity * 100}%` }}
            />
          </div>
        </>
      ) : (
        <div className="text-cyber-dim text-[11px] text-center py-2">SCANNING...</div>
      )}

      {/* History */}
      {emotionHistory.length > 0 && (
        <div className="space-y-0.5 max-h-28 overflow-y-auto">
          {[...emotionHistory].reverse().slice(0, 8).map((entry, idx) => (
            <div key={idx} className="flex items-center gap-1.5 text-[10px] py-0.5">
              <span className="text-cyber-dim w-6 shrink-0 tabular-nums">t{String(entry.turn).padStart(2, '0')}</span>
              <span className="shrink-0">{EMOTION_EMOJI[entry.emotion] || '😐'}</span>
              <span className="text-cyber-text truncate flex-1">{entry.emotion}</span>
              <div className="w-10 h-0.5 bg-cyber-border rounded-sm overflow-hidden shrink-0">
                <div className="h-full bg-cyber-fuchsia/70 rounded-sm" style={{ width: `${entry.intensity * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </CyberPanel>
  );
}
