import { Wifi, WifiOff, Terminal, Activity } from 'lucide-react';
import type { FeatureData, EmotionState, WarningState } from '../types';

interface TopBarProps {
  isConnected: boolean;
  turnCount: number;
  featureData: FeatureData | null;
  emotion: EmotionState | null;
  warning: WarningState | null;
  reconnectAttempt: number;
  maxReconnect: number;
}

export default function TopBar({
  isConnected, turnCount, featureData, emotion, warning,
  reconnectAttempt, maxReconnect,
}: TopBarProps) {
  // Build ticker items from feature deltas
  const tickerItems: string[] = [];
  if (featureData) {
    const sorted = Object.entries(featureData.confidences)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
    sorted.forEach(([k, v]) => {
      tickerItems.push(`${k.replace(/_/g, ' ')}: ${Math.round(v * 100)}%`);
    });
  }
  if (emotion) {
    tickerItems.push(`EMO:${emotion.emotion.toUpperCase()} @${Math.round(emotion.intensity * 100)}%`);
  }
  if (warning) {
    tickerItems.push(`⚠ ${warning.level.toUpperCase()}: ${warning.message.slice(0, 40)}`);
  }

  return (
    <div className="h-8 bg-cyber-bg border-b border-cyber-border flex items-center px-3 gap-3 text-[11px] shrink-0 overflow-hidden">
      {/* Logo */}
      <div className="flex items-center gap-1.5 shrink-0">
        <Terminal size={12} className="text-cyber-cyan" />
        <span className="text-cyber-cyan font-semibold tracking-wider text-glow-cyan">SOULMATCH_OS</span>
      </div>

      {/* Connection status */}
      <div className="flex items-center gap-1 shrink-0">
        {isConnected ? (
          <>
            <Wifi size={10} className="text-cyber-green" />
            <span className="text-cyber-green">ONLINE</span>
          </>
        ) : (
          <>
            <WifiOff size={10} className="text-cyber-red animate-glow-pulse" />
            <span className="text-cyber-red">
              {reconnectAttempt >= maxReconnect ? 'LOST' : `RECON ${reconnectAttempt}/${maxReconnect}`}
            </span>
          </>
        )}
      </div>

      {/* Separator */}
      <div className="text-cyber-dim">──</div>

      {/* Ticker */}
      <div className="flex-1 overflow-hidden relative">
        {tickerItems.length > 0 && (
          <div className="animate-ticker whitespace-nowrap text-cyber-muted">
            {tickerItems.map((item, i) => (
              <span key={i}>
                <span className="text-cyber-dim mx-2">│</span>
                <span>{item}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Separator */}
      <div className="text-cyber-dim">──</div>

      {/* Turn counter */}
      <div className="flex items-center gap-1 shrink-0">
        <Activity size={10} className="text-cyber-fuchsia" />
        <span className="text-cyber-fuchsia">T:{turnCount}</span>
      </div>

      {/* Ping indicator */}
      <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${isConnected ? 'bg-cyber-green animate-glow-pulse' : 'bg-cyber-red'}`} />
    </div>
  );
}
