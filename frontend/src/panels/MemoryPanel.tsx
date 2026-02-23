import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import CyberPanel from '../ui/CyberPanel';
import type { MemoryStats, MemoryItem } from '../types';

interface MemoryPanelProps {
  memoryStats: MemoryStats | null;
  memories: MemoryItem[];
}

export default function MemoryPanel({ memoryStats, memories }: MemoryPanelProps) {
  const [expandEpisodic, setExpandEpisodic] = useState(false);
  const [expandSemantic, setExpandSemantic] = useState(false);

  return (
    <CyberPanel title="MEMORY LAYERS">
      {memoryStats ? (
        <>
          {/* Three-layer bars */}
          <div className="space-y-2 mb-2">
            {/* Working Memory */}
            <div>
              <div className="flex justify-between text-[10px] mb-0.5">
                <span className="text-cyber-cyan">WORKING</span>
                <span className="text-cyber-muted tabular-nums">{memoryStats.working_memory_size}/20</span>
              </div>
              <div className="h-1.5 bg-cyber-border rounded-sm overflow-hidden">
                <div className="h-full bg-cyber-cyan transition-all duration-500 rounded-sm"
                  style={{ width: `${(memoryStats.working_memory_size / 20) * 100}%` }} />
              </div>
            </div>
            {/* Episodic Memory */}
            <div>
              <div className="flex justify-between text-[10px] mb-0.5">
                <button onClick={() => setExpandEpisodic(!expandEpisodic)}
                  className="flex items-center gap-0.5 text-cyber-amber hover:text-cyber-amber/80">
                  {expandEpisodic ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                  EPISODIC
                </button>
                <span className="text-cyber-muted tabular-nums">{memoryStats.episodic_memory_count} ep</span>
              </div>
              <div className="h-1.5 bg-cyber-border rounded-sm overflow-hidden">
                <div className="h-full bg-cyber-amber transition-all duration-500 rounded-sm"
                  style={{ width: `${Math.min(memoryStats.episodic_memory_count * 20, 100)}%` }} />
              </div>
            </div>
            {/* Semantic Memory */}
            <div>
              <div className="flex justify-between text-[10px] mb-0.5">
                <button onClick={() => setExpandSemantic(!expandSemantic)}
                  className="flex items-center gap-0.5 text-cyber-fuchsia hover:text-cyber-fuchsia/80">
                  {expandSemantic ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                  SEMANTIC
                </button>
                <span className="text-cyber-muted tabular-nums">{memoryStats.semantic_memory_count} ins</span>
              </div>
              <div className="h-1.5 bg-cyber-border rounded-sm overflow-hidden">
                <div className="h-full bg-cyber-fuchsia transition-all duration-500 rounded-sm"
                  style={{ width: `${Math.min(memoryStats.semantic_memory_count * 25, 100)}%` }} />
              </div>
            </div>
          </div>

          {/* Compression ratio */}
          {memoryStats.compression_ratio > 0 && (
            <div className="text-[9px] text-cyber-dim text-right mb-2">
              COMPRESS: {(memoryStats.compression_ratio * 100).toFixed(0)}%
            </div>
          )}

          {/* Expanded episodic */}
          {expandEpisodic && memoryStats.episodic_memories && memoryStats.episodic_memories.length > 0 && (
            <div className="space-y-1.5 mb-2 max-h-32 overflow-y-auto">
              {memoryStats.episodic_memories.map((ep) => (
                <div key={ep.episode_id} className="bg-cyber-bg border-l-2 border-cyber-amber p-1.5 text-[10px]">
                  <div className="flex justify-between mb-0.5">
                    <span className="text-cyber-amber">T{ep.turn_range[0]}-{ep.turn_range[1]}</span>
                    <span className={
                      ep.emotion_trend === 'improving' ? 'text-cyber-green' :
                      ep.emotion_trend === 'declining' ? 'text-cyber-red' : 'text-cyber-amber'
                    }>
                      {ep.emotion_trend === 'improving' ? '↑' : ep.emotion_trend === 'declining' ? '↓' : '→'}
                    </span>
                  </div>
                  <div className="text-cyber-text leading-relaxed">{ep.summary}</div>
                  {ep.key_events.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {ep.key_events.map((evt, i) => (
                        <span key={i} className="px-1 bg-cyber-amber/10 text-cyber-amber text-[8px]">{evt}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Expanded semantic */}
          {expandSemantic && memoryStats.semantic_memories && memoryStats.semantic_memories.length > 0 && (
            <div className="space-y-1.5 mb-2 max-h-32 overflow-y-auto">
              {memoryStats.semantic_memories.map((sem) => (
                <div key={sem.reflection_id} className="bg-cyber-bg border-l-2 border-cyber-fuchsia p-1.5 text-[10px]">
                  <div className="text-cyber-fuchsia mb-0.5">T{sem.turn_range[0]}-{sem.turn_range[1]}</div>
                  <div className="text-cyber-text leading-relaxed">{sem.reflection}</div>
                  {sem.relationship_insights && (
                    <div className="text-cyber-muted mt-0.5 italic">{sem.relationship_insights}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="text-cyber-dim text-[11px] text-center py-2">INITIALIZING...</div>
      )}

      {/* Retrieved memories */}
      {memories.length > 0 && (
        <div className="mt-1 space-y-1 max-h-20 overflow-y-auto">
          <div className="text-[9px] text-cyber-dim uppercase">Retrieved</div>
          {memories.slice(0, 3).map((mem, idx) => (
            <div key={idx} className="text-[10px] text-cyber-text bg-cyber-bg p-1 border-l border-cyber-cyan/30 leading-relaxed">
              {mem.content.slice(0, 80)}{mem.content.length > 80 ? '...' : ''}
            </div>
          ))}
        </div>
      )}
    </CyberPanel>
  );
}
