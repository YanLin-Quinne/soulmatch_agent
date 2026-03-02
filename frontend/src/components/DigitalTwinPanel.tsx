import { useState } from 'react';

interface TwinMessage {
  sender: string;
  content: string;
}

interface DigitalTwinPanelProps {
  twinData: any;
  twinMessages: TwinMessage[];
  turnCount: number;
  onCreateTwin: () => void;
  onSendTwinMessage: (msg: string) => void;
  onComparePerception: (perception: Record<string, number>) => void;
}

const BIG_FIVE_DIMS = [
  { key: 'big_five_openness', label: 'Openness' },
  { key: 'big_five_conscientiousness', label: 'Conscientiousness' },
  { key: 'big_five_extraversion', label: 'Extraversion' },
  { key: 'big_five_agreeableness', label: 'Agreeableness' },
  { key: 'big_five_neuroticism', label: 'Neuroticism' },
];

const sectionLabel: React.CSSProperties = {
  fontSize: 11, color: 'var(--text-muted)', marginBottom: 8,
  fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
};

export default function DigitalTwinPanel({
  twinData,
  twinMessages,
  turnCount,
  onCreateTwin,
  onSendTwinMessage,
  onComparePerception,
}: DigitalTwinPanelProps) {
  const [chatInput, setChatInput] = useState('');
  const [showPerception, setShowPerception] = useState(false);
  const [sliders, setSliders] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    BIG_FIVE_DIMS.forEach(d => { init[d.key] = 0.5; });
    return init;
  });

  // --- Sub-view 1: Not enough data yet ---
  if (turnCount < 15 && !twinData) {
    const progress = Math.min(turnCount / 15, 1);
    return (
      <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
        <div style={{ fontSize: 32, marginBottom: 10 }}>&#129302;</div>
        <p style={{ fontWeight: 500, color: 'var(--text-muted)', marginBottom: 8 }}>
          Collect more conversation data
        </p>
        <p style={{ fontSize: 11, marginBottom: 12, color: 'var(--text-dim)' }}>
          {turnCount}/15 turns completed
        </p>
        <div style={{
          height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', width: `${progress * 100}%`, borderRadius: 3,
            background: 'var(--gradient-1)', transition: 'width 0.4s ease',
          }} />
        </div>
        <p style={{ fontSize: 11, marginTop: 10, color: 'var(--text-dim)' }}>
          The digital twin needs at least 15 turns of conversation to build an accurate personality model.
        </p>
      </div>
    );
  }

  // --- Sub-view 3: Ready to create ---
  if (!twinData) {
    return (
      <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
        <div style={{ fontSize: 32, marginBottom: 10 }}>&#10024;</div>
        <p style={{ fontWeight: 500, color: 'var(--text-muted)', marginBottom: 12 }}>
          Enough data collected!
        </p>
        <button
          onClick={onCreateTwin}
          style={{
            padding: '10px 24px', borderRadius: 8, border: 'none',
            background: 'var(--gradient-1)', color: '#fff', fontWeight: 600,
            fontSize: 13, cursor: 'pointer', transition: 'opacity 0.2s',
          }}
          onMouseOver={e => (e.currentTarget.style.opacity = '0.85')}
          onMouseOut={e => (e.currentTarget.style.opacity = '1')}
        >
          Create Digital Twin
        </button>
        <p style={{ fontSize: 11, marginTop: 10, color: 'var(--text-dim)' }}>
          Generate an AI clone based on your inferred personality.
        </p>
      </div>
    );
  }

  // --- Sub-view 2: Twin created ---
  const handleSend = () => {
    const text = chatInput.trim();
    if (!text) return;
    onSendTwinMessage(text);
    setChatInput('');
  };

  const handleCompare = () => {
    onComparePerception(sliders);
    setShowPerception(false);
  };

  return (
    <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Twin Profile Card */}
      <div className="twin-profile-card" style={{
        background: 'var(--bg-hover)', borderRadius: 10, padding: 14,
        border: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            background: 'var(--gradient-1)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            fontSize: 18, color: '#fff', fontWeight: 700,
          }}>
            {(twinData.name?.[0] || 'T').toUpperCase()}
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text)' }}>
              {twinData.name}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              Created at turn {twinData.source_turn}
            </div>
          </div>
        </div>

        <div style={sectionLabel}>Personality</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: 8 }}>
          {twinData.personality_summary}
        </p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {twinData.interests?.map((interest: string) => (
            <span key={interest} style={{
              padding: '2px 8px', borderRadius: 10, fontSize: 11,
              background: 'var(--accent-dim)', color: 'var(--accent)',
            }}>
              {interest}
            </span>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-dim)' }}>
          {twinData.mbti && (
            <span>MBTI: <span style={{ color: 'var(--accent)' }}>{twinData.mbti}</span></span>
          )}
          {twinData.attachment_style && (
            <span>Attachment: <span style={{ color: 'var(--accent)' }}>{twinData.attachment_style}</span></span>
          )}
          <span>Style: <span style={{ color: 'var(--accent)' }}>{twinData.communication_style}</span></span>
        </div>
      </div>

      {/* Mini Chat Area */}
      <div className="twin-chat-area" style={{
        background: 'var(--bg-hover)', borderRadius: 10, padding: 10,
        border: '1px solid var(--border)', maxHeight: 220, display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={sectionLabel}>Chat with Twin</div>
        <div style={{
          flex: 1, overflowY: 'auto', marginBottom: 8, minHeight: 80,
          display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          {twinMessages.length === 0 && (
            <div style={{ fontSize: 11, color: 'var(--text-dim)', textAlign: 'center', padding: 12 }}>
              Send a message to chat with your digital twin
            </div>
          )}
          {twinMessages.map((msg, i) => (
            <div key={i} style={{
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%', padding: '6px 10px', borderRadius: 8,
              fontSize: 12, lineHeight: 1.4,
              background: msg.sender === 'user' ? 'var(--accent-dim)' : 'var(--bg-card)',
              color: msg.sender === 'user' ? 'var(--accent)' : 'var(--text)',
            }}>
              {msg.content}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <input
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask your twin..."
            style={{
              flex: 1, padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)',
              background: 'var(--bg-card)', color: 'var(--text)', fontSize: 12,
              outline: 'none',
            }}
          />
          <button
            onClick={handleSend}
            disabled={!chatInput.trim()}
            style={{
              padding: '6px 12px', borderRadius: 6, border: 'none',
              background: 'var(--accent)', color: '#fff', fontSize: 12,
              cursor: chatInput.trim() ? 'pointer' : 'default',
              opacity: chatInput.trim() ? 1 : 0.5,
            }}
          >
            Send
          </button>
        </div>
      </div>

      {/* Perception Comparison */}
      {!showPerception ? (
        <button
          onClick={() => setShowPerception(true)}
          style={{
            padding: '8px 16px', borderRadius: 8, border: '1px solid var(--border)',
            background: 'transparent', color: 'var(--text-muted)', fontSize: 12,
            cursor: 'pointer', transition: 'all 0.2s',
          }}
          onMouseOver={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
          onMouseOut={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
        >
          Compare Perception
        </button>
      ) : (
        <div style={{
          background: 'var(--bg-hover)', borderRadius: 10, padding: 12,
          border: '1px solid var(--border)',
        }}>
          <div style={sectionLabel}>Your Self-Perception (Big Five)</div>
          <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10 }}>
            Slide each trait to where you think you are, then compare with the AI's prediction.
          </p>
          {BIG_FIVE_DIMS.map(dim => (
            <div key={dim.key} className="perception-slider" style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                <span style={{ color: 'var(--text-muted)' }}>{dim.label}</span>
                <span style={{ color: 'var(--accent)' }}>{(sliders[dim.key] * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min={0} max={1} step={0.01}
                value={sliders[dim.key]}
                onChange={e => setSliders(prev => ({ ...prev, [dim.key]: parseFloat(e.target.value) }))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
            </div>
          ))}
          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button
              onClick={handleCompare}
              style={{
                flex: 1, padding: '7px 0', borderRadius: 6, border: 'none',
                background: 'var(--gradient-1)', color: '#fff', fontSize: 12,
                fontWeight: 600, cursor: 'pointer',
              }}
            >
              Compare
            </button>
            <button
              onClick={() => setShowPerception(false)}
              style={{
                padding: '7px 14px', borderRadius: 6, border: '1px solid var(--border)',
                background: 'transparent', color: 'var(--text-dim)', fontSize: 12,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
