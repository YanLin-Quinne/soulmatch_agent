import React, { useState, useEffect } from 'react';
import { Persona } from '../data/personas';

interface ProfilingModeProps {
  persona: Persona;
  onBack: () => void;
}

export const ProfilingMode: React.FC<ProfilingModeProps> = ({ persona, onBack }) => {
  const [sessionId, setSessionId] = useState<string>('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [inference, setInference] = useState<any>(null);

  useEffect(() => {
    startSession();
  }, [persona]);

  const startSession = async () => {
    try {
      const res = await fetch('/api/profiling/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ persona_id: persona.id })
      });
      const data = await res.json();
      if (data.success) {
        setSessionId(data.session_id);
      }
    } catch (err) {
      console.error('Failed to start session:', err);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/profiling/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: input })
      });
      const data = await res.json();
      
      if (data.success) {
        setMessages(prev => [...prev, { role: 'bot', content: data.bot_message }]);
        setInference(data.inferred_traits);
      }
    } catch (err) {
      console.error('Failed to send message:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="profiling-mode">
      <div className="header">
        <button onClick={onBack}>← 返回</button>
        <h2>画像模式: {persona.name}</h2>
        <span>轮次: {messages.filter(m => m.role === 'user').length}/30</span>
      </div>
      
      <div className="content">
        <div className="chat-panel">
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
            {loading && <div className="message bot loading">思考中...</div>}
          </div>
          <div className="input-area">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="输入消息..."
            />
            <button onClick={sendMessage} disabled={loading}>发送</button>
          </div>
        </div>

        <div className="inference-panel">
          <h3>推断结果</h3>
          {inference && (
            <div className="traits">
              {Object.entries(inference).map(([key, value]: [string, any]) => (
                <div key={key} className="trait">
                  <span className="trait-name">{key}</span>
                  <span className="trait-value">{JSON.stringify(value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
