import React, { useState, useEffect } from 'react';
import { PERSONAS } from '../data/personas';

interface PlaygroundModeProps {
  onBack: () => void;
}

export const PlaygroundMode: React.FC<PlaygroundModeProps> = ({ onBack }) => {
  const [sessionId, setSessionId] = useState<string>('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [input, setInput] = useState('');
  const [guessMode, setGuessMode] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    startGame();
  }, []);

  const startGame = async () => {
    try {
      const res = await fetch('/api/playground/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: `player_${Date.now()}` })
      });
      const data = await res.json();
      if (data.success) {
        setSessionId(data.session_id);
        setMessages([{ role: 'system', content: data.message }]);
      }
    } catch (err) {
      console.error('Failed to start game:', err);
    }
  };

  const submitGuess = async (personaId: number) => {
    try {
      const res = await fetch('/api/playground/guess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, guess_persona_id: personaId })
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('Failed to submit guess:', err);
    }
  };

  return (
    <div className="playground-mode">
      <div className="header">
        <button onClick={onBack}>← 返回</button>
        <h2>推理模式: 猜猜我是谁</h2>
        <span>消息: {messages.filter(m => m.role === 'user').length}/10</span>
      </div>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>

      {!result && (
        <>
          <div className="input-area">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="问我问题..."
            />
            <button onClick={() => setGuessMode(true)}>提交猜测</button>
          </div>

          {guessMode && (
            <div className="guess-grid">
              {PERSONAS.map((p) => (
                <div key={p.id} className="guess-card" onClick={() => submitGuess(p.id)}>
                  <div>{p.avatar}</div>
                  <div>{p.name}</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {result && (
        <div className="result">
          <h3>{result.correct ? '🎉 猜对了!' : '❌ 猜错了'}</h3>
          <p>正确答案: {PERSONAS[result.actual_persona_id].name}</p>
          <p>得分: {result.score}</p>
          <button onClick={startGame}>再玩一次</button>
        </div>
      )}
    </div>
  );
};
