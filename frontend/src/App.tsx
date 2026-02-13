import { useState, useEffect, useCallback, useRef } from 'react';

const WS_BASE = 'ws://localhost:8000';

// Character display data (matching 8 bot personas)
interface Character {
  id: string;
  name: string;
  emoji: string;
  job: string;
  city: string;
  age: number;
  status: 'Chat' | 'Browsing' | 'Online';
}

const CHARACTERS: Character[] = [
  { id: 'bot_0', name: 'Mina', emoji: '🍳', job: 'Policy Analyst', city: 'San Francisco', age: 29, status: 'Browsing' },
  { id: 'bot_1', name: 'Jade', emoji: '🎵', job: 'Office Admin', city: 'San Francisco', age: 25, status: 'Chat' },
  { id: 'bot_2', name: 'Sierra', emoji: '🏔️', job: 'Math Teacher', city: 'Oakland', age: 34, status: 'Browsing' },
  { id: 'bot_3', name: 'Kevin', emoji: '💻', job: 'Software Dev', city: 'San Francisco', age: 33, status: 'Online' },
  { id: 'bot_4', name: 'Marcus', emoji: '🎸', job: 'VP Operations', city: 'Castro Valley', age: 33, status: 'Chat' },
  { id: 'bot_5', name: 'Derek', emoji: '📊', job: 'Financial Analyst', city: 'Oakland', age: 39, status: 'Browsing' },
  { id: 'bot_6', name: 'Luna', emoji: '🎨', job: 'Freelancer', city: 'Hayward', age: 26, status: 'Chat' },
  { id: 'bot_7', name: 'Travis', emoji: '✈️', job: 'Sales Director', city: 'San Francisco', age: 42, status: 'Online' },
];

// Age filter groups
interface AgeGroup {
  label: string;
  min: number;
  max: number;
}

const AGE_GROUPS: AgeGroup[] = [
  { label: 'All', min: 0, max: 999 },
  { label: '20s', min: 20, max: 29 },
  { label: '30s', min: 30, max: 39 },
  { label: '40+', min: 40, max: 999 },
];

interface BotInfo {
  profile_id: string;
  age: number | null;
  sex: string | null;
  location: string | null;
  communication_style: string;
  core_values: string[];
  interests: string[];
  relationship_goals: string;
  personality_summary: string;
}

interface Message {
  id: string;
  sender: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
}

interface EmotionState {
  emotion: string;
  confidence: number;
  intensity: number;
}

interface WarningState {
  level: string;
  message: string;
  risk_score: number;
}

const EMOTION_EMOJI: Record<string, string> = {
  joy: '😄', sadness: '😢', anger: '😠', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐', love: '😍',
  excitement: '🤩', anxiety: '😰',
};

function App() {
  const userId = useRef(`user_${Date.now()}`);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Page state
  const [page, setPage] = useState<'select' | 'chat'>('select');
  const [selectedChar, setSelectedChar] = useState<Character | null>(null);
  const [ageFilter, setAgeFilter] = useState<string>('All');

  // Chat state
  const [, setCurrentBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [warning, setWarning] = useState<WarningState | null>(null);
  const [turnCount, setTurnCount] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    const websocket = new WebSocket(`${WS_BASE}/ws/${userId.current}`);

    websocket.onopen = () => {
      setIsConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'welcome':
          break;

        case 'conversation_started':
          setIsTyping(false);
          if (data.data) {
            const d = data.data;
            if (d.bot_profile) {
              setCurrentBot(prev => ({ ...prev!, ...d.bot_profile }));
            }
            if (d.greeting) {
              setMessages(prev => [...prev, {
                id: `bot-${Date.now()}`,
                sender: 'bot',
                content: d.greeting,
                timestamp: new Date(),
              }]);
            }
            if (d.match_explanation) {
              setMessages(prev => [...prev, {
                id: `sys-${Date.now()}`,
                sender: 'system',
                content: `Match: ${d.match_explanation} (score: ${(d.compatibility_score * 100).toFixed(0)}%)`,
                timestamp: new Date(),
              }]);
            }
          }
          break;

        case 'bot_message':
          setIsTyping(false);
          if (data.message) {
            setMessages(prev => [...prev, {
              id: `bot-${Date.now()}`,
              sender: 'bot',
              content: data.message,
              timestamp: new Date(),
            }]);
          }
          if (data.turn) setTurnCount(data.turn);
          break;

        case 'emotion':
          if (data.data?.current_emotion) {
            setEmotion(data.data.current_emotion);
          }
          break;

        case 'warning':
          if (data.data) {
            setWarning({
              level: data.data.level,
              message: data.data.message,
              risk_score: data.data.risk_score,
            });
            setTimeout(() => setWarning(null), 10000);
          }
          break;

        case 'feature_update':
          break;

        case 'context':
          if (data.data?.turn_count) setTurnCount(data.data.turn_count);
          break;

        case 'error':
          setIsTyping(false);
          setMessages(prev => [...prev, {
            id: `err-${Date.now()}`,
            sender: 'system',
            content: `Error: ${data.message}`,
            timestamp: new Date(),
          }]);
          break;
      }
    };

    websocket.onerror = () => setIsConnected(false);
    websocket.onclose = () => setIsConnected(false);

    setWs(websocket);
    return websocket;
  }, []);

  useEffect(() => {
    const websocket = connectWebSocket();
    return () => { websocket.close(); };
  }, [connectWebSocket]);

  // Handle card click
  const handleCardClick = (char: Character) => {
    setSelectedChar(char);
    setPage('chat');
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setIsTyping(true);

    // Create placeholder bot info
    const botInfo: BotInfo = {
      profile_id: char.id,
      age: char.age,
      sex: null,
      location: char.city,
      communication_style: 'casual',
      core_values: [],
      interests: [],
      relationship_goals: 'unsure',
      personality_summary: '',
    };
    setCurrentBot(botInfo);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'start' }));
    }
  };

  // Send message
  const handleSend = () => {
    if (!inputText.trim() || !ws || ws.readyState !== WebSocket.OPEN) return;

    const content = inputText.trim();
    setMessages(prev => [...prev, {
      id: `user-${Date.now()}`,
      sender: 'user',
      content,
      timestamp: new Date(),
    }]);
    setInputText('');
    setIsTyping(true);

    ws.send(JSON.stringify({ action: 'message', content }));
  };

  // Back to selection page
  const handleBack = () => {
    setPage('select');
    setSelectedChar(null);
    setCurrentBot(null);
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setInputText('');
  };

  // Filter characters
  const filteredCharacters = CHARACTERS.filter(char => {
    const group = AGE_GROUPS.find(g => g.label === ageFilter);
    if (!group) return true;
    return char.age >= group.min && char.age <= group.max;
  });

  return (
    <div>
      {page === 'select' ? (
        // Character Selection Page
        <div className="select-page">
          <div className="page-header">
            <h1 className="page-title">SoulMatch</h1>
            <p className="page-description">
              Choose someone to start chatting. After 30 exchanges, the system will predict their personality, psychology, and social traits.<br/>
              Note — some profiles are AI-powered.
            </p>
          </div>

          <div className="age-filter-group">
            {AGE_GROUPS.map(group => (
              <button
                key={group.label}
                className={`age-filter-btn ${ageFilter === group.label ? 'active' : ''}`}
                onClick={() => setAgeFilter(group.label)}
              >
                {group.label}
              </button>
            ))}
          </div>

          <div className="character-grid">
            {filteredCharacters.map(char => (
              <div
                key={char.id}
                className="character-card"
                onClick={() => handleCardClick(char)}
              >
                <div className="card-emoji">{char.emoji}</div>
                <div className="card-name">{char.name}</div>
                <div className="card-job">{char.job} · {char.city}</div>
                <div className="card-tags">
                  <span className="tag tag-age">{char.age}</span>
                  <span className="tag tag-city">{char.city}</span>
                  <span className={`tag tag-status-${char.status.toLowerCase()}`}>{char.status}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="page-footer">
            🎭 Some of these {CHARACTERS.length} profiles are AI bots — can you tell which?
          </div>
        </div>
      ) : (
        // Chat Page
        <div className="chat-page">
          <div className="chat-header">
            <button className="back-btn" onClick={handleBack}>
              ← Back
            </button>
            {selectedChar && (
              <div className="chat-bot-info">
                <span className="chat-bot-emoji">{selectedChar.emoji}</span>
                <div>
                  <div className="chat-bot-name">{selectedChar.name}</div>
                  <div className="chat-bot-detail">{selectedChar.job} · {selectedChar.city}</div>
                </div>
              </div>
            )}
            <div className="chat-header-right">
              {emotion && (
                <div className="emotion-badge">
                  {EMOTION_EMOJI[emotion.emotion] || '😐'} {emotion.emotion}
                </div>
              )}
              <span className="turn-count">Turn: {turnCount}</span>
            </div>
          </div>

          {warning && (
            <div className={`warning-banner ${warning.level}`}>
              ⚠️ Scam Warning ({warning.level}): {warning.message}
            </div>
          )}

          <div className="message-list">
            {messages.length === 0 && (
              <div className="empty-state">
                <div style={{ fontSize: '3rem' }}>💬</div>
                <h2>Start a conversation</h2>
                <p>Send a message to begin chatting</p>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`message-bubble ${msg.sender}`}>
                {msg.content}
              </div>
            ))}
            {isTyping && (
              <div className="message-bubble bot typing-bubble">
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
                <span className="typing-dot"></span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-bar">
            <input
              className="input-field"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder="Type a message..."
              disabled={!isConnected}
            />
            <button className="send-btn" onClick={handleSend} disabled={!isConnected || !inputText.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
