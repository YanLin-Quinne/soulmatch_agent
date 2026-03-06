import { useState, useEffect, useCallback, useRef } from 'react';
import RelationshipTab from './components/RelationshipTab';
import SentimentBanner from './components/SentimentBanner';
import ConversationHints from './components/ConversationHints';
import DigitalTwinPanel from './components/DigitalTwinPanel';
import HomeScreen from './components/HomeScreen';
import DigitalTwinSetup from './components/DigitalTwinSetup';
import ComparisonView from './components/ComparisonView';
import { PERSONAS, Persona } from './data/personas';

const WS_BASE = window.location.protocol === 'https:'
  ? `wss://${window.location.host}`
  : 'ws://localhost:8000';

interface Character {
  id: string;
  name: string;
  avatar: string;
  job: string;
  city: string;
  age: number;
  status: 'Online' | 'Away' | 'Busy';
  interests: string[];
  bio: string;
}

const CHARACTERS: Character[] = PERSONAS.map(p => ({
  id: `bot_${p.id}`,
  name: p.name,
  avatar: `https://api.dicebear.com/9.x/notionists/svg?seed=${p.name}`,
  job: p.occupation,
  city: 'Remote',
  age: p.age,
  status: 'Online' as const,
  interests: p.interests,
  bio: p.bio
}));

const AGE_FILTERS = ['All', '20s', '30s', '40+'] as const;

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
  quotedMessage?: { id: string; content: string };
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

interface FeatureData {
  features: Record<string, any>;
  confidences: Record<string, number>;
  turn: number;
  low_confidence: string[];
  average_confidence: number;
  conformal?: {
    coverage_guarantee: number;
    avg_set_size: number;
    singletons: number;
    total_dims: number;
    prediction_sets: Record<string, {
      set: string[];
      point: string;
      size: number;
      llm_conf: number;
      calibrated_conf: number;
    }>;
  };
}

interface MemoryItem {
  content: string;
  relevance?: number;
}

interface EmotionEntry {
  turn: number;
  emotion: string;
  intensity: number;
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

  const [page, setPage] = useState<'home' | 'chat' | 'twin-setup' | 'twin-chat' | 'comparison'>('home');
  const [selectedChar, setSelectedChar] = useState<Character | null>(null);
  const [friendGuess, setFriendGuess] = useState<any>(null);
  const [systemInference, setSystemInference] = useState<any>(null);

  const [, setCurrentBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [peerTyping, setPeerTyping] = useState(false);
  const [quotedMessage, setQuotedMessage] = useState<{ id: string; content: string } | null>(null);

  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [warning, setWarning] = useState<WarningState | null>(null);
  const [turnCount, setTurnCount] = useState(0);

  // Theme and sidebar state
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('soulmate-theme') as 'dark' | 'light') || 'dark';
  });
  const [activeTab, setActiveTab] = useState<'predict' | 'emotion' | 'safety' | 'memory' | 'relationship' | 'twin'>('predict');

  // New data states for sidebar
  const [featureData, setFeatureData] = useState<FeatureData | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionEntry[]>([]);
  const [scamData, setScamData] = useState<{ level: string; message: string; risk_score: number } | null>(null);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [contextData, setContextData] = useState<any>(null);
  const [sidebarOpen] = useState(true);

  // v2.0: Relationship prediction states
  const [relationshipData, setRelationshipData] = useState<{
    rel_status: string;
    rel_type: string;
    sentiment: string;
    can_advance: boolean;
    advance_prediction_set: string[];
    social_votes?: { agent: string; vote: string; rel_status: string; confidence: number; reasoning: string }[];
    vote_distribution?: Record<string, number>;
  } | null>(null);
  const [milestoneReport, setMilestoneReport] = useState<any>(null);
  const [trustHistory, setTrustHistory] = useState<{ turn: number; trust: number }[]>([]);

  // EMNLP demo: conversation sentiment
  const [conversationSentiment, setConversationSentiment] = useState<{
    label: string; score: number; trend: string; hints: string[];
  } | null>(null);
  const [conversationHints, setConversationHints] = useState<{text: string, type: string}[] | null>(null);
  const [loveDetail, setLoveDetail] = useState<any>(null);
  const [twinData, setTwinData] = useState<any>(null);
  const [twinMessages, setTwinMessages] = useState<{sender: string, content: string}[]>([]);
  const [memoryStats, setMemoryStats] = useState<{
    current_turn: number;
    working_memory_size: number;
    episodic_memory_count: number;
    semantic_memory_count: number;
    compression_ratio: number;
    episodic_memories?: { episode_id: string; turn_range: number[]; summary: string; key_events: string[]; emotion_trend: string }[];
    semantic_memories?: { reflection_id: string; turn_range: number[]; reflection: string; feature_updates: Record<string, any>; relationship_insights: string }[];
  } | null>(null);

  const typingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isUnmountedRef = useRef(false);
  const selectedCharRef = useRef<Character | null>(null);
  const reconnectAttemptRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 10;
  const BASE_RECONNECT_DELAY = 2000; // 2s base, doubles each attempt, max 30s

  // Theme effect
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('soulmate-theme', theme);
  }, [theme]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const connectWebSocket = useCallback(() => {
    const websocket = new WebSocket(`${WS_BASE}/ws/${userId.current}`);

    websocket.onopen = () => {
      setIsConnected(true);
      reconnectAttemptRef.current = 0; // Reset on successful connect
      // Reconnect: re-send start if user was in a chat
      if (selectedCharRef.current) {
        websocket.send(JSON.stringify({ action: 'start', bot_id: selectedCharRef.current.id }));
      }
      // Start ping heartbeat every 20s
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = setInterval(() => {
        if (websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({ action: 'ping' }));
        }
      }, 20000);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'welcome':
          break;
        case 'heartbeat':
        case 'pong':
          break;
        case 'typing_indicator':
          if (data.data) {
            setPeerTyping(data.data.is_typing);
          }
          break;
        case 'conversation_started':
          setIsTyping(false);
          if (data.data) {
            const d = data.data;
            if (d.bot_profile) setCurrentBot(prev => ({ ...prev!, ...d.bot_profile }));
            // Only show greeting on first connect, not on reconnect
            if (d.greeting) {
              setMessages(prev => {
                if (prev.length === 0) {
                  const msgs: typeof prev = [{ id: `bot-${Date.now()}`, sender: 'bot', content: d.greeting, timestamp: new Date() }];
                  if (d.match_explanation) {
                    msgs.push({ id: `sys-${Date.now()}`, sender: 'system', content: `Match: ${d.match_explanation} (score: ${(d.compatibility_score * 100).toFixed(0)}%)`, timestamp: new Date() });
                  }
                  return msgs;
                }
                return prev;
              });
            }
          }
          break;
        case 'bot_message':
          setIsTyping(false);
          if (data.message) {
            const botMsg: Message = { id: `bot-${Date.now()}`, sender: 'bot', content: data.message, timestamp: new Date() };
            if (data.quoted) {
              botMsg.quotedMessage = { id: data.quoted.id, content: data.quoted.content };
            }
            setMessages(prev => [...prev, botMsg]);
          }
          if (data.turn) setTurnCount(data.turn);
          // Capture memory updates if present
          if (data.memory_update) {
            setMemories(data.memory_update);
          }
          break;
        case 'emotion':
          if (data.data?.current_emotion) {
            const emo = data.data.current_emotion;
            setEmotion(emo);
            // Add to emotion history
            setEmotionHistory(prev => [...prev, {
              turn: turnCount + 1,
              emotion: emo.emotion,
              intensity: emo.intensity
            }]);
          }
          break;
        case 'warning':
          if (data.data) {
            setWarning({ level: data.data.level, message: data.data.message, risk_score: data.data.risk_score });
            setScamData({ level: data.data.level, message: data.data.message, risk_score: data.data.risk_score });
            setTimeout(() => setWarning(null), 10000);
          }
          break;
        case 'feature_update':
          if (data.data) {
            setFeatureData(data.data);
          }
          break;
        case 'relationship_prediction':
          if (data.data) {
            setRelationshipData(data.data);
            // Track trust score history
            const trust = data.data.trust_score ?? 0.5;
            setTrustHistory((prev) => [...prev, { turn: data.data.turn ?? 0, trust }]);
          }
          break;
        case 'milestone_report':
          if (data.data) {
            setMilestoneReport(data.data);
            setActiveTab('relationship');
          }
          break;
        case 'context':
          if (data.data) {
            setContextData(data.data);
            if (data.data.turn_count) setTurnCount(data.data.turn_count);
          }
          break;
        case 'conversation_sentiment':
          if (data.data) {
            setConversationSentiment(data.data);
          }
          break;
        case 'conversation_hints':
          if (data.data) {
            setConversationHints(data.data);
          }
          break;
        case 'threshold_reached':
          if (data.data) {
            setMessages(prev => [...prev, {
              id: `sys-threshold-${Date.now()}`,
              sender: 'system',
              content: data.data.message || '30-turn threshold reached! Predictions are now more frequent.',
              timestamp: new Date(),
            }]);
            setActiveTab('relationship');
          }
          break;
        case 'memory_stats':
          if (data.data) {
            setMemoryStats(data.data);
          }
          break;
        case 'love_prediction':
          if (data.data) {
            setLoveDetail(data.data);
          }
          break;
        case 'digital_twin_created':
          if (data.data) {
            setTwinData(data.data);
            setActiveTab('twin');
          }
          break;
        case 'twin_message':
          if (data.data) {
            setTwinMessages(prev => [...prev, data.data]);
          }
          break;
        case 'perception_comparison':
          if (data.data) {
            const rate = Math.round((data.data.overall_match_rate || 0) * 100);
            setMessages(prev => [...prev, {
              id: `sys-perception-${Date.now()}`,
              sender: 'system',
              content: `Perception comparison: ${rate}% match (${data.data.matched}/${data.data.total} dimensions within tolerance).`,
              timestamp: new Date(),
            }]);
          }
          break;
        case 'error':
          setIsTyping(false);
          setMessages(prev => [...prev, { id: `err-${Date.now()}`, sender: 'system', content: `Error: ${data.message}`, timestamp: new Date() }]);
          break;
      }
    };

    websocket.onerror = () => setIsConnected(false);
    websocket.onclose = () => {
      setIsConnected(false);
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      if (!isUnmountedRef.current && reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttemptRef.current), 30000);
        reconnectAttemptRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      }
    };
    setWs(websocket);
    return websocket;
  }, []);

  useEffect(() => {
    isUnmountedRef.current = false;
    const websocket = connectWebSocket();
    return () => {
      isUnmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      websocket.close();
    };
  }, [connectWebSocket]);

  const handleCardClick = (char: Character) => {
    setSelectedChar(char);
    selectedCharRef.current = char;
    setPage('chat');
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setIsTyping(true);
    setQuotedMessage(null);
    setPeerTyping(false);
    // Reset new states
    setFeatureData(null);
    setEmotionHistory([]);
    setScamData(null);
    setMemories([]);
    setContextData(null);
    setActiveTab('predict');
    setConversationHints(null);
    setTwinData(null);
    setTwinMessages([]);
    setLoveDetail(null);
    setCurrentBot({
      profile_id: char.id, age: char.age, sex: null, location: char.city,
      communication_style: 'casual', core_values: [], interests: char.interests,
      relationship_goals: 'unsure', personality_summary: '',
    });
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'start', bot_id: char.id }));
    }
  };

  const handleSend = () => {
    if (!inputText.trim() || !ws || ws.readyState !== WebSocket.OPEN) return;
    const content = inputText.trim();
    setMessages(prev => [...prev, {
      id: `user-${Date.now()}`, sender: 'user', content, timestamp: new Date(),
      quotedMessage: quotedMessage ?? undefined,
    }]);
    setInputText('');
    setIsTyping(true);
    setConversationHints(null);
    const payload: any = { action: 'message', content };
    if (quotedMessage) {
      payload.quote_id = quotedMessage.id;
      payload.quote_text = quotedMessage.content;
    }
    ws.send(JSON.stringify(payload));
    setQuotedMessage(null);
  };

  const handleBack = () => {
    setPage('home');
    setSelectedChar(null);
    selectedCharRef.current = null;
    setCurrentBot(null);
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setInputText('');
    setQuotedMessage(null);
    setPeerTyping(false);
    setFeatureData(null);
    setEmotionHistory([]);
    setScamData(null);
    setMemories([]);
    setContextData(null);
    setActiveTab('predict');
    setConversationSentiment(null);
    setConversationHints(null);
    setTwinData(null);
    setTwinMessages([]);
    setLoveDetail(null);
    setFriendGuess(null);
    setSystemInference(null);
  };

  const progress = Math.min(turnCount / 30, 1);

  const toggleTheme = () => {
    setTheme(t => t === 'dark' ? 'light' : 'dark');
  };

  if (page === 'chat' && selectedChar) {
    return (
      <div className="chat-page">
        <div className="chat-header">
          <button className="back-btn" onClick={handleBack}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5"/><path d="M12 19l-7-7 7-7"/></svg>
          </button>
          <img className="chat-avatar" src={selectedChar.avatar} alt={selectedChar.name} />
          <div className="chat-user-info">
            <div className="chat-user-name">{selectedChar.name}</div>
            <div className="chat-user-detail">{selectedChar.job} &middot; {selectedChar.city}</div>
          </div>
          <div className="chat-header-right">
            <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
              {theme === 'dark' ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </button>
            {emotion && (
              <span className="emotion-pill">{EMOTION_EMOJI[emotion.emotion] || '😐'} {emotion.emotion}</span>
            )}
            <div className="progress-ring" title={`${turnCount}/30 turns`}>
              <svg viewBox="0 0 36 36">
                <path className="progress-bg" d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path className="progress-fill" strokeDasharray={`${progress * 100}, 100`} d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831" />
              </svg>
              <span className="progress-text">{turnCount}</span>
            </div>
          </div>
        </div>

        {!isConnected && (
          <div className="connection-banner">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 1l22 22"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.56 9"/><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>
            {reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS
              ? 'Connection lost. Please refresh the page.'
              : `Reconnecting... (attempt ${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})`}
          </div>
        )}

        {warning && (
          <div className={`warning-bar warning-${warning.level}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            {warning.level}: {warning.message}
          </div>
        )}

        <SentimentBanner sentiment={conversationSentiment} turnCount={turnCount} />

        <div className="chat-layout">
          <div className="chat-main">
            <div className="messages">
              {messages.length === 0 && !isTyping && (
                <div className="empty-chat">
                  <div className="empty-chat-avatar">
                    <img src={selectedChar.avatar} alt="" />
                  </div>
                  <h3>Chat with {selectedChar.name}</h3>
                  <p>Say hi to start the conversation. After 30 exchanges, we'll reveal personality insights.</p>
                </div>
              )}
              {messages.map(msg => (
                <div key={msg.id} className={`msg msg-${msg.sender}`}>
                  {msg.quotedMessage && (
                    <div className="msg-quote">
                      <span className="msg-quote-label">Replying to:</span> {msg.quotedMessage.content.slice(0, 80)}{msg.quotedMessage.content.length > 80 ? '...' : ''}
                    </div>
                  )}
                  <div className="msg-content">
                    {msg.content}
                    {msg.sender !== 'system' && (
                      <button
                        className="msg-quote-btn"
                        title="Quote this message"
                        onClick={() => setQuotedMessage({ id: msg.id, content: msg.content })}
                      >↩</button>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="msg msg-bot">
                  <div className="msg-content typing">
                    <span /><span /><span />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <ConversationHints
              hints={conversationHints}
              onSelectHint={(text) => { setInputText(text); setConversationHints(null); }}
              onDismiss={() => setConversationHints(null)}
            />

            <div className="input-area">
              {quotedMessage && (
                <div className="quote-preview">
                  <span>Replying to: {quotedMessage.content.slice(0, 60)}{quotedMessage.content.length > 60 ? '...' : ''}</span>
                  <button onClick={() => setQuotedMessage(null)} style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 16, padding: '0 4px' }}>&times;</button>
                </div>
              )}
              <input
                value={inputText}
                onChange={e => {
                  setInputText(e.target.value);
                  // Typing indicator debounce
                  if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: 'typing', is_typing: true }));
                    if (typingTimerRef.current) clearTimeout(typingTimerRef.current);
                    typingTimerRef.current = setTimeout(() => {
                      ws.send(JSON.stringify({ action: 'typing', is_typing: false }));
                    }, 2000);
                  }
                }}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Type something..."
                disabled={!isConnected}
              />
              <button onClick={handleSend} disabled={!isConnected || !inputText.trim()}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
          </div>

          {sidebarOpen && (
            <aside className="analysis-sidebar">
              <div className="sidebar-tabs">
                <button 
                  className={`sidebar-tab ${activeTab === 'predict' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('predict')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>
                  </svg>
                  Predict
                </button>
                <button 
                  className={`sidebar-tab ${activeTab === 'emotion' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('emotion')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                  </svg>
                  Emotion
                </button>
                <button 
                  className={`sidebar-tab ${activeTab === 'safety' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('safety')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                  </svg>
                  Safety
                </button>
                <button
                  className={`sidebar-tab ${activeTab === 'memory' ? 'active' : ''}`}
                  onClick={() => setActiveTab('memory')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>
                  </svg>
                  Memory
                </button>
                <button
                  className={`sidebar-tab ${activeTab === 'relationship' ? 'active' : ''}`}
                  onClick={() => setActiveTab('relationship')}
                  style={{ position: 'relative' }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                  </svg>
                  Relation
                  {milestoneReport && <span style={{ position: 'absolute', top: 2, right: 2, width: 6, height: 6, borderRadius: '50%', background: '#a855f7' }} />}
                </button>
                <button
                  className={`sidebar-tab ${activeTab === 'twin' ? 'active' : ''}`}
                  onClick={() => setActiveTab('twin')}
                  style={{ position: 'relative' }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
                  </svg>
                  Twin
                  {twinData && <span style={{ position: 'absolute', top: 2, right: 2, width: 6, height: 6, borderRadius: '50%', background: '#06b6d4' }} />}
                </button>
              </div>

              <div className="sidebar-panel">
                {activeTab === 'predict' && (
                  <>
                    {featureData ? (
                      <>
                        <div className="overall-confidence">
                          <h4 style={{ textAlign: 'center', marginBottom: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>Overall Confidence</h4>
                          <div className="big-number">{Math.round(featureData.average_confidence * 100)}%</div>
                          <div className="confidence-bar">
                            <div className="confidence-bar-track">
                              <div className="confidence-bar-fill" style={{ width: `${featureData.average_confidence * 100}%` }} />
                            </div>
                          </div>
                        </div>

                        <div className="sidebar-section">
                          <h4>Personality (Big Five)</h4>
                          {Object.entries(featureData.features)
                            .filter(([key]) => key.startsWith('big_five_'))
                            .map(([key, value]) => {
                              const trait = key.replace('big_five_', '').replace(/_/g, ' ');
                              const confidence = featureData.confidences[key] || 0;
                              return (
                                <div key={key} className="confidence-bar">
                                  <div className="confidence-bar-label">
                                    <span style={{ textTransform: 'capitalize' }}>{trait}</span>
                                    <span>{Math.round(confidence * 100)}%</span>
                                  </div>
                                  <div className="confidence-bar-track">
                                    <div className="confidence-bar-fill" style={{ width: `${confidence * 100}%` }} />
                                  </div>
                                </div>
                              );
                            })}
                        </div>

                        <div className="sidebar-section">
                          <h4>Demographics</h4>
                          {featureData.features.age !== undefined && (
                            <div className="stat-row">
                              <span className="label">Age</span>
                              <span className="value">{featureData.features.age}</span>
                            </div>
                          )}
                          {featureData.features.sex && (
                            <div className="stat-row">
                              <span className="label">Sex</span>
                              <span className="value">{featureData.features.sex}</span>
                            </div>
                          )}
                          {featureData.features.orientation && (
                            <div className="stat-row">
                              <span className="label">Orientation</span>
                              <span className="value">{featureData.features.orientation}</span>
                            </div>
                          )}
                          {featureData.features.location && (
                            <div className="stat-row">
                              <span className="label">Location</span>
                              <span className="value">{featureData.features.location}</span>
                            </div>
                          )}
                        </div>

                        <div className="sidebar-section">
                          <h4>Interests</h4>
                          {Object.entries(featureData.features)
                            .filter(([key]) => key.startsWith('interest_'))
                            .map(([key]) => {
                              const interest = key.replace('interest_', '').replace(/_/g, ' ');
                              const confidence = featureData.confidences[key] || 0;
                              return (
                                <div key={key} className="confidence-bar">
                                  <div className="confidence-bar-label">
                                    <span style={{ textTransform: 'capitalize' }}>{interest}</span>
                                    <span>{Math.round(confidence * 100)}%</span>
                                  </div>
                                  <div className="confidence-bar-track">
                                    <div className="confidence-bar-fill" style={{ width: `${confidence * 100}%` }} />
                                  </div>
                                </div>
                              );
                            })}
                        </div>

                        {featureData.low_confidence && featureData.low_confidence.length > 0 && (
                          <div className="sidebar-section">
                            <h4>Low Confidence Features</h4>
                            <div className="low-confidence-tags">
                              {featureData.low_confidence.map((feature, idx) => (
                                <span key={idx} className="low-confidence-tag">{feature}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Conformal Prediction Sets */}
                        {featureData.conformal && (
                          <div className="sidebar-section">
                            <h4>Conformal Prediction</h4>
                            <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                              <span className="cp-badge cp-badge-coverage">
                                Coverage: {Math.round(featureData.conformal.coverage_guarantee * 100)}%
                              </span>
                              <span className="cp-badge cp-badge-sets">
                                Avg Set: {featureData.conformal.avg_set_size.toFixed(1)}
                              </span>
                              <span className="cp-badge cp-badge-singletons">
                                Singletons: {featureData.conformal.singletons}/{featureData.conformal.total_dims}
                              </span>
                            </div>
                            <div className="cp-sets-list">
                              {Object.entries(featureData.conformal.prediction_sets).map(([dim, ps]) => (
                                <div key={dim} className="cp-set-item">
                                  <div className="cp-set-header">
                                    <span className="cp-dim-name">{dim.replace(/_/g, ' ')}</span>
                                    <span className="cp-point-pred">{ps.point}</span>
                                  </div>
                                  <div className="cp-set-tags">
                                    {ps.set.map((val, i) => (
                                      <span key={i} className={`cp-set-tag ${val === ps.point ? 'cp-set-tag-point' : ''}`}>
                                        {val}
                                      </span>
                                    ))}
                                  </div>
                                  <div className="cp-conf-row">
                                    <span>LLM: {Math.round(ps.llm_conf * 100)}%</span>
                                    <span style={{ color: 'var(--accent)' }}>Calibrated: {Math.round(ps.calibrated_conf * 100)}%</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="placeholder">Waiting for prediction data...</div>
                    )}
                  </>
                )}

                {activeTab === 'emotion' && (
                  <>
                    {emotion && (
                      <div className="current-emotion-display">
                        <div className="current-emotion-emoji">{EMOTION_EMOJI[emotion.emotion] || '😐'}</div>
                        <div className="current-emotion-name">{emotion.emotion}</div>
                        <div className="current-emotion-intensity">
                          <div className="confidence-bar-label">
                            <span>Intensity</span>
                            <span>{Math.round(emotion.intensity * 100)}%</span>
                          </div>
                          <div className="confidence-bar-track">
                            <div className="confidence-bar-fill" style={{ width: `${emotion.intensity * 100}%` }} />
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="sidebar-section">
                      <h4>Emotion Timeline</h4>
                      {emotionHistory.length > 0 ? (
                        <div className="emotion-timeline">
                          {[...emotionHistory].reverse().map((entry, idx) => (
                            <div key={idx} className="emotion-entry">
                              <span className="turn">t{String(entry.turn).padStart(2, '0')}</span>
                              <span className="emoji">{EMOTION_EMOJI[entry.emotion] || '😐'}</span>
                              <span className="name">{entry.emotion}</span>
                              <div className="intensity-bar">
                                <div className="intensity-fill" style={{ width: `${entry.intensity * 100}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="placeholder">No emotion data yet</div>
                      )}
                    </div>
                  </>
                )}

                {activeTab === 'safety' && (
                  <>
                    {scamData ? (
                      <>
                        <div className={`safety-indicator ${scamData.level}`}>
                          <div className="dot" />
                          <div className="text">Risk Level: {scamData.level}</div>
                          <div className="score">{Math.round(scamData.risk_score * 100)}%</div>
                        </div>
                        <div className="sidebar-section">
                          <h4>Warning Message</h4>
                          <p style={{ fontSize: '13px', lineHeight: '1.6', color: 'var(--text)' }}>{scamData.message}</p>
                        </div>
                      </>
                    ) : (
                      <div className="safety-indicator safe">
                        <div className="dot" />
                        <div className="text">No risks detected</div>
                      </div>
                    )}

                    {contextData && contextData.risk_level && (
                      <div className="sidebar-section">
                        <h4>Context Risk Level</h4>
                        <div className="stat-row">
                          <span className="label">Current Status</span>
                          <span className="value">{contextData.risk_level}</span>
                        </div>
                        {contextData.state && (
                          <div className="stat-row">
                            <span className="label">State</span>
                            <span className="value">{contextData.state}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}

                {activeTab === 'memory' && (
                  <>
                    {/* Three-Layer Memory Architecture */}
                    {memoryStats && (
                      <div className="sidebar-section">
                        <h4>Three-Layer Memory</h4>
                        <div className="stat-row">
                          <span className="label">Working Memory</span>
                          <span className="value" style={{ color: 'var(--accent)' }}>
                            {memoryStats.working_memory_size} turns
                          </span>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
                          Recent conversation buffer (max 20)
                        </div>
                        <div className="stat-row">
                          <span className="label">Episodic Memory</span>
                          <span className="value" style={{ color: memoryStats.episodic_memory_count > 0 ? '#facc15' : 'var(--text-dim)' }}>
                            {memoryStats.episodic_memory_count} episodes
                          </span>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
                          Compressed conversation summaries
                        </div>
                        <div className="stat-row">
                          <span className="label">Semantic Memory</span>
                          <span className="value" style={{ color: memoryStats.semantic_memory_count > 0 ? '#c084fc' : 'var(--text-dim)' }}>
                            {memoryStats.semantic_memory_count} insights
                          </span>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
                          High-level user understanding
                        </div>
                        {memoryStats.compression_ratio > 0 && (
                          <div className="stat-row" style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                            <span className="label">Compression</span>
                            <span className="value">{(memoryStats.compression_ratio * 100).toFixed(0)}%</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Episodic Memory Content (Layer 2) */}
                    {memoryStats?.episodic_memories && memoryStats.episodic_memories.length > 0 && (
                      <div className="sidebar-section">
                        <h4>Episodic Memories</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {memoryStats.episodic_memories.map((ep) => (
                            <div key={ep.episode_id} className="episodic-card">
                              <div className="episodic-header">
                                <span className="episodic-range">Turns {ep.turn_range[0]}-{ep.turn_range[1]}</span>
                                <span className={`episodic-trend episodic-trend-${ep.emotion_trend}`}>
                                  {ep.emotion_trend === 'improving' ? '↑' : ep.emotion_trend === 'declining' ? '↓' : '→'} {ep.emotion_trend}
                                </span>
                              </div>
                              <div className="episodic-summary">{ep.summary}</div>
                              {ep.key_events.length > 0 && (
                                <div className="episodic-events">
                                  {ep.key_events.map((evt, i) => (
                                    <span key={i} className="episodic-event-tag">{evt}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Semantic Memory Content (Layer 3) */}
                    {memoryStats?.semantic_memories && memoryStats.semantic_memories.length > 0 && (
                      <div className="sidebar-section">
                        <h4>Semantic Reflections</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {memoryStats.semantic_memories.map((sem) => (
                            <div key={sem.reflection_id} className="semantic-card">
                              <div className="semantic-header">
                                Turns {sem.turn_range[0]}-{sem.turn_range[1]}
                              </div>
                              <div className="semantic-reflection">{sem.reflection}</div>
                              {sem.relationship_insights && (
                                <div className="semantic-insights">
                                  <span className="semantic-insights-label">Relationship:</span> {sem.relationship_insights}
                                </div>
                              )}
                              {Object.keys(sem.feature_updates).length > 0 && (
                                <div className="semantic-features">
                                  {Object.entries(sem.feature_updates).map(([key, val]) => (
                                    <div key={key} className="semantic-feature-item">
                                      <span>{key}</span>
                                      <span style={{ color: 'var(--accent)' }}>
                                        {typeof val === 'object' ? val.change : String(val)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="sidebar-section">
                      <h4>Retrieved Memories</h4>
                      {memories.length > 0 ? (
                        <div className="memory-list">
                          {memories.map((mem, idx) => (
                            <div key={idx} className="memory-item">
                              {mem.content}
                              {mem.relevance !== undefined && (
                                <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-dim)' }}>
                                  Relevance: {Math.round(mem.relevance * 100)}%
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="placeholder">No memories retrieved yet</div>
                      )}
                    </div>

                    {contextData && (
                      <div className="sidebar-section">
                        <h4>Context Info</h4>
                        {contextData.state && (
                          <div className="stat-row">
                            <span className="label">State</span>
                            <span className="value">{contextData.state}</span>
                          </div>
                        )}
                        {contextData.user_emotion && (
                          <div className="stat-row">
                            <span className="label">User Emotion</span>
                            <span className="value">{contextData.user_emotion}</span>
                          </div>
                        )}
                        {contextData.avg_feature_confidence !== undefined && (
                          <div className="stat-row">
                            <span className="label">Avg Confidence</span>
                            <span className="value">{Math.round(contextData.avg_feature_confidence * 100)}%</span>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}

                {activeTab === 'relationship' && (
                  <RelationshipTab
                    relationshipData={relationshipData}
                    milestoneReport={milestoneReport}
                    trustHistory={trustHistory}
                    turnCount={turnCount}
                    loveDetail={loveDetail}
                  />
                )}

                {activeTab === 'twin' && (
                  <DigitalTwinPanel
                    twinData={twinData}
                    twinMessages={twinMessages}
                    turnCount={turnCount}
                    onCreateTwin={() => {
                      if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: 'create_twin' }));
                      }
                    }}
                    onSendTwinMessage={(msg) => {
                      if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: 'twin_message', content: msg }));
                      }
                    }}
                    onComparePerception={(perception) => {
                      if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: 'compare_perception', perception }));
                      }
                    }}
                  />
                )}
              </div>
            </aside>
          )}
        </div>
      </div>
    );
  }

  if (page === 'home') {
    return (
      <HomeScreen
        onSelectPersona={(persona) => {
          const char = CHARACTERS.find(c => c.id === `bot_${persona.id}`);
          if (char) handleCardClick(char);
        }}
        onEnterDigitalTwin={() => setPage('twin-setup')}
      />
    );
  }

  if (page === 'twin-setup') {
    return (
      <DigitalTwinSetup
        onStartChat={(guess) => {
          setFriendGuess(guess);
          setPage('twin-chat');
          // TODO: 启动与 AI 分身的聊天
        }}
      />
    );
  }

  if (page === 'twin-chat') {
    // AI 分身聊天：复用现有聊天界面，20句后跳转对比
    const twinProgress = Math.min(turnCount / 20, 1);

    // 20句后自动跳转到对比页面
    useEffect(() => {
      if (turnCount >= 20 && friendGuess && page === 'twin-chat') {
        setSystemInference({
          gender: featureData?.features?.sex || 'Unknown',
          age_range: featureData?.features?.age || 'Unknown',
          occupation_guess: featureData?.features?.occupation || 'Unknown',
          mbti: featureData?.features?.mbti || 'Unknown',
          bigFive: {
            E: featureData?.features?.big_five_extraversion || 0.5,
            A: featureData?.features?.big_five_agreeableness || 0.5
          },
          communication_style: featureData?.features?.communication_style || 'Unknown'
        });
        setPage('comparison');
      }
    }, [turnCount, friendGuess, page, featureData]);

    return (
      <div className="chat-page">
        <div className="chat-header">
          <button className="back-btn" onClick={handleBack}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5"/><path d="M12 19l-7-7 7-7"/></svg>
          </button>
          <div className="chat-user-info">
            <div className="chat-user-name">AI Digital Twin</div>
            <div className="chat-user-detail">Chat 20 turns to compare</div>
          </div>
          <div className="chat-header-right">
            <div className="progress-ring" title={`${turnCount}/20 turns`}>
              <svg viewBox="0 0 36 36">
                <path className="progress-bg" d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path className="progress-fill" strokeDasharray={`${twinProgress * 100}, 100`} d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831" />
              </svg>
              <span className="progress-text">{turnCount}</span>
            </div>
          </div>
        </div>

        {!isConnected && (
          <div className="connection-banner">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 1l22 22"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.56 9"/><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>
            {reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS
              ? 'Connection lost. Please refresh the page.'
              : `Reconnecting... (attempt ${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})`}
          </div>
        )}

        <div className="chat-layout">
          <div className="chat-main">
            <div className="messages">
              {messages.length === 0 && !isTyping && (
                <div className="empty-chat">
                  <h3>Chat with AI Digital Twin</h3>
                  <p>Start chatting. After 20 exchanges, we'll compare your friend's guess with system inference.</p>
                </div>
              )}
              {messages.map(msg => (
                <div key={msg.id} className={`msg msg-${msg.sender}`}>
                  <div className="msg-content">{msg.content}</div>
                </div>
              ))}
              {isTyping && (
                <div className="msg msg-bot">
                  <div className="msg-content typing">
                    <span /><span /><span />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <input
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Type something..."
                disabled={!isConnected}
              />
              <button onClick={handleSend} disabled={!isConnected || !inputText.trim()}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (page === 'comparison') {
    return (
      <ComparisonView
        systemInference={systemInference || {}}
        friendGuess={friendGuess || {}}
        onRestart={handleBack}
      />
    );
  }

  return null;
}

export default App;
