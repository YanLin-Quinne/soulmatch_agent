import { useState, useEffect, useCallback, useRef } from 'react';
import {
  BarChart3, Heart, Brain, Users, Shield, Hexagon,
  Send, ArrowLeft, MapPin,
} from 'lucide-react';
import TopBar from './layouts/TopBar';
import ConfidencePanel from './panels/ConfidencePanel';
import TraitMatrix from './panels/TraitMatrix';
import EmotionPanel from './panels/EmotionPanel';
import MemoryPanel from './panels/MemoryPanel';
import RelationPanel from './panels/RelationPanel';
import ConformalPanel from './panels/ConformalPanel';
import { CHARACTERS, AGE_FILTERS, EMOTION_EMOJI, STATUS_COLORS } from './constants';
import type {
  Character, BotInfo, Message, EmotionState, WarningState,
  FeatureData, MemoryItem, EmotionEntry, RelationshipData,
  MilestoneReport, TrustPoint, MemoryStats, ContextData,
} from './types';

const WS_BASE = window.location.protocol === 'https:'
  ? `wss://${window.location.host}`
  : 'ws://localhost:8000';

type NavTab = 'confidence' | 'traits' | 'emotion' | 'memory' | 'relation' | 'conformal';

const NAV_ITEMS: { id: NavTab; icon: typeof BarChart3; label: string }[] = [
  { id: 'confidence', icon: BarChart3, label: 'CONF' },
  { id: 'traits', icon: Hexagon, label: 'TRAIT' },
  { id: 'emotion', icon: Heart, label: 'EMO' },
  { id: 'memory', icon: Brain, label: 'MEM' },
  { id: 'relation', icon: Users, label: 'REL' },
  { id: 'conformal', icon: Shield, label: 'CP' },
];

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function App() {
  const userId = useRef(`user_${Date.now()}`);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const [page, setPage] = useState<'select' | 'chat'>('select');
  const [selectedChar, setSelectedChar] = useState<Character | null>(null);
  const [ageFilter, setAgeFilter] = useState<string>('All');

  const [, setCurrentBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [warning, setWarning] = useState<WarningState | null>(null);
  const [turnCount, setTurnCount] = useState(0);

  const [activeTab, setActiveTab] = useState<NavTab>('confidence');
  const [featureData, setFeatureData] = useState<FeatureData | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionEntry[]>([]);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [contextData, setContextData] = useState<ContextData | null>(null);
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>([]);

  const [relationshipData, setRelationshipData] = useState<RelationshipData | null>(null);
  const [milestoneReport, setMilestoneReport] = useState<MilestoneReport | null>(null);
  const [trustHistory, setTrustHistory] = useState<TrustPoint[]>([]);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isUnmountedRef = useRef(false);
  const selectedCharRef = useRef<Character | null>(null);
  const reconnectAttemptRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 10;
  const BASE_RECONNECT_DELAY = 2000;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const connectWebSocket = useCallback(() => {
    const websocket = new WebSocket(`${WS_BASE}/ws/${userId.current}`);

    websocket.onopen = () => {
      setIsConnected(true);
      reconnectAttemptRef.current = 0;
      if (selectedCharRef.current) {
        websocket.send(JSON.stringify({ action: 'start', bot_id: selectedCharRef.current.id }));
      }
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
        case 'conversation_started':
          setIsTyping(false);
          if (data.data) {
            const d = data.data;
            if (d.bot_profile) setCurrentBot(prev => ({ ...prev!, ...d.bot_profile }));
            if (d.greeting) {
              setMessages(prev => {
                if (prev.length === 0) {
                  const msgs: Message[] = [{ id: `bot-${Date.now()}`, sender: 'bot', content: d.greeting, timestamp: new Date() }];
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
            setMessages(prev => [...prev, { id: `bot-${Date.now()}`, sender: 'bot', content: data.message, timestamp: new Date() }]);
          }
          if (data.turn) setTurnCount(data.turn);
          if (data.memory_update) setMemories(data.memory_update);
          break;
        case 'emotion':
          if (data.data?.current_emotion) {
            const emo = data.data.current_emotion;
            setEmotion(emo);
            setEmotionHistory(prev => [...prev, { turn: turnCount + 1, emotion: emo.emotion, intensity: emo.intensity }]);
          }
          break;
        case 'warning':
          if (data.data) {
            setWarning({ level: data.data.level, message: data.data.message, risk_score: data.data.risk_score });
            setTimeout(() => setWarning(null), 10000);
          }
          break;
        case 'feature_update':
          if (data.data) {
            setFeatureData(data.data);
            if (data.data.average_confidence !== undefined) {
              setConfidenceHistory(prev => [...prev, data.data.average_confidence]);
            }
          }
          break;
        case 'relationship_prediction':
          if (data.data) {
            setRelationshipData(data.data);
            const trust = data.data.trust_score ?? 0.5;
            setTrustHistory(prev => [...prev, { turn: data.data.turn ?? 0, trust }]);
          }
          break;
        case 'milestone_report':
          if (data.data) {
            setMilestoneReport(data.data);
            setActiveTab('relation');
          }
          break;
        case 'context':
          if (data.data) {
            setContextData(data.data);
            if (data.data.turn_count) setTurnCount(data.data.turn_count);
          }
          break;
        case 'memory_stats':
          if (data.data) setMemoryStats(data.data);
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
        reconnectTimerRef.current = setTimeout(() => connectWebSocket(), delay);
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

  const resetState = () => {
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setFeatureData(null);
    setEmotionHistory([]);
    setMemories([]);
    setContextData(null);
    setConfidenceHistory([]);
    setActiveTab('confidence');
    setRelationshipData(null);
    setMilestoneReport(null);
    setTrustHistory([]);
    setMemoryStats(null);
  };

  const handleCardClick = (char: Character) => {
    setSelectedChar(char);
    selectedCharRef.current = char;
    setPage('chat');
    resetState();
    setIsTyping(true);
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
    setMessages(prev => [...prev, { id: `user-${Date.now()}`, sender: 'user', content, timestamp: new Date() }]);
    setInputText('');
    setIsTyping(true);
    ws.send(JSON.stringify({ action: 'message', content }));
  };

  const handleBack = () => {
    setPage('select');
    setSelectedChar(null);
    selectedCharRef.current = null;
    setCurrentBot(null);
    setInputText('');
    resetState();
  };

  const filteredCharacters = CHARACTERS.filter(char => {
    if (ageFilter === 'All') return true;
    if (ageFilter === '20s') return char.age >= 20 && char.age <= 29;
    if (ageFilter === '30s') return char.age >= 30 && char.age <= 39;
    if (ageFilter === '40+') return char.age >= 40;
    return true;
  });

  // ===== SELECT PAGE =====
  if (page === 'select') {
    return (
      <div className="min-h-screen bg-cyber-bg p-8 max-w-5xl mx-auto">
        {/* Hero */}
        <header className="text-center mb-10">
          <div className="inline-block px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-cyber-cyan border border-cyber-cyan/20 bg-cyber-cyan/5 mb-4">
            AI-POWERED MATCHING SYSTEM
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyber-cyan to-cyber-fuchsia bg-clip-text text-transparent mb-3">
            SOULMATCH_OS
          </h1>
          <p className="text-cyber-muted text-sm max-w-lg mx-auto leading-relaxed">
            Select a node to initiate connection. After 30 exchanges, the system predicts personality traits,
            emotional patterns, and social tendencies.
          </p>
        </header>

        {/* Filters */}
        <div className="flex justify-center gap-2 mb-8">
          {AGE_FILTERS.map(f => (
            <button
              key={f}
              className={`px-4 py-1.5 text-xs border transition-all ${
                ageFilter === f
                  ? 'border-cyber-cyan text-cyber-cyan bg-cyber-cyan/10'
                  : 'border-cyber-border text-cyber-muted hover:border-cyber-dim'
              }`}
              onClick={() => setAgeFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>

        {/* Grid */}
        <div className="grid grid-cols-4 gap-4 max-lg:grid-cols-3 max-md:grid-cols-2 max-sm:grid-cols-1">
          {filteredCharacters.map((char, i) => (
            <div
              key={char.id}
              className="relative bg-cyber-panel border border-cyber-border p-5 cursor-pointer transition-all hover:border-cyber-cyan/50 hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(0,240,255,0.1)] animate-fade-in group"
              style={{ animationDelay: `${i * 60}ms` }}
              onClick={() => handleCardClick(char)}
            >
              <div className="cyber-corner absolute inset-0 pointer-events-none before:border-cyber-cyan/30 after:border-cyber-cyan/30 opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="flex justify-center mb-3 relative">
                <img className="w-16 h-16 rounded-full bg-cyber-border" src={char.avatar} alt={char.name} />
                <span
                  className="absolute bottom-0 right-[calc(50%-36px)] w-3 h-3 rounded-full border-2 border-cyber-panel"
                  style={{ background: STATUS_COLORS[char.status] }}
                />
              </div>
              <h3 className="text-center text-sm font-semibold text-cyber-text mb-0.5">
                {char.name}, {char.age}
              </h3>
              <p className="text-center text-[11px] text-cyber-muted mb-1">{char.job}</p>
              <p className="text-center text-[10px] text-cyber-dim flex items-center justify-center gap-1 mb-2">
                <MapPin size={10} />
                {char.city}
              </p>
              <p className="text-center text-[11px] text-cyber-muted italic mb-3">{char.bio}</p>
              <div className="flex justify-center flex-wrap gap-1">
                {char.interests.map(tag => (
                  <span key={tag} className="px-2 py-0.5 text-[10px] bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/20">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <footer className="flex justify-center items-center gap-3 mt-10 text-[11px] text-cyber-dim">
          <span>8 NODES</span>
          <span className="w-1 h-1 rounded-full bg-cyber-dim" />
          <span>SOME ARE AI</span>
          <span className="w-1 h-1 rounded-full bg-cyber-dim" />
          <span>30 TURNS TO PREDICT</span>
        </footer>
      </div>
    );
  }

  // ===== CHAT PAGE =====
  if (!selectedChar) return null;

  const renderRightPanel = () => {
    switch (activeTab) {
      case 'confidence':
        return <ConfidencePanel featureData={featureData} contextData={contextData} confidenceHistory={confidenceHistory} />;
      case 'traits':
        return <TraitMatrix featureData={featureData} />;
      case 'emotion':
        return <EmotionPanel emotion={emotion} emotionHistory={emotionHistory} />;
      case 'memory':
        return <MemoryPanel memoryStats={memoryStats} memories={memories} />;
      case 'relation':
        return <RelationPanel relationshipData={relationshipData} milestoneReport={milestoneReport} trustHistory={trustHistory} turnCount={turnCount} />;
      case 'conformal':
        return <ConformalPanel featureData={featureData} />;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-cyber-bg overflow-hidden">
      {/* Top Bar */}
      <TopBar
        isConnected={isConnected}
        turnCount={turnCount}
        featureData={featureData}
        emotion={emotion}
        warning={warning}
        reconnectAttempt={reconnectAttemptRef.current}
        maxReconnect={MAX_RECONNECT_ATTEMPTS}
      />

      {/* Warning bar */}
      {warning && (
        <div className={`px-3 py-1.5 text-[11px] flex items-center gap-2 border-b ${
          warning.level === 'critical' ? 'bg-cyber-red/10 text-cyber-red border-cyber-red/20' :
          warning.level === 'high' ? 'bg-orange-500/10 text-orange-400 border-orange-500/20' :
          warning.level === 'medium' ? 'bg-cyber-amber/10 text-cyber-amber border-cyber-amber/20' :
          'bg-cyber-dim/10 text-cyber-muted border-cyber-border'
        }`}>
          <span className="animate-glow-pulse">⚠</span>
          [{warning.level.toUpperCase()}] {warning.message}
        </div>
      )}

      {/* Main layout */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Icon Nav */}
        <nav className="w-12 bg-cyber-surface border-r border-cyber-border flex flex-col items-center py-2 gap-1 shrink-0">
          <button
            onClick={handleBack}
            className="w-8 h-8 flex items-center justify-center text-cyber-muted hover:text-cyber-cyan hover:bg-cyber-cyan/10 transition-all mb-2"
            title="Back"
          >
            <ArrowLeft size={14} />
          </button>
          <div className="w-6 h-px bg-cyber-border mb-2" />
          {NAV_ITEMS.map(item => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-8 h-8 flex flex-col items-center justify-center gap-0.5 transition-all ${
                  isActive
                    ? 'text-cyber-cyan bg-cyber-cyan/10'
                    : 'text-cyber-muted hover:text-cyber-text hover:bg-cyber-panel'
                }`}
                title={item.label}
              >
                <Icon size={12} />
                <span className="text-[7px] leading-none">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Center: Terminal Chat */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Chat header */}
          <div className="h-9 bg-cyber-surface border-b border-cyber-border flex items-center px-3 gap-2 shrink-0">
            <img className="w-5 h-5 rounded-full bg-cyber-border" src={selectedChar.avatar} alt="" />
            <span className="text-xs text-cyber-text font-semibold">{selectedChar.name}</span>
            <span className="text-[10px] text-cyber-dim">{selectedChar.job} · {selectedChar.city}</span>
            {emotion && (
              <span className="ml-auto text-[10px] px-1.5 py-0.5 bg-cyber-fuchsia/10 text-cyber-fuchsia border border-cyber-fuchsia/20">
                {EMOTION_EMOJI[emotion.emotion] || '😐'} {emotion.emotion}
              </span>
            )}
          </div>

          {/* Terminal messages */}
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
            {messages.length === 0 && !isTyping && (
              <div className="text-center py-16">
                <div className="text-cyber-dim text-[11px] mb-2">// TERMINAL READY</div>
                <div className="text-cyber-muted text-[11px]">
                  Initiate conversation with <span className="text-cyber-cyan">{selectedChar.name}</span>
                </div>
                <div className="text-cyber-dim text-[10px] mt-1">30 exchanges → personality prediction</div>
              </div>
            )}
            {messages.map(msg => {
              const time = formatTime(msg.timestamp);
              if (msg.sender === 'user') {
                return (
                  <div key={msg.id} className="animate-slide-up mb-1">
                    <span className="text-cyber-dim">[{time}]</span>{' '}
                    <span className="text-cyber-cyan">USER&gt;</span>{' '}
                    <span className="text-cyber-text">{msg.content}</span>
                  </div>
                );
              }
              if (msg.sender === 'bot') {
                return (
                  <div key={msg.id} className="animate-slide-up mb-1">
                    <span className="text-cyber-dim">[{time}]</span>{' '}
                    <span className="text-cyber-fuchsia">{selectedChar.name.toUpperCase()}&gt;</span>{' '}
                    <span className="text-cyber-text">{msg.content}</span>
                  </div>
                );
              }
              // system
              return (
                <div key={msg.id} className="animate-slide-up mb-1">
                  <span className="text-cyber-dim">[{time}]</span>{' '}
                  <span className="text-cyber-amber">SYS&gt;</span>{' '}
                  <span className="text-cyber-muted italic">{msg.content}</span>
                </div>
              );
            })}
            {isTyping && (
              <div className="mb-1 animate-glow-pulse">
                <span className="text-cyber-dim">[{formatTime(new Date())}]</span>{' '}
                <span className="text-cyber-fuchsia">{selectedChar.name.toUpperCase()}&gt;</span>{' '}
                <span className="text-cyber-dim">processing</span>
                <span className="text-cyber-cyan animate-blink">_</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-cyber-border bg-cyber-surface flex items-center px-3 py-2 gap-2 shrink-0">
            <span className="text-cyber-cyan text-xs shrink-0">&gt;</span>
            <input
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder={isConnected ? 'Type command...' : 'DISCONNECTED'}
              disabled={!isConnected}
              className="flex-1 bg-transparent text-xs text-cyber-text placeholder:text-cyber-dim outline-none font-mono"
            />
            <button
              onClick={handleSend}
              disabled={!isConnected || !inputText.trim()}
              className="w-7 h-7 flex items-center justify-center text-cyber-cyan hover:bg-cyber-cyan/10 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              <Send size={12} />
            </button>
          </div>
        </div>

        {/* Right Panel */}
        <aside className="w-72 bg-cyber-surface border-l border-cyber-border overflow-y-auto shrink-0 p-2 space-y-2 max-xl:hidden">
          {renderRightPanel()}
        </aside>
      </div>
    </div>
  );
}

export default App;
