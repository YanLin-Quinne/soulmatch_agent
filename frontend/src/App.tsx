import React, { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// 15 个角色的展示数据
interface Character {
  id: string;
  name: string;
  emoji: string;
  job: string;
  city: string;
  age: number;
  status: string;
}

const CHARACTERS: Character[] = [
  { id: 'char_1', name: '王大力', emoji: '⚽', job: '健身教练', city: '长沙', age: 26, status: '来聊天' },
  { id: 'char_2', name: '张伟', emoji: '💼', job: '产品经理', city: '北京', age: 28, status: '闲逛中' },
  { id: 'char_3', name: '李思涵', emoji: '📚', job: '社会学研究生', city: '南京', age: 24, status: '来聊天' },
  { id: 'char_4', name: '刘建国', emoji: '🏛️', job: '某局副局长', city: '济南', age: 52, status: '闲逛中' },
  { id: 'char_5', name: 'Patricia Chen', emoji: '🌐', job: '外企亚太区VP', city: '上海', age: 45, status: '闲逛中' },
  { id: 'char_6', name: '赵磊', emoji: '🔧', job: '外卖骑手', city: '深圳', age: 35, status: '来聊天' },
  { id: 'char_7', name: '老周', emoji: '🍵', job: '中学数学老师', city: '武汉', age: 58, status: '来聊天' },
  { id: 'char_8', name: 'Helen Wu', emoji: '✈️', job: '退休(前外企CFO)', city: '环游世界中', age: 72, status: '随缘聊' },
  { id: 'char_9', name: '小K', emoji: '🎮', job: '高中生', city: '广州', age: 17, status: '在线中' },
  { id: 'char_10', name: '林小雨', emoji: '🌸', job: '大学生', city: '成都', age: 20, status: '随缘聊' },
  { id: 'char_11', name: '苏曼', emoji: '🧘', job: '瑜伽馆主/心理咨询师', city: '大理', age: 38, status: '随缘聊' },
  { id: 'char_12', name: '陈美琪', emoji: '🎨', job: '自由插画师', city: '杭州', age: 25, status: '闲逛中' },
  { id: 'char_13', name: 'Amy', emoji: '🚀', job: '跨境电商创业者', city: '义乌', age: 31, status: '闲逛中' },
  { id: 'char_14', name: '大卫', emoji: '🎸', job: '酒吧驻唱', city: '厦门', age: 42, status: '闲逛中' },
  { id: 'char_15', name: '王德明', emoji: '🎵', job: '退休干部', city: '西安', age: 67, status: '在线中' },
];

// 年龄筛选分组
interface AgeGroup {
  label: string;
  range: [number, number] | null;
}

const AGE_GROUPS: AgeGroup[] = [
  { label: '全部', range: null },
  { label: '10-20s', range: [10, 29] },
  { label: '30-40s', range: [30, 49] },
  { label: '50-60s', range: [50, 69] },
  { label: '70+', range: [70, 999] },
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

  // 页面状态
  const [page, setPage] = useState<'select' | 'chat'>('select');
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [ageFilter, setAgeFilter] = useState<string>('全部');

  // 聊天状态
  const [currentBot, setCurrentBot] = useState<BotInfo | null>(null);
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

  // 角色选择 - 切换到聊天页面
  const handleCharacterSelect = (character: Character) => {
    setSelectedCharacter(character);
    setPage('chat');
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setIsTyping(true);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'start' }));
    }
  };

  // 发送消息
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

  // 返回角色选择页面
  const handleBack = () => {
    setPage('select');
    setSelectedCharacter(null);
    setCurrentBot(null);
    setMessages([]);
    setEmotion(null);
    setWarning(null);
    setTurnCount(0);
    setInputText('');
  };

  // 过滤角色
  const filteredCharacters = CHARACTERS.filter(char => {
    const group = AGE_GROUPS.find(g => g.label === ageFilter);
    if (!group || !group.range) return true;
    return char.age >= group.range[0] && char.age <= group.range[1];
  });

  // 获取状态标签的 class
  const getStatusClass = (status: string): string => {
    if (status === '来聊天' || status === '在线中') return 'tag-status-chat';
    if (status === '闲逛中') return 'tag-status-idle';
    if (status === '随缘聊') return 'tag-status-random';
    return 'tag-status-chat';
  };

  return (
    <div className="app-container">
      {page === 'select' ? (
        // 角色选择页面
        <div className="select-page">
          <div className="page-header">
            <p className="page-description">
              选择一个人开始聊天。30 句对话后系统将推断对方的性格、心理、社会特征。注意——部分角色是 AI 伪装的。
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
                onClick={() => handleCharacterSelect(char)}
              >
                <div className="card-emoji">{char.emoji}</div>
                <div className="card-name">{char.name}</div>
                <div className="card-job">{char.job} · {char.city}</div>
                <div className="card-tags">
                  <span className="tag tag-age">{char.age}岁</span>
                  <span className="tag tag-city">{char.city}</span>
                  <span className={`tag ${getStatusClass(char.status)}`}>{char.status}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="page-footer">
            🎭 15人中有10个AI角色，你能分辨吗？
          </div>
        </div>
      ) : (
        // 聊天页面
        <div className="chat-page">
          <div className="chat-header">
            <button className="back-btn" onClick={handleBack}>
              ← 返回
            </button>
            {selectedCharacter && (
              <div className="chat-bot-info">
                <span className="chat-bot-emoji">{selectedCharacter.emoji}</span>
                <div>
                  <div className="chat-bot-name">{selectedCharacter.name}</div>
                  <div className="chat-bot-detail">{selectedCharacter.job} · {selectedCharacter.city}</div>
                </div>
              </div>
            )}
            <div className="turn-count">第 {turnCount} 轮</div>
            {emotion && (
              <div className="emotion-badge">
                {EMOTION_EMOJI[emotion.emotion] || '😐'} {emotion.emotion}
              </div>
            )}
          </div>

          {warning && (
            <div className={`warning-banner ${warning.level}`}>
              ⚠️ 诈骗警告 ({warning.level}): {warning.message}
              <span style={{ marginLeft: 8, fontSize: '0.8rem' }}>
                风险: {(warning.risk_score * 100).toFixed(0)}%
              </span>
            </div>
          )}

          <div className="message-list">
            {messages.length === 0 && !isTyping && (
              <div className="empty-state">
                <div style={{ fontSize: '3rem' }}>💬</div>
                <h2>开始对话</h2>
                <p>发送消息开始与 {selectedCharacter?.name} 聊天</p>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`message-bubble ${msg.sender}`}>
                {msg.content}
              </div>
            ))}
            {isTyping && (
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
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
              placeholder="输入消息..."
              disabled={!isConnected}
            />
            <button className="send-btn" onClick={handleSend} disabled={!isConnected}>
              发送
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
