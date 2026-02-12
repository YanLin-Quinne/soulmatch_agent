import React, { useState, useEffect, useCallback } from 'react';
import CharacterCard, { BotInfo } from './components/CharacterCard';
import ChatWindow, { Message } from './components/ChatWindow';
import EmotionDisplay from './components/EmotionDisplay';
import WarningBanner from './components/WarningBanner';

// Bot emoji pool for random assignment
const BOT_EMOJIS = ['🤖', '👾', '🎭', '🎨', '🎵', '📚', '🌟', '💫', '🦋', '🌸'];

interface EmotionState {
  emotion: string;
  intensity: number;
  trend: 'improving' | 'declining' | 'stable';
}

interface WarningState {
  level: 'low' | 'medium' | 'high' | 'critical';
  message: string;
}

function App() {
  const [currentUser] = useState('user_001'); // Could be configurable
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  const [availableBots, setAvailableBots] = useState<BotInfo[]>([]);
  const [currentBot, setCurrentBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [warning, setWarning] = useState<WarningState | null>(null);

  // Initialize demo bots (in real app, fetch from API)
  useEffect(() => {
    const demoBots: BotInfo[] = [
      {
        bot_id: 'Emma',
        persona_summary: 'A creative artist who loves painting and exploring emotions through art. Empathetic and insightful.',
        compatibility_score: 0.92,
        emoji: BOT_EMOJIS[0],
      },
      {
        bot_id: 'Alex',
        persona_summary: 'Tech enthusiast and problem solver. Enjoys deep conversations about innovation and the future.',
        compatibility_score: 0.87,
        emoji: BOT_EMOJIS[1],
      },
      {
        bot_id: 'Luna',
        persona_summary: 'Nature lover and mindfulness coach. Calm, patient, and always ready to listen.',
        compatibility_score: 0.85,
        emoji: BOT_EMOJIS[2],
      },
      {
        bot_id: 'Max',
        persona_summary: 'Adventure seeker and storyteller. Brings energy and excitement to every conversation.',
        compatibility_score: 0.81,
        emoji: BOT_EMOJIS[3],
      },
    ];
    setAvailableBots(demoBots);
  }, []);

  // WebSocket connection
  const connectWebSocket = useCallback((userId: string) => {
    const websocket = new WebSocket(`ws://localhost:8000/ws/${userId}`);
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);

      switch (data.type) {
        case 'conversation_started':
          // Bot info received when conversation starts
          if (data.bot_info) {
            setCurrentBot((prev) => ({
              ...prev!,
              ...data.bot_info,
            }));
          }
          break;

        case 'bot_message':
          setIsTyping(false);
          if (data.message) {
            const botMessage: Message = {
              id: `bot-${Date.now()}`,
              sender: 'bot',
              content: data.message,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, botMessage]);
          }
          break;

        case 'emotion':
          setEmotion({
            emotion: data.emotion || 'neutral',
            intensity: data.intensity || 0.5,
            trend: data.trend || 'stable',
          });
          break;

        case 'warning':
          if (data.level && data.message) {
            setWarning({
              level: data.level,
              message: data.message,
            });
          }
          break;

        case 'feature_update':
          console.log('Feature update:', data);
          break;

        default:
          console.log('Unknown message type:', data.type);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    setWs(websocket);
    return websocket;
  }, []);

  // Connect on mount
  useEffect(() => {
    const websocket = connectWebSocket(currentUser);

    return () => {
      websocket.close();
    };
  }, [currentUser, connectWebSocket]);

  // Handle bot selection
  const handleBotSelect = (bot: BotInfo) => {
    setCurrentBot(bot);
    setMessages([]);
    setEmotion(null);
    setWarning(null);

    // Send start conversation message
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        action: 'start',
        bot_id: bot.bot_id,
      }));
    }
  };

  // Handle sending message
  const handleSendMessage = (content: string) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return;
    }

    // Add user message to UI
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      sender: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Send to server
    ws.send(JSON.stringify({
      action: 'message',
      content,
    }));

    // Show typing indicator
    setIsTyping(true);
  };

  return (
    <div className="app-container">
      {/* Sidebar with bot list */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>SoulMatch 💕</h1>
          <p>Choose your AI companion</p>
        </div>
        <div className="bot-list">
          {availableBots.map((bot) => (
            <CharacterCard
              key={bot.bot_id}
              bot={bot}
              isActive={currentBot?.bot_id === bot.bot_id}
              onClick={() => handleBotSelect(bot)}
            />
          ))}
        </div>
      </div>

      {/* Main chat area */}
      <div className="main-area">
        <div className="chat-header">
          <div className="chat-header-left">
            {currentBot ? (
              <>
                <span style={{ fontSize: '24px' }}>{currentBot.emoji}</span>
                <div>
                  <h2>{currentBot.bot_id}</h2>
                  <div className="status">
                    {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                  </div>
                </div>
              </>
            ) : (
              <h2>Select a companion to start chatting</h2>
            )}
          </div>
          {emotion && currentBot && (
            <EmotionDisplay
              emotion={emotion.emotion}
              intensity={emotion.intensity}
              trend={emotion.trend}
            />
          )}
        </div>

        {warning && (
          <WarningBanner
            level={warning.level}
            message={warning.message}
            onClose={() => setWarning(null)}
          />
        )}

        {currentBot ? (
          <ChatWindow
            messages={messages}
            onSendMessage={handleSendMessage}
            isTyping={isTyping}
            botEmoji={currentBot.emoji}
          />
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">💬</div>
            <h2>Welcome to SoulMatch</h2>
            <p>Select an AI companion from the sidebar to begin your conversation</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
