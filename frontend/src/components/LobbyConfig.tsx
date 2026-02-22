import { useState } from 'react';

interface Bot {
  id: string;
  name: string;
  avatar: string;
  age: number;
  city: string;
  job: string;
  interests: string[];
}

const ALL_BOTS: Bot[] = [
  { id: 'bot_0', name: 'Mina', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Mina', age: 29, city: 'San Francisco', job: 'Policy Analyst', interests: ['Food', 'Travel', 'Books'] },
  { id: 'bot_1', name: 'Jade', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Jade', age: 25, city: 'San Francisco', job: 'Office Admin', interests: ['Music', 'Food', 'Arts'] },
  { id: 'bot_2', name: 'Sierra', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Sierra', age: 34, city: 'Oakland', job: 'Math Teacher', interests: ['Travel', 'Outdoors', 'Music'] },
  { id: 'bot_3', name: 'Kevin', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Kevin', age: 33, city: 'San Francisco', job: 'Software Dev', interests: ['Music', 'Tech', 'Outdoors'] },
  { id: 'bot_4', name: 'Marcus', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Marcus', age: 33, city: 'Castro Valley', job: 'VP Operations', interests: ['Music', 'Food', 'Sports'] },
  { id: 'bot_5', name: 'Derek', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Derek', age: 39, city: 'Oakland', job: 'Financial Analyst', interests: ['Food', 'Outdoors', 'Sports'] },
  { id: 'bot_6', name: 'Luna', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Luna', age: 26, city: 'Hayward', job: 'Freelancer', interests: ['Food', 'Travel', 'Music'] },
  { id: 'bot_7', name: 'Travis', avatar: 'https://api.dicebear.com/9.x/notionists/svg?seed=Travis', age: 42, city: 'San Francisco', job: 'Sales Director', interests: ['Travel', 'Food', 'Outdoors'] },
];

interface LobbyConfigProps {
  onStart: (selectedBot: Bot, includeHuman: boolean) => void;
}

function BotCard({ bot, selected, onClick }: { bot: Bot; selected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-xl border transition-all text-left ${
        selected
          ? 'border-pink-500 bg-pink-900/20 ring-1 ring-pink-500'
          : 'border-gray-700 bg-gray-800/50 hover:border-gray-500'
      }`}
    >
      <div className="flex items-center gap-3">
        <img
          src={bot.avatar}
          alt={bot.name}
          className={`w-12 h-12 rounded-full border-2 ${selected ? 'border-pink-400' : 'border-gray-600'}`}
        />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-white">{bot.name}</div>
          <div className="text-xs text-gray-400">{bot.age} · {bot.job}</div>
          <div className="text-xs text-gray-500 truncate">{bot.city}</div>
        </div>
        {selected && <span className="text-pink-400 text-sm">✓</span>}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {bot.interests.map((interest) => (
          <span
            key={interest}
            className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-700 text-gray-400"
          >
            {interest}
          </span>
        ))}
      </div>
    </button>
  );
}

export default function LobbyConfig({ onStart }: LobbyConfigProps) {
  const [selectedBot, setSelectedBot] = useState<Bot | null>(null);
  const [includeHuman, setIncludeHuman] = useState(false);

  const handleStart = () => {
    if (!selectedBot) return;
    onStart(selectedBot, includeHuman);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            Soul<span className="text-pink-400">Match</span>
          </h1>
          <p className="text-gray-400 text-sm">
            Multi-agent relationship prediction · v2.0
          </p>
        </div>

        {/* Session Config */}
        <div className="bg-gray-800 rounded-2xl p-6 mb-6">
          <h2 className="text-sm font-medium text-gray-300 mb-4 uppercase tracking-wide">
            Session Configuration
          </h2>

          {/* Human participant toggle */}
          <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-xl mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full border-2 border-dashed border-blue-400 bg-blue-900/20 flex items-center justify-center text-lg">
                👤
              </div>
              <div>
                <div className="text-sm text-white">You (real person)</div>
                <div className="text-xs text-gray-400">
                  {includeHuman
                    ? 'Profile inferred from conversation'
                    : 'Click to include yourself'}
                </div>
              </div>
            </div>
            <button
              onClick={() => setIncludeHuman((v) => !v)}
              className={`w-12 h-6 rounded-full transition-all relative ${
                includeHuman ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-all ${
                  includeHuman ? 'left-6' : 'left-0.5'
                }`}
              />
            </button>
          </div>

          {includeHuman && (
            <div className="p-3 bg-blue-900/20 border border-blue-800 rounded-lg mb-4 text-xs text-blue-300">
              Your profile will be inferred from conversation — no forms to fill out.
              Relationship analysis begins at turn 5.
            </div>
          )}
        </div>

        {/* Bot Selection */}
        <div className="bg-gray-800 rounded-2xl p-6 mb-6">
          <h2 className="text-sm font-medium text-gray-300 mb-4 uppercase tracking-wide">
            Select Your Match
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {ALL_BOTS.map((bot) => (
              <BotCard
                key={bot.id}
                bot={bot}
                selected={selectedBot?.id === bot.id}
                onClick={() => setSelectedBot(bot)}
              />
            ))}
          </div>
        </div>

        {/* Start Button */}
        <button
          onClick={handleStart}
          disabled={!selectedBot}
          className={`w-full py-4 rounded-xl text-white font-medium text-lg transition-all ${
            selectedBot
              ? 'bg-gradient-to-r from-pink-500 to-purple-600 hover:opacity-90 shadow-lg shadow-pink-500/20'
              : 'bg-gray-700 opacity-50 cursor-not-allowed'
          }`}
        >
          {selectedBot
            ? `Start with ${selectedBot.name} →`
            : 'Select a match to continue'}
        </button>

        <p className="text-center text-xs text-gray-600 mt-4">
          Multi-agent system · Conformal prediction · 42-dim features
        </p>
      </div>
    </div>
  );
}
