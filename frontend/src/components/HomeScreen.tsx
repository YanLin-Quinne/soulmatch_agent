import { PERSONAS, Persona } from '../data/personas';

interface HomeScreenProps {
  onSelectPersona: (persona: Persona) => void;
  onEnterDigitalTwin: () => void;
}

export default function HomeScreen({ onSelectPersona, onEnterDigitalTwin }: HomeScreenProps) {
  return (
    <div className="home-screen">
      <div className="home-header">
        <h1 className="home-title">SoulMatch Agent</h1>
        <p className="home-subtitle">选择一个人聊天30句，系统将推断你的性格画像</p>
      </div>

      <div className="persona-grid">
        {PERSONAS.map(p => (
          <div key={p.id} className="persona-card" onClick={() => onSelectPersona(p)}>
            <img className="persona-avatar" src={`https://api.dicebear.com/9.x/notionists/svg?seed=${p.name}`} alt={p.name} />
            <div className="persona-name">{p.name}</div>
            <div className="persona-hint">{p.occupation} · {p.age}</div>
            <div className="persona-tags">
              <span className="persona-tag age">{p.age}</span>
              {p.isAI && <span className="persona-tag bot">🤖 AI</span>}
            </div>
          </div>
        ))}
      </div>

      <div className="digital-twin-entry" onClick={onEnterDigitalTwin}>
        <div className="twin-entry-icon">🧬</div>
        <div className="twin-entry-content">
          <h3>AI分身模式</h3>
          <p>预判朋友性格 → 与AI分身聊天 → 对比准确度</p>
        </div>
        <div className="twin-entry-arrow">→</div>
      </div>

      <p className="home-note">💡 10 AI personas based on multi-agent architecture and Bayesian inference</p>
    </div>
  );
}
