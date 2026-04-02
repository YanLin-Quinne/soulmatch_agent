import { useState } from 'react';

interface Persona {
  id: number;
  name: string;
  emoji: string;
  isBot: boolean;
  profile: {
    age: number;
    gender: string;
    occupation: string;
    location: string;
    mbti?: string;
  };
  tags: string[];
}

interface PersonalityModeProps {
  personas: Persona[];
  onSelectPersona: (persona: Persona) => void;
}

const AGE_FILTERS = [
  { label: 'All', value: 'all' },
  { label: '10-20s', value: 'young' },
  { label: '30-40s', value: 'mid' },
  { label: '50-60s', value: 'senior' },
  { label: '70+', value: 'elder' }
];

export default function PersonalityMode({ personas, onSelectPersona }: PersonalityModeProps) {
  const [ageFilter, setAgeFilter] = useState('all');

  const filteredPersonas = personas.filter(p => {
    if (ageFilter === 'all') return true;
    const age = p.profile.age;
    if (ageFilter === 'young') return age < 30;
    if (ageFilter === 'mid') return age >= 30 && age < 50;
    if (ageFilter === 'senior') return age >= 50 && age < 70;
    if (ageFilter === 'elder') return age >= 70;
    return true;
  });

  return (
    <div className="personality-mode">
      <div className="hero-section">
        <h1 className="hero-title">AI YOU</h1>
        <p className="hero-subtitle">Choose someone to chat with. After 30 turns, the system will infer your personality profile</p>
      </div>

      <div className="filter-bar">
        {AGE_FILTERS.map(f => (
          <button
            key={f.value}
            className={`filter-btn ${ageFilter === f.value ? 'active' : ''}`}
            onClick={() => setAgeFilter(f.value)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="persona-grid">
        {filteredPersonas.map(p => (
          <div
            key={p.id}
            className="persona-card"
            onClick={() => onSelectPersona(p)}
          >
            <div className="persona-avatar">{p.emoji}</div>
            <div className="persona-name">{p.name}</div>
            <div className="persona-hint">
              {p.profile.occupation} · {p.profile.location}
            </div>
            <div className="persona-tags">
              <span className="persona-tag age">{p.profile.age} yrs</span>
              <span className="persona-tag loc">{p.profile.location}</span>
              {p.isBot && <span className="persona-tag bot">🤖 AI</span>}
            </div>
          </div>
        ))}
      </div>

      <p className="home-note">🎭 10 out of 15 are AI personas — can you tell which?</p>
    </div>
  );
}
