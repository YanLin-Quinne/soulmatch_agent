import React from 'react';
import { PERSONAS, Persona } from '../data/personas';

interface PersonaGridProps {
  onSelectPersona: (persona: Persona) => void;
}

export const PersonaGrid: React.FC<PersonaGridProps> = ({ onSelectPersona }) => {
  return (
    <div className="persona-grid">
      <h2>Choose a persona to start chatting</h2>
      <div className="grid">
        {PERSONAS.map((persona) => (
          <div
            key={persona.id}
            className="persona-card"
            onClick={() => onSelectPersona(persona)}
          >
            <div className="avatar">{persona.avatar}</div>
            <h3>{persona.name}</h3>
            <p className="age-gender">{persona.age} · {persona.gender}</p>
            <p className="bio">{persona.bio}</p>
            <div className="tags">
              {persona.personality.slice(0, 2).map((trait, i) => (
                <span key={i} className="tag">{trait}</span>
              ))}
            </div>
            <div className="badge">{persona.isAI ? '🤖 AI' : '👤 Human'}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
