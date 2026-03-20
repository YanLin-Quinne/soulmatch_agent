import { useState } from 'react';

interface FriendGuess {
  gender: string;
  age_range: string;
  mbti: string;
  occupation: string;
  EI: number;
  TF: number;
  description: string;
}

interface DigitalTwinSetupProps {
  onStartChat: (guess: FriendGuess) => void;
}

export default function DigitalTwinSetup({ onStartChat }: DigitalTwinSetupProps) {
  const [guess, setGuess] = useState<FriendGuess>({
    gender: 'Male',
    age_range: '25-30',
    mbti: '',
    occupation: '',
    EI: 50,
    TF: 50,
    description: ''
  });

  const handleSubmit = () => {
    onStartChat(guess);
  };

  return (
    <div className="clone-setup-screen">
      <div className="clone-setup-content">
        <h2 className="clone-title">🧬 Chat with Their AI Digital Twin</h2>
        <p className="clone-subtitle">
          Before chatting with the AI twin, fill in what you think this person is like.
          <br />
          After the chat, the system will compare your guess vs its inference to see how well you know them.
        </p>

        <div className="clone-form">
          <div className="form-row">
            <div className="form-group">
              <label>Their Gender (your guess)</label>
              <select
                value={guess.gender}
                onChange={e => setGuess({ ...guess, gender: e.target.value })}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Non-binary">Non-binary</option>
                <option value="Unsure">Unsure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Their Age Range (your guess)</label>
              <select
                value={guess.age_range}
                onChange={e => setGuess({ ...guess, age_range: e.target.value })}
              >
                <option value="18-24">18-24</option>
                <option value="25-30">25-30</option>
                <option value="31-40">31-40</option>
                <option value="41-50">41-50</option>
                <option value="51-60">51-60</option>
                <option value="60+">60+</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Their MBTI (your guess)</label>
              <input
                type="text"
                value={guess.mbti}
                onChange={e => setGuess({ ...guess, mbti: e.target.value.toUpperCase() })}
                placeholder="e.g. ENFP (leave blank if unsure)"
                maxLength={4}
              />
            </div>
            <div className="form-group">
              <label>Their Occupation (your guess)</label>
              <input
                type="text"
                value={guess.occupation}
                onChange={e => setGuess({ ...guess, occupation: e.target.value })}
                placeholder="e.g. Engineer, Student..."
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Extroverted ← → Introverted</label>
              <input
                type="range"
                min="0"
                max="100"
                value={guess.EI}
                onChange={e => setGuess({ ...guess, EI: +e.target.value })}
              />
              <span className="range-value">{guess.EI}%</span>
            </div>
            <div className="form-group">
              <label>Rational ← → Emotional</label>
              <input
                type="range"
                min="0"
                max="100"
                value={guess.TF}
                onChange={e => setGuess({ ...guess, TF: +e.target.value })}
              />
              <span className="range-value">{guess.TF}%</span>
            </div>
          </div>

          <div className="form-group">
            <label>Describe what you think they are like in a few sentences</label>
            <textarea
              value={guess.description}
              onChange={e => setGuess({ ...guess, description: e.target.value })}
              placeholder="e.g. I think they are introverted but insightful, enjoys deep thinking..."
              rows={4}
            />
          </div>

          <p className="form-note">
            💡 After submitting, you'll chat with the AI twin for 20 turns, then compare your guess with system inference
          </p>

          <button className="submit-btn" onClick={handleSubmit}>
            Start Chatting with AI Twin →
          </button>
        </div>
      </div>
    </div>
  );
}
