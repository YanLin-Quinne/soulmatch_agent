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
    gender: '男',
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
        <h2 className="clone-title">🧬 与TA的AI分身聊天</h2>
        <p className="clone-subtitle">
          在和AI分身聊天之前，先填写你认为这个人是什么样的。
          <br />
          聊完后系统会对比你的预判 vs 系统推断，看你有多了解TA。
        </p>

        <div className="clone-form">
          <div className="form-row">
            <div className="form-group">
              <label>你觉得TA的性别</label>
              <select
                value={guess.gender}
                onChange={e => setGuess({ ...guess, gender: e.target.value })}
              >
                <option value="男">男</option>
                <option value="女">女</option>
                <option value="非二元">非二元</option>
                <option value="不确定">不确定</option>
              </select>
            </div>
            <div className="form-group">
              <label>你觉得TA的年龄段</label>
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
              <label>你觉得TA的MBTI</label>
              <input
                type="text"
                value={guess.mbti}
                onChange={e => setGuess({ ...guess, mbti: e.target.value.toUpperCase() })}
                placeholder="如 ENFP（不确定可留空）"
                maxLength={4}
              />
            </div>
            <div className="form-group">
              <label>你觉得TA的职业</label>
              <input
                type="text"
                value={guess.occupation}
                onChange={e => setGuess({ ...guess, occupation: e.target.value })}
                placeholder="如 程序员、学生..."
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>外向 ← → 内向</label>
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
              <label>理性 ← → 感性</label>
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
            <label>用几句话描述你觉得TA是什么样的人</label>
            <textarea
              value={guess.description}
              onChange={e => setGuess({ ...guess, description: e.target.value })}
              placeholder="比如：我觉得TA是一个比较内向但很有想法的人，喜欢深度思考..."
              rows={4}
            />
          </div>

          <p className="form-note">
            💡 填完后你将与AI分身聊天20句，最后对比你的预判和系统推断
          </p>

          <button className="submit-btn" onClick={handleSubmit}>
            开始与AI分身聊天 →
          </button>
        </div>
      </div>
    </div>
  );
}
