interface ModeSelectorProps {
  onSelectMode: (mode: 'personality' | 'playground') => void;
}

export default function ModeSelector({ onSelectMode }: ModeSelectorProps) {
  return (
    <div className="mode-selector-screen">
      <div className="mode-selector-content">
        <h1 className="mode-title">SoulMatch Agent</h1>
        <p className="mode-subtitle">选择你的体验模式</p>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('personality')}>
            <div className="mode-icon">🎭</div>
            <h3>性格推断模式</h3>
            <p>选择一个人聊天30句，系统将推断你的性格画像</p>
            <div className="mode-features">
              <span>✓ 15个角色可选</span>
              <span>✓ 10个AI + 5个真人</span>
              <span>✓ Big Five性格分析</span>
            </div>
          </div>

          <div className="mode-card" onClick={() => onSelectMode('playground')}>
            <div className="mode-icon">🧬</div>
            <h3>AI分身模式</h3>
            <p>先预判朋友的性格，再与TA的AI分身聊天，最后对比准确度</p>
            <div className="mode-features">
              <span>✓ 预判朋友特征</span>
              <span>✓ 与AI分身对话</span>
              <span>✓ 匹配度分析</span>
            </div>
          </div>
        </div>

        <p className="mode-note">💡 两种模式都基于多智能体架构和贝叶斯推理</p>
      </div>
    </div>
  );
}
