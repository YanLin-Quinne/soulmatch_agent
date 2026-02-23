interface CyberPanelProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  accent?: 'cyan' | 'fuchsia';
}

export default function CyberPanel({ title, children, className = '', accent = 'cyan' }: CyberPanelProps) {
  const borderColor = accent === 'cyan' ? 'border-cyber-cyan/20' : 'border-cyber-fuchsia/20';
  const titleColor = accent === 'cyan' ? 'text-cyber-cyan' : 'text-cyber-fuchsia';
  const glowClass = accent === 'cyan' ? 'text-glow-cyan' : 'text-glow-fuchsia';
  const cornerColor = accent === 'cyan' ? 'before:border-cyber-cyan/40 after:border-cyber-cyan/40' : 'before:border-cyber-fuchsia/40 after:border-cyber-fuchsia/40';

  return (
    <div className={`relative bg-cyber-panel border ${borderColor} p-3 ${className}`}>
      {/* Corner decorations */}
      <div className={`cyber-corner absolute inset-0 pointer-events-none ${cornerColor}`} />
      {/* Title bar */}
      <div className={`text-[10px] uppercase tracking-widest ${titleColor} ${glowClass} mb-2 flex items-center gap-2`}>
        <span className="opacity-50">[</span>
        {title}
        <span className="opacity-50">]</span>
        <div className="flex-1 h-px bg-gradient-to-r from-current to-transparent opacity-20" />
      </div>
      {/* Content */}
      <div className="relative">
        {children}
      </div>
    </div>
  );
}
