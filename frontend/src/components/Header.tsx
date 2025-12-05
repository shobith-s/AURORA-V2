import { Activity, BarChart3, Sparkles, Code } from 'lucide-react';

interface HeaderProps {
  onToggleMetrics: () => void;
  onToggleIDE: () => void;
}

export default function Header({ onToggleMetrics, onToggleIDE }: HeaderProps) {
  return (
    <header className="glass-card mx-4 mt-4 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center shadow-lg">
            <Sparkles className="w-7 h-7 text-brand-black" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-primary">AURORA</h1>
            <p className="text-xs text-brand-cool-gray">Intelligent Data Preprocessing</p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-4">
          {/* Live Status */}
          <div className="flex items-center gap-2 px-4 py-2 bg-success/10 rounded-lg border border-success/30">
            <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-success">System Online</span>
          </div>

          {/* Custom Script IDE */}
          <button
            onClick={onToggleIDE}
            className="btn-secondary flex items-center gap-2 border-warning/30 hover:border-warning/60 text-warning hover:text-warning"
          >
            <Code className="w-4 h-4" />
            Custom Script
          </button>

          {/* Metrics Toggle */}
          <button
            onClick={onToggleMetrics}
            className="btn-secondary flex items-center gap-2"
          >
            <BarChart3 className="w-4 h-4" />
            Performance Metrics
          </button>
        </div>
      </div>
    </header>
  );
}
