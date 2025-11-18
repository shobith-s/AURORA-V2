import { Activity, BarChart3, Sparkles } from 'lucide-react';

interface HeaderProps {
  onToggleMetrics: () => void;
}

export default function Header({ onToggleMetrics }: HeaderProps) {
  return (
    <header className="glass-card mx-4 mt-4 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold gradient-text">AURORA</h1>
            <p className="text-xs text-slate-600">Intelligent Data Preprocessing</p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-4">
          {/* Live Status */}
          <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-lg border border-green-200">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-green-700">System Online</span>
          </div>

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
