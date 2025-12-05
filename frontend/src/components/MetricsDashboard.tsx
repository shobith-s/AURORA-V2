import { useEffect, useState } from 'react';
import { Activity, Zap, Brain, Target, TrendingUp, Database } from 'lucide-react';
import axios from 'axios';
import { LearnedRulesPanel } from './LearnedRulesPanel';

export default function MetricsDashboard() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 3000); // Refresh every 3s
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats');
      setStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="glass-card p-12 text-center">
          <div className="loading-dots flex gap-2 justify-center">
            <span className="w-3 h-3 bg-primary rounded-full animate-bounce"></span>
            <span className="w-3 h-3 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
            <span className="w-3 h-3 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
          </div>
          <p className="mt-4 text-brand-cool-gray">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="glass-card p-12 text-center">
          <p className="text-brand-cool-gray">No metrics available</p>
        </div>
      </div>
    );
  }

  const totalDecisions = stats.total_decisions || 0;
  const symbolicPct = totalDecisions > 0 ? ((stats.symbolic_decisions || 0) / totalDecisions * 100).toFixed(1) : 0;
  const neuralPct = totalDecisions > 0 ? ((stats.neural_decisions || 0) / totalDecisions * 100).toFixed(1) : 0;
  // Calculate fallback as remainder
  const metaPct = totalDecisions > 0 ? (100 - parseFloat(symbolicPct as string) - parseFloat(neuralPct as string)).toFixed(1) : 0;

  return (
    <div className="container mx-auto px-4 py-6 animate-in slide-in-from-top">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total Decisions */}
        <div className="stats-card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
              <Activity className="w-6 h-6 text-brand-white" />
            </div>
            <span className="text-xs text-brand-cool-gray uppercase tracking-wider">Total</span>
          </div>
          <div className="text-3xl font-bold text-brand-black mb-1">{totalDecisions.toLocaleString()}</div>
          <div className="text-sm text-brand-cool-gray">Decisions Made</div>
        </div>

        {/* Average Confidence */}
        <div className="stats-card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-gradient-to-br bg-success rounded-xl flex items-center justify-center">
              <Target className="w-6 h-6 text-brand-white" />
            </div>
            <span className="text-xs text-brand-cool-gray uppercase tracking-wider">Accuracy</span>
          </div>
          <div className="text-3xl font-bold text-brand-black mb-1">{stats.avg_confidence}%</div>
          <div className="text-sm text-brand-cool-gray">Avg Confidence</div>
        </div>

        {/* Active Rules */}
        <div className="stats-card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
              <Database className="w-6 h-6 text-brand-white" />
            </div>
            <span className="text-xs text-brand-cool-gray uppercase tracking-wider">Rules</span>
          </div>
          <div className="text-3xl font-bold text-brand-black mb-1">{stats.active_rules}</div>
          <div className="text-sm text-brand-cool-gray">Active Rules</div>
        </div>

        {/* Avg Response Time */}
        <div className="stats-card">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-warning rounded-xl flex items-center justify-center">
              <Zap className="w-6 h-6 text-brand-white" />
            </div>
            <span className="text-xs text-brand-cool-gray uppercase tracking-wider">Speed</span>
          </div>
          <div className="text-3xl font-bold text-brand-black mb-1">{stats.avg_time_ms?.toFixed(1) || 0}</div>
          <div className="text-sm text-brand-cool-gray">ms per decision</div>
        </div>
      </div>

      {/* Decision Sources Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Symbolic Engine */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-brand-black">Symbolic Engine</h3>
              <p className="text-xs text-brand-cool-gray">Rule-based decisions</p>
            </div>
          </div>
          <div className="flex items-end gap-2">
            <div className="text-4xl font-bold text-primary">{symbolicPct}%</div>
            <div className="text-sm text-brand-cool-gray mb-1">{stats.symbolic_decisions} decisions</div>
          </div>
          <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r bg-primary rounded-full transition-all duration-500"
              style={{ width: `${symbolicPct}%` }}
            ></div>
          </div>
        </div>

        {/* Fallback / Safety */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-brand-white0/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-brand-cool-gray" />
            </div>
            <div>
              <h3 className="font-semibold text-brand-black">Fallback / Safety</h3>
              <p className="text-xs text-brand-cool-gray">Conservative defaults</p>
            </div>
          </div>
          <div className="flex items-end gap-2">
            <div className="text-4xl font-bold text-brand-cool-gray">{metaPct}%</div>
            <div className="text-sm text-brand-cool-gray mb-1">decisions</div>
          </div>
          <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-border-dark rounded-full transition-all duration-500"
              style={{ width: `${metaPct}%` }}
            ></div>
          </div>
        </div>

        {/* Neural Oracle */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-pink-500/20 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-pink-400" />
            </div>
            <div>
              <h3 className="font-semibold text-brand-black">Neural Oracle</h3>
              <p className="text-xs text-brand-cool-gray">ML predictions</p>
            </div>
          </div>
          <div className="flex items-end gap-2">
            <div className="text-4xl font-bold text-pink-400">{neuralPct}%</div>
            <div className="text-sm text-brand-cool-gray mb-1">{stats.neural_decisions} decisions</div>
          </div>
          <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-error rounded-full transition-all duration-500"
              style={{ width: `${neuralPct}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Learned Rules Section */}
      <div className="mt-8">
        <LearnedRulesPanel />
      </div>
    </div>
  );
}
