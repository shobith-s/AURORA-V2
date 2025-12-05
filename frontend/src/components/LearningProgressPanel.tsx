/**
 * Learning Progress Panel - Shows V3 adaptive learning progress
 * Displays training/production phases and rule creation status
 */

import { useState, useEffect } from 'react';
import { BookOpen, Target, TrendingUp, CheckCircle, Clock, Zap } from 'lucide-react';
import axios from 'axios';

interface LearningProgress {
  approach: string;
  total_corrections: number;
  patterns_tracked: number;
  pattern_corrections: number;
  corrections_needed_for_training: number;
  corrections_needed_for_production: number;
  production_ready: boolean;
  rule_created: boolean;
  rule_name?: string;
  total_symbolic_rules?: number;
  message?: string;
}

export default function LearningProgressPanel() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats'); // Use relative path
      setStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-background-muted rounded w-1/2 mb-4"></div>
          <div className="h-32 bg-background-muted rounded"></div>
        </div>
      </div>
    );
  }

  const adaptiveLearning = stats?.adaptive_learning;
  if (!adaptiveLearning) {
    return null;
  }

  const totalCorrections = adaptiveLearning.total_corrections || 0;
  const patternsTracked = adaptiveLearning.patterns_tracked || 0;
  const activeAdjustments = adaptiveLearning.active_adjustments || 0;

  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
          <Target className="w-5 h-5 text-brand-white" />
        </div>
        <div>
          <h3 className="text-lg font-bold text-brand-black">Adaptive Learning (V3)</h3>
          <p className="text-xs text-foreground-muted">Creates symbolic rules from your corrections</p>
        </div>
      </div>

      {/* Progress Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-primary/10 rounded-lg border border-primary/50">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium text-primary-dark">Total Corrections</span>
          </div>
          <div className="text-2xl font-bold text-blue-900">{totalCorrections}</div>
        </div>

        <div className="p-4 bg-primary/10 rounded-lg border border-primary/50">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium text-primary-dark">Patterns Tracked</span>
          </div>
          <div className="text-2xl font-bold text-purple-900">{patternsTracked}</div>
        </div>

        <div className="p-4 bg-gradient-to-br bg-success/10 rounded-lg border border-success/30">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-green-600" />
            <span className="text-xs font-medium text-success">Rules Created</span>
          </div>
          <div className="text-2xl font-bold text-green-900">{activeAdjustments}</div>
        </div>
      </div>

      {/* Adjustments Detail */}
      {adaptiveLearning.adjustments && Object.keys(adaptiveLearning.adjustments).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Active Learned Rules</h4>
          <div className="space-y-2">
            {Object.entries(adaptiveLearning.adjustments).map(([pattern, adj]: [string, any]) => (
              <div key={pattern} className="p-3 bg-brand-white rounded-lg border border-brand-warm-gray">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-brand-black">
                    {pattern.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <span className="px-2 py-1 bg-green-100 text-success text-xs rounded-full font-medium">
                    {adj.corrections} corrections
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs text-foreground-muted">
                  <CheckCircle className="w-3 h-3 text-green-600" />
                  <span>Action: {adj.action}</span>
                  <span className="text-green-600 font-medium">{adj.confidence_delta}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Phase Explanation */}
      <div className="mt-6 p-4 bg-brand-white rounded-lg border border-brand-warm-gray">
        <div className="flex items-start gap-3">
          <Clock className="w-4 h-4 text-foreground-muted mt-0.5" />
          <div className="text-xs text-foreground-muted">
            <p className="font-medium text-brand-black mb-1">How It Works:</p>
            <p className="mb-1">
              <strong>Training (2-9 corrections):</strong> Analyzes your corrections to identify patterns
            </p>
            <p>
              <strong>Production (10+ corrections):</strong> Creates new symbolic rules and injects them into the engine
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
