import { useEffect, useState } from 'react';
import { Activity, Zap, Brain, BookOpen, Clock, TrendingUp, Database, AlertTriangle, Sparkles, GraduationCap } from 'lucide-react';
import axios from 'axios';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend, BarChart, Bar } from 'recharts';

export default function MetricsDashboard() {
  const [dashboardMetrics, setDashboardMetrics] = useState<any>(null);
  const [realtime, setRealtime] = useState<any>(null);
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [driftStatus, setDriftStatus] = useState<any>(null);
  const [layerMetrics, setLayerMetrics] = useState<any>(null);

  useEffect(() => {
    fetchDashboardMetrics();
    fetchCacheStats();
    fetchDriftStatus();
    fetchLayerMetrics();
    const interval = setInterval(() => {
      fetchRealtime();
      fetchCacheStats();
      fetchLayerMetrics();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardMetrics = async () => {
    try {
      const response = await axios.get('/api/metrics/dashboard');
      setDashboardMetrics(response.data);
    } catch (error) {
      console.error('Failed to fetch dashboard metrics:', error);
    }
  };

  const fetchRealtime = async () => {
    try {
      const response = await axios.get('/api/metrics/realtime');
      setRealtime(response.data);
    } catch (error) {
      // Silently fail - realtime metrics are not critical
    }
  };

  const fetchCacheStats = async () => {
    try {
      const response = await axios.get('/api/cache/stats');
      setCacheStats(response.data);
    } catch (error) {
      // Silently fail - cache stats are not critical
    }
  };

  const fetchDriftStatus = async () => {
    try {
      const response = await axios.get('/api/drift/status');
      setDriftStatus(response.data);
    } catch (error) {
      // Silently fail - drift status is not critical
    }
  };

  const fetchLayerMetrics = async () => {
    try {
      const response = await axios.get('/api/metrics/layers');
      setLayerMetrics(response.data);
    } catch (error) {
      // Silently fail - layer metrics are not critical
    }
  };

  if (!dashboardMetrics) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="glass-card p-12 text-center">
          <div className="loading-dots flex gap-2 justify-center">
            <span className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></span>
            <span className="w-3 h-3 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
            <span className="w-3 h-3 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
          </div>
          <p className="mt-4 text-slate-600">Loading metrics...</p>
        </div>
      </div>
    );
  }

  const { overview, decision_sources, neural_oracle, learning, performance_by_component } = dashboardMetrics;

  // Prepare data for charts
  const sourceData = Object.entries(decision_sources.percentages || {}).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value: Number(value),
    count: decision_sources.counts[name]
  }));

  const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b'];

  // Calculate total decisions
  const totalDecisions = Object.values(decision_sources?.counts || {}).reduce(
    (sum: number, count: any) => sum + Number(count),
    0
  );

  // Cache level data for chart
  const cacheData = cacheStats ? [
    { name: 'L1 (Exact)', hits: cacheStats.l1_hits, fill: '#10b981' },
    { name: 'L2 (Similar)', hits: cacheStats.l2_hits, fill: '#3b82f6' },
    { name: 'L3 (Pattern)', hits: cacheStats.l3_hits, fill: '#8b5cf6' },
    { name: 'Misses', hits: cacheStats.misses, fill: '#ef4444' }
  ] : [];

  return (
    <div className="container mx-auto px-4 py-4 animate-in slide-in-from-top">
      {/* Real-time Stats - Compact */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">LIVE</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {realtime?.cpu_percent?.toFixed(1) || overview.system_cpu?.toFixed(1) || 0}%
          </div>
          <div className="text-xs text-slate-600">CPU Usage</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-5 h-5 text-purple-600" />
            <span className="text-xs font-medium text-purple-600 bg-purple-50 px-2 py-1 rounded">LIVE</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {realtime?.memory_percent?.toFixed(1) || overview.system_memory?.toFixed(1) || 0}%
          </div>
          <div className="text-xs text-slate-600">Memory Usage</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-1 rounded">AVG</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {overview.avg_latency_ms?.toFixed(2) || 0}ms
          </div>
          <div className="text-xs text-slate-600">Avg Latency</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-5 h-5 text-yellow-600" />
            <span className="text-xs font-medium text-yellow-600 bg-yellow-50 px-2 py-1 rounded">TIME</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {overview.uptime_hours?.toFixed(1) || 0}h
          </div>
          <div className="text-xs text-slate-600">Uptime</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            <span className="text-xs font-medium text-indigo-600 bg-indigo-50 px-2 py-1 rounded">TOTAL</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {overview.total_decisions || 0}
          </div>
          <div className="text-xs text-slate-600">Decisions</div>
        </div>
      </div>

      {/* 3-Column Dashboard Grid: Neural + Learning + Cache */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        {/* Neural Oracle Status */}
        {neural_oracle && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-600" />
                <h3 className="font-semibold text-slate-800 text-sm">Neural Oracle</h3>
              </div>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                neural_oracle.model_loaded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {neural_oracle.model_loaded ? '✓' : '✗'}
              </span>
            </div>

            {neural_oracle.model_loaded ? (
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-purple-50 rounded p-2 border border-purple-200">
                  <div className="text-lg font-bold text-purple-600">
                    {neural_oracle.usage_stats?.total_calls || 0}
                  </div>
                  <div className="text-xs text-purple-700">Calls</div>
                </div>
                <div className="bg-green-50 rounded p-2 border border-green-200">
                  <div className="text-lg font-bold text-green-600">
                    {neural_oracle.usage_stats?.avg_latency_ms?.toFixed(1) || 0}ms
                  </div>
                  <div className="text-xs text-green-700">Latency</div>
                </div>
              </div>
            ) : (
              <div className="bg-yellow-50 rounded p-2 border border-yellow-200">
                <p className="text-xs text-yellow-700">Train to enable</p>
              </div>
            )}
          </div>
        )}

        {/* Learning Progress - Compact */}
        {learning && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <GraduationCap className="w-4 h-4 text-green-600" />
                <h3 className="font-semibold text-slate-800 text-sm">Learning</h3>
              </div>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                learning.corrections.total >= 50 ? 'bg-green-100 text-green-700' :
                learning.corrections.total >= 20 ? 'bg-yellow-100 text-yellow-700' :
                'bg-slate-100 text-slate-700'
              }`}>
                {learning.corrections.total}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-green-50 rounded p-2 border border-green-200">
                <div className="text-lg font-bold text-green-600">{learning.corrections.total}</div>
                <div className="text-xs text-green-700">Corrections</div>
              </div>
              <div className="bg-purple-50 rounded p-2 border border-purple-200">
                <div className="text-lg font-bold text-purple-600">{learning.learned_rules.total}</div>
                <div className="text-xs text-purple-700">Rules</div>
              </div>
            </div>
          </div>
        )}

        {/* Cache Statistics - Compact */}
        {cacheStats && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-600" />
                <h3 className="font-semibold text-slate-800 text-sm">Cache</h3>
              </div>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                cacheStats.hit_rate > 0.7 ? 'bg-green-100 text-green-700' :
                cacheStats.hit_rate > 0.5 ? 'bg-yellow-100 text-yellow-700' :
                'bg-red-100 text-red-700'
              }`}>
                {(cacheStats.hit_rate * 100).toFixed(0)}%
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-green-50 rounded p-2 border border-green-200">
                <div className="text-lg font-bold text-green-600">{cacheStats.l1_hits}</div>
                <div className="text-xs text-green-700">L1 Hits</div>
              </div>
              <div className="bg-blue-50 rounded p-2 border border-blue-200">
                <div className="text-lg font-bold text-blue-600">{cacheStats.l2_hits}</div>
                <div className="text-xs text-blue-700">L2 Hits</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 2-Column: Charts + Drift */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        {/* Decision Sources Chart */}
        <div className="glass-card p-4">
          <h3 className="font-semibold text-slate-800 mb-3 text-sm">Decision Sources</h3>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={sourceData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.value.toFixed(0)}%`}
                outerRadius={60}
                fill="#8884d8"
                dataKey="value"
              >
                {sourceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: any) => `${value.toFixed(1)}%`} />
            </PieChart>
          </ResponsiveContainer>
          <div className="text-center text-xs text-slate-600 mt-2">
            Total: {totalDecisions} decisions
          </div>
        </div>

        {/* Data Drift Monitoring - Compact */}
        {driftStatus && driftStatus.monitored_columns > 0 && (
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
                <h3 className="font-semibold text-slate-800 text-sm">Drift Monitor</h3>
              </div>
              {driftStatus.requires_retraining && (
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                  ⚠️
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-50 rounded p-2 border border-slate-200">
                <div className="text-lg font-bold text-slate-600">{driftStatus.monitored_columns}</div>
                <div className="text-xs text-slate-700">Monitored</div>
              </div>
              <div className="bg-yellow-50 rounded p-2 border border-yellow-200">
                <div className="text-lg font-bold text-yellow-600">{driftStatus.columns_with_drift}</div>
                <div className="text-xs text-yellow-700">Drift</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
