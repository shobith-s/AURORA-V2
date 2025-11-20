import { useEffect, useState } from 'react';
import { Activity, Zap, Brain, BookOpen, Clock, TrendingUp, Database, AlertTriangle, Sparkles, GraduationCap } from 'lucide-react';
import axios from 'axios';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend, BarChart, Bar } from 'recharts';

export default function MetricsDashboard() {
  const [dashboardMetrics, setDashboardMetrics] = useState<any>(null);
  const [realtime, setRealtime] = useState<any>(null);
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [driftStatus, setDriftStatus] = useState<any>(null);

  useEffect(() => {
    fetchDashboardMetrics();
    fetchCacheStats();
    fetchDriftStatus();
    const interval = setInterval(() => {
      fetchRealtime();
      fetchCacheStats();
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

  // Cache level data for chart
  const cacheData = cacheStats ? [
    { name: 'L1 (Exact)', hits: cacheStats.l1_hits, fill: '#10b981' },
    { name: 'L2 (Similar)', hits: cacheStats.l2_hits, fill: '#3b82f6' },
    { name: 'L3 (Pattern)', hits: cacheStats.l3_hits, fill: '#8b5cf6' },
    { name: 'Misses', hits: cacheStats.misses, fill: '#ef4444' }
  ] : [];

  return (
    <div className="container mx-auto px-4 py-6 animate-in slide-in-from-top">
      {/* Real-time Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
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

      {/* Neural Oracle Status */}
      {neural_oracle && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-600" />
              <h3 className="font-bold text-slate-800">Neural Oracle Status</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              neural_oracle.model_loaded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
              {neural_oracle.model_loaded ? '✓ Model Loaded' : '✗ Not Loaded'}
            </span>
          </div>

          {neural_oracle.model_loaded ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                <div className="text-2xl font-bold text-purple-600">
                  {neural_oracle.model_info?.model_size_kb?.toFixed(1) || 0}KB
                </div>
                <div className="text-xs text-purple-700">Model Size</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <div className="text-2xl font-bold text-blue-600">
                  {neural_oracle.model_info?.num_actions || 0}
                </div>
                <div className="text-xs text-blue-700">Actions</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                <div className="text-2xl font-bold text-green-600">
                  {neural_oracle.usage_stats?.avg_latency_ms?.toFixed(2) || 0}ms
                </div>
                <div className="text-xs text-green-700">Avg Latency</div>
              </div>
              <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                <div className="text-2xl font-bold text-indigo-600">
                  {neural_oracle.usage_stats?.total_calls || 0}
                </div>
                <div className="text-xs text-indigo-700">Total Calls</div>
              </div>
            </div>
          ) : (
            <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200 text-center">
              <p className="text-sm text-yellow-700">
                Train the neural oracle to enable meta-learning:
                <code className="block mt-2 bg-yellow-100 px-3 py-1 rounded text-xs">
                  python scripts/train_neural_oracle.py
                </code>
              </p>
            </div>
          )}

          {neural_oracle.training_history && Object.keys(neural_oracle.training_history).length > 0 && (
            <div className="mt-4 bg-slate-50 rounded-lg p-4 border border-slate-200">
              <div className="text-sm font-medium text-slate-700 mb-2">Last Training:</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div>
                  <span className="text-slate-600">Date: </span>
                  <span className="font-mono">
                    {new Date(neural_oracle.training_history.training_date).toLocaleDateString()}
                  </span>
                </div>
                <div>
                  <span className="text-slate-600">Val Accuracy: </span>
                  <span className="font-mono text-green-600">
                    {(neural_oracle.training_history.val_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-600">Samples: </span>
                  <span className="font-mono">{neural_oracle.training_history.num_samples}</span>
                </div>
                <div>
                  <span className="text-slate-600">Real Data: </span>
                  <span className="font-mono text-purple-600">
                    {neural_oracle.training_history.num_real_corrections || 0}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Learning Progress */}
      {learning && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-green-600" />
              <h3 className="font-bold text-slate-800">Adaptive Learning Progress</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              learning.corrections.total >= 50 ? 'bg-green-100 text-green-700' :
              learning.corrections.total >= 20 ? 'bg-yellow-100 text-yellow-700' :
              'bg-slate-100 text-slate-700'
            }`}>
              {learning.corrections.total} Corrections
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="text-2xl font-bold text-green-600">{learning.corrections.total}</div>
              <div className="text-xs text-green-700">Total Corrections</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{learning.corrections.last_7_days}</div>
              <div className="text-xs text-blue-700">Last 7 Days</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">{learning.learned_rules.total}</div>
              <div className="text-xs text-purple-700">Learned Rules</div>
            </div>
            <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
              <div className="text-2xl font-bold text-indigo-600">
                {learning.corrections.velocity_per_day?.toFixed(1) || 0}
              </div>
              <div className="text-xs text-indigo-700">Daily Velocity</div>
            </div>
          </div>

          {/* Training Recommendation */}
          {learning.corrections.total >= 50 && (
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-green-600" />
                <div className="text-sm font-medium text-green-700">Ready to Retrain!</div>
              </div>
              <p className="text-xs text-green-600 mb-2">
                You have {learning.corrections.total} corrections. Retraining will improve accuracy.
              </p>
              <code className="block bg-green-100 px-3 py-1 rounded text-xs text-green-800">
                python scripts/train_from_corrections.py
              </code>
            </div>
          )}

          {/* Top Learned Rules */}
          {learning.learned_rules.top_rules && learning.learned_rules.top_rules.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-slate-700 mb-2">Top Learned Rules:</div>
              <div className="space-y-2">
                {learning.learned_rules.top_rules.slice(0, 5).map((rule: any, idx: number) => (
                  <div key={idx} className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-700">{rule.rule_name}</span>
                      <span className="text-xs font-mono text-slate-600">
                        {rule.accuracy?.toFixed(1)}% accurate
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-slate-600">
                      <span>Action: <span className="font-mono">{rule.action}</span></span>
                      <span>Support: {rule.support_count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cache Statistics */}
      {cacheStats && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-600" />
              <h3 className="font-bold text-slate-800">Intelligent Cache Performance</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              cacheStats.hit_rate > 0.7 ? 'bg-green-100 text-green-700' :
              cacheStats.hit_rate > 0.5 ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              {(cacheStats.hit_rate * 100).toFixed(1)}% Hit Rate
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="text-2xl font-bold text-green-600">{cacheStats.l1_hits}</div>
              <div className="text-xs text-green-700">L1 Exact Hits</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{cacheStats.l2_hits}</div>
              <div className="text-xs text-blue-700">L2 Similar Hits</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">{cacheStats.l3_hits}</div>
              <div className="text-xs text-purple-700">L3 Pattern Hits</div>
            </div>
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
              <div className="text-2xl font-bold text-slate-600">{cacheStats.cache_size}</div>
              <div className="text-xs text-slate-700">Cache Size</div>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={cacheData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.hits}`}
                outerRadius={80}
                dataKey="hits"
              >
                {cacheData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Data Drift Monitoring */}
      {driftStatus && driftStatus.monitored_columns > 0 && (
        <div className="glass-card p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              <h3 className="font-bold text-slate-800">Data Drift Monitoring</h3>
            </div>
            {driftStatus.requires_retraining && (
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700">
                ⚠️ Retraining Recommended
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
              <div className="text-2xl font-bold text-slate-600">{driftStatus.monitored_columns}</div>
              <div className="text-xs text-slate-700">Monitored Columns</div>
            </div>
            <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
              <div className="text-2xl font-bold text-yellow-600">{driftStatus.columns_with_drift}</div>
              <div className="text-xs text-yellow-700">Columns with Drift</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
              <div className="text-2xl font-bold text-orange-600">{driftStatus.high_priority_columns.length}</div>
              <div className="text-xs text-orange-700">High Priority</div>
            </div>
            <div className="bg-red-50 rounded-lg p-4 border border-red-200">
              <div className="text-2xl font-bold text-red-600">{driftStatus.critical_columns.length}</div>
              <div className="text-xs text-red-700">Critical</div>
            </div>
          </div>

          {(driftStatus.critical_columns.length > 0 || driftStatus.high_priority_columns.length > 0) && (
            <div className="mt-4 space-y-2">
              {driftStatus.critical_columns.length > 0 && (
                <div className="bg-red-50 rounded-lg p-3 border border-red-200">
                  <div className="text-sm font-medium text-red-700 mb-1">Critical Columns:</div>
                  <div className="text-xs text-red-600">{driftStatus.critical_columns.join(', ')}</div>
                </div>
              )}
              {driftStatus.high_priority_columns.length > 0 && (
                <div className="bg-orange-50 rounded-lg p-3 border border-orange-200">
                  <div className="text-sm font-medium text-orange-700 mb-1">High Priority Columns:</div>
                  <div className="text-xs text-orange-600">{driftStatus.high_priority_columns.join(', ')}</div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decision Sources */}
        <div className="glass-card p-6">
          <h3 className="font-bold text-slate-800 mb-4">Decision Sources</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={sourceData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.value.toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sourceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: any) => `${value.toFixed(1)}%`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 text-center text-xs text-slate-600">
            Total: {Object.values(decision_sources.counts).reduce((a: any, b: any) => a + b, 0)} decisions
          </div>
        </div>

        {/* Component Performance */}
        <div className="glass-card p-6">
          <h3 className="font-bold text-slate-800 mb-4">Component Latency</h3>
          <div className="space-y-4">
            {Object.entries(performance_by_component || {}).map(([name, data]: [string, any]) => (
              <div key={name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700 capitalize">
                    {name.replace(/_/g, ' ')}
                  </span>
                  <span className="text-sm font-bold text-slate-800">
                    {data?.avg_latency_ms?.toFixed(2) || 0}ms
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      data?.avg_latency_ms < 1 ? 'bg-green-500' :
                      data?.avg_latency_ms < 5 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min((data?.avg_latency_ms || 0) * 10, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
