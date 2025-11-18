import { useEffect, useState } from 'react';
import { Activity, Zap, Brain, BookOpen, Clock, TrendingUp } from 'lucide-react';
import axios from 'axios';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend } from 'recharts';

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState<any>(null);
  const [realtime, setRealtime] = useState<any>(null);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchRealtime, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('/api/metrics/performance');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const fetchRealtime = async () => {
    try {
      const response = await axios.get('/api/metrics/realtime');
      setRealtime(response.data);
    } catch (error) {
      console.error('Failed to fetch realtime metrics:', error);
    }
  };

  if (!metrics) {
    return (
      <div className="container mx-auto px-4 py-6">
        <div className="glass-card p-12 text-center">
          <div className="loading-dots flex gap-2 justify-center">
            <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
            <span className="w-3 h-3 bg-purple-500 rounded-full"></span>
            <span className="w-3 h-3 bg-pink-500 rounded-full"></span>
          </div>
          <p className="mt-4 text-slate-600">Loading metrics...</p>
        </div>
      </div>
    );
  }

  const sourceData = Object.entries(metrics.decision_sources || {}).map(([name, value]) => ({
    name,
    value
  }));

  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

  return (
    <div className="container mx-auto px-4 py-6 animate-in slide-in-from-top">
      {/* Real-time Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">LIVE</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {realtime?.cpu_percent?.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-600">CPU Usage</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-5 h-5 text-purple-600" />
            <span className="text-xs font-medium text-purple-600 bg-purple-50 px-2 py-1 rounded">LIVE</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {realtime?.memory_percent?.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-600">Memory Usage</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-1 rounded">RATE</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {metrics.overview.success_rate ? (metrics.overview.success_rate * 100).toFixed(1) : 0}%
          </div>
          <div className="text-xs text-slate-600">Success Rate</div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-5 h-5 text-yellow-600" />
            <span className="text-xs font-medium text-yellow-600 bg-yellow-50 px-2 py-1 rounded">AVG</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {metrics.component_metrics?.overall_pipeline?.avg_latency_ms?.toFixed(2) || 0}ms
          </div>
          <div className="text-xs text-slate-600">Avg Latency</div>
        </div>
      </div>

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
                label={(entry) => `${entry.name}: ${entry.value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sourceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Component Performance */}
        <div className="glass-card p-6">
          <h3 className="font-bold text-slate-800 mb-4">Component Latency</h3>
          <div className="space-y-4">
            {Object.entries(metrics.component_metrics || {}).map(([name, data]: [string, any]) => (
              <div key={name}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700 capitalize">
                    {name.replace(/_/g, ' ')}
                  </span>
                  <span className="text-sm font-bold text-slate-800">
                    {data?.avg_latency_ms?.toFixed(2)}ms
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
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

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div className="glass-card p-4">
          <div className="text-3xl font-bold gradient-text mb-1">
            {metrics.overview.total_decisions}
          </div>
          <div className="text-sm text-slate-600">Total Decisions</div>
        </div>

        <div className="glass-card p-4">
          <div className="text-3xl font-bold text-green-600 mb-1">
            {metrics.overview.successful_calls}
          </div>
          <div className="text-sm text-slate-600">Successful</div>
        </div>

        <div className="glass-card p-4">
          <div className="text-3xl font-bold text-blue-600 mb-1">
            {metrics.confidence_stats?.avg ? (metrics.confidence_stats.avg * 100).toFixed(0) : 0}%
          </div>
          <div className="text-sm text-slate-600">Avg Confidence</div>
        </div>

        <div className="glass-card p-4">
          <div className="text-3xl font-bold text-purple-600 mb-1">
            {metrics.learning?.learned_rules || 0}
          </div>
          <div className="text-sm text-slate-600">Learned Rules</div>
        </div>
      </div>
    </div>
  );
}
