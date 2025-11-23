import { useState, useEffect } from 'react';
import { TrendingUp, Clock, Users, Star, CheckCircle, Award, Zap, ThumbsUp } from 'lucide-react';
import axios from 'axios';

interface KeyStat {
  label: string;
  value: string;
  icon: string;
  color: string;
  description: string;
}

interface ProofPoint {
  text: string;
}

interface Testimonial {
  user_id: string;
  rating: number;
  testimonial: string;
  use_case: string;
  timestamp: string;
}

interface DashboardData {
  overview: {
    total_decisions: number;
    total_users: number;
    total_sessions: number;
    time_saved_hours: number;
    average_confidence: number;
  };
  performance: {
    acceptance_rate: number;
    time_vs_manual_improvement_percentage: number;
    average_time_saved_per_decision_seconds: number;
  };
  user_satisfaction: {
    average_rating: number;
    recommendation_rate: number;
    learning_rate: number;
  };
  key_stats: KeyStat[];
  proof_points: string[];
  testimonials: Testimonial[];
}

export default function ValidationDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('/api/validation/dashboard');
      if (response.data.success) {
        setDashboardData(response.data.dashboard);
      }
    } catch (error) {
      console.error('Error fetching validation dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const getIconComponent = (iconName: string) => {
    const iconMap: Record<string, any> = {
      'clock': Clock,
      'star': Star,
      'thumbs-up': ThumbsUp,
      'graduation-cap': Award,
      'check': CheckCircle,
      'zap': Zap,
      'users': Users,
      'trending-up': TrendingUp,
    };
    return iconMap[iconName] || Star;
  };

  const getColorClass = (color: string) => {
    const colorMap: Record<string, string> = {
      'blue': 'from-blue-500 to-blue-600',
      'green': 'from-green-500 to-green-600',
      'yellow': 'from-yellow-500 to-yellow-600',
      'purple': 'from-purple-500 to-purple-600',
      'red': 'from-red-500 to-red-600',
    };
    return colorMap[color] || 'from-gray-500 to-gray-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h3 className="text-yellow-800 font-semibold">No Validation Data Yet</h3>
          <p className="text-yellow-700 mt-2">
            Start using AURORA to generate validation metrics!
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Hero Stats */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          AURORA Validation Dashboard
        </h1>
        <p className="text-slate-600 text-lg">
          Real metrics proving AURORA's value
        </p>
      </div>

      {/* Key Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {dashboardData.key_stats.map((stat, index) => {
          const IconComponent = getIconComponent(stat.icon);
          const colorClass = getColorClass(stat.color);

          return (
            <div
              key={index}
              className="bg-white rounded-lg shadow-lg p-6 border border-slate-200 hover:shadow-xl transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-slate-600 text-sm font-medium mb-1">
                    {stat.label}
                  </p>
                  <p className="text-3xl font-bold text-slate-900">
                    {stat.value}
                  </p>
                  <p className="text-slate-500 text-xs mt-2">
                    {stat.description}
                  </p>
                </div>
                <div className={`p-3 rounded-lg bg-gradient-to-br ${colorClass}`}>
                  <IconComponent className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <Users className="w-8 h-8 mb-2 opacity-80" />
          <p className="text-sm opacity-90">Total Users</p>
          <p className="text-3xl font-bold">{dashboardData.overview.total_users}</p>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white">
          <CheckCircle className="w-8 h-8 mb-2 opacity-80" />
          <p className="text-sm opacity-90">Decisions Made</p>
          <p className="text-3xl font-bold">{dashboardData.overview.total_decisions}</p>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white">
          <Clock className="w-8 h-8 mb-2 opacity-80" />
          <p className="text-sm opacity-90">Time Saved</p>
          <p className="text-3xl font-bold">{dashboardData.overview.time_saved_hours.toFixed(1)}h</p>
        </div>

        <div className="bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-lg p-6 text-white">
          <Star className="w-8 h-8 mb-2 opacity-80" />
          <p className="text-sm opacity-90">Avg Confidence</p>
          <p className="text-3xl font-bold">{(dashboardData.overview.average_confidence * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Proof Points */}
      {dashboardData.proof_points && dashboardData.proof_points.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <Award className="w-6 h-6 text-blue-600" />
            Proven Results
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {dashboardData.proof_points.map((point, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg"
              >
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <p className="text-slate-700">{point}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Acceptance Rate</h3>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-3xl font-bold text-green-600">
                  {dashboardData.performance.acceptance_rate.toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-green-200">
              <div
                style={{ width: `${dashboardData.performance.acceptance_rate}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-green-500"
              ></div>
            </div>
            <p className="text-sm text-slate-600">Decisions accepted without modification</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Time Improvement</h3>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-3xl font-bold text-blue-600">
                  {dashboardData.performance.time_vs_manual_improvement_percentage.toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
              <div
                style={{ width: `${Math.min(100, dashboardData.performance.time_vs_manual_improvement_percentage)}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
              ></div>
            </div>
            <p className="text-sm text-slate-600">Faster than manual preprocessing</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">User Rating</h3>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-3xl font-bold text-yellow-600">
                  {dashboardData.user_satisfaction.average_rating.toFixed(1)}/5
                </span>
              </div>
            </div>
            <div className="flex gap-1 mb-4">
              {[1, 2, 3, 4, 5].map((star) => (
                <Star
                  key={star}
                  className={`w-6 h-6 ${
                    star <= dashboardData.user_satisfaction.average_rating
                      ? 'fill-yellow-400 text-yellow-400'
                      : 'text-gray-300'
                  }`}
                />
              ))}
            </div>
            <p className="text-sm text-slate-600">Average user satisfaction</p>
          </div>
        </div>
      </div>

      {/* Testimonials */}
      {dashboardData.testimonials && dashboardData.testimonials.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6 border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center gap-2">
            <Star className="w-6 h-6 text-yellow-500" />
            User Testimonials
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {dashboardData.testimonials.map((testimonial, index) => (
              <div
                key={index}
                className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg p-6 border border-slate-200"
              >
                <div className="flex items-center gap-2 mb-3">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <p className="text-slate-700 italic mb-4">"{testimonial.testimonial}"</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">
                    <span className="font-medium">Use case:</span> {testimonial.use_case}
                  </span>
                  <span className="text-slate-500">{testimonial.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 text-center text-slate-500 text-sm">
        <p>Metrics updated in real-time • All data anonymized and aggregated</p>
      </div>
    </div>
  );
}
