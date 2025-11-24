import { TrendingUp, Target, Clock, Sparkles, ThumbsUp, ThumbsDown, Edit2, BookOpen } from 'lucide-react';
import { useState } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ExplanationModal from './ExplanationModal';

interface ResultCardProps {
  result: any;
}

export default function ResultCard({ result }: ResultCardProps) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctAction, setCorrectAction] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const getSourceColor = (source: string) => {
    // Check if learned pattern has limited training
    const isLimitedTraining = source === 'learned' && result.explanation?.includes('Limited training');

    const colors = {
      symbolic: 'bg-blue-100 text-blue-700 border-blue-200',
      neural: 'bg-purple-100 text-purple-700 border-purple-200',
      learned: isLimitedTraining
        ? 'bg-amber-100 text-amber-700 border-amber-200'  // Warning color for limited training
        : 'bg-green-100 text-green-700 border-green-200',
      meta_learning: 'bg-slate-100 text-slate-700 border-slate-200',
      conservative_fallback: 'bg-gray-100 text-gray-700 border-gray-200'
    };
    return colors[source as keyof typeof colors] || 'bg-gray-100 text-gray-700';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const handleCorrection = async () => {
    if (!correctAction) {
      toast.error('Please enter the correct action');
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await axios.post('/api/correct', {
        column_data: [], // In real app, would pass actual data
        column_name: result.column_name || 'column',
        wrong_action: result.action,
        correct_action: correctAction,
        confidence: result.confidence
      });

      toast.success('Correction submitted! System is learning...');
      setShowCorrection(false);
    } catch (error) {
      toast.error('Failed to submit correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="glass-card p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-slate-800">Recommendation</h3>
        <div className={`px-3 py-1 rounded-full text-xs font-medium border ${getSourceColor(result.source)}`}>
          {result.source === 'symbolic' && '⚡ Symbolic'}
          {result.source === 'neural' && '🧠 Neural'}
          {result.source === 'learned' && '🎓 Learned'}
        </div>
      </div>

      {/* Main Action */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-100">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white rounded-lg flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600">Recommended Action</p>
              <h4 className="text-2xl font-bold gradient-text">
                {result.action.replace(/_/g, ' ').toUpperCase()}
              </h4>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowExplanation(true)}
              className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg text-sm font-medium transition shadow-md"
            >
              <BookOpen className="w-4 h-4" />
              Explain
            </button>
            <button
              onClick={() => setShowCorrection(!showCorrection)}
              className="flex items-center gap-2 px-3 py-2 bg-white hover:bg-blue-50 text-blue-600 rounded-lg text-sm font-medium border border-blue-200 transition"
            >
              <Edit2 className="w-4 h-4" />
              Override
            </button>
          </div>
        </div>
        <p className="text-sm text-slate-700">{result.explanation}</p>
      </div>

      {/* Learned Pattern Training Warning */}
      {result.source === 'learned' && result.explanation?.includes('Limited training') && (
        <div className="rounded-lg p-4 border bg-amber-50 border-amber-300">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-amber-100">
              <span className="text-lg">🎓</span>
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-amber-800">
                Limited Training Data
              </p>
              <p className="text-xs mt-1 text-amber-700">
                This learned pattern is based on limited corrections (&lt;20). The system has reduced its confidence accordingly. Consider reviewing this recommendation carefully and submit corrections to improve future predictions.
              </p>
              <div className="mt-2 flex items-center gap-2">
                <div className="flex-1 bg-amber-200 rounded-full h-2">
                  <div
                    className="bg-amber-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, (parseInt(result.explanation.match(/(\d+)\/20/)?.[1] || '0') / 20) * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-amber-700">
                  {result.explanation.match(/(\d+)\/20/)?.[1] || '?'}/20
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Confidence Warning (Phase 3) */}
      {result.warning && (
        <div className={`rounded-lg p-4 border ${result.require_manual_review
          ? 'bg-red-50 border-red-300'
          : 'bg-yellow-50 border-yellow-300'
          }`}>
          <div className="flex items-start gap-3">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${result.require_manual_review ? 'bg-red-100' : 'bg-yellow-100'
              }`}>
              <span className="text-lg">{result.require_manual_review ? '⚠️' : '⚡'}</span>
            </div>
            <div className="flex-1">
              <p className={`text-sm font-semibold ${result.require_manual_review ? 'text-red-800' : 'text-yellow-800'
                }`}>
                {result.require_manual_review ? 'Manual Review Required' : 'Low Confidence Warning'}
              </p>
              <p className={`text-xs mt-1 ${result.require_manual_review ? 'text-red-700' : 'text-yellow-700'
                }`}>
                {result.warning}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Confidence */}
        <div className="bg-white/50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-slate-600" />
            <span className="text-xs font-medium text-slate-600">Confidence</span>
          </div>
          <div className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}%
          </div>
          <div className="w-full bg-slate-200 rounded-full h-2 mt-2">
            <div
              className={`h-2 rounded-full ${result.confidence >= 0.9 ? 'bg-green-500' :
                result.confidence >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Decision ID */}
        <div className="bg-white/50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-slate-600" />
            <span className="text-xs font-medium text-slate-600">Decision ID</span>
          </div>
          <div className="text-xs font-mono text-slate-700 truncate">
            {result.decision_id?.slice(0, 16)}...
          </div>
        </div>
      </div>

      {/* Alternatives */}
      {result.alternatives && result.alternatives.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-slate-700 mb-3">Alternative Actions</h4>
          <div className="space-y-2">
            {result.alternatives.slice(0, 3).map((alt: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-white/50 rounded-lg border border-slate-200"
              >
                <span className="text-sm text-slate-700">
                  {alt.action.replace(/_/g, ' ')}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-slate-200 rounded-full h-1.5">
                    <div
                      className="bg-slate-400 h-1.5 rounded-full"
                      style={{ width: `${alt.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-600 w-12 text-right">
                    {(alt.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feedback Section */}
      <div className="border-t border-slate-200 pt-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm text-slate-600">Was this recommendation helpful?</p>
          <div className="flex gap-2">
            <button className="p-2 hover:bg-green-50 rounded-lg transition">
              <ThumbsUp className="w-4 h-4 text-green-600" />
            </button>
            <button
              onClick={() => setShowCorrection(!showCorrection)}
              className="p-2 hover:bg-red-50 rounded-lg transition"
            >
              <ThumbsDown className="w-4 h-4 text-red-600" />
            </button>
          </div>
        </div>

        {/* Override/Correction Form */}
        {showCorrection && (
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200 animate-in slide-in-from-top mt-3">
            <div className="flex items-start gap-2 mb-3">
              <Edit2 className="w-5 h-5 text-blue-600 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-blue-800">Override Recommendation</p>
                <p className="text-xs text-blue-600 mt-0.5">
                  Provide the correct action and help AURORA learn from your expertise
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <select
                value={correctAction}
                onChange={(e) => setCorrectAction(e.target.value)}
                className="flex-1 px-3 py-2 rounded-lg border border-blue-300 text-sm bg-white"
              >
                <option value="">-- Select Action --</option>
                <option value="keep">Keep (No preprocessing)</option>
                <option value="standard_scale">Standard Scale</option>
                <option value="min_max_scale">Min-Max Scale</option>
                <option value="robust_scale">Robust Scale</option>
                <option value="log_transform">Log Transform</option>
                <option value="box_cox">Box-Cox Transform</option>
                <option value="yeo_johnson">Yeo-Johnson Transform</option>
                <option value="one_hot_encode">One-Hot Encode</option>
                <option value="label_encode">Label Encode</option>
                <option value="target_encode">Target Encode</option>
                <option value="fill_null_mean">Fill Nulls (Mean)</option>
                <option value="fill_null_median">Fill Nulls (Median)</option>
                <option value="fill_null_mode">Fill Nulls (Mode)</option>
                <option value="drop_column">Drop Column</option>
              </select>
              <button
                onClick={handleCorrection}
                disabled={isSubmitting || !correctAction}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting ? 'Submitting...' : 'Apply Override'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Explanation Modal */}
      <ExplanationModal
        isOpen={showExplanation}
        onClose={() => setShowExplanation(false)}
        columnData={[]} // Will need to pass actual data
        columnName={result.column_name || 'column'}
      />
    </div>
  );
}
