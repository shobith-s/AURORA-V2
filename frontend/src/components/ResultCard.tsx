import { Target, Clock, Sparkles, ThumbsUp, ThumbsDown, Edit2, BookOpen } from 'lucide-react';
import { useState } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ExplanationModal from './ExplanationModal';
import ActionLibraryModal from './ActionLibraryModal';

interface ResultData {
  action: string;
  confidence: number;
  source: string;
  column_name?: string;
  explanation?: string;
  alternatives?: Array<{ action: string; confidence: number }>;
  transformed_data?: unknown[];
  warning?: string;
  require_manual_review?: boolean;
  decision_id?: string;
  latency?: number;
}

interface ResultCardProps {
  result: ResultData;
}

export default function ResultCard({ result }: ResultCardProps) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctAction, setCorrectAction] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showActionLibrary, setShowActionLibrary] = useState(false);

  const getSourceColor = (source: string) => {
    // Check if learned pattern has limited training
    const isLimitedTraining = source === 'learned' && result.explanation?.includes('Limited training');

    const colors = {
      symbolic: 'bg-primary/20 text-primary-dark border-primary/50',
      neural: 'bg-primary/20 text-primary-dark border-primary/50',
      learned: isLimitedTraining
        ? 'bg-amber-100 text-amber-700 border-amber-200'  // Warning color for limited training
        : 'bg-green-100 text-success border-success/30',
      meta_learning: 'bg-background-muted text-foreground border-brand-warm-gray',
      conservative_fallback: 'bg-background-muted text-foreground border-brand-warm-gray'
    };
    return colors[source as keyof typeof colors] || 'bg-background-muted text-foreground';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-error';
  };

  const handleCorrection = async () => {
    if (!correctAction) {
      toast.error('Please enter the correct action');
      return;
    }

    setIsSubmitting(true);
    try {
      await axios.post('/api/correct', {
        column_data: [], // In real app, would pass actual data
        column_name: result.column_name || 'column',
        wrong_action: result.action,
        correct_action: correctAction,
        confidence: result.confidence
      });

      toast.success('Correction submitted! System is learning...');
      setShowCorrection(false);
    } catch {
      toast.error('Failed to submit correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="glass-card p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-brand-black">Recommendation</h3>
        <div
          className={`px-3 py-1 rounded-full text-xs font-medium border ${getSourceColor(result.source)} cursor-help`}
          title={result.source === 'neural' ? 'Decision made by Neural Oracle (XGBoost) based on learned patterns' : 'Decision made by Symbolic Engine based on explicit rules'}
        >
          {result.source === 'symbolic' && '⚡ Symbolic'}
          {result.source === 'neural' && '🧠 Neural'}
          {result.source === 'learned' && '🎓 Learned'}
        </div>
      </div>

      {/* Main Action */}
      <div className="bg-primary/10 rounded-xl p-6 border border-primary/30">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-brand-white rounded-lg flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-primary" />
            </div>
            <div>
              <p className="text-sm text-foreground-muted">Recommended Action</p>
              <h4 className="text-2xl font-bold gradient-text">
                {result.action.replace(/_/g, ' ').toUpperCase()}
              </h4>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowExplanation(true)}
              className={`flex items-center gap-2 px-3 py-2 bg-primary hover:bg-primary-hover text-brand-white rounded-lg text-sm font-medium transition shadow-md ${result.source === 'neural' ? 'animate-pulse' : ''}`}
            >
              <BookOpen className="w-4 h-4" />
              Explain
            </button>
            <button
              onClick={() => setShowCorrection(!showCorrection)}
              className="flex items-center gap-2 px-3 py-2 bg-brand-white hover:bg-primary/10 text-primary rounded-lg text-sm font-medium border border-primary/50 transition"
            >
              <Edit2 className="w-4 h-4" />
              Override
            </button>
          </div>
        </div>
        <p className="text-sm text-foreground">{result.explanation}</p>
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
        <div className="bg-brand-white/50 rounded-lg p-4 border border-brand-warm-gray">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs font-medium text-foreground-muted">Confidence</span>
          </div>
          <div className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}%
          </div>
          <div className="w-full bg-background-muted rounded-full h-2 mt-2">
            <div
              className={`h-2 rounded-full ${result.confidence >= 0.9 ? 'bg-success' :
                result.confidence >= 0.7 ? 'bg-warning' : 'bg-error'
                }`}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>

        {/* Decision ID */}
        <div className="bg-brand-white/50 rounded-lg p-4 border border-brand-warm-gray">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs font-medium text-foreground-muted">Decision ID</span>
          </div>
          <div className="text-xs font-mono text-foreground truncate">
            {result.decision_id?.slice(0, 16)}...
          </div>
        </div>
      </div>

      {/* Alternatives */}
      {result.alternatives && result.alternatives.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Alternative Actions</h4>
          <div className="space-y-2">
            {result.alternatives.slice(0, 3).map((alt: { action: string; confidence: number }, idx: number) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-brand-white/50 rounded-lg border border-brand-warm-gray"
              >
                <span className="text-sm text-foreground">
                  {alt.action.replace(/_/g, ' ')}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-background-muted rounded-full h-1.5">
                    <div
                      className="bg-slate-400 h-1.5 rounded-full"
                      style={{ width: `${alt.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-foreground-muted w-12 text-right">
                    {(alt.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feedback Section */}
      <div className="border-t border-brand-warm-gray pt-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm text-foreground-muted">Was this recommendation helpful?</p>
          <div className="flex gap-2">
            <button className="p-2 hover:bg-green-50 rounded-lg transition">
              <ThumbsUp className="w-4 h-4 text-green-600" />
            </button>
            <button
              onClick={() => setShowCorrection(!showCorrection)}
              className="p-2 hover:bg-red-50 rounded-lg transition"
            >
              <ThumbsDown className="w-4 h-4 text-error" />
            </button>
          </div>
        </div>

        {/* Override/Correction Form */}
        {showCorrection && (
          <div className="bg-primary/10 rounded-lg p-4 border border-primary/50 animate-in slide-in-from-top mt-3">
            <div className="flex items-start gap-2 mb-3">
              <Edit2 className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-blue-800">Override Recommendation</p>
                <p className="text-xs text-primary mt-0.5">
                  Provide the correct action and help AURORA learn from your expertise
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setShowActionLibrary(true)}
                className="flex-1 px-4 py-2 bg-brand-white border border-primary rounded-lg text-left text-sm text-foreground hover:border-primary hover:ring-2 hover:ring-blue-100 transition-all flex items-center justify-between group"
              >
                <span className={correctAction ? 'text-primary-dark font-medium' : 'text-brand-cool-gray'}>
                  {correctAction
                    ? correctAction.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                    : 'Select an action...'}
                </span>
                <div className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs font-medium group-hover:bg-primary/20 transition-colors">
                  Browse Library
                </div>
              </button>

              <button
                onClick={handleCorrection}
                disabled={isSubmitting || !correctAction}
                className="px-4 py-2 bg-primary text-brand-white rounded-lg text-sm font-medium hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
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

      {/* Smart Action Library Modal */}
      <ActionLibraryModal
        isOpen={showActionLibrary}
        onClose={() => setShowActionLibrary(false)}
        onSelectAction={(action) => {
          setCorrectAction(action);
          setShowActionLibrary(false);
        }}
        currentAction={correctAction}
      />
    </div>
  );
}
