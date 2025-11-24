import { X, BookOpen, Lightbulb, AlertTriangle, TrendingUp, Zap } from 'lucide-react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

interface ExplanationModalProps {
    isOpen: boolean;
    onClose: () => void;
    columnData: any[];
    columnName: string;
}

export default function ExplanationModal({ isOpen, onClose, columnData, columnName }: ExplanationModalProps) {
    const [explanation, setExplanation] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<'explanation' | 'alternatives' | 'impact'>('explanation');

    // Fetch explanation when modal opens
    useEffect(() => {
        if (isOpen && !explanation) {
            fetchExplanation();
        }
    }, [isOpen]);

    const fetchExplanation = async () => {
        setIsLoading(true);
        try {
            const response = await axios.post('/api/explain/enhanced', {
                column_data: columnData,
                column_name: columnName
            });
            setExplanation(response.data);
        } catch (error: any) {
            toast.error('Failed to load explanation');
            console.error(error);
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="relative w-full max-w-4xl max-h-[90vh] overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700 rounded-2xl shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-700">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                            <BookOpen className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-slate-900">Enhanced Explanation</h2>
                            <p className="text-sm text-slate-400">Deep dive into the decision for "{columnName}"</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-slate-400" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 px-6 pt-4 border-b border-slate-700">
                    <button
                        onClick={() => setActiveTab('explanation')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${activeTab === 'explanation'
                                ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-400'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <Lightbulb className="w-4 h-4" />
                            Explanation
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('alternatives')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${activeTab === 'alternatives'
                                ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-400'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Alternatives
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('impact')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${activeTab === 'impact'
                                ? 'bg-slate-800 text-blue-400 border-b-2 border-blue-400'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            Impact
                        </div>
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="flex flex-col items-center gap-3">
                                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                <p className="text-slate-400">Generating detailed explanation...</p>
                            </div>
                        </div>
                    ) : explanation ? (
                        <div className="space-y-6">
                            {activeTab === 'explanation' && (
                                <div className="prose prose-invert max-w-none">
                                    {/* Decision Summary */}
                                    <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg mb-6">
                                        <div className="flex items-center justify-between mb-2">
                                            <h3 className="text-lg font-bold text-slate-900 m-0">Recommended Action</h3>
                                            <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-medium">
                                                {(explanation.decision.confidence * 100).toFixed(0)}% Confident
                                            </span>
                                        </div>
                                        <p className="text-2xl font-bold text-blue-400 m-0">
                                            {explanation.decision.action.replace(/_/g, ' ').toUpperCase()}
                                        </p>
                                        <p className="text-sm text-slate-400 mt-1 m-0">
                                            Source: {explanation.decision.source}
                                        </p>
                                    </div>

                                    {/* Markdown Explanation */}
                                    <div
                                        className="text-slate-300 leading-relaxed"
                                        dangerouslySetInnerHTML={{ __html: formatMarkdown(explanation.markdown_report) }}
                                    />
                                </div>
                            )}

                            {activeTab === 'alternatives' && (
                                <div className="space-y-4">
                                    <h3 className="text-lg font-bold text-slate-900">Alternative Approaches</h3>
                                    <p className="text-slate-400">
                                        Here are other preprocessing techniques that could work, with trade-offs explained.
                                    </p>
                                    {/* Placeholder - will be populated from enhanced_explanation */}
                                    <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                                        <p className="text-slate-400 text-sm">
                                            Alternative analysis coming from enhanced_explanation.alternatives
                                        </p>
                                    </div>
                                </div>
                            )}

                            {activeTab === 'impact' && (
                                <div className="space-y-4">
                                    <h3 className="text-lg font-bold text-slate-900">Expected Impact</h3>
                                    <p className="text-slate-400">
                                        How this preprocessing decision will affect your model's performance.
                                    </p>
                                    {/* Placeholder - will be populated from enhanced_explanation */}
                                    <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                                        <p className="text-slate-400 text-sm">
                                            Impact predictions coming from enhanced_explanation.impact_prediction
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-12">
                            <p className="text-slate-400">No explanation available</p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-slate-700 bg-slate-900/50">
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                        <AlertTriangle className="w-4 h-4" />
                        <span>This explanation is AI-generated. Always validate with domain knowledge.</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-slate-900 rounded-lg font-medium transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}

// Simple markdown to HTML converter (basic implementation)
function formatMarkdown(markdown: string): string {
    if (!markdown) return '';

    return markdown
        // Headers
        .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-slate-900 mt-4 mb-2">$1</h3>')
        .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-slate-900 mt-6 mb-3">$1</h2>')
        .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold text-slate-900 mt-8 mb-4">$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-900 font-semibold">$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em class="text-slate-300">$1</em>')
        // Lists
        .replace(/^\* (.*$)/gim, '<li class="ml-4 text-slate-300">$1</li>')
        .replace(/^- (.*$)/gim, '<li class="ml-4 text-slate-300">$1</li>')
        // Line breaks
        .replace(/\n\n/g, '<br/><br/>')
        .replace(/\n/g, '<br/>');
}
