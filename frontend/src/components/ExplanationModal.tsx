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
            console.log('🔍 Explanation Modal - Fetching explanation...');
            console.log('Column Name:', columnName);
            console.log('Column Data (first 10):', columnData?.slice(0, 10));
            console.log('Column Data length:', columnData?.length);

            const payload = {
                column_data: columnData,
                column_name: columnName
            };

            console.log('📤 API Request Payload:', payload);

            const response = await axios.post('/api/explain/enhanced', payload);

            console.log('✅ API Response:', response.data);
            setExplanation(response.data);
        } catch (error: any) {
            console.error('❌ Explanation Modal Error:', error);
            console.error('Error Response:', error.response?.data);
            console.error('Error Status:', error.response?.status);
            toast.error('Failed to load explanation: ' + (error.response?.data?.detail || error.message));
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
            <div className="relative w-full max-w-4xl max-h-[90vh] overflow-hidden bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-slate-800 dark:to-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-lg">
                            <BookOpen className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Enhanced Explanation</h2>
                            <p className="text-sm text-slate-600 dark:text-slate-400">Deep dive into the decision for "{columnName}"</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 px-6 pt-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                    <button
                        onClick={() => setActiveTab('explanation')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-all ${activeTab === 'explanation'
                            ? 'bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 shadow-sm'
                            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <Lightbulb className="w-4 h-4" />
                            Explanation
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('alternatives')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-all ${activeTab === 'alternatives'
                            ? 'bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 shadow-sm'
                            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Alternatives
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('impact')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-all ${activeTab === 'impact'
                            ? 'bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 shadow-sm'
                            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            Impact
                        </div>
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)] bg-white dark:bg-slate-900">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="flex flex-col items-center gap-3">
                                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                <p className="text-slate-600 dark:text-slate-400">Generating detailed explanation...</p>
                            </div>
                        </div>
                    ) : explanation ? (
                        <div className="space-y-6">
                            {activeTab === 'explanation' && (
                                <div className="prose prose-slate dark:prose-invert max-w-none">
                                    {/* Decision Summary Card */}
                                    <div className="p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-800 dark:to-slate-800 border border-blue-200 dark:border-slate-700 rounded-xl mb-6 shadow-sm">
                                        <div className="flex items-center justify-between mb-4">
                                            <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 m-0">
                                                Recommended Action
                                            </h3>
                                            <span className="px-4 py-1.5 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-full text-sm font-semibold shadow-md">
                                                {(explanation.decision.confidence * 100).toFixed(0)}% Confident
                                            </span>
                                        </div>
                                        <p className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent m-0 mb-2">
                                            {explanation.decision.action.replace(/_/g, ' ').toUpperCase()}
                                        </p>
                                        <div className="flex items-center gap-2 mt-3">
                                            <span className="text-xs px-2 py-1 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded font-medium">
                                                Source: {explanation.decision.source}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Markdown Explanation */}
                                    <div
                                        className="text-slate-700 dark:text-slate-300 leading-relaxed prose-headings:text-slate-900 dark:prose-headings:text-white prose-strong:text-slate-900 dark:prose-strong:text-white prose-p:text-slate-700 dark:prose-p:text-slate-300"
                                        dangerouslySetInnerHTML={{ __html: formatMarkdown(explanation.markdown_report) }}
                                    />
                                </div>
                            )}

                            {activeTab === 'alternatives' && (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">Alternative Approaches</h3>
                                        <p className="text-slate-600 dark:text-slate-400">
                                            Here are other preprocessing techniques that could work, with trade-offs explained.
                                        </p>
                                    </div>

                                    {explanation.alternatives && explanation.alternatives.length > 0 ? (
                                        <div className="space-y-3">
                                            {explanation.alternatives.map((alt: any, idx: number) => (
                                                <div key={idx} className="p-4 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg hover:border-blue-300 dark:hover:border-blue-600 transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <span className="font-semibold text-slate-900 dark:text-white">
                                                            {alt.action?.replace(/_/g, ' ').toUpperCase() || 'Unknown Action'}
                                                        </span>
                                                        <span className="text-sm px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full font-medium">
                                                            {(alt.confidence * 100).toFixed(0)}% confidence
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="p-6 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg text-center">
                                            <p className="text-slate-500 dark:text-slate-400">No alternative approaches available for this decision.</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {activeTab === 'impact' && (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">Expected Impact</h3>
                                        <p className="text-slate-600 dark:text-slate-400">
                                            How this preprocessing decision will affect your model's performance.
                                        </p>
                                    </div>
                                    <div className="p-6 bg-gradient-to-br from-green-50 to-blue-50 dark:from-slate-800 dark:to-slate-800 border border-green-200 dark:border-slate-700 rounded-lg">
                                        <p className="text-slate-700 dark:text-slate-300">
                                            Impact analysis will be enhanced in future updates with predicted performance metrics and data quality improvements.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-12">
                            <p className="text-slate-600 dark:text-slate-400">No explanation available</p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                    <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                        <span>This explanation is AI-generated. Always validate with domain knowledge.</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg font-medium transition-all shadow-md hover:shadow-lg"
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
        .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-slate-900 dark:text-white mt-4 mb-2">$1</h3>')
        .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-slate-900 dark:text-white mt-6 mb-3">$1</h2>')
        .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold text-slate-900 dark:text-white mt-8 mb-4">$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-900 dark:text-white font-semibold">$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em class="text-slate-700 dark:text-slate-300">$1</em>')
        // Lists
        .replace(/^\* (.*$)/gim, '<li class="ml-4 text-slate-700 dark:text-slate-300">$1</li>')
        .replace(/^- (.*$)/gim, '<li class="ml-4 text-slate-700 dark:text-slate-300">$1</li>')
        // Line breaks
        .replace(/\n\n/g, '<br/><br/>')
        .replace(/\n/g, '<br/>');
}
