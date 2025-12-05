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
            <div className="relative w-full max-w-4xl max-h-[90vh] overflow-hidden bg-brand-white dark:bg-background-dark border border-brand-warm-gray dark:border-border-dark rounded-2xl shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-brand-warm-gray dark:border-border-dark bg-primary/10 dark:bg-background-dark">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary rounded-lg shadow-lg">
                            <BookOpen className="w-6 h-6 text-brand-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-brand-black dark:text-brand-white">Enhanced Explanation</h2>
                            <p className="text-sm text-foreground-muted dark:text-brand-cool-gray">Deep dive into the decision for "{columnName}"</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-background-muted dark:hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-foreground-muted dark:text-brand-cool-gray" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 px-6 pt-4 border-b border-brand-warm-gray dark:border-border-dark bg-brand-white dark:bg-background-dark/50">
                    <button
                        onClick={() => setActiveTab('explanation')}
                        className={`px-4 py-2 rounded-t-lg font-medium transition-all ${activeTab === 'explanation'
                            ? 'bg-brand-white dark:bg-background-dark text-primary dark:text-primary border-b-2 border-primary dark:border-primary shadow-sm'
                            : 'text-foreground-muted dark:text-brand-cool-gray hover:text-brand-black dark:hover:text-brand-warm-gray hover:bg-background-muted dark:hover:bg-slate-700/50'
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
                            ? 'bg-brand-white dark:bg-background-dark text-primary dark:text-primary border-b-2 border-primary dark:border-primary shadow-sm'
                            : 'text-foreground-muted dark:text-brand-cool-gray hover:text-brand-black dark:hover:text-brand-warm-gray hover:bg-background-muted dark:hover:bg-slate-700/50'
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
                            ? 'bg-brand-white dark:bg-background-dark text-primary dark:text-primary border-b-2 border-primary dark:border-primary shadow-sm'
                            : 'text-foreground-muted dark:text-brand-cool-gray hover:text-brand-black dark:hover:text-brand-warm-gray hover:bg-background-muted dark:hover:bg-slate-700/50'
                            }`}
                    >
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            Impact
                        </div>
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)] bg-brand-white dark:bg-background-dark">
                    {isLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <div className="flex flex-col items-center gap-3">
                                <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                                <p className="text-foreground-muted dark:text-brand-cool-gray">Generating detailed explanation...</p>
                            </div>
                        </div>
                    ) : explanation ? (
                        <div className="space-y-6">
                            {activeTab === 'explanation' && (
                                <div className="prose prose-slate dark:prose-invert max-w-none">
                                    {/* Decision Summary Card */}
                                    <div className="p-6 bg-primary/10 dark:bg-background-dark border border-primary/50 dark:border-border-dark rounded-xl mb-6 shadow-sm">
                                        <div className="flex items-center justify-between mb-4">
                                            <h3 className="text-lg font-semibold text-brand-black dark:text-brand-white m-0">
                                                Recommended Action
                                            </h3>
                                            <span className="px-4 py-1.5 bg-primary text-brand-white rounded-full text-sm font-semibold shadow-md">
                                                {(explanation.decision.confidence * 100).toFixed(0)}% Confident
                                            </span>
                                        </div>
                                        <p className="text-3xl font-bold bg-primary dark:text-primary bg-clip-text text-transparent m-0 mb-2">
                                            {explanation.decision.action.replace(/_/g, ' ').toUpperCase()}
                                        </p>
                                        <div className="flex items-center gap-2 mt-3">
                                            <span className="text-xs px-2 py-1 bg-background-muted dark:bg-slate-700 text-foreground dark:text-foreground-muted rounded font-medium">
                                                Source: {explanation.decision.source}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Markdown Explanation */}
                                    <div
                                        className="text-foreground dark:text-foreground-muted leading-relaxed prose-headings:text-brand-black dark:prose-headings:text-brand-white prose-strong:text-brand-black dark:prose-strong:text-brand-white prose-p:text-foreground dark:prose-p:text-foreground-muted"
                                        dangerouslySetInnerHTML={{ __html: formatMarkdown(explanation.markdown_report) }}
                                    />
                                </div>
                            )}

                            {activeTab === 'alternatives' && (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-xl font-bold text-brand-black dark:text-brand-white mb-2">Alternative Approaches</h3>
                                        <p className="text-foreground-muted dark:text-brand-cool-gray">
                                            Here are other preprocessing techniques that could work, with trade-offs explained.
                                        </p>
                                    </div>

                                    {explanation.alternatives && explanation.alternatives.length > 0 ? (
                                        <div className="space-y-3">
                                            {explanation.alternatives.map((alt: any, idx: number) => (
                                                <div key={idx} className="p-4 bg-brand-white dark:bg-background-dark/50 border border-brand-warm-gray dark:border-border-dark rounded-lg hover:border-primary dark:hover:border-primary transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <span className="font-semibold text-brand-black dark:text-brand-white">
                                                            {alt.action?.replace(/_/g, ' ').toUpperCase() || 'Unknown Action'}
                                                        </span>
                                                        <span className="text-sm px-3 py-1 bg-primary/20 dark:bg-blue-900/30 text-primary-dark dark:text-blue-300 rounded-full font-medium">
                                                            {(alt.confidence * 100).toFixed(0)}% confidence
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="p-6 bg-brand-white dark:bg-background-dark/50 border border-brand-warm-gray dark:border-border-dark rounded-lg text-center">
                                            <p className="text-brand-cool-gray dark:text-brand-cool-gray">No alternative approaches available for this decision.</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {activeTab === 'impact' && (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-xl font-bold text-brand-black dark:text-brand-white mb-2">Expected Impact</h3>
                                        <p className="text-foreground-muted dark:text-brand-cool-gray">
                                            How this preprocessing decision will affect your model's performance.
                                        </p>
                                    </div>
                                    <div className="p-6 bg-success/10 dark:bg-background-dark border border-success/30 dark:border-border-dark rounded-lg">
                                        <p className="text-foreground dark:text-foreground-muted">
                                            Impact analysis will be enhanced in future updates with predicted performance metrics and data quality improvements.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-12">
                            <p className="text-foreground-muted dark:text-brand-cool-gray">No explanation available</p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-brand-warm-gray dark:border-border-dark bg-brand-white dark:bg-background-dark/50">
                    <div className="flex items-center gap-2 text-sm text-foreground-muted dark:text-brand-cool-gray">
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                        <span>This explanation is AI-generated. Always validate with domain knowledge.</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-primary hover:bg-primary-hover text-brand-white rounded-lg font-medium transition-all shadow-md hover:shadow-lg"
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
        .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-brand-black dark:text-brand-white mt-4 mb-2">$1</h3>')
        .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-brand-black dark:text-brand-white mt-6 mb-3">$1</h2>')
        .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold text-brand-black dark:text-brand-white mt-8 mb-4">$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-brand-black dark:text-brand-white font-semibold">$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em class="text-foreground dark:text-foreground-muted">$1</em>')
        // Lists
        .replace(/^\* (.*$)/gim, '<li class="ml-4 text-foreground dark:text-foreground-muted">$1</li>')
        .replace(/^- (.*$)/gim, '<li class="ml-4 text-foreground dark:text-foreground-muted">$1</li>')
        // Line breaks
        .replace(/\n\n/g, '<br/><br/>')
        .replace(/\n/g, '<br/>');
}
