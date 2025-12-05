import { useState, useMemo } from 'react';
import {
    Search, X, Ruler, Zap, Scissors, Hash,
    Type, Database, Wand2, Filter, Check,
    Calculator, Globe, Calendar,
    Trash2, Percent
} from 'lucide-react';

interface ActionLibraryModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectAction: (action: string) => void;
    currentAction?: string;
}

type Category = 'All' | 'Scaling' | 'Transformation' | 'Outlier Handling' | 'Encoding' | 'Null Filling' | 'Type Conversion' | 'Feature Engineering' | 'Data Quality';

interface ActionItem {
    id: string;
    label: string;
    category: Category;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
    popular?: boolean;
}

const ACTIONS: ActionItem[] = [
    // Scaling
    { id: 'standard_scale', label: 'Standard Scale', category: 'Scaling', description: 'Zero mean, unit variance (Z-score). Best for normal distributions.', icon: Ruler, popular: true },
    { id: 'minmax_scale', label: 'Min-Max Scale', category: 'Scaling', description: 'Scales data to [0, 1] range. Preserves zero values.', icon: Ruler },
    { id: 'robust_scale', label: 'Robust Scale', category: 'Scaling', description: 'Scales using statistics that are robust to outliers (IQR).', icon: Ruler, popular: true },
    { id: 'maxabs_scale', label: 'MaxAbs Scale', category: 'Scaling', description: 'Scales by dividing by max absolute value. Preserves sparsity.', icon: Ruler },
    { id: 'normalize_l1', label: 'Normalize L1', category: 'Scaling', description: 'Scales samples to have unit norm (sum of values = 1).', icon: Ruler },
    { id: 'normalize_l2', label: 'Normalize L2', category: 'Scaling', description: 'Scales samples to have unit norm (sum of squares = 1).', icon: Ruler },

    // Transformation
    { id: 'log_transform', label: 'Log Transform', category: 'Transformation', description: 'Applies natural log. Good for right-skewed data.', icon: Zap, popular: true },
    { id: 'log1p_transform', label: 'Log1p Transform', category: 'Transformation', description: 'Log(1+x). Safe for zero values.', icon: Zap },
    { id: 'sqrt_transform', label: 'Sqrt Transform', category: 'Transformation', description: 'Square root. Milder than log transform.', icon: Zap },
    { id: 'box_cox', label: 'Box-Cox', category: 'Transformation', description: 'Power transform to make data normal. Requires positive values.', icon: Zap },
    { id: 'yeo_johnson', label: 'Yeo-Johnson', category: 'Transformation', description: 'Power transform like Box-Cox but supports negative values.', icon: Zap },
    { id: 'quantile_transform', label: 'Quantile Transform', category: 'Transformation', description: 'Maps data to a uniform or normal distribution.', icon: Zap },
    { id: 'power_transform', label: 'Power Transform', category: 'Transformation', description: 'Apply a power transform to make data more Gaussian.', icon: Zap },

    // Outlier Handling
    { id: 'clip_outliers', label: 'Clip Outliers', category: 'Outlier Handling', description: 'Caps values at specific percentiles (e.g., 1st and 99th).', icon: Scissors, popular: true },
    { id: 'winsorize', label: 'Winsorize', category: 'Outlier Handling', description: 'Replaces outliers with the nearest non-outlier value.', icon: Scissors },
    { id: 'cap_floor_outliers', label: 'Cap/Floor', category: 'Outlier Handling', description: 'Caps values based on mean +/- 3 standard deviations.', icon: Scissors },
    { id: 'remove_outliers', label: 'Remove Outliers', category: 'Outlier Handling', description: 'Removes rows containing outliers.', icon: Trash2 },

    // Encoding
    { id: 'onehot_encode', label: 'One-Hot Encode', category: 'Encoding', description: 'Creates binary columns for each category.', icon: Hash, popular: true },
    { id: 'label_encode', label: 'Label Encode', category: 'Encoding', description: 'Encodes categories as integers (0, 1, 2...).', icon: Hash },
    { id: 'ordinal_encode', label: 'Ordinal Encode', category: 'Encoding', description: 'Encodes categories based on order/rank.', icon: Hash },
    { id: 'frequency_encode', label: 'Frequency Encode', category: 'Encoding', description: 'Replaces category with its frequency count.', icon: Hash },
    { id: 'binary_encode', label: 'Binary Encode', category: 'Encoding', description: 'Encodes categories into binary digits.', icon: Hash },
    { id: 'target_encode', label: 'Target Encode', category: 'Encoding', description: 'Encodes categories using target mean.', icon: Hash },
    { id: 'hash_encode', label: 'Hash Encode', category: 'Encoding', description: 'Hashes categories to a fixed number of columns.', icon: Hash },

    // Null Filling
    { id: 'fill_null_mean', label: 'Fill Mean', category: 'Null Filling', description: 'Fills missing values with the column mean.', icon: Database, popular: true },
    { id: 'fill_null_median', label: 'Fill Median', category: 'Null Filling', description: 'Fills missing values with the column median.', icon: Database },
    { id: 'fill_null_mode', label: 'Fill Mode', category: 'Null Filling', description: 'Fills missing values with the most frequent value.', icon: Database },
    { id: 'fill_null_forward', label: 'Forward Fill', category: 'Null Filling', description: 'Propagates last valid observation forward.', icon: Database },
    { id: 'fill_null_backward', label: 'Backward Fill', category: 'Null Filling', description: 'Propagates next valid observation backward.', icon: Database },
    { id: 'fill_null_interpolate', label: 'Interpolate', category: 'Null Filling', description: 'Interpolates values linearly.', icon: Database },
    { id: 'drop_if_mostly_null', label: 'Drop if Mostly Null', category: 'Null Filling', description: 'Drops column if >50% values are missing.', icon: Trash2 },

    // Type Conversion
    { id: 'parse_numeric', label: 'Parse Numeric', category: 'Type Conversion', description: 'Converts text to numbers, handling errors.', icon: Type },
    { id: 'parse_datetime', label: 'Parse Datetime', category: 'Type Conversion', description: 'Converts text to datetime objects.', icon: Calendar },
    { id: 'parse_boolean', label: 'Parse Boolean', category: 'Type Conversion', description: 'Converts text (yes/no, true/false) to boolean.', icon: Type },
    { id: 'datetime_extract_year', label: 'Extract Year', category: 'Type Conversion', description: 'Extracts year component from datetime.', icon: Calendar },

    // Feature Engineering
    { id: 'binning_equal_width', label: 'Binning (Width)', category: 'Feature Engineering', description: 'Divides range into equal-sized bins.', icon: Calculator },
    { id: 'binning_equal_freq', label: 'Binning (Freq)', category: 'Feature Engineering', description: 'Divides data into bins with equal count.', icon: Calculator },
    { id: 'text_clean', label: 'Text Clean', category: 'Feature Engineering', description: 'Removes special chars, extra spaces.', icon: Wand2 },
    { id: 'text_lowercase', label: 'Text Lowercase', category: 'Feature Engineering', description: 'Converts all text to lowercase.', icon: Type },
    { id: 'currency_normalize', label: 'Currency Normalize', category: 'Feature Engineering', description: 'Cleans currency symbols and formatting.', icon: Globe },
    { id: 'percentage_to_decimal', label: 'Percent to Decimal', category: 'Feature Engineering', description: 'Converts "50%" to 0.5.', icon: Percent },

    // Data Quality
    { id: 'keep_as_is', label: 'Keep As Is', category: 'Data Quality', description: 'Do not apply any preprocessing.', icon: Check, popular: true },
    { id: 'drop_column', label: 'Drop Column', category: 'Data Quality', description: 'Remove this column from the dataset.', icon: Trash2 },
];

export default function ActionLibraryModal({ isOpen, onClose, onSelectAction, currentAction }: ActionLibraryModalProps) {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<Category>('All');

    const filteredActions = useMemo(() => {
        return ACTIONS.filter(action => {
            const matchesSearch = action.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
                action.description.toLowerCase().includes(searchQuery.toLowerCase());
            const matchesCategory = selectedCategory === 'All' || action.category === selectedCategory;
            return matchesSearch && matchesCategory;
        });
    }, [searchQuery, selectedCategory]);

    const categories: Category[] = ['All', 'Scaling', 'Transformation', 'Outlier Handling', 'Encoding', 'Null Filling', 'Type Conversion', 'Feature Engineering', 'Data Quality'];

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background-dark/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-brand-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">

                {/* Header */}
                <div className="p-6 border-b border-slate-100 bg-brand-white z-10">
                    <div className="flex items-center justify-between mb-4">
                        <div>
                            <h2 className="text-2xl font-bold text-brand-black flex items-center gap-2">
                                <Wand2 className="w-6 h-6 text-primary" />
                                Smart Action Library
                            </h2>
                            <p className="text-brand-cool-gray text-sm mt-1">Browse and select the perfect preprocessing transformation</p>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-background-muted rounded-full transition-colors">
                            <X className="w-6 h-6 text-brand-cool-gray" />
                        </button>
                    </div>

                    {/* Search Bar */}
                    <div className="relative">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-brand-cool-gray" />
                        <input
                            type="text"
                            placeholder="Search actions (e.g., 'log', 'scale', 'missing')..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-12 pr-4 py-3 bg-brand-white border border-brand-warm-gray rounded-xl focus:ring-2 focus:ring-primary focus:border-primary outline-none transition-all text-lg"
                            autoFocus
                        />
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex flex-1 overflow-hidden">

                    {/* Sidebar - Categories */}
                    <div className="w-64 bg-brand-white border-r border-brand-warm-gray overflow-y-auto p-4 hidden md:block">
                        <h3 className="text-xs font-semibold text-brand-cool-gray uppercase tracking-wider mb-3 px-2">Categories</h3>
                        <div className="space-y-1">
                            {categories.map(category => (
                                <button
                                    key={category}
                                    onClick={() => setSelectedCategory(category)}
                                    className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all ${selectedCategory === category
                                            ? 'bg-primary/20 text-primary-dark shadow-sm'
                                            : 'text-foreground-muted hover:bg-background-muted/50'
                                        }`}
                                >
                                    <div className="flex items-center justify-between">
                                        {category}
                                        {category === 'All' && <span className="text-xs bg-background-muted text-foreground-muted px-1.5 py-0.5 rounded-full">{ACTIONS.length}</span>}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Action Grid */}
                    <div className="flex-1 overflow-y-auto p-6 bg-brand-white/30">
                        {filteredActions.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-full text-brand-cool-gray">
                                <Filter className="w-12 h-12 mb-4 opacity-20" />
                                <p className="text-lg font-medium">No actions found</p>
                                <p className="text-sm">Try a different search term</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {filteredActions.map(action => (
                                    <button
                                        key={action.id}
                                        onClick={() => onSelectAction(action.id)}
                                        className={`group relative flex items-start gap-4 p-4 rounded-xl border text-left transition-all duration-200 hover:shadow-md ${currentAction === action.id
                                                ? 'bg-primary/10 border-primary ring-1 ring-blue-500'
                                                : 'bg-brand-white border-brand-warm-gray hover:border-primary hover:bg-primary/10'
                                            }`}
                                    >
                                        <div className={`p-3 rounded-lg ${currentAction === action.id ? 'bg-primary/30 text-primary-dark' : 'bg-background-muted text-foreground-muted group-hover:bg-primary/20 group-hover:text-primary'
                                            } transition-colors`}>
                                            <action.icon className="w-6 h-6" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className={`font-bold ${currentAction === action.id ? 'text-blue-800' : 'text-brand-black'}`}>
                                                    {action.label}
                                                </h4>
                                                {action.popular && (
                                                    <span className="px-1.5 py-0.5 bg-amber-100 text-amber-700 text-[10px] font-bold uppercase tracking-wide rounded-full">
                                                        Popular
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-sm text-brand-cool-gray leading-relaxed line-clamp-2">
                                                {action.description}
                                            </p>
                                            <div className="mt-2 flex items-center gap-2">
                                                <span className="text-[10px] px-2 py-0.5 bg-background-muted text-brand-cool-gray rounded-full border border-brand-warm-gray">
                                                    {action.category}
                                                </span>
                                            </div>
                                        </div>
                                        {currentAction === action.id && (
                                            <div className="absolute top-4 right-4">
                                                <Check className="w-5 h-5 text-primary" />
                                            </div>
                                        )}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-slate-100 bg-brand-white flex justify-between items-center text-sm text-brand-cool-gray">
                    <div className="flex items-center gap-4">
                        <span className="flex items-center gap-1"><span className="px-1.5 py-0.5 bg-background-muted rounded border border-brand-warm-gray text-xs font-mono">↑↓</span> Navigate</span>
                        <span className="flex items-center gap-1"><span className="px-1.5 py-0.5 bg-background-muted rounded border border-brand-warm-gray text-xs font-mono">Enter</span> Select</span>
                        <span className="flex items-center gap-1"><span className="px-1.5 py-0.5 bg-background-muted rounded border border-brand-warm-gray text-xs font-mono">Esc</span> Close</span>
                    </div>
                    <div>
                        Showing {filteredActions.length} actions
                    </div>
                </div>

            </div>
        </div>
    );
}
