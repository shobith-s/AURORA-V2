import { useState } from 'react';
import { Play, Save, Trash2, Code, Terminal, X, ChevronRight } from 'lucide-react';

interface ScriptIDEProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ScriptIDE({ isOpen, onClose }: ScriptIDEProps) {
    const [code, setCode] = useState(`# Custom Preprocessing Script
# You can access the dataframe as 'df'

def custom_preprocess(df):
    # Example: Create a new feature
    # df['bmi'] = df['weight'] / (df['height'] ** 2)
    
    # Example: Filter rows
    # df = df[df['price'] > 0]
    
    return df
`);
    const [output, setOutput] = useState<string>('');
    const [isRunning, setIsRunning] = useState(false);

    const handleRun = () => {
        setIsRunning(true);
        setOutput('Running script...');

        // Simulate execution
        setTimeout(() => {
            setIsRunning(false);
            setOutput(`> Script executed successfully\n> Processed 800 rows\n> Added 1 new feature: 'bmi'\n> Memory usage: 12.4 MB`);
        }, 1500);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="w-full max-w-2xl bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col h-full animate-in slide-in-from-right duration-300">

                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-orange-600 rounded-lg flex items-center justify-center shadow-lg">
                            <Code className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-slate-900">Custom Script IDE</h2>
                            <p className="text-xs text-slate-400">Python Preprocessing Environment</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Toolbar */}
                <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-b border-slate-700">
                    <button
                        onClick={handleRun}
                        disabled={isRunning}
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isRunning ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                        Run Cell
                    </button>
                    <div className="w-px h-6 bg-slate-700 mx-2" />
                    <button className="p-2 hover:bg-slate-700 rounded-md text-slate-400 hover:text-white transition-colors" title="Save Script">
                        <Save className="w-4 h-4" />
                    </button>
                    <button className="p-2 hover:bg-slate-700 rounded-md text-slate-400 hover:text-red-400 transition-colors" title="Clear">
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>

                {/* Editor Area */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <div className="flex-1 relative bg-[#1e1e1e] font-mono text-sm">
                        <div className="absolute left-0 top-0 bottom-0 w-12 bg-[#2d2d2d] border-r border-[#404040] flex flex-col items-end py-4 pr-2 text-slate-500 select-none">
                            {code.split('\n').map((_, i) => (
                                <div key={i} className="leading-6">{i + 1}</div>
                            ))}
                        </div>
                        <textarea
                            value={code}
                            onChange={(e) => setCode(e.target.value)}
                            className="absolute inset-0 left-12 w-[calc(100%-3rem)] h-full bg-transparent text-slate-200 p-4 resize-none focus:outline-none leading-6 selection:bg-blue-500/30"
                            spellCheck={false}
                        />
                    </div>

                    {/* Output Console */}
                    <div className="h-1/3 bg-slate-950 border-t border-slate-700 flex flex-col">
                        <div className="flex items-center gap-2 px-4 py-2 bg-slate-900 border-b border-slate-800 text-xs font-medium text-slate-400 uppercase tracking-wider">
                            <Terminal className="w-3 h-3" />
                            Console Output
                        </div>
                        <div className="flex-1 p-4 font-mono text-sm text-green-400 overflow-auto whitespace-pre-wrap">
                            {output || <span className="text-slate-600 italic">Ready to execute...</span>}
                        </div>
                    </div>
                </div>

                {/* Status Bar */}
                <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 text-xs text-slate-500 flex justify-between">
                    <span>Python 3.9.13</span>
                    <span>Ln {code.split('\n').length}, Col 1</span>
                </div>
            </div>
        </div>
    );
}
