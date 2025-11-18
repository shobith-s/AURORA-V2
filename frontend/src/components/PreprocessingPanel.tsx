import { useState } from 'react';
import { Upload, Play, Download, CheckCircle, AlertCircle, Info } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ResultCard from './ResultCard';

export default function PreprocessingPanel() {
  const [columnData, setColumnData] = useState('');
  const [columnName, setColumnName] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handlePreprocess = async () => {
    if (!columnData.trim()) {
      toast.error('Please enter column data');
      return;
    }

    setIsProcessing(true);
    try {
      // Parse column data (comma or newline separated)
      const data = columnData
        .split(/[,\n]/)
        .map(v => v.trim())
        .filter(v => v !== '')
        .map(v => {
          // Try to parse as number
          const num = parseFloat(v);
          return isNaN(num) ? v : num;
        });

      const response = await axios.post('/api/preprocess', {
        column_data: data,
        column_name: columnName || 'unnamed_column',
        target_available: false,
        metadata: {}
      });

      setResult(response.data);
      toast.success('Column analyzed successfully!');
    } catch (error: any) {
      console.error('Error:', error);
      toast.error(error.response?.data?.detail || 'Failed to process column');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSampleData = (type: string) => {
    const samples = {
      skewed: '10,15,20,12,18,500,700,1000,25,30,2000',
      categorical: 'A,B,C,A,B,A,C,B,A,C',
      dates: '2024-01-01,2024-01-02,2024-01-03,2024-01-04',
      currency: '$10.99,$25.50,$100.00,$5.99,$75.25'
    };
    setColumnData(samples[type as keyof typeof samples] || '');
    setColumnName(type + '_sample');
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-800">Data Preprocessing</h2>
          <div className="flex gap-2">
            <button
              onClick={() => handleSampleData('skewed')}
              className="text-xs px-3 py-1 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100"
            >
              Skewed Data
            </button>
            <button
              onClick={() => handleSampleData('categorical')}
              className="text-xs px-3 py-1 bg-purple-50 text-purple-600 rounded-lg hover:bg-purple-100"
            >
              Categorical
            </button>
            <button
              onClick={() => handleSampleData('dates')}
              className="text-xs px-3 py-1 bg-green-50 text-green-600 rounded-lg hover:bg-green-100"
            >
              Dates
            </button>
          </div>
        </div>

        {/* Column Name */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Column Name
          </label>
          <input
            type="text"
            value={columnName}
            onChange={(e) => setColumnName(e.target.value)}
            placeholder="e.g., revenue, age, category"
            className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition"
          />
        </div>

        {/* Column Data */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Column Data (comma or newline separated)
          </label>
          <textarea
            value={columnData}
            onChange={(e) => setColumnData(e.target.value)}
            placeholder="Enter your data here... (e.g., 10, 20, 30, 100, 200)"
            className="w-full h-32 px-4 py-2 rounded-lg border border-slate-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition font-mono text-sm"
          />
          <p className="mt-1 text-xs text-slate-500">
            💡 Tip: Paste data from CSV or Excel directly
          </p>
        </div>

        {/* Action Button */}
        <button
          onClick={handlePreprocess}
          disabled={isProcessing}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {isProcessing ? (
            <>
              <div className="loading-dots flex gap-1">
                <span className="w-2 h-2 bg-white rounded-full"></span>
                <span className="w-2 h-2 bg-white rounded-full"></span>
                <span className="w-2 h-2 bg-white rounded-full"></span>
              </div>
              Processing...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Analyze & Recommend
            </>
          )}
        </button>
      </div>

      {/* Result Section */}
      {result && <ResultCard result={result} />}

      {/* Info Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-card p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-800 text-sm">Privacy-First</h3>
              <p className="text-xs text-slate-600 mt-1">
                Your data is processed in real-time and never stored
              </p>
            </div>
          </div>
        </div>

        <div className="glass-card p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <Info className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-800 text-sm">Lightning Fast</h3>
              <p className="text-xs text-slate-600 mt-1">
                Sub-millisecond decisions with symbolic rules
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
