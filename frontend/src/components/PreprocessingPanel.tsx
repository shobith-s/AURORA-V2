import { useState } from 'react';
import { Upload, Play, Download, CheckCircle, AlertCircle, Info, FileSpreadsheet, X } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ResultCard from './ResultCard';

interface BatchResults {
  results: Record<string, any>;
  summary: {
    total_columns: number;
    processed_columns: number;
    avg_confidence: number;
  };
}

export default function PreprocessingPanel() {
  const [mode, setMode] = useState<'single' | 'file'>('single');
  const [columnData, setColumnData] = useState('');
  const [columnName, setColumnName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [batchResults, setBatchResults] = useState<BatchResults | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        toast.error('Please upload a CSV file');
        return;
      }
      setSelectedFile(file);
      setResult(null);
      setBatchResults(null);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setBatchResults(null);
  };

  const parseCSV = (text: string): Record<string, any[]> => {
    const lines = text.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('CSV must have at least a header row and one data row');
    }

    const headers = lines[0].split(',').map(h => h.trim());
    const columns: Record<string, any[]> = {};

    headers.forEach(header => {
      columns[header] = [];
    });

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim());
      headers.forEach((header, index) => {
        const value = values[index] || '';
        // Try to parse as number
        if (value === '') {
          columns[header].push(null);
        } else {
          const num = parseFloat(value);
          columns[header].push(isNaN(num) ? value : num);
        }
      });
    }

    return columns;
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      toast.error('Please select a CSV file');
      return;
    }

    setIsProcessing(true);
    try {
      const text = await selectedFile.text();
      const columns = parseCSV(text);

      const response = await axios.post('/api/batch', {
        columns,
        target_column: null
      });

      setBatchResults(response.data);
      toast.success(`Analyzed ${Object.keys(columns).length} columns successfully!`);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || error.message || 'Failed to process file');
    } finally {
      setIsProcessing(false);
    }
  };

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
      setBatchResults(null);
      toast.success('Column analyzed successfully!');
    } catch (error: any) {
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
    setMode('single');
  };

  return (
    <div className="space-y-6">
      {/* Mode Selector */}
      <div className="glass-card p-4">
        <div className="flex gap-4">
          <button
            onClick={() => setMode('single')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
              mode === 'single'
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                : 'bg-white/50 text-slate-600 hover:bg-white/80'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <Play className="w-5 h-5" />
              Single Column
            </div>
          </button>
          <button
            onClick={() => setMode('file')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
              mode === 'file'
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                : 'bg-white/50 text-slate-600 hover:bg-white/80'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <FileSpreadsheet className="w-5 h-5" />
              Upload CSV File
            </div>
          </button>
        </div>
      </div>

      {/* Input Section */}
      <div className="glass-card p-6">
        {mode === 'single' ? (
          <>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-slate-800">Single Column Analysis</h2>
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
                  Analyze Column
                </>
              )}
            </button>
          </>
        ) : (
          <>
            <h2 className="text-xl font-bold text-slate-800 mb-4">CSV File Upload</h2>

            {!selectedFile ? (
              <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                <label className="cursor-pointer">
                  <span className="text-slate-700 font-medium">
                    Click to upload or drag and drop
                  </span>
                  <p className="text-sm text-slate-500 mt-1">CSV files only</p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <FileSpreadsheet className="w-8 h-8 text-blue-600" />
                    <div>
                      <p className="font-medium text-slate-800">{selectedFile.name}</p>
                      <p className="text-sm text-slate-600">
                        {(selectedFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={removeFile}
                    className="p-2 hover:bg-blue-100 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-slate-600" />
                  </button>
                </div>

                <button
                  onClick={handleFileUpload}
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
                      Analyzing CSV...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Analyze All Columns
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Single Column Result */}
      {result && !batchResults && <ResultCard result={result} />}

      {/* Batch Results */}
      {batchResults && (
        <div className="space-y-4">
          {/* Summary Card */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Analysis Summary</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {batchResults.summary.total_columns}
                </div>
                <div className="text-sm text-slate-600 mt-1">Total Columns</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {batchResults.summary.processed_columns}
                </div>
                <div className="text-sm text-slate-600 mt-1">Processed</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {(batchResults.summary.avg_confidence * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-slate-600 mt-1">Avg Confidence</div>
              </div>
            </div>
          </div>

          {/* Results for Each Column */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">
              Column Recommendations
            </h3>
            <div className="space-y-4">
              {Object.entries(batchResults.results).map(([columnName, columnResult]) => (
                <div key={columnName} className="border border-slate-200 rounded-lg p-4 bg-white/50">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-slate-800">{columnName}</h4>
                      <p className="text-sm text-slate-600 mt-1">{columnResult.explanation}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                        columnResult.confidence >= 0.9
                          ? 'bg-green-100 text-green-700'
                          : columnResult.confidence >= 0.7
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {(columnResult.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-600">Action:</span>
                      <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-sm font-medium">
                        {columnResult.action}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-600">Source:</span>
                      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-lg text-sm font-medium">
                        {columnResult.source}
                      </span>
                    </div>
                  </div>

                  {columnResult.alternatives && columnResult.alternatives.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <p className="text-xs text-slate-600 mb-2">Alternative actions:</p>
                      <div className="flex gap-2 flex-wrap">
                        {columnResult.alternatives.slice(0, 3).map((alt: any, idx: number) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-slate-100 text-slate-600 rounded text-xs"
                          >
                            {alt.action} ({(alt.confidence * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

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
