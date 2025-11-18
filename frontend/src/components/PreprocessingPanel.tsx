import { useState } from 'react';
import { Upload, Play, Download, CheckCircle, AlertCircle, Info, FileSpreadsheet, X, Edit2, Activity, TrendingUp, TrendingDown, MinusCircle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ResultCard from './ResultCard';

interface ColumnHealthMetrics {
  column_name: string;
  data_type: string;
  health_score: number;
  anomalies: string[];
  null_count: number;
  null_pct: number;
  duplicate_count: number;
  duplicate_pct: number;
  unique_count: number;
  unique_ratio: number;
  outlier_count?: number;
  outlier_pct?: number;
  skewness?: number;
  kurtosis?: number;
  mean?: number;
  std?: number;
  cv?: number;
  cardinality?: number;
  is_imbalanced?: boolean;
  severity: 'healthy' | 'warning' | 'critical';
}

interface BatchHealthResponse {
  overall_health_score: number;
  healthy_columns: number;
  warning_columns: number;
  critical_columns: number;
  column_health: Record<string, ColumnHealthMetrics>;
}

interface BatchResults {
  results: Record<string, any>;
  summary: {
    total_columns: number;
    processed_columns: number;
    avg_confidence: number;
  };
  health?: BatchHealthResponse;
}

export default function PreprocessingPanel() {
  const [mode, setMode] = useState<'single' | 'file'>('single');
  const [columnData, setColumnData] = useState('');
  const [columnName, setColumnName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [batchResults, setBatchResults] = useState<BatchResults | null>(null);
  const [showCorrectionFor, setShowCorrectionFor] = useState<string | null>(null);
  const [correctActions, setCorrectActions] = useState<Record<string, string>>({});
  const [isSubmittingCorrection, setIsSubmittingCorrection] = useState<Record<string, boolean>>({});
  const [expandedHealthColumn, setExpandedHealthColumn] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [originalData, setOriginalData] = useState<Record<string, any[]> | null>(null);

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

      // Store original data for pipeline execution
      setOriginalData(columns);

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

  const handleExecutePipeline = async () => {
    if (!batchResults || !originalData) {
      toast.error('No data to process');
      return;
    }

    setIsExecuting(true);
    try {
      // Build actions map from results
      const actions: Record<string, string> = {};
      Object.entries(batchResults.results).forEach(([colName, result]: [string, any]) => {
        actions[colName] = result.action;
      });

      const response = await axios.post('/api/execute', {
        columns: originalData,
        actions
      });

      const data = response.data;

      if (!data.success) {
        const errorCount = Object.keys(data.errors).length;
        toast.error(`Pipeline completed with ${errorCount} errors. Check console for details.`);
        console.error('Pipeline errors:', data.errors);
      } else {
        toast.success('Pipeline executed successfully!');
      }

      // Convert to CSV and download
      downloadCSV(data.processed_data);

    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to execute pipeline');
      console.error('Pipeline execution error:', error);
    } finally {
      setIsExecuting(false);
    }
  };

  const downloadCSV = (data: Record<string, any[]>) => {
    // Convert data to CSV format
    const columns = Object.keys(data);
    if (columns.length === 0) {
      toast.error('No data to download');
      return;
    }

    const rowCount = data[columns[0]].length;

    // Create CSV header
    let csv = columns.join(',') + '\n';

    // Create CSV rows
    for (let i = 0; i < rowCount; i++) {
      const row = columns.map(col => {
        const value = data[col][i];
        // Handle null/undefined
        if (value === null || value === undefined) return '';
        // Escape values with commas or quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csv += row.join(',') + '\n';
    }

    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', 'processed_data.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    toast.success('File downloaded successfully!');
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

  const handleBatchCorrection = async (columnName: string, wrongAction: string, confidence: number) => {
    const correctAction = correctActions[columnName];
    if (!correctAction) {
      toast.error('Please select the correct action');
      return;
    }

    setIsSubmittingCorrection(prev => ({ ...prev, [columnName]: true }));
    try {
      await axios.post('/api/correct', {
        column_data: [],
        column_name: columnName,
        wrong_action: wrongAction,
        correct_action: correctAction,
        confidence: confidence
      });

      toast.success(`Correction for "${columnName}" submitted! System is learning...`);
      setShowCorrectionFor(null);
      setCorrectActions(prev => {
        const newActions = { ...prev };
        delete newActions[columnName];
        return newActions;
      });
    } catch (error) {
      toast.error('Failed to submit correction');
    } finally {
      setIsSubmittingCorrection(prev => ({ ...prev, [columnName]: false }));
    }
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
                  {Object.values(batchResults.results).filter((r: any) => r.action === 'keep_as_is').length}
                </div>
                <div className="text-sm text-slate-600 mt-1">Healthy Columns</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {(batchResults.summary.avg_confidence * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-slate-600 mt-1">Avg Confidence</div>
              </div>
            </div>
            {Object.values(batchResults.results).filter((r: any) => r.action === 'keep_as_is').length === batchResults.summary.total_columns && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-700 text-center">
                  🎉 All columns are healthy! No preprocessing needed.
                </p>
              </div>
            )}

            {/* Execute Pipeline Button */}
            <div className="flex justify-center">
              <button
                onClick={handleExecutePipeline}
                disabled={isExecuting}
                className="btn-primary flex items-center gap-3 px-8 py-4 text-lg"
              >
                {isExecuting ? (
                  <>
                    <div className="loading-dots flex gap-1">
                      <span className="w-2 h-2 bg-white rounded-full"></span>
                      <span className="w-2 h-2 bg-white rounded-full"></span>
                      <span className="w-2 h-2 bg-white rounded-full"></span>
                    </div>
                    Processing & Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-6 h-6" />
                    Execute Pipeline & Download Results
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Data Health Dashboard */}
          {batchResults.health && (
            <div className="glass-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <Activity className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-bold text-slate-800">Data Health Analysis</h3>
              </div>

              {/* Overall Health Score */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 mb-6 border border-blue-100">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600 mb-1">Overall Dataset Health</p>
                    <div className="flex items-center gap-3">
                      <div className="text-4xl font-bold" style={{
                        color: batchResults.health.overall_health_score >= 80 ? '#10b981' :
                               batchResults.health.overall_health_score >= 50 ? '#f59e0b' : '#ef4444'
                      }}>
                        {batchResults.health.overall_health_score.toFixed(1)}
                      </div>
                      <div className="text-2xl text-slate-400">/100</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{batchResults.health.healthy_columns}</div>
                      <div className="text-xs text-slate-600">Healthy</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">{batchResults.health.warning_columns}</div>
                      <div className="text-xs text-slate-600">Warning</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">{batchResults.health.critical_columns}</div>
                      <div className="text-xs text-slate-600">Critical</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Column Health Details */}
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-slate-700 mb-3">Column-Level Health Details</h4>
                {Object.values(batchResults.health.column_health).map((health: ColumnHealthMetrics) => (
                  <div key={health.column_name} className={`border rounded-lg p-4 transition-all ${
                    health.severity === 'healthy' ? 'border-green-200 bg-green-50/30' :
                    health.severity === 'warning' ? 'border-yellow-200 bg-yellow-50/30' :
                    'border-red-200 bg-red-50/30'
                  }`}>
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-3 flex-1">
                        {health.severity === 'healthy' && <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />}
                        {health.severity === 'warning' && <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0" />}
                        {health.severity === 'critical' && <X className="w-5 h-5 text-red-600 flex-shrink-0" />}
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <h5 className="font-semibold text-slate-800">{health.column_name}</h5>
                            <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs">
                              {health.data_type}
                            </span>
                          </div>
                          {health.anomalies.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {health.anomalies.map((anomaly, idx) => (
                                <span key={idx} className={`text-xs px-2 py-0.5 rounded ${
                                  health.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                  health.severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                                  'bg-blue-100 text-blue-700'
                                }`}>
                                  {anomaly}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${
                            health.severity === 'healthy' ? 'text-green-600' :
                            health.severity === 'warning' ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>
                            {health.health_score.toFixed(0)}
                          </div>
                          <div className="text-xs text-slate-500">health</div>
                        </div>
                        <button
                          onClick={() => setExpandedHealthColumn(expandedHealthColumn === health.column_name ? null : health.column_name)}
                          className="px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded transition"
                        >
                          {expandedHealthColumn === health.column_name ? 'Hide' : 'Details'}
                        </button>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {expandedHealthColumn === health.column_name && (
                      <div className="mt-4 pt-4 border-t border-slate-200">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {/* Quality Metrics */}
                          <div className="bg-white rounded-lg p-3 shadow-sm">
                            <div className="text-xs text-slate-600 mb-1">Null Values</div>
                            <div className="text-lg font-semibold text-slate-800">
                              {health.null_pct > 0 ? `${(health.null_pct * 100).toFixed(1)}%` : 'None'}
                            </div>
                            <div className="text-xs text-slate-500">{health.null_count} rows</div>
                          </div>

                          <div className="bg-white rounded-lg p-3 shadow-sm">
                            <div className="text-xs text-slate-600 mb-1">Unique Values</div>
                            <div className="text-lg font-semibold text-slate-800">{health.unique_count}</div>
                            <div className="text-xs text-slate-500">{(health.unique_ratio * 100).toFixed(1)}% unique</div>
                          </div>

                          <div className="bg-white rounded-lg p-3 shadow-sm">
                            <div className="text-xs text-slate-600 mb-1">Duplicates</div>
                            <div className="text-lg font-semibold text-slate-800">
                              {health.duplicate_pct > 0 ? `${(health.duplicate_pct * 100).toFixed(1)}%` : 'None'}
                            </div>
                            <div className="text-xs text-slate-500">{health.duplicate_count} rows</div>
                          </div>

                          {/* Numeric-specific */}
                          {health.data_type === 'numeric' && (
                            <>
                              {health.outlier_pct !== null && health.outlier_pct !== undefined && (
                                <div className="bg-white rounded-lg p-3 shadow-sm">
                                  <div className="text-xs text-slate-600 mb-1">Outliers</div>
                                  <div className="text-lg font-semibold text-slate-800">
                                    {(health.outlier_pct * 100).toFixed(1)}%
                                  </div>
                                  <div className="text-xs text-slate-500">{health.outlier_count} values</div>
                                </div>
                              )}
                              {health.skewness !== null && health.skewness !== undefined && (
                                <div className="bg-white rounded-lg p-3 shadow-sm">
                                  <div className="text-xs text-slate-600 mb-1">Skewness</div>
                                  <div className="text-lg font-semibold text-slate-800 flex items-center gap-1">
                                    {health.skewness.toFixed(2)}
                                    {health.skewness > 0.5 && <TrendingUp className="w-4 h-4 text-orange-500" />}
                                    {health.skewness < -0.5 && <TrendingDown className="w-4 h-4 text-blue-500" />}
                                    {Math.abs(health.skewness) <= 0.5 && <MinusCircle className="w-4 h-4 text-green-500" />}
                                  </div>
                                  <div className="text-xs text-slate-500">
                                    {Math.abs(health.skewness) < 0.5 ? 'Normal' : Math.abs(health.skewness) < 1 ? 'Moderate' : 'High'}
                                  </div>
                                </div>
                              )}
                              {health.mean !== null && health.mean !== undefined && (
                                <div className="bg-white rounded-lg p-3 shadow-sm">
                                  <div className="text-xs text-slate-600 mb-1">Mean ± Std</div>
                                  <div className="text-sm font-semibold text-slate-800">
                                    {health.mean.toFixed(2)} ± {health.std?.toFixed(2) || '0'}
                                  </div>
                                  {health.cv && <div className="text-xs text-slate-500">CV: {health.cv.toFixed(2)}</div>}
                                </div>
                              )}
                            </>
                          )}

                          {/* Categorical-specific */}
                          {health.data_type === 'categorical' && (
                            <>
                              {health.cardinality !== null && health.cardinality !== undefined && (
                                <div className="bg-white rounded-lg p-3 shadow-sm">
                                  <div className="text-xs text-slate-600 mb-1">Cardinality</div>
                                  <div className="text-lg font-semibold text-slate-800">{health.cardinality}</div>
                                  <div className="text-xs text-slate-500">categories</div>
                                </div>
                              )}
                              {health.is_imbalanced !== null && health.is_imbalanced !== undefined && (
                                <div className="bg-white rounded-lg p-3 shadow-sm">
                                  <div className="text-xs text-slate-600 mb-1">Balance</div>
                                  <div className="text-lg font-semibold text-slate-800">
                                    {health.is_imbalanced ? 'Imbalanced' : 'Balanced'}
                                  </div>
                                  <div className="text-xs text-slate-500">
                                    {health.is_imbalanced ? 'Skewed distribution' : 'Good distribution'}
                                  </div>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results for Each Column */}
          {Object.keys(batchResults.results).length > 0 && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold text-slate-800 mb-4">
                Column Recommendations ({Object.keys(batchResults.results).length})
              </h3>
              <div className="space-y-4">
                {Object.entries(batchResults.results).map(([columnName, columnResult]) => (
                <div key={columnName} className="border border-slate-200 rounded-lg p-4 bg-white/50">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
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
                        {(columnResult.confidence * 100).toFixed(0)}%
                      </div>
                      <button
                        onClick={() => setShowCorrectionFor(showCorrectionFor === columnName ? null : columnName)}
                        className="flex items-center gap-1 px-2 py-1 bg-white hover:bg-blue-50 text-blue-600 rounded-lg text-xs font-medium border border-blue-200 transition"
                      >
                        <Edit2 className="w-3 h-3" />
                        Override
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-wrap mb-3">
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
                    <div className="mb-3 pb-3 border-b border-slate-200">
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

                  {/* Override Form */}
                  {showCorrectionFor === columnName && (
                    <div className="bg-blue-50 rounded-lg p-3 border border-blue-200 mt-3">
                      <div className="flex items-start gap-2 mb-2">
                        <Edit2 className="w-4 h-4 text-blue-600 mt-0.5" />
                        <div className="flex-1">
                          <p className="text-xs font-semibold text-blue-800">Override Recommendation</p>
                          <p className="text-xs text-blue-600 mt-0.5">
                            Select the correct action for "{columnName}"
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <select
                          value={correctActions[columnName] || ''}
                          onChange={(e) => setCorrectActions(prev => ({ ...prev, [columnName]: e.target.value }))}
                          className="flex-1 px-2 py-1.5 rounded-lg border border-blue-300 text-xs bg-white"
                        >
                          <option value="">-- Select Action --</option>
                          <option value="keep_as_is">Keep (No preprocessing)</option>
                          <option value="standard_scale">Standard Scale</option>
                          <option value="minmax_scale">Min-Max Scale</option>
                          <option value="robust_scale">Robust Scale</option>
                          <option value="log_transform">Log Transform</option>
                          <option value="box_cox">Box-Cox Transform</option>
                          <option value="yeo_johnson">Yeo-Johnson Transform</option>
                          <option value="onehot_encode">One-Hot Encode</option>
                          <option value="label_encode">Label Encode</option>
                          <option value="target_encode">Target Encode</option>
                          <option value="fill_null_mean">Fill Nulls (Mean)</option>
                          <option value="fill_null_median">Fill Nulls (Median)</option>
                          <option value="fill_null_mode">Fill Nulls (Mode)</option>
                          <option value="drop_column">Drop Column</option>
                        </select>
                        <button
                          onClick={() => handleBatchCorrection(columnName, columnResult.action, columnResult.confidence)}
                          disabled={isSubmittingCorrection[columnName] || !correctActions[columnName]}
                          className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isSubmittingCorrection[columnName] ? 'Submitting...' : 'Apply'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                ))}
              </div>
            </div>
          )}
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
