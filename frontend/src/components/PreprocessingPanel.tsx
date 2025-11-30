import { useState } from 'react';
import { Upload, Play, Download, CheckCircle, AlertCircle, Info, FileSpreadsheet, X, Edit2, Activity, TrendingUp, TrendingDown, MinusCircle, ChevronDown, ChevronRight, Layers, Zap, Brain, BookOpen, Target, Shield } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ResultCard from './ResultCard';
import ExplanationModal from './ExplanationModal';

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
  const [showExplanationFor, setShowExplanationFor] = useState<string | null>(null);
  const [explanationColumnName, setExplanationColumnName] = useState<string | null>(null);
  const [explanationColumnData, setExplanationColumnData] = useState<any>(null);
  const [correctActions, setCorrectActions] = useState<Record<string, string>>({});
  const [isSubmittingCorrection, setIsSubmittingCorrection] = useState<Record<string, boolean>>({});
  const [expandedHealthColumn, setExpandedHealthColumn] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [originalData, setOriginalData] = useState<Record<string, any[]> | null>(null);

  // New state for expandable panels
  const [expandedPanels, setExpandedPanels] = useState<Record<string, boolean>>({
    architecture: true,
    dataHealth: true,
    recommendations: true,
  });

  const togglePanel = (panel: string) => {
    setExpandedPanels(prev => ({ ...prev, [panel]: !prev[panel] }));
  };

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
        } else if (!isNaN(Number(value))) {
          columns[header].push(Number(value));
        } else {
          columns[header].push(value);
        }
      });
    }

    return columns;
  };

  const handleSampleData = (type: string) => {
    const samples = {
      skewed: {
        name: 'revenue',
        data: '100, 150, 200, 250, 300, 500, 1000, 5000, 10000, 25000'
      },
      categorical: {
        name: 'category',
        data: 'A, B, C, A, B, C, A, B, A, A'
      },
      dates: {
        name: 'date',
        data: '2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05'
      }
    };
    const sample = samples[type as keyof typeof samples];
    setColumnName(sample.name);
    setColumnData(sample.data);
  };

  const handlePreprocess = async () => {
    if (!columnData.trim()) {
      toast.error('Please enter column data');
      return;
    }

    setIsProcessing(true);
    try {
      const values = columnData
        .split(/[,\n]/)
        .map(v => v.trim())
        .filter(v => v !== '');

      const response = await axios.post('/api/preprocess', {
        data: values,
        column_name: columnName || 'unnamed_column'
      });

      setResult(response.data);
      toast.success('Analysis complete!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Analysis failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    try {
      const text = await selectedFile.text();
      const columns = parseCSV(text);
      setOriginalData(columns);

      const response = await axios.post('/api/batch', { columns });
      setBatchResults(response.data);
      toast.success(`Analyzed ${Object.keys(columns).length} columns!`);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBatchCorrection = async (columnName: string, wrongAction: string, confidence: number) => {
    const correctAction = correctActions[columnName];
    if (!correctAction || !originalData || !originalData[columnName]) {
      toast.error('Invalid correction data');
      return;
    }

    setIsSubmittingCorrection(prev => ({ ...prev, [columnName]: true }));
    try {
      await axios.post('/api/correct', {
        column_data: originalData[columnName],
        column_name: columnName,
        wrong_action: wrongAction,
        correct_action: correctAction,
        confidence: confidence
      });

      // Update the UI to reflect the corrected action
      if (batchResults) {
        setBatchResults({
          ...batchResults,
          results: {
            ...batchResults.results,
            [columnName]: {
              ...batchResults.results[columnName],
              action: correctAction,
              confidence: 1.0,
              source: 'user_override',
              explanation: `User override: ${wrongAction} → ${correctAction}`
            }
          }
        });
      }

      toast.success(`✓ Learned correction for "${columnName}"! UI updated.`);
      setShowCorrectionFor(null);
      setCorrectActions(prev => {
        const newActions = { ...prev };
        delete newActions[columnName];
        return newActions;
      });
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Correction failed');
    } finally {
      setIsSubmittingCorrection(prev => ({ ...prev, [columnName]: false }));
    }
  };

  const handleExecutePipeline = async () => {
    if (!batchResults || !originalData) {
      toast.error('No data to process');
      return;
    }

    setIsExecuting(true);
    try {
      const actions: Record<string, string> = {};
      Object.entries(batchResults.results).forEach(([colName, result]) => {
        actions[colName] = result.action;
      });

      const response = await axios.post('/api/execute', {
        columns: originalData,
        actions
      }, {
        responseType: 'blob'
      });

      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'preprocessed_data.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();

      toast.success('✓ Pipeline executed! Downloading results...');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Pipeline execution failed');
    } finally {
      setIsExecuting(false);
    }
  };

  // Calculate decision source breakdown from batch results
  const getDecisionSourceBreakdown = () => {
    if (!batchResults) return null;

    const sources = {
      learned: 0,
      symbolic: 0,
      neural: 0,
      conservative_fallback: 0
    };

    Object.values(batchResults.results).forEach((result: any) => {
      const source = result.source || 'symbolic';
      if (sources.hasOwnProperty(source)) {
        sources[source as keyof typeof sources]++;
      }
    });

    return sources;
  };

  const sourceBreakdown = getDecisionSourceBreakdown();

  return (
    <div className="space-y-6">
      {/* Mode Selector */}
      <div className="glass-card p-4">
        <div className="flex gap-4">
          <button
            onClick={() => setMode('single')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${mode === 'single'
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
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${mode === 'file'
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
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
                <div className="text-3xl font-bold text-blue-600">
                  {batchResults.summary.total_columns}
                </div>
                <div className="text-sm text-slate-600 mt-1">Total Columns</div>
              </div>
              <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
                <div className="text-3xl font-bold text-green-600">
                  {Object.values(batchResults.results).filter((r: any) => r.action === 'keep_as_is').length}
                </div>
                <div className="text-sm text-slate-600 mt-1">Healthy Columns</div>
              </div>
              <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
                <div className="text-3xl font-bold text-purple-600">
                  {(batchResults.summary.avg_confidence * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-slate-600 mt-1">Avg Confidence</div>
              </div>
            </div>

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

          {/* Intelligent 2-Panel Layout: Health + Recommendations */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-stretch">
            {/* PANEL 2: Data Health (30% on XL screens) */}
            {batchResults.health && (
              <div className="xl:col-span-1 glass-card overflow-hidden flex flex-col">
                <button
                  onClick={() => togglePanel('dataHealth')}
                  className="w-full p-6 flex items-center justify-between hover:bg-slate-50/50 transition-colors"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-3">
                      <Activity className="w-6 h-6 text-blue-600" />
                      <h3 className="text-base font-bold text-slate-800">Health</h3>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full font-medium w-fit ${batchResults.health.overall_health_score >= 80 ? 'bg-green-100 text-green-700' :
                      batchResults.health.overall_health_score >= 50 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                      {batchResults.health.overall_health_score.toFixed(0)}/100
                    </span>
                  </div>
                  {expandedPanels.dataHealth ? (
                    <ChevronDown className="w-5 h-5 text-slate-600 flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-slate-600 flex-shrink-0" />
                  )}
                </button>

                {expandedPanels.dataHealth && (
                  <div className="px-4 pb-4 border-t border-slate-200 flex-1">
                    {/* Overall Health Score - Compact */}
                    <div className="mt-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-100">
                      <p className="text-xs text-slate-600 mb-2">Dataset Health</p>
                      <div className="flex items-center gap-2 mb-3">
                        <div className="text-3xl font-bold" style={{
                          color: batchResults.health.overall_health_score >= 80 ? '#10b981' :
                            batchResults.health.overall_health_score >= 50 ? '#f59e0b' : '#ef4444'
                        }}>
                          {batchResults.health.overall_health_score.toFixed(0)}
                        </div>
                        <div className="text-lg text-slate-400">/100</div>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-green-600">{batchResults.health.healthy_columns}</div>
                          <div className="text-xs text-slate-600">OK</div>
                        </div>
                        <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-yellow-600">{batchResults.health.warning_columns}</div>
                          <div className="text-xs text-slate-600">Warn</div>
                        </div>
                        <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-red-600">{batchResults.health.critical_columns}</div>
                          <div className="text-xs text-slate-600">Crit</div>
                        </div>
                      </div>
                    </div>

                    {/* Column Health Details - Compact */}
                    <div className="mt-4 space-y-2 max-h-[600px] overflow-y-auto pr-2">
                      <h4 className="text-xs font-semibold text-slate-700">Columns</h4>
                      {Object.values(batchResults.health.column_health).map((health: ColumnHealthMetrics) => (
                        <div key={health.column_name} className={`border rounded-lg p-2 transition-all ${health.severity === 'healthy' ? 'border-green-200 bg-green-50/30' :
                          health.severity === 'warning' ? 'border-yellow-200 bg-yellow-50/30' :
                            'border-red-200 bg-red-50/30'
                          }`}>
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 flex-1 min-w-0">
                              {health.severity === 'healthy' && <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />}
                              {health.severity === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-600 flex-shrink-0" />}
                              {health.severity === 'critical' && <X className="w-4 h-4 text-red-600 flex-shrink-0" />}
                              <div className="flex-1 min-w-0">
                                <h5 className="font-semibold text-slate-800 text-xs truncate">{health.column_name}</h5>
                                <span className="text-xs text-slate-500">{health.data_type}</span>
                              </div>
                            </div>
                            <div className={`text-lg font-bold ${health.severity === 'healthy' ? 'text-green-600' :
                              health.severity === 'warning' ? 'text-yellow-600' :
                                'text-red-600'
                              }`}>
                              {health.health_score.toFixed(0)}
                            </div>
                          </div>
                          {health.anomalies.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {health.anomalies.slice(0, 2).map((anomaly, idx) => (
                                <span key={idx} className={`text-xs px-1 py-0.5 rounded ${health.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                  health.severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-blue-100 text-blue-700'
                                  }`}>
                                  {anomaly}
                                </span>
                              ))}
                              {health.anomalies.length > 2 && (
                                <span className="text-xs text-slate-500">+{health.anomalies.length - 2}</span>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* PANEL 3: Column Recommendations (30% on XL screens) */}
            {Object.keys(batchResults.results).length > 0 && (
              <div className="xl:col-span-1 glass-card overflow-hidden flex flex-col">
                <button
                  onClick={() => togglePanel('recommendations')}
                  className="w-full p-6 flex items-center justify-between hover:bg-slate-50/50 transition-colors"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-3">
                      <Target className="w-6 h-6 text-blue-600" />
                      <h3 className="text-base font-bold text-slate-800">Recommendations</h3>
                    </div>
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full font-medium w-fit">
                      {Object.keys(batchResults.results).length} columns
                    </span>
                  </div>
                  {expandedPanels.recommendations ? (
                    <ChevronDown className="w-5 h-5 text-slate-600 flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-slate-600 flex-shrink-0" />
                  )}
                </button>

                {expandedPanels.recommendations && (
                  <div className="px-4 pb-4 border-t border-slate-200 flex-1">
                    <div className="mt-4 space-y-2 max-h-[600px] overflow-y-auto pr-2">
                      {Object.entries(batchResults.results).map(([columnName, columnResult]) => (
                        <div key={columnName} className="border border-slate-200 rounded-lg p-2 bg-white/50">
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <h4 className="font-semibold text-slate-800 text-xs truncate flex-1 min-w-0">{columnName}</h4>
                            <div className={`px-2 py-0.5 rounded-full text-xs font-medium flex-shrink-0 ${columnResult.confidence >= 0.9
                              ? 'bg-green-100 text-green-700'
                              : columnResult.confidence >= 0.7
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-red-100 text-red-700'
                              }`}>
                              {(columnResult.confidence * 100).toFixed(0)}%
                            </div>
                          </div>

                          <div className="flex flex-col gap-1 mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-600">Action:</span>
                              <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                                {columnResult.action}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-600">Source:</span>
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${columnResult.source === 'user_override'
                                ? 'bg-green-100 text-green-700'
                                : columnResult.source === 'conservative_fallback'
                                  ? 'bg-slate-100 text-slate-700'
                                  : columnResult.source === 'neural'
                                    ? 'bg-pink-100 text-pink-700'
                                    : columnResult.source === 'learned'
                                      ? 'bg-green-100 text-green-700'
                                      : 'bg-blue-100 text-blue-700'
                                }`}>
                                {columnResult.source}
                              </span>
                            </div>
                          </div>

                          <div className="flex gap-2 mb-2">
                            <button
                              onClick={() => {
                                setShowExplanationFor(columnName);
                                setExplanationColumnName(columnName);
                                setExplanationColumnData(originalData?.[columnName] || []);
                              }}
                              className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-white hover:bg-purple-50 text-purple-600 rounded text-xs font-medium border border-purple-200 transition"
                            >
                              <BookOpen className="w-3 h-3" />
                              Explain
                            </button>
                            <button
                              onClick={() => setShowCorrectionFor(showCorrectionFor === columnName ? null : columnName)}
                              className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-white hover:bg-blue-50 text-blue-600 rounded text-xs font-medium border border-blue-200 transition"
                            >
                              <Edit2 className="w-3 h-3" />
                              Override
                            </button>
                          </div>

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
              <h3 className="font-semibold text-slate-800 text-sm">Universal Coverage</h3>
              <p className="text-xs text-slate-600 mt-1">
                95-99% autonomous on ANY domain (financial, medical, IoT, web)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation Modal */}
      {showExplanationFor && explanationColumnData && explanationColumnName && (
        <ExplanationModal
          isOpen={!!showExplanationFor}
          onClose={() => {
            setShowExplanationFor(null);
            setExplanationColumnName(null);
            setExplanationColumnData(null);
          }}
          columnData={explanationColumnData}
          columnName={explanationColumnName}
        />
      )}
    </div>
  );
}
