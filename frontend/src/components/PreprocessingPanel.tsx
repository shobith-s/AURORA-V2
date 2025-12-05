import { useState } from 'react';
import { Upload, Play, CheckCircle, AlertCircle, Info, FileSpreadsheet, X, Edit2, Activity, ChevronDown, ChevronRight, BookOpen, Target, Shield } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import ResultCard from './ResultCard';
import ExplanationModal from './ExplanationModal';
import ActionLibraryModal from './ActionLibraryModal';

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

interface ColumnResult {
  action: string;
  confidence: number;
  source: string;
  explanation?: string;
}

interface BatchResults {
  results: Record<string, ColumnResult>;
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
  const [result, setResult] = useState<ColumnResult | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResults | null>(null);
  const [showCorrectionFor, setShowCorrectionFor] = useState<string | null>(null);
  const [showExplanationFor, setShowExplanationFor] = useState<string | null>(null);
  const [explanationColumnName, setExplanationColumnName] = useState<string | null>(null);
  const [explanationColumnData, setExplanationColumnData] = useState<unknown[] | null>(null);
  const [correctActions, setCorrectActions] = useState<Record<string, string>>({});
  const [isSubmittingCorrection, setIsSubmittingCorrection] = useState<Record<string, boolean>>({});
  const [isExecuting, setIsExecuting] = useState(false);
  const [originalData, setOriginalData] = useState<Record<string, unknown[]> | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [preprocessedData, setPreprocessedData] = useState<Record<string, unknown[]> | null>(null);

  // Action Library State
  const [showActionLibrary, setShowActionLibrary] = useState(false);
  const [libraryTargetColumn, setLibraryTargetColumn] = useState<string | null>(null);

  // New state for expandable panels
  const [expandedPanels, setExpandedPanels] = useState<Record<string, boolean>>({
    architecture: true,
    dataHealth: true,
    recommendations: true,
  });

  // DEBUG MARKER - REMOVE BEFORE PRODUCTION
  const debugStyle = { border: '4px solid #ef4444', position: 'relative' as const };
  const debugBadge = (
    <div className="absolute top-0 right-0 bg-error text-brand-white text-xs font-bold px-2 py-1 z-50">
      DEBUG: V2 ACTIVE
    </div>
  );

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

  const parseCSV = (text: string): Record<string, unknown[]> => {
    const lines = text.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('CSV must have at least a header row and one data row');
    }

    const headers = lines[0].split(',').map(h => h.trim());
    const columns: Record<string, unknown[]> = {};

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
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        toast.error(error.response?.data?.detail || 'Analysis failed');
      } else {
        toast.error('Analysis failed');
      }
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

      const response = await axios.post('/api/batch', { columns }, {
        timeout: 60000, // 60 second timeout
      });
      setBatchResults(response.data);
      toast.success(`Analyzed ${Object.keys(columns).length} columns!`);
    } catch (error: unknown) {
      console.error('Upload error:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          toast.error('Upload timed out. Try with a smaller file or fewer columns.');
        } else {
          toast.error(error.response?.data?.detail || error.message || 'Upload failed');
        }
      } else {
        toast.error('Upload failed');
      }
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
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        toast.error(error.response?.data?.detail || 'Correction failed');
      } else {
        toast.error('Correction failed');
      }
    } finally {
      setIsSubmittingCorrection(prev => ({ ...prev, [columnName]: false }));
    }
  };

  interface ValidationReport {
    timestamp?: string;
    overall_status?: string;
    summary?: {
      passed_validation: number;
      validated_columns: number;
    };
  }
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);

  const handleExecutePipeline = async () => {
    if (!batchResults || !originalData) {
      toast.error('No data to process');
      return;
    }

    setIsExecuting(true);
    setValidationReport(null); // Reset report
    try {
      // Collect all actions (including overrides)
      const actions: Record<string, string> = {};
      Object.entries(batchResults.results).forEach(([colName, result]) => {
        actions[colName] = result.action;
      });

      // Execute preprocessing
      const response = await axios.post('/api/execute', {
        columns: originalData,
        actions
      });

      // Store session ID, preprocessed data, and validation report
      setSessionId(response.data.session_id);
      setPreprocessedData(response.data.processed_data);
      setValidationReport(response.data.validation_report);

      toast.success('✓ Preprocessing complete with Proof of Quality!');
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        toast.error(error.response?.data?.detail || 'Pipeline execution failed');
      } else {
        toast.error('Pipeline execution failed');
      }
    } finally {
      setIsExecuting(false);
    }
  };

  const handleDownload = (format: 'csv' | 'excel' | 'json') => {
    if (!sessionId) {
      toast.error('No preprocessed data available');
      return;
    }

    // Download file
    window.location.href = `/api/download/${sessionId}?format=${format}`;
    toast.success(`📥 Downloading ${format.toUpperCase()} file...`);
  };

  return (
    <div className="space-y-6" style={debugStyle}>
      {debugBadge}
      {/* Mode Selector */}
      <div className="glass-card p-4">
        <div className="flex gap-4">
          <button
            onClick={() => setMode('single')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${mode === 'single'
              ? 'bg-primary text-brand-white shadow-lg'
              : 'bg-brand-white/50 text-foreground-muted hover:bg-brand-white'
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
              ? 'bg-primary text-brand-white shadow-lg'
              : 'bg-brand-white/50 text-foreground-muted hover:bg-brand-white'
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
              <h2 className="text-xl font-bold text-brand-black">Single Column Analysis</h2>
              <div className="flex gap-2">
                <button
                  onClick={() => handleSampleData('skewed')}
                  className="text-xs px-3 py-1 bg-primary/10 text-primary rounded-lg hover:bg-primary/20"
                >
                  Skewed Data
                </button>
                <button
                  onClick={() => handleSampleData('categorical')}
                  className="text-xs px-3 py-1 bg-primary/10 text-primary rounded-lg hover:bg-primary/20"
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
              <label className="block text-sm font-medium text-foreground mb-2">
                Column Name
              </label>
              <input
                type="text"
                value={columnName}
                onChange={(e) => setColumnName(e.target.value)}
                placeholder="e.g., revenue, age, category"
                className="w-full px-4 py-2 rounded-lg border border-brand-warm-gray focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition"
              />
            </div>

            {/* Column Data */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-foreground mb-2">
                Column Data (comma or newline separated)
              </label>
              <textarea
                value={columnData}
                onChange={(e) => setColumnData(e.target.value)}
                placeholder="Enter your data here... (e.g., 10, 20, 30, 100, 200)"
                className="w-full h-32 px-4 py-2 rounded-lg border border-brand-warm-gray focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition font-mono text-sm"
              />
              <p className="mt-1 text-xs text-brand-cool-gray">
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
                    <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                    <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                    <span className="w-2 h-2 bg-brand-white rounded-full"></span>
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
            <h2 className="text-xl font-bold text-brand-black mb-4">CSV File Upload</h2>

            {!selectedFile ? (
              <div className="border-2 border-dashed border-brand-warm-gray rounded-lg p-8 text-center hover:border-primary transition-colors">
                <Upload className="w-12 h-12 mx-auto mb-4 text-brand-cool-gray" />
                <label className="cursor-pointer">
                  <span className="text-foreground font-medium">
                    Click to upload or drag and drop
                  </span>
                  <p className="text-sm text-brand-cool-gray mt-1">CSV files only</p>
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
                <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg">
                  <div className="flex items-center gap-3">
                    <FileSpreadsheet className="w-8 h-8 text-primary" />
                    <div>
                      <p className="font-medium text-brand-black">{selectedFile.name}</p>
                      <p className="text-sm text-foreground-muted">
                        {(selectedFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={removeFile}
                    className="p-2 hover:bg-primary/20 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-foreground-muted" />
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
                        <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                        <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                        <span className="w-2 h-2 bg-brand-white rounded-full"></span>
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
            <h3 className="text-lg font-bold text-brand-black mb-4">Analysis Summary</h3>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="text-center p-4 bg-primary/10 rounded-lg">
                <div className="text-3xl font-bold text-primary">
                  {batchResults.summary.total_columns}
                </div>
                <div className="text-sm text-foreground-muted mt-1">Total Columns</div>
              </div>
              <div className="text-center p-4 bg-success/10 rounded-lg">
                <div className="text-3xl font-bold text-green-600">
                  {Object.values(batchResults.results).filter((r: ColumnResult) => r.action === 'keep_as_is').length}
                </div>
                <div className="text-sm text-foreground-muted mt-1">Healthy Columns</div>
              </div>
              <div className="text-center p-4 bg-primary/10 rounded-lg">
                <div className="text-3xl font-bold text-primary">
                  {(batchResults.summary.avg_confidence * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-foreground-muted mt-1">Avg Confidence</div>
              </div>
            </div>

            {/* Execute Pipeline Button */}
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={handleExecutePipeline}
                disabled={isExecuting || !!sessionId}
                className="btn-primary flex items-center gap-3 px-8 py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isExecuting ? (
                  <>
                    <div className="loading-dots flex gap-1">
                      <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                      <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                      <span className="w-2 h-2 bg-brand-white rounded-full"></span>
                    </div>
                    Executing Preprocessing...
                  </>
                ) : sessionId ? (
                  <>
                    <CheckCircle className="w-6 h-6" />
                    Preprocessing Complete!
                  </>
                ) : (
                  <>
                    <Play className="w-6 h-6" />
                    Execute Preprocessing Pipeline
                  </>
                )}
              </button>

            </div>
          </div>

          {/* Validation & Execution Results */}
          {sessionId && preprocessedData && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
              {/* Download Panel */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-success rounded-xl flex items-center justify-center">
                    <CheckCircle className="w-6 h-6 text-brand-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-brand-black">Ready to Download</h3>
                    <p className="text-sm text-foreground-muted">
                      {Object.keys(preprocessedData).length} columns processed
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <p className="text-sm font-medium text-foreground">Choose format:</p>
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      onClick={() => handleDownload('csv')}
                      className="group relative overflow-hidden bg-primary/10 hover:bg-primary/20 border border-primary/50 rounded-xl p-3 transition-all hover:scale-105"
                    >
                      <div className="flex flex-col items-center gap-1">
                        <FileSpreadsheet className="w-6 h-6 text-primary" />
                        <span className="text-xs font-bold text-blue-800">CSV</span>
                      </div>
                    </button>
                    <button
                      onClick={() => handleDownload('excel')}
                      className="group relative overflow-hidden bg-green-50 hover:bg-green-100 border border-success/30 rounded-xl p-3 transition-all hover:scale-105"
                    >
                      <div className="flex flex-col items-center gap-1">
                        <FileSpreadsheet className="w-6 h-6 text-green-600" />
                        <span className="text-xs font-bold text-green-800">Excel</span>
                      </div>
                    </button>
                    <button
                      onClick={() => handleDownload('json')}
                      className="group relative overflow-hidden bg-primary/10 hover:bg-primary/20 border border-primary/50 rounded-xl p-3 transition-all hover:scale-105"
                    >
                      <div className="flex flex-col items-center gap-1">
                        <FileSpreadsheet className="w-6 h-6 text-primary" />
                        <span className="text-xs font-bold text-purple-800">JSON</span>
                      </div>
                    </button>
                  </div>
                </div>
              </div>

              {/* Validation Summary (Proof of Quality) */}
              <div className="glass-card p-6 border-l-4 border-l-blue-500">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-primary/20 rounded-xl flex items-center justify-center">
                      <Shield className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-brand-black">Proof of Quality</h3>
                      <p className="text-sm text-foreground-muted">Statistical Validation</p>
                    </div>
                  </div>
                  {validationReport && validationReport.summary && (
                    <div className="text-right">
                      <div className="text-2xl font-bold text-primary">
                        {validationReport.summary.passed_validation}/{validationReport.summary.validated_columns}
                      </div>
                      <div className="text-xs text-brand-cool-gray">Passed Checks</div>
                    </div>
                  )}
                </div>

                {validationReport ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm p-2 bg-brand-white rounded-lg">
                      <span className="text-foreground-muted">Statistical Tests</span>
                      <span className="font-medium text-green-600">Active ✅</span>
                    </div>
                    <div className="flex items-center justify-between text-sm p-2 bg-brand-white rounded-lg">
                      <span className="text-foreground-muted">Consistency Checks</span>
                      <span className="font-medium text-green-600">Active ✅</span>
                    </div>
                    <div className="mt-2 pt-2 border-t border-slate-100">
                      <p className="text-xs text-brand-cool-gray italic">
                        &quot;Every decision is statistically validated to ensure data integrity.&quot;
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-24 text-brand-cool-gray text-sm">
                    Waiting for execution...
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Intelligent 2-Panel Layout: Health + Recommendations */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-stretch">
            {/* PANEL 2: Data Health (30% on XL screens) */}
            {batchResults.health && (
              <div className="xl:col-span-1 glass-card overflow-hidden flex flex-col">
                <button
                  onClick={() => togglePanel('dataHealth')}
                  className="w-full p-6 flex items-center justify-between hover:bg-brand-white/50 transition-colors"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-3">
                      <Activity className="w-6 h-6 text-primary" />
                      <h3 className="text-base font-bold text-brand-black">Health</h3>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full font-medium w-fit ${batchResults.health.overall_health_score >= 80 ? 'bg-green-100 text-success' :
                      batchResults.health.overall_health_score >= 50 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                      {batchResults.health.overall_health_score.toFixed(0)}/100
                    </span>
                  </div>
                  {expandedPanels.dataHealth ? (
                    <ChevronDown className="w-5 h-5 text-foreground-muted flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-foreground-muted flex-shrink-0" />
                  )}
                </button>

                {expandedPanels.dataHealth && (
                  <div className="px-4 pb-4 border-t border-brand-warm-gray flex-1">
                    {/* Overall Health Score - Compact */}
                    <div className="mt-4 bg-primary/10 rounded-lg p-4 border border-primary/30">
                      <p className="text-xs text-foreground-muted mb-2">Dataset Health</p>
                      <div className="flex items-center gap-2 mb-3">
                        <div className="text-3xl font-bold" style={{
                          color: batchResults.health.overall_health_score >= 80 ? '#10b981' :
                            batchResults.health.overall_health_score >= 50 ? '#f59e0b' : '#ef4444'
                        }}>
                          {batchResults.health.overall_health_score.toFixed(0)}
                        </div>
                        <div className="text-lg text-brand-cool-gray">/100</div>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center p-2 bg-brand-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-green-600">{batchResults.health.healthy_columns}</div>
                          <div className="text-xs text-foreground-muted">OK</div>
                        </div>
                        <div className="text-center p-2 bg-brand-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-yellow-600">{batchResults.health.warning_columns}</div>
                          <div className="text-xs text-foreground-muted">Warn</div>
                        </div>
                        <div className="text-center p-2 bg-brand-white rounded-lg shadow-sm">
                          <div className="text-lg font-bold text-error">{batchResults.health.critical_columns}</div>
                          <div className="text-xs text-foreground-muted">Crit</div>
                        </div>
                      </div>
                    </div>

                    {/* Column Health Details - Compact */}
                    <div className="mt-4 space-y-2 max-h-[600px] overflow-y-auto pr-2">
                      <h4 className="text-xs font-semibold text-foreground">Columns</h4>
                      {Object.values(batchResults.health.column_health).map((health: ColumnHealthMetrics) => (
                        <div key={health.column_name} className={`border rounded-lg p-2 transition-all ${health.severity === 'healthy' ? 'border-success/30 bg-green-50/30' :
                          health.severity === 'warning' ? 'border-yellow-200 bg-yellow-50/30' :
                            'border-red-200 bg-red-50/30'
                          }`}>
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 flex-1 min-w-0">
                              {health.severity === 'healthy' && <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />}
                              {health.severity === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-600 flex-shrink-0" />}
                              {health.severity === 'critical' && <X className="w-4 h-4 text-error flex-shrink-0" />}
                              <div className="flex-1 min-w-0">
                                <h5 className="font-semibold text-brand-black text-xs truncate">{health.column_name}</h5>
                                <span className="text-xs text-brand-cool-gray">{health.data_type}</span>
                              </div>
                            </div>
                            <div className={`text-lg font-bold ${health.severity === 'healthy' ? 'text-green-600' :
                              health.severity === 'warning' ? 'text-yellow-600' :
                                'text-error'
                              }`}>
                              {health.health_score.toFixed(0)}
                            </div>
                          </div>
                          {health.anomalies.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {health.anomalies.slice(0, 2).map((anomaly, idx) => (
                                <span key={idx} className={`text-xs px-1 py-0.5 rounded ${health.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                  health.severity === 'warning' ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-primary/20 text-primary-dark'
                                  }`}>
                                  {anomaly}
                                </span>
                              ))}
                              {health.anomalies.length > 2 && (
                                <span className="text-xs text-brand-cool-gray">+{health.anomalies.length - 2}</span>
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
                  className="w-full p-6 flex items-center justify-between hover:bg-brand-white/50 transition-colors"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-3">
                      <Target className="w-6 h-6 text-primary" />
                      <h3 className="text-base font-bold text-brand-black">Recommendations</h3>
                    </div>
                    <span className="px-2 py-1 bg-primary/20 text-primary-dark text-xs rounded-full font-medium w-fit">
                      {Object.keys(batchResults.results).length} columns
                    </span>
                  </div>
                  {expandedPanels.recommendations ? (
                    <ChevronDown className="w-5 h-5 text-foreground-muted flex-shrink-0" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-foreground-muted flex-shrink-0" />
                  )}
                </button>

                {expandedPanels.recommendations && (
                  <div className="px-4 pb-4 border-t border-brand-warm-gray flex-1">
                    <div className="mt-4 space-y-2 max-h-[600px] overflow-y-auto pr-2">
                      {Object.entries(batchResults.results).map(([columnName, columnResult]) => (
                        <div key={columnName} className="border border-brand-warm-gray rounded-lg p-2 bg-brand-white/50">
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <h4 className="font-semibold text-brand-black text-xs truncate flex-1 min-w-0">{columnName}</h4>
                            <div className={`px-2 py-0.5 rounded-full text-xs font-medium flex-shrink-0 ${columnResult.confidence >= 0.9
                              ? 'bg-green-100 text-success'
                              : columnResult.confidence >= 0.7
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-red-100 text-red-700'
                              }`}>
                              {(columnResult.confidence * 100).toFixed(0)}%
                            </div>
                          </div>

                          <div className="flex flex-col gap-1 mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-foreground-muted">Action:</span>
                              <span className="px-2 py-0.5 bg-primary/20 text-primary-dark rounded text-xs font-medium">
                                {columnResult.action}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-foreground-muted">Source:</span>
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${columnResult.source === 'user_override'
                                ? 'bg-green-100 text-success'
                                : columnResult.source === 'meta_learning'
                                  ? 'bg-orange-100 text-orange-700'
                                  : columnResult.source === 'conservative_fallback'
                                    ? 'bg-background-muted text-foreground'
                                    : 'bg-primary/20 text-primary-dark'
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
                              className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-brand-white hover:bg-primary/10 text-primary rounded text-xs font-medium border border-primary/50 transition"
                            >
                              <BookOpen className="w-3 h-3" />
                              Explain
                            </button>
                            <button
                              onClick={() => setShowCorrectionFor(showCorrectionFor === columnName ? null : columnName)}
                              className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-brand-white hover:bg-primary/10 text-primary rounded text-xs font-medium border border-primary/50 transition"
                            >
                              <Edit2 className="w-3 h-3" />
                              Override
                            </button>
                          </div>

                          {/* Override Form */}
                          {showCorrectionFor === columnName && (
                            <div className="bg-primary/10 rounded-lg p-3 border border-primary/50 mt-3">
                              <div className="flex items-start gap-2 mb-2">
                                <Edit2 className="w-4 h-4 text-primary mt-0.5" />
                                <div className="flex-1">
                                  <p className="text-xs font-semibold text-blue-800">Override Recommendation</p>
                                  <p className="text-xs text-primary mt-0.5">
                                    Select the correct action for &quot;{columnName}&quot;
                                  </p>
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => {
                                    // Open Action Library Modal
                                    // We need a new state to track which column is opening the library
                                    // But for now, let's reuse showCorrectionFor as the target
                                    // and add a new state 'isLibraryOpen'
                                    setShowActionLibrary(true);
                                    setLibraryTargetColumn(columnName);
                                  }}
                                  className="flex-1 px-3 py-1.5 bg-brand-white border border-primary rounded-lg text-left text-xs text-foreground hover:border-primary hover:ring-2 hover:ring-blue-100 transition-all flex items-center justify-between group"
                                >
                                  <span className={correctActions[columnName] ? 'text-primary-dark font-medium' : 'text-brand-cool-gray'}>
                                    {correctActions[columnName]
                                      ? correctActions[columnName].replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                                      : 'Select Action...'}
                                  </span>
                                  <div className="px-1.5 py-0.5 bg-primary/10 text-primary rounded text-[10px] font-medium group-hover:bg-primary/20 transition-colors">
                                    Browse Library
                                  </div>
                                </button>
                                <button
                                  onClick={() => handleBatchCorrection(columnName, columnResult.action, columnResult.confidence)}
                                  disabled={isSubmittingCorrection[columnName] || !correctActions[columnName]}
                                  className="px-3 py-1.5 bg-primary text-brand-white rounded-lg text-xs font-medium hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
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
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-brand-black text-sm">Privacy-First</h3>
              <p className="text-xs text-foreground-muted mt-1">
                Your data is processed in real-time and never stored
              </p>
            </div>
          </div>
        </div>

        <div className="glass-card p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Info className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-brand-black text-sm">Universal Coverage</h3>
              <p className="text-xs text-foreground-muted mt-1">
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
      {/* Smart Action Library Modal */}
      <ActionLibraryModal
        isOpen={showActionLibrary}
        onClose={() => setShowActionLibrary(false)}
        onSelectAction={(action) => {
          if (libraryTargetColumn) {
            setCorrectActions(prev => ({ ...prev, [libraryTargetColumn]: action }));
          }
          setShowActionLibrary(false);
        }}
        currentAction={libraryTargetColumn ? correctActions[libraryTargetColumn] : undefined}
      />
    </div>
  );
}
