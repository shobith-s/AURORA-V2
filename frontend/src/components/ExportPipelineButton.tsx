import React, { useState } from 'react';

interface Decision {
    column_name: string;
    action: string;
    confidence: number;
    explanation: string;
    parameters?: Record<string, unknown>;
}

interface ExportPipelineButtonProps {
    decisions: Decision[];
}

export default function ExportPipelineButton({ decisions }: ExportPipelineButtonProps) {
    const [format, setFormat] = useState<'python' | 'sklearn' | 'json'>('python');
    const [isExporting, setIsExporting] = useState(false);

    const handleExport = async () => {
        if (decisions.length === 0) {
            alert('No preprocessing decisions to export');
            return;
        }

        setIsExporting(true);

        try {
            const response = await fetch('http://localhost:8000/export-pipeline', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    decisions,
                    format,
                }),
            });

            if (!response.ok) {
                throw new Error('Export failed');
            }

            const data = await response.json();

            // Create and download file
            const blob = new Blob([data.content], {
                type: format === 'json' ? 'application/json' : 'text/plain',
            });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Export error:', error);
            alert('Failed to export pipeline. Please try again.');
        } finally {
            setIsExporting(false);
        }
    };

    return (
        <div className="flex items-center gap-3 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Export Format
                </label>
                <select
                    value={format}
                    onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setFormat(e.target.value as 'python' | 'sklearn' | 'json')}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <option value="python">Python Code</option>
                    <option value="sklearn">Sklearn Pipeline (pkl)</option>
                    <option value="json">JSON Config</option>
                </select>
            </div>

            <button
                onClick={handleExport}
                disabled={isExporting || decisions.length === 0}
                className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors mt-6"
            >
                <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                </svg>
                {isExporting ? 'Exporting...' : 'Export Pipeline'}
            </button>
        </div>
    );
}
