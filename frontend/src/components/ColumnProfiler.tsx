import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ColumnProfilerProps {
    columnData: number[];
    columnName: string;
    action?: string;
}

interface ProfileData {
    column_name: string;
    data_type: string;
    statistics: {
        mean?: number;
        median?: number;
        std?: number;
        min?: number;
        max?: number;
        skewness?: number;
        count?: number;
    };
    distribution: {
        histogram?: Array<{
            bin_label: string;
            count: number;
            percentage: number;
        }>;
    };
    outliers?: {
        count: number;
        percentage: number;
    };
}

export default function ColumnProfiler({ columnData, columnName, action }: ColumnProfilerProps) {
    const [profile, setProfile] = useState<ProfileData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (columnData && columnData.length > 0) {
            fetchProfile();
        }
    }, [columnData, columnName]);

    const fetchProfile = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/api/profile-column', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    column_data: columnData,
                    column_name: columnName,
                    bins: 15,
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to fetch profile');
            }

            const data = await response.json();
            setProfile(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center p-8">
                <div className="text-gray-600">Loading profile...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600">Error: {error}</p>
            </div>
        );
    }

    if (!profile) {
        return null;
    }

    const stats = profile.statistics;

    return (
        <div className="space-y-6">
            {/* Statistics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {stats.mean !== undefined && (
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="text-sm text-gray-600">Mean</div>
                        <div className="text-2xl font-bold text-gray-900">{stats.mean.toFixed(2)}</div>
                    </div>
                )}
                {stats.median !== undefined && (
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="text-sm text-gray-600">Median</div>
                        <div className="text-2xl font-bold text-gray-900">{stats.median.toFixed(2)}</div>
                    </div>
                )}
                {stats.std !== undefined && (
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="text-sm text-gray-600">Std Dev</div>
                        <div className="text-2xl font-bold text-gray-900">{stats.std.toFixed(2)}</div>
                    </div>
                )}
                {stats.skewness !== undefined && (
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="text-sm text-gray-600">Skewness</div>
                        <div className="text-2xl font-bold text-gray-900">{stats.skewness.toFixed(2)}</div>
                    </div>
                )}
            </div>

            {/* Outliers Alert */}
            {profile.outliers && profile.outliers.count > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                        <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <span className="font-medium text-yellow-800">
                            {profile.outliers.count} outliers detected ({profile.outliers.percentage.toFixed(1)}%)
                        </span>
                    </div>
                </div>
            )}

            {/* Distribution Histogram */}
            {profile.distribution.histogram && profile.distribution.histogram.length > 0 && (
                <div className="bg-white p-6 rounded-lg border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={profile.distribution.histogram}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="bin_label"
                                angle={-45}
                                textAnchor="end"
                                height={80}
                                tick={{ fontSize: 10 }}
                            />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="count" fill="#3b82f6" name="Frequency" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Range Info */}
            {stats.min !== undefined && stats.max !== undefined && (
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm text-gray-600">Min</div>
                            <div className="text-lg font-semibold">{stats.min.toFixed(2)}</div>
                        </div>
                        <div className="flex-1 mx-4">
                            <div className="h-2 bg-gray-200 rounded-full">
                                <div className="h-2 bg-blue-500 rounded-full" style={{ width: '100%' }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="text-sm text-gray-600">Max</div>
                            <div className="text-lg font-semibold">{stats.max.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
