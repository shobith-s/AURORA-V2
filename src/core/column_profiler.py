"""
Column profiling utilities for generating distribution statistics and visualizations.
Provides comprehensive statistical analysis for data profiling.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class ColumnProfiler:
    """Generate statistical profiles and distribution data for columns."""
    
    def __init__(self, bins: int = 20):
        """
        Initialize the profiler.
        
        Args:
            bins: Number of bins for histogram generation
        """
        self.bins = bins
    
    def profile_column(
        self, 
        data: List[Any], 
        column_name: str = "column"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive profile for a column.
        
        Args:
            data: Column data as list
            column_name: Name of the column
            
        Returns:
            Dictionary with statistics, distribution, and outlier information
        """
        series = pd.Series(data)
        
        # Determine data type
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        if is_numeric:
            return self._profile_numeric(series, column_name)
        else:
            return self._profile_categorical(series, column_name)
    
    def _profile_numeric(
        self, 
        series: pd.Series, 
        column_name: str
    ) -> Dict[str, Any]:
        """Profile numeric column."""
        # Remove nulls for statistics
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return self._empty_profile(column_name, "numeric")
        
        # Basic statistics
        statistics = {
            "count": int(len(clean_data)),
            "null_count": int(series.isnull().sum()),
            "mean": float(clean_data.mean()),
            "median": float(clean_data.median()),
            "std": float(clean_data.std()),
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "q1": float(clean_data.quantile(0.25)),
            "q3": float(clean_data.quantile(0.75)),
            "skewness": float(clean_data.skew()),
            "kurtosis": float(clean_data.kurtosis()),
        }
        
        # Distribution data
        distribution = self._generate_histogram(clean_data)
        
        # Outlier detection
        outliers = self._detect_outliers(clean_data)
        
        return {
            "column_name": column_name,
            "data_type": "numeric",
            "statistics": statistics,
            "distribution": distribution,
            "outliers": outliers
        }
    
    def _profile_categorical(
        self, 
        series: pd.Series, 
        column_name: str
    ) -> Dict[str, Any]:
        """Profile categorical column."""
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return self._empty_profile(column_name, "categorical")
        
        # Value counts
        value_counts = clean_data.value_counts()
        
        statistics = {
            "count": int(len(clean_data)),
            "null_count": int(series.isnull().sum()),
            "unique_count": int(clean_data.nunique()),
            "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        }
        
        # Distribution (top categories)
        distribution = {
            "categories": [
                {
                    "category": str(cat),
                    "count": int(count),
                    "percentage": float(count / len(clean_data) * 100)
                }
                for cat, count in value_counts.head(20).items()
            ]
        }
        
        return {
            "column_name": column_name,
            "data_type": "categorical",
            "statistics": statistics,
            "distribution": distribution,
            "outliers": None
        }
    
    def _generate_histogram(self, data: pd.Series) -> Dict[str, Any]:
        """Generate histogram data."""
        # Create bins
        counts, bin_edges = np.histogram(data, bins=self.bins)
        
        # Format histogram data
        histogram = []
        for i in range(len(counts)):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])
            histogram.append({
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_label": f"{bin_start:.2f}-{bin_end:.2f}",
                "count": int(counts[i]),
                "percentage": float(counts[i] / len(data) * 100)
            })
        
        # Percentiles
        percentiles = {
            "p5": float(data.quantile(0.05)),
            "p25": float(data.quantile(0.25)),
            "p50": float(data.quantile(0.50)),
            "p75": float(data.quantile(0.75)),
            "p95": float(data.quantile(0.95)),
        }
        
        return {
            "histogram": histogram,
            "percentiles": percentiles
        }
    
    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_values = data[outlier_mask].tolist()
        outlier_indices = data[outlier_mask].index.tolist()
        
        return {
            "count": len(outlier_values),
            "percentage": float(len(outlier_values) / len(data) * 100) if len(data) > 0 else 0,
            "values": [float(v) for v in outlier_values[:50]],  # Limit to 50
            "indices": [int(i) for i in outlier_indices[:50]],
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    
    def _empty_profile(self, column_name: str, data_type: str) -> Dict[str, Any]:
        """Return empty profile for columns with no data."""
        return {
            "column_name": column_name,
            "data_type": data_type,
            "statistics": {},
            "distribution": {},
            "outliers": None,
            "error": "No data available"
        }
    
    def compare_preprocessing(
        self,
        original_data: List[Any],
        transformed_data: List[Any],
        action: str
    ) -> Dict[str, Any]:
        """
        Compare original and transformed data.
        
        Args:
            original_data: Original column data
            transformed_data: Transformed column data
            action: Preprocessing action applied
            
        Returns:
            Comparison with before/after profiles and improvement metrics
        """
        original_profile = self.profile_column(original_data, "original")
        transformed_profile = self.profile_column(transformed_data, "transformed")
        
        # Calculate improvement metrics
        improvement = self._calculate_improvements(
            original_profile.get("statistics", {}),
            transformed_profile.get("statistics", {})
        )
        
        return {
            "action": action,
            "original": original_profile,
            "transformed": transformed_profile,
            "improvement_metrics": improvement
        }
    
    def _calculate_improvements(
        self,
        original_stats: Dict[str, Any],
        transformed_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate improvement metrics between original and transformed data."""
        improvements = {}
        
        # Skewness reduction
        if "skewness" in original_stats and "skewness" in transformed_stats:
            orig_skew = abs(original_stats["skewness"])
            trans_skew = abs(transformed_stats["skewness"])
            if orig_skew > 0:
                improvements["skewness_reduction"] = float(
                    (orig_skew - trans_skew) / orig_skew * 100
                )
        
        # Standard deviation change
        if "std" in original_stats and "std" in transformed_stats:
            improvements["std_change"] = float(
                (transformed_stats["std"] - original_stats["std"]) / 
                original_stats["std"] * 100
            ) if original_stats["std"] > 0 else 0
        
        # Normality score (using kurtosis as proxy)
        if "kurtosis" in transformed_stats:
            # Closer to 0 is more normal
            improvements["normality_score"] = float(
                max(0, 1 - abs(transformed_stats["kurtosis"]) / 10)
            )
        
        return improvements
