"""
Data Drift Detection - Phase 1 Improvements.

Detects when data distribution changes, triggering retraining if needed.
Uses statistical tests (Kolmogorov-Smirnov, Chi-square) to detect shifts.

Expected impact: Maintain accuracy over time in production.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
import json
from pathlib import Path


@dataclass
class DistributionProfile:
    """Statistical profile of a column's distribution."""

    column_name: str
    dtype: str  # 'numeric' or 'categorical'
    timestamp: float

    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Categorical statistics
    value_counts: Optional[Dict[str, int]] = None
    num_categories: Optional[int] = None
    top_categories: Optional[List[str]] = field(default_factory=list)

    # Common statistics
    null_count: int = 0
    total_count: int = 0
    null_ratio: float = 0.0

    # Histogram for numeric data
    histogram: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class DriftReport:
    """Report of detected drift."""

    column_name: str
    drift_detected: bool
    drift_score: float  # 0-1, higher = more drift
    p_value: float  # Statistical significance
    test_used: str  # 'ks_test', 'chi_square', etc.
    changes: Dict[str, Any]
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    recommendation: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'column_name': self.column_name,
            'drift_detected': self.drift_detected,
            'drift_score': self.drift_score,
            'p_value': self.p_value,
            'test_used': self.test_used,
            'changes': self.changes,
            'severity': self.severity,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp
        }


class DriftDetector:
    """
    Monitor for data distribution changes.

    Compares new data against reference distributions and detects drift.
    """

    def __init__(self, significance_level: float = 0.05,
                 drift_threshold: float = 0.3):
        """
        Initialize drift detector.

        Args:
            significance_level: P-value threshold for statistical tests (default: 0.05)
            drift_threshold: Minimum drift score to trigger alert (0-1)
        """
        self.significance_level = significance_level
        self.drift_threshold = drift_threshold

        # Store reference distributions
        self.reference_profiles: Dict[str, DistributionProfile] = {}

        # Store raw data samples for statistical tests
        self.reference_samples: Dict[str, np.ndarray] = {}

        # Drift history
        self.drift_history: List[DriftReport] = []

    def set_reference(self, column_name: str, column_data: pd.Series,
                     max_sample_size: int = 5000):
        """
        Set reference distribution for a column.

        Args:
            column_name: Name of the column
            column_data: Reference data
            max_sample_size: Maximum sample size to store
        """
        profile = self._create_profile(column_name, column_data)
        self.reference_profiles[column_name] = profile

        # Store sample for statistical tests
        clean_data = column_data.dropna()
        if len(clean_data) > max_sample_size:
            # Random sample
            clean_data = clean_data.sample(n=max_sample_size, random_state=42)

        if pd.api.types.is_numeric_dtype(column_data):
            self.reference_samples[column_name] = clean_data.values
        else:
            # For categorical, store value counts
            self.reference_samples[column_name] = clean_data.values

    def detect_drift(self, column_name: str, new_data: pd.Series) -> DriftReport:
        """
        Detect if new data has drifted from reference.

        Args:
            column_name: Name of column
            new_data: New data to compare

        Returns:
            DriftReport with drift detection results
        """
        if column_name not in self.reference_profiles:
            return DriftReport(
                column_name=column_name,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                test_used='none',
                changes={},
                severity='none',
                recommendation='No reference distribution set',
                timestamp=datetime.now().timestamp()
            )

        ref_profile = self.reference_profiles[column_name]
        ref_sample = self.reference_samples[column_name]

        # Perform appropriate drift test based on data type
        if ref_profile.dtype == 'numeric':
            return self._detect_numeric_drift(column_name, new_data,
                                              ref_profile, ref_sample)
        else:
            return self._detect_categorical_drift(column_name, new_data,
                                                  ref_profile, ref_sample)

    def _detect_numeric_drift(self, column_name: str, new_data: pd.Series,
                             ref_profile: DistributionProfile,
                             ref_sample: np.ndarray) -> DriftReport:
        """Detect drift in numeric data using Kolmogorov-Smirnov test."""
        new_clean = new_data.dropna().values

        if len(new_clean) == 0:
            return DriftReport(
                column_name=column_name,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                test_used='none',
                changes={'error': 'No valid data'},
                severity='none',
                recommendation='Column has no valid values',
                timestamp=datetime.now().timestamp()
            )

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_sample, new_clean)

        # Calculate changes in statistics
        new_mean = new_clean.mean()
        new_std = new_clean.std()
        new_median = np.median(new_clean)

        changes = {
            'mean_shift': float(new_mean - ref_profile.mean) if ref_profile.mean else 0.0,
            'mean_shift_pct': float((new_mean - ref_profile.mean) / (ref_profile.mean + 1e-10) * 100) if ref_profile.mean else 0.0,
            'std_ratio': float(new_std / (ref_profile.std + 1e-10)) if ref_profile.std else 0.0,
            'median_shift': float(new_median - ref_profile.median) if ref_profile.median else 0.0,
            'ks_statistic': float(ks_stat),
            'distribution_divergence': float(ks_stat)
        }

        # Determine drift severity
        drift_detected = p_value < self.significance_level
        drift_score = ks_stat  # KS statistic is the drift score

        if not drift_detected:
            severity = 'none'
            recommendation = 'No action needed'
        elif drift_score < 0.1:
            severity = 'low'
            recommendation = 'Monitor closely'
        elif drift_score < 0.2:
            severity = 'medium'
            recommendation = 'Consider investigating changes'
        elif drift_score < 0.3:
            severity = 'high'
            recommendation = 'Retrain model recommended'
        else:
            severity = 'critical'
            recommendation = 'Retrain model immediately'

        report = DriftReport(
            column_name=column_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            test_used='kolmogorov_smirnov',
            changes=changes,
            severity=severity,
            recommendation=recommendation,
            timestamp=datetime.now().timestamp()
        )

        self.drift_history.append(report)
        return report

    def _detect_categorical_drift(self, column_name: str, new_data: pd.Series,
                                  ref_profile: DistributionProfile,
                                  ref_sample: np.ndarray) -> DriftReport:
        """Detect drift in categorical data using Chi-square test."""
        new_clean = new_data.dropna()

        if len(new_clean) == 0:
            return DriftReport(
                column_name=column_name,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                test_used='none',
                changes={'error': 'No valid data'},
                severity='none',
                recommendation='Column has no valid values',
                timestamp=datetime.now().timestamp()
            )

        # Get value counts
        new_value_counts = new_clean.value_counts()
        ref_value_counts = pd.Series(ref_profile.value_counts) if ref_profile.value_counts else pd.Series()

        # Align categories (union of both)
        all_categories = set(new_value_counts.index) | set(ref_value_counts.index)

        ref_counts = np.array([ref_value_counts.get(cat, 0) for cat in all_categories])
        new_counts = np.array([new_value_counts.get(cat, 0) for cat in all_categories])

        # Normalize to probabilities
        ref_probs = ref_counts / (ref_counts.sum() + 1e-10)
        new_probs = new_counts / (new_counts.sum() + 1e-10)

        # Chi-square test
        try:
            chi2_stat, p_value = stats.chisquare(new_counts + 1, ref_counts + 1)  # +1 for smoothing
        except:
            chi2_stat, p_value = 0.0, 1.0

        # Calculate KL divergence as drift score
        kl_div = stats.entropy(new_probs + 1e-10, ref_probs + 1e-10)
        drift_score = min(kl_div / 5.0, 1.0)  # Normalize to 0-1

        # Find new categories
        new_categories = set(new_value_counts.index) - set(ref_value_counts.index)
        missing_categories = set(ref_value_counts.index) - set(new_value_counts.index)

        changes = {
            'new_categories': list(new_categories),
            'missing_categories': list(missing_categories),
            'num_new_categories': len(new_categories),
            'num_missing_categories': len(missing_categories),
            'kl_divergence': float(kl_div),
            'chi2_statistic': float(chi2_stat)
        }

        # Determine drift severity
        drift_detected = p_value < self.significance_level

        if not drift_detected:
            severity = 'none'
            recommendation = 'No action needed'
        elif drift_score < 0.2:
            severity = 'low'
            recommendation = 'Monitor closely'
        elif drift_score < 0.4:
            severity = 'medium'
            recommendation = 'Consider investigating changes'
        elif drift_score < 0.6:
            severity = 'high'
            recommendation = 'Retrain model recommended'
        else:
            severity = 'critical'
            recommendation = 'Retrain model immediately'

        report = DriftReport(
            column_name=column_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            test_used='chi_square',
            changes=changes,
            severity=severity,
            recommendation=recommendation,
            timestamp=datetime.now().timestamp()
        )

        self.drift_history.append(report)
        return report

    def _create_profile(self, column_name: str, column_data: pd.Series) -> DistributionProfile:
        """Create statistical profile of a column."""
        total_count = len(column_data)
        null_count = column_data.isna().sum()
        null_ratio = null_count / total_count if total_count > 0 else 0.0

        clean_data = column_data.dropna()

        if pd.api.types.is_numeric_dtype(column_data):
            # Numeric profile
            if len(clean_data) > 0:
                quantiles = clean_data.quantile([0.25, 0.5, 0.75])
                hist, bin_edges = np.histogram(clean_data, bins=50)

                profile = DistributionProfile(
                    column_name=column_name,
                    dtype='numeric',
                    timestamp=datetime.now().timestamp(),
                    mean=float(clean_data.mean()),
                    std=float(clean_data.std()),
                    median=float(clean_data.median()),
                    q25=float(quantiles[0.25]),
                    q75=float(quantiles[0.75]),
                    min_val=float(clean_data.min()),
                    max_val=float(clean_data.max()),
                    skewness=float(clean_data.skew()),
                    kurtosis=float(clean_data.kurtosis()),
                    null_count=null_count,
                    total_count=total_count,
                    null_ratio=null_ratio,
                    histogram=(hist, bin_edges)
                )
            else:
                profile = DistributionProfile(
                    column_name=column_name,
                    dtype='numeric',
                    timestamp=datetime.now().timestamp(),
                    null_count=null_count,
                    total_count=total_count,
                    null_ratio=null_ratio
                )
        else:
            # Categorical profile
            value_counts = clean_data.value_counts()
            top_categories = value_counts.head(20).index.tolist()

            profile = DistributionProfile(
                column_name=column_name,
                dtype='categorical',
                timestamp=datetime.now().timestamp(),
                value_counts=value_counts.to_dict(),
                num_categories=len(value_counts),
                top_categories=top_categories,
                null_count=null_count,
                total_count=total_count,
                null_ratio=null_ratio
            )

        return profile

    def detect_dataset_drift(self, df: pd.DataFrame,
                            reference_df: Optional[pd.DataFrame] = None) -> Dict[str, DriftReport]:
        """
        Detect drift across entire dataset.

        Args:
            df: New dataset
            reference_df: Optional reference dataset (if not already set)

        Returns:
            Dictionary of column_name -> DriftReport
        """
        if reference_df is not None:
            # Set all references
            for col in reference_df.columns:
                self.set_reference(col, reference_df[col])

        # Detect drift for all columns
        reports = {}
        for col in df.columns:
            if col in self.reference_profiles:
                report = self.detect_drift(col, df[col])
                reports[col] = report

        return reports

    def get_drift_summary(self, reports: Dict[str, DriftReport]) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        total_columns = len(reports)
        drifted_columns = sum(1 for r in reports.values() if r.drift_detected)

        severity_counts = {}
        for report in reports.values():
            severity_counts[report.severity] = severity_counts.get(report.severity, 0) + 1

        return {
            'total_columns': total_columns,
            'drifted_columns': drifted_columns,
            'drift_percentage': drifted_columns / total_columns * 100 if total_columns > 0 else 0.0,
            'severity_counts': severity_counts,
            'requires_retraining': any(r.severity in ['high', 'critical'] for r in reports.values()),
            'columns_by_severity': {
                'critical': [r.column_name for r in reports.values() if r.severity == 'critical'],
                'high': [r.column_name for r in reports.values() if r.severity == 'high'],
                'medium': [r.column_name for r in reports.values() if r.severity == 'medium'],
                'low': [r.column_name for r in reports.values() if r.severity == 'low']
            }
        }

    def save_reference(self, file_path: str):
        """Save reference distributions to file."""
        data = {
            'profiles': {name: {
                'column_name': p.column_name,
                'dtype': p.dtype,
                'timestamp': p.timestamp,
                'mean': p.mean,
                'std': p.std,
                'median': p.median,
                'q25': p.q25,
                'q75': p.q75,
                'min_val': p.min_val,
                'max_val': p.max_val,
                'skewness': p.skewness,
                'kurtosis': p.kurtosis,
                'value_counts': p.value_counts,
                'num_categories': p.num_categories,
                'top_categories': p.top_categories,
                'null_count': p.null_count,
                'total_count': p.total_count,
                'null_ratio': p.null_ratio
            } for name, p in self.reference_profiles.items()},
            'samples': {name: sample.tolist() for name, sample in self.reference_samples.items()}
        }

        Path(file_path).write_text(json.dumps(data, indent=2))

    def load_reference(self, file_path: str):
        """Load reference distributions from file."""
        data = json.loads(Path(file_path).read_text())

        # Load profiles
        for name, profile_dict in data['profiles'].items():
            profile = DistributionProfile(**profile_dict)
            self.reference_profiles[name] = profile

        # Load samples
        for name, sample_list in data['samples'].items():
            self.reference_samples[name] = np.array(sample_list)


# Global drift detector instance
_global_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create global drift detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = DriftDetector()
    return _global_detector
