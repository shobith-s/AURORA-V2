"""
Enhanced Feature Extractor - Phase 1 Improvements.

Adds statistical tests and semantic pattern detection for better decision making.
Expected impact: +10-15% accuracy on edge cases.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import re
from scipy import stats
from .minimal_extractor import MinimalFeatureExtractor, MinimalFeatures


@dataclass
class EnhancedFeatures:
    """Enhanced feature set with statistical tests and pattern detection."""

    # Original 10 features (for backward compatibility)
    minimal_features: MinimalFeatures

    # Statistical test features (5 new features)
    is_normal: bool  # Shapiro-Wilk test
    is_bimodal: bool  # Hartigan's dip test approximation
    is_lognormal: bool  # Log-normal test
    normality_p_value: float  # P-value from normality test
    kurtosis: float  # Fourth moment (tail heaviness)

    # Semantic pattern features (7 new features)
    email_ratio: float  # Ratio of values matching email pattern
    url_ratio: float  # Ratio matching URL pattern
    phone_ratio: float  # Ratio matching phone pattern
    date_ratio: float  # Ratio matching date patterns
    currency_ratio: float  # Ratio with currency symbols
    id_pattern_ratio: float  # Ratio matching ID patterns
    code_pattern_ratio: float  # Ratio matching code/alphanumeric patterns

    # Distribution shape features (5 new features)
    num_modes: int  # Number of peaks in distribution
    left_tail_heaviness: float  # Heavy left tail indicator
    right_tail_heaviness: float  # Heavy right tail indicator
    range_compression: float  # IQR / total range
    coefficient_variation: float  # CV = std / mean

    # Temporal features (3 new features)
    is_likely_temporal: bool  # Column name suggests temporal
    is_datetime_parseable: float  # Ratio of values parseable as datetime
    date_range_days: float  # Range in days if datetime

    def to_array(self) -> np.ndarray:
        """Convert to numpy array (27 total features)."""
        # Original 10 features
        base = self.minimal_features.to_array()

        # Additional 17 features
        additional = np.array([
            float(self.is_normal),
            float(self.is_bimodal),
            float(self.is_lognormal),
            self.normality_p_value,
            self.kurtosis,
            self.email_ratio,
            self.url_ratio,
            self.phone_ratio,
            self.date_ratio,
            self.currency_ratio,
            self.id_pattern_ratio,
            self.code_pattern_ratio,
            self.num_modes,
            self.left_tail_heaviness,
            self.right_tail_heaviness,
            self.range_compression,
            self.coefficient_variation,
            float(self.is_likely_temporal),
            self.is_datetime_parseable,
            self.date_range_days
        ], dtype=np.float32)

        return np.concatenate([base, additional])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.minimal_features.to_dict()
        result.update({
            "is_normal": self.is_normal,
            "is_bimodal": self.is_bimodal,
            "is_lognormal": self.is_lognormal,
            "normality_p_value": self.normality_p_value,
            "kurtosis": self.kurtosis,
            "email_ratio": self.email_ratio,
            "url_ratio": self.url_ratio,
            "phone_ratio": self.phone_ratio,
            "date_ratio": self.date_ratio,
            "currency_ratio": self.currency_ratio,
            "id_pattern_ratio": self.id_pattern_ratio,
            "code_pattern_ratio": self.code_pattern_ratio,
            "num_modes": self.num_modes,
            "left_tail_heaviness": self.left_tail_heaviness,
            "right_tail_heaviness": self.right_tail_heaviness,
            "range_compression": self.range_compression,
            "coefficient_variation": self.coefficient_variation,
            "is_likely_temporal": self.is_likely_temporal,
            "is_datetime_parseable": self.is_datetime_parseable,
            "date_range_days": self.date_range_days
        })
        return result


class EnhancedFeatureExtractor:
    """Feature extractor with statistical tests and pattern detection."""

    def __init__(self):
        self.minimal_extractor = MinimalFeatureExtractor()

        # Compile regex patterns once for performance
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        )
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.date_patterns = [
            re.compile(r'\d{4}-\d{2}-\d{2}'),  # ISO
            re.compile(r'\d{2}/\d{2}/\d{4}'),  # US
            re.compile(r'\d{2}-\d{2}-\d{4}'),  # EU
        ]
        self.currency_pattern = re.compile(r'[$£€¥]\s*\d+')
        self.id_pattern = re.compile(r'^[A-Z]{2,4}-?\d{4,}$')
        self.code_pattern = re.compile(r'^[A-Za-z0-9_-]{8,}$')

        # Temporal keywords
        self.temporal_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated',
            'year', 'month', 'day', 'hour', 'minute', 'second'
        ]

    def extract(self, column: pd.Series, column_name: str = "") -> EnhancedFeatures:
        """
        Extract enhanced features from a column.

        Args:
            column: Pandas Series to analyze
            column_name: Name of the column

        Returns:
            EnhancedFeatures object with 27 features
        """
        # Get base features
        minimal_features = self.minimal_extractor.extract(column)

        # Extract enhanced features
        stat_features = self._extract_statistical_tests(column)
        pattern_features = self._extract_semantic_patterns(column)
        dist_features = self._extract_distribution_features(column)
        temporal_features = self._extract_temporal_features(column, column_name)

        return EnhancedFeatures(
            minimal_features=minimal_features,
            **stat_features,
            **pattern_features,
            **dist_features,
            **temporal_features
        )

    def _extract_statistical_tests(self, column: pd.Series) -> Dict[str, Any]:
        """Extract statistical test results."""
        features = {
            'is_normal': False,
            'is_bimodal': False,
            'is_lognormal': False,
            'normality_p_value': 0.0,
            'kurtosis': 0.0
        }

        if pd.api.types.is_numeric_dtype(column):
            clean = column.dropna()

            if len(clean) >= 3:
                try:
                    # Normality test (Shapiro-Wilk)
                    # For large samples (>5000), use D'Agostino-Pearson test
                    if len(clean) > 5000:
                        _, p_value = stats.normaltest(clean)
                    else:
                        _, p_value = stats.shapiro(clean)

                    features['normality_p_value'] = float(p_value)
                    features['is_normal'] = p_value > 0.05

                    # Kurtosis (tail heaviness)
                    features['kurtosis'] = float(stats.kurtosis(clean))

                    # Bimodality test (simple heuristic: kurtosis < 1)
                    # Proper Hartigan's dip test is more complex
                    features['is_bimodal'] = features['kurtosis'] < -1.0

                    # Log-normal test
                    if clean.min() > 0:
                        log_data = np.log(clean)
                        _, log_p = stats.normaltest(log_data)
                        features['is_lognormal'] = log_p > 0.05 and not features['is_normal']

                except Exception:
                    # If tests fail, keep defaults
                    pass

        return features

    def _extract_semantic_patterns(self, column: pd.Series) -> Dict[str, float]:
        """Extract semantic pattern ratios."""
        features = {
            'email_ratio': 0.0,
            'url_ratio': 0.0,
            'phone_ratio': 0.0,
            'date_ratio': 0.0,
            'currency_ratio': 0.0,
            'id_pattern_ratio': 0.0,
            'code_pattern_ratio': 0.0
        }

        if pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            sample = column.dropna().astype(str).head(1000)  # Sample for performance

            if len(sample) == 0:
                return features

            try:
                # Email pattern
                features['email_ratio'] = sample.str.contains(
                    self.email_pattern, regex=True
                ).mean()

                # URL pattern
                features['url_ratio'] = sample.str.contains(
                    self.url_pattern, regex=True
                ).mean()

                # Phone pattern
                features['phone_ratio'] = sample.str.contains(
                    self.phone_pattern, regex=True
                ).mean()

                # Date patterns (max across all patterns)
                date_matches = []
                for pattern in self.date_patterns:
                    date_matches.append(
                        sample.str.contains(pattern, regex=True).mean()
                    )
                features['date_ratio'] = max(date_matches) if date_matches else 0.0

                # Currency pattern
                features['currency_ratio'] = sample.str.contains(
                    self.currency_pattern, regex=True
                ).mean()

                # ID pattern
                features['id_pattern_ratio'] = sample.str.contains(
                    self.id_pattern, regex=True
                ).mean()

                # Code pattern
                features['code_pattern_ratio'] = sample.str.contains(
                    self.code_pattern, regex=True
                ).mean()

            except Exception:
                # If pattern matching fails, keep defaults
                pass

        return features

    def _extract_distribution_features(self, column: pd.Series) -> Dict[str, Any]:
        """Extract distribution shape characteristics."""
        features = {
            'num_modes': 1,
            'left_tail_heaviness': 0.0,
            'right_tail_heaviness': 0.0,
            'range_compression': 0.0,
            'coefficient_variation': 0.0
        }

        if pd.api.types.is_numeric_dtype(column):
            clean = column.dropna()

            if len(clean) < 3:
                return features

            try:
                # Percentile analysis
                percentiles = clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

                iqr = percentiles[0.75] - percentiles[0.25]

                if iqr > 0:
                    # Tail heaviness
                    features['left_tail_heaviness'] = float(
                        (percentiles[0.25] - percentiles[0.01]) / iqr
                    )
                    features['right_tail_heaviness'] = float(
                        (percentiles[0.99] - percentiles[0.75]) / iqr
                    )

                    # Range compression (how compressed is the middle 50%?)
                    total_range = clean.max() - clean.min()
                    if total_range > 0:
                        features['range_compression'] = float(iqr / total_range)

                # Coefficient of variation
                mean_val = clean.mean()
                if abs(mean_val) > 1e-10:
                    features['coefficient_variation'] = float(clean.std() / abs(mean_val))

                # Simple modality detection using histogram
                # Count peaks in histogram
                if len(clean) >= 20:
                    hist, bin_edges = np.histogram(clean, bins=min(20, len(clean) // 10))

                    # Find local maxima
                    peaks = 0
                    for i in range(1, len(hist) - 1):
                        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                            # Only count significant peaks (>10% of max)
                            if hist[i] > 0.1 * hist.max():
                                peaks += 1

                    features['num_modes'] = max(1, peaks)

            except Exception:
                # If calculation fails, keep defaults
                pass

        return features

    def _extract_temporal_features(self, column: pd.Series,
                                   column_name: str) -> Dict[str, Any]:
        """Extract temporal characteristics."""
        features = {
            'is_likely_temporal': False,
            'is_datetime_parseable': 0.0,
            'date_range_days': 0.0
        }

        # Check column name for temporal keywords
        if any(kw in column_name.lower() for kw in self.temporal_keywords):
            features['is_likely_temporal'] = True

            try:
                # Try to parse as datetime
                dt_column = pd.to_datetime(column, errors='coerce')
                valid_ratio = dt_column.notna().mean()
                features['is_datetime_parseable'] = float(valid_ratio)

                if valid_ratio > 0.8:
                    valid_dates = dt_column.dropna()

                    if len(valid_dates) > 1:
                        # Calculate date range in days
                        date_range = (valid_dates.max() - valid_dates.min()).days
                        features['date_range_days'] = float(date_range)
            except Exception:
                # If parsing fails, keep defaults
                pass

        return features


class BackwardCompatibleExtractor(EnhancedFeatureExtractor):
    """
    Wrapper that provides backward compatibility with MinimalFeatureExtractor.

    Returns 10 features for old models, 27 for new models.
    """

    def extract(self, column: pd.Series, column_name: str = "",
                use_enhanced: bool = True) -> Any:
        """
        Extract features with optional enhancement.

        Args:
            column: Column to analyze
            column_name: Name of column
            use_enhanced: If False, return only 10 minimal features

        Returns:
            EnhancedFeatures if use_enhanced=True, else MinimalFeatures
        """
        if not use_enhanced:
            # Backward compatibility: return only 10 features
            return self.minimal_extractor.extract(column)

        # Return full 27 features
        return super().extract(column, column_name)
