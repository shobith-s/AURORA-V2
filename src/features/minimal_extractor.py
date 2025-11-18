"""
Minimal Feature Extractor for edge cases.
Extracts only 10 essential features when symbolic engine has low confidence.
Optimized for speed with numba and caching.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import jit
import hashlib


@dataclass
class MinimalFeatures:
    """Minimal feature set for neural oracle (only 10 features)."""

    # Basic statistics (4 features)
    null_percentage: float
    unique_ratio: float
    skewness: float  # 0.0 for non-numeric
    outlier_percentage: float  # 0.0 for non-numeric

    # Pattern detection (3 features)
    entropy: float  # Information content
    pattern_complexity: float  # Number of patterns detected
    multimodality_score: float  # 0.0 for non-numeric

    # Metadata (3 features)
    cardinality_bucket: int  # 0=low, 1=medium, 2=high, 3=unique
    detected_dtype: int  # 0=numeric, 1=categorical, 2=text, 3=temporal, 4=other
    column_name_signal: float  # Signal from column name tokens

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.null_percentage,
            self.unique_ratio,
            self.skewness,
            self.outlier_percentage,
            self.entropy,
            self.pattern_complexity,
            self.multimodality_score,
            float(self.cardinality_bucket),
            float(self.detected_dtype),
            self.column_name_signal
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "null_percentage": self.null_percentage,
            "unique_ratio": self.unique_ratio,
            "skewness": self.skewness,
            "outlier_percentage": self.outlier_percentage,
            "entropy": self.entropy,
            "pattern_complexity": self.pattern_complexity,
            "multimodality_score": self.multimodality_score,
            "cardinality_bucket": self.cardinality_bucket,
            "detected_dtype": self.detected_dtype,
            "column_name_signal": self.column_name_signal
        }


@jit(nopython=True)
def compute_entropy_numba(value_counts: np.ndarray, total: int) -> float:
    """
    Compute Shannon entropy using numba for speed.

    Args:
        value_counts: Array of value counts
        total: Total number of values

    Returns:
        Shannon entropy
    """
    entropy = 0.0
    for count in value_counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


@jit(nopython=True)
def compute_outliers_iqr_numba(values: np.ndarray) -> float:
    """
    Compute outlier percentage using IQR method with numba.

    Args:
        values: Numeric array

    Returns:
        Outlier percentage
    """
    if len(values) == 0:
        return 0.0

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = np.sum((values < lower_bound) | (values > upper_bound))
    return outliers / len(values)


@jit(nopython=True)
def compute_bimodality_numba(values: np.ndarray) -> float:
    """
    Compute bimodality coefficient using numba.

    Args:
        values: Numeric array

    Returns:
        Bimodality score (0-1, higher = more bimodal)
    """
    if len(values) < 3:
        return 0.0

    # Compute standardized skewness and kurtosis
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return 0.0

    n = len(values)
    skew = np.sum(((values - mean) / std) ** 3) / n
    kurt = np.sum(((values - mean) / std) ** 4) / n - 3

    # Bimodality coefficient
    numerator = skew ** 2 + 1
    denominator = kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))

    if denominator == 0:
        return 0.0

    return numerator / denominator


class MinimalFeatureExtractor:
    """
    Fast feature extractor that computes only 10 essential features.
    Uses numba for numerical computations and caching for repeated calls.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize the minimal feature extractor.

        Args:
            use_cache: Whether to cache computed features
        """
        self.use_cache = use_cache
        self._cache: Dict[str, MinimalFeatures] = {}

        # Column name keyword signals
        self.name_signals = {
            'id': -1.0,  # Likely to drop
            'key': -1.0,
            'uuid': -1.0,
            'timestamp': 0.8,  # Temporal
            'date': 0.8,
            'time': 0.8,
            'price': 0.6,  # Likely numeric transformation
            'amount': 0.6,
            'revenue': 0.6,
            'cost': 0.6,
            'age': 0.3,
            'count': 0.3,
            'category': 0.5,  # Categorical encoding
            'type': 0.5,
            'status': 0.5,
        }

    def _compute_cache_key(self, column: pd.Series, column_name: str) -> str:
        """Compute a cache key for a column."""
        # Hash the column data and name
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(column, index=False).values
        ).hexdigest()
        return f"{column_name}_{data_hash}"

    def extract(
        self,
        column: pd.Series,
        column_name: str = "",
        force_recompute: bool = False
    ) -> MinimalFeatures:
        """
        Extract minimal features from a column.

        Args:
            column: The column data
            column_name: Name of the column
            force_recompute: Force recomputation even if cached

        Returns:
            MinimalFeatures object
        """
        # Check cache
        if self.use_cache and not force_recompute:
            cache_key = self._compute_cache_key(column, column_name)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Compute features
        features = self._compute_features(column, column_name)

        # Cache result
        if self.use_cache:
            cache_key = self._compute_cache_key(column, column_name)
            self._cache[cache_key] = features

        return features

    def _compute_features(
        self,
        column: pd.Series,
        column_name: str
    ) -> MinimalFeatures:
        """Compute all 10 minimal features."""

        # Basic statistics
        total_count = len(column)
        null_count = column.isnull().sum()
        null_percentage = null_count / total_count if total_count > 0 else 0.0

        unique_count = column.nunique()
        unique_ratio = unique_count / total_count if total_count > 0 else 0.0

        # Type detection
        is_numeric = pd.api.types.is_numeric_dtype(column)
        is_categorical = pd.api.types.is_categorical_dtype(column) or \
                        (pd.api.types.is_object_dtype(column) and unique_ratio < 0.5)
        is_temporal = pd.api.types.is_datetime64_any_dtype(column)
        is_text = pd.api.types.is_string_dtype(column) or pd.api.types.is_object_dtype(column)

        # Detected dtype (encoded)
        if is_numeric:
            detected_dtype = 0
        elif is_categorical:
            detected_dtype = 1
        elif is_temporal:
            detected_dtype = 3
        elif is_text:
            detected_dtype = 2
        else:
            detected_dtype = 4

        # Cardinality bucket
        if unique_ratio > 0.95:
            cardinality_bucket = 3  # unique
        elif unique_count > 50:
            cardinality_bucket = 2  # high
        elif unique_count > 10:
            cardinality_bucket = 1  # medium
        else:
            cardinality_bucket = 0  # low

        # Numeric features (lazy computation)
        skewness = 0.0
        outlier_percentage = 0.0
        multimodality_score = 0.0

        if is_numeric and null_percentage < 1.0:
            non_null = column.dropna().values.astype(np.float64)

            if len(non_null) > 0:
                # Skewness
                mean = np.mean(non_null)
                std = np.std(non_null)
                if std > 0:
                    skewness = float(np.mean(((non_null - mean) / std) ** 3))

                # Outlier percentage using numba-optimized function
                outlier_percentage = float(compute_outliers_iqr_numba(non_null))

                # Multimodality score
                if len(non_null) >= 3:
                    multimodality_score = float(compute_bimodality_numba(non_null))

        # Entropy (information content)
        entropy = self._compute_entropy(column)

        # Pattern complexity
        pattern_complexity = self._compute_pattern_complexity(column)

        # Column name signal
        column_name_signal = self._extract_name_signal(column_name)

        return MinimalFeatures(
            null_percentage=null_percentage,
            unique_ratio=unique_ratio,
            skewness=skewness,
            outlier_percentage=outlier_percentage,
            entropy=entropy,
            pattern_complexity=pattern_complexity,
            multimodality_score=multimodality_score,
            cardinality_bucket=cardinality_bucket,
            detected_dtype=detected_dtype,
            column_name_signal=column_name_signal
        )

    def _compute_entropy(self, column: pd.Series) -> float:
        """Compute Shannon entropy."""
        if len(column) == 0:
            return 0.0

        value_counts = column.value_counts().values
        total = len(column)

        return float(compute_entropy_numba(value_counts, total))

    def _compute_pattern_complexity(self, column: pd.Series) -> float:
        """
        Compute pattern complexity score.
        Returns number of different patterns detected (0-1 normalized).
        """
        import re

        if not pd.api.types.is_string_dtype(column) and not pd.api.types.is_object_dtype(column):
            return 0.0

        sample = column.dropna().head(min(100, len(column))).astype(str)

        if len(sample) == 0:
            return 0.0

        patterns_detected = 0

        # Check for various patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO date
            r'\d+\.\d+',  # Decimal
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
            r'https?://',  # URL
            r'[$£€¥]',  # Currency symbols
            r'%',  # Percentage
            r'\(\d{3}\)',  # Phone
        ]

        for pattern in patterns:
            if sample.str.contains(pattern, regex=True).any():
                patterns_detected += 1

        # Normalize to 0-1
        return min(1.0, patterns_detected / len(patterns))

    def _extract_name_signal(self, column_name: str) -> float:
        """
        Extract signal from column name.
        Returns relevance score (-1 to 1).
        """
        if not column_name:
            return 0.0

        name_lower = column_name.lower()

        # Check for keyword matches
        for keyword, signal in self.name_signals.items():
            if keyword in name_lower:
                return signal

        # Default neutral signal
        return 0.0

    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Singleton instance for global use
_extractor_instance: Optional[MinimalFeatureExtractor] = None


def get_feature_extractor() -> MinimalFeatureExtractor:
    """Get the global feature extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MinimalFeatureExtractor()
    return _extractor_instance
