"""
Symbolic Engine - Core rule-based preprocessing decision engine.
Handles 80% of preprocessing decisions with deterministic rules.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import re
from datetime import datetime
import json

from .rules import get_all_rules, Rule, RuleCategory
from ..core.actions import PreprocessingAction, PreprocessingResult


@dataclass
class ColumnStatistics:
    """Statistics extracted from a column for rule evaluation."""

    # Basic properties
    row_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_ratio: float
    dtype: str

    # Type indicators
    is_numeric: bool = False
    is_categorical: bool = False
    is_text: bool = False
    is_temporal: bool = False
    is_boolean: bool = False

    # Numeric statistics (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Numeric properties
    all_positive: bool = False
    has_zeros: bool = False
    has_outliers: bool = False
    outlier_pct: float = 0.0
    is_already_scaled: bool = False
    has_natural_bounds: bool = False
    is_smooth: bool = False
    is_bimodal: bool = False

    # Categorical properties
    cardinality: int = 0
    is_ordinal: bool = False

    # Data quality
    duplicate_ratio: float = 0.0
    null_has_pattern: bool = False

    # NEW: Additional statistical metrics for universal coverage
    cv: Optional[float] = None  # Coefficient of variation
    entropy: Optional[float] = None  # Shannon entropy (information content)
    target_correlation: Optional[float] = None  # Correlation with target (if available)
    range_size: Optional[float] = None  # Max - Min
    iqr: Optional[float] = None  # Interquartile range

    # Pattern matching
    matches_iso_datetime: float = 0.0
    matches_date_pattern: float = 0.0
    matches_boolean_tf: float = 0.0
    matches_boolean_yn: float = 0.0
    matches_boolean_01: float = 0.0
    matches_numeric_pattern: float = 0.0
    matches_json: float = 0.0
    matches_currency_pattern: float = 0.0
    matches_phone_pattern: float = 0.0
    matches_email_pattern: float = 0.0
    matches_url_pattern: float = 0.0

    # Domain-specific
    has_currency_symbols: bool = False
    has_percentage_symbol: bool = False
    has_extra_whitespace: bool = False
    has_special_chars: bool = False
    looks_like_percentage: bool = False
    looks_like_categorical_code: bool = False

    # Target availability (for supervised learning)
    target_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for rule evaluation."""
        return {
            "row_count": self.row_count,
            "null_count": self.null_count,
            "null_pct": self.null_pct,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "dtype": self.dtype,
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "is_text": self.is_text,
            "is_temporal": self.is_temporal,
            "is_boolean": self.is_boolean,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "all_positive": self.all_positive,
            "has_zeros": self.has_zeros,
            "has_outliers": self.has_outliers,
            "outlier_pct": self.outlier_pct,
            "is_already_scaled": self.is_already_scaled,
            "has_natural_bounds": self.has_natural_bounds,
            "is_smooth": self.is_smooth,
            "is_bimodal": self.is_bimodal,
            "cardinality": self.cardinality,
            "is_ordinal": self.is_ordinal,
            "duplicate_ratio": self.duplicate_ratio,
            "null_has_pattern": self.null_has_pattern,
            "cv": self.cv,
            "entropy": self.entropy,
            "target_correlation": self.target_correlation,
            "range_size": self.range_size,
            "iqr": self.iqr,
            "matches_iso_datetime": self.matches_iso_datetime,
            "matches_date_pattern": self.matches_date_pattern,
            "matches_boolean_tf": self.matches_boolean_tf,
            "matches_boolean_yn": self.matches_boolean_yn,
            "matches_boolean_01": self.matches_boolean_01,
            "matches_numeric_pattern": self.matches_numeric_pattern,
            "matches_json": self.matches_json,
            "matches_currency_pattern": self.matches_currency_pattern,
            "matches_phone_pattern": self.matches_phone_pattern,
            "matches_email_pattern": self.matches_email_pattern,
            "matches_url_pattern": self.matches_url_pattern,
            "has_currency_symbols": self.has_currency_symbols,
            "has_percentage_symbol": self.has_percentage_symbol,
            "has_extra_whitespace": self.has_extra_whitespace,
            "has_special_chars": self.has_special_chars,
            "looks_like_percentage": self.looks_like_percentage,
            "looks_like_categorical_code": self.looks_like_categorical_code,
            "target_available": self.target_available,
        }


class SymbolicEngine:
    """
    Symbolic rule-based preprocessing engine.
    Uses 100+ deterministic rules to handle 80% of preprocessing decisions.
    """

    def __init__(self, confidence_threshold: float = 0.9):
        """
        Initialize the symbolic engine.

        Args:
            confidence_threshold: Minimum confidence to make a decision (default 0.9)
        """
        self.confidence_threshold = confidence_threshold
        self.rules = get_all_rules()
        self.decision_count = 0
        self.high_confidence_count = 0

    def compute_column_statistics(
        self,
        column: pd.Series,
        column_name: str = "",
        target_available: bool = False
    ) -> ColumnStatistics:
        """
        Compute comprehensive statistics for a column.

        Args:
            column: The column data
            column_name: Name of the column
            target_available: Whether target variable is available

        Returns:
            ColumnStatistics object
        """
        # Basic statistics
        row_count = len(column)
        null_count = column.isnull().sum()
        null_pct = null_count / row_count if row_count > 0 else 0
        unique_count = column.nunique()
        unique_ratio = unique_count / row_count if row_count > 0 else 0
        dtype = str(column.dtype)

        # Type detection
        is_numeric = pd.api.types.is_numeric_dtype(column)
        is_categorical = pd.api.types.is_categorical_dtype(column) or \
                        (pd.api.types.is_object_dtype(column) and unique_ratio < 0.5)
        is_text = pd.api.types.is_string_dtype(column) or pd.api.types.is_object_dtype(column)

        stats = ColumnStatistics(
            row_count=row_count,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            dtype=dtype,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_text=is_text,
            cardinality=unique_count,
            target_available=target_available
        )

        # Numeric statistics
        if is_numeric and null_pct < 1.0:
            non_null = column.dropna()
            stats.min_value = float(non_null.min())
            stats.max_value = float(non_null.max())
            stats.mean = float(non_null.mean())
            stats.median = float(non_null.median())
            stats.std = float(non_null.std())
            stats.skewness = float(non_null.skew())
            stats.kurtosis = float(non_null.kurtosis())

            stats.all_positive = stats.min_value > 0
            stats.has_zeros = (non_null == 0).any()

            # NEW: Additional statistical metrics
            stats.range_size = stats.max_value - stats.min_value

            # Coefficient of variation (CV = std / mean) - measures relative variability
            if stats.mean != 0:
                stats.cv = abs(stats.std / stats.mean)
            else:
                stats.cv = 0.0

            # Outlier detection using IQR
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            stats.iqr = float(iqr)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (non_null < lower_bound) | (non_null > upper_bound)
            stats.has_outliers = outliers.any()
            stats.outlier_pct = outliers.sum() / len(non_null) if len(non_null) > 0 else 0

            # Check if already scaled
            stats.is_already_scaled = (
                abs(stats.mean) < 0.1 and
                0.8 < stats.std < 1.2
            )

            # Check for natural bounds
            stats.has_natural_bounds = (
                (stats.min_value >= 0 and stats.max_value <= 1) or
                (stats.min_value >= 0 and stats.max_value <= 100)
            )

            # Check for bimodality (simple heuristic)
            stats.is_bimodal = abs(stats.kurtosis) < -1

        # Pattern matching for non-numeric columns
        if is_text and null_pct < 1.0:
            sample = column.dropna().head(min(1000, len(column)))
            sample_str = sample.astype(str)

            # DateTime patterns
            stats.matches_iso_datetime = self._check_pattern(sample_str, r'^\d{4}-\d{2}-\d{2}')
            stats.matches_date_pattern = self._check_pattern(sample_str,
                r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}')

            # Boolean patterns
            stats.matches_boolean_tf = self._check_pattern(sample_str, r'^(T|F|True|False|true|false)$', ignore_case=True)
            stats.matches_boolean_yn = self._check_pattern(sample_str, r'^(Y|N|Yes|No|yes|no)$', ignore_case=True)
            stats.matches_boolean_01 = self._check_pattern(sample_str, r'^[01]$')

            # Numeric pattern
            stats.matches_numeric_pattern = self._check_pattern(sample_str, r'^-?\d+\.?\d*$')

            # JSON pattern
            stats.matches_json = self._check_json(sample_str)

            # Currency pattern
            stats.has_currency_symbols = sample_str.str.contains(r'[$£€¥]', regex=True).any()
            stats.matches_currency_pattern = self._check_pattern(sample_str, r'^\$?£?€?¥?\d+[,\.]?\d*')

            # Percentage
            stats.has_percentage_symbol = sample_str.str.contains('%').any()

            # Contact patterns
            stats.matches_phone_pattern = self._check_pattern(sample_str,
                r'^\+?1?\d{9,15}$|^\(\d{3}\)\s*\d{3}-\d{4}$')
            stats.matches_email_pattern = self._check_pattern(sample_str,
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            stats.matches_url_pattern = self._check_pattern(sample_str,
                r'^https?://[^\s]+$')

            # Text quality
            stats.has_extra_whitespace = sample_str.str.contains(r'\s{2,}').any()
            stats.has_special_chars = sample_str.str.contains(r'[^\w\s]').any()

        # Duplicate ratio
        if row_count > 0:
            duplicates = row_count - unique_count
            stats.duplicate_ratio = duplicates / row_count

        # NEW: Compute Shannon entropy (information content)
        # Works for both categorical and numeric (discretized)
        if row_count > 0:
            value_counts = column.value_counts(normalize=True, dropna=True)
            if len(value_counts) > 0:
                # Shannon entropy: -sum(p * log2(p))
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                # Normalize by max entropy (log2(n)) to get 0-1 scale
                max_entropy = np.log2(len(value_counts))
                stats.entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            else:
                stats.entropy = 0.0

        # Check for ordinal pattern
        if is_categorical:
            stats.is_ordinal = self._detect_ordinal(column.dropna())

        return stats

    def _check_pattern(self, series: pd.Series, pattern: str, ignore_case: bool = False) -> float:
        """Check what fraction of values match a regex pattern."""
        flags = re.IGNORECASE if ignore_case else 0
        matches = series.str.match(pattern, flags=flags)
        return matches.sum() / len(series) if len(series) > 0 else 0.0

    def _check_json(self, series: pd.Series) -> float:
        """Check what fraction of values are valid JSON."""
        def is_json(s):
            try:
                json.loads(s)
                return True
            except:
                return False

        matches = series.apply(is_json)
        return matches.sum() / len(series) if len(series) > 0 else 0.0

    def _detect_ordinal(self, series: pd.Series) -> bool:
        """Detect if categorical variable is ordinal."""
        # Common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['never', 'sometimes', 'always'],
            ['bad', 'good', 'excellent'],
            ['poor', 'fair', 'good', 'excellent'],
        ]

        unique_values = set(str(v).lower() for v in series.unique())

        for pattern in ordinal_patterns:
            if unique_values.issubset(set(pattern)):
                return True

        # Check for numeric-like ordering
        values_list = [str(v) for v in series.unique()]
        if all(v.isdigit() for v in values_list):
            return True

        return False

    def evaluate(
        self,
        column: pd.Series,
        column_name: str = "",
        target_available: bool = False
    ) -> PreprocessingResult:
        """
        Evaluate column and return preprocessing recommendation.

        Args:
            column: The column data
            column_name: Name of the column
            target_available: Whether target variable is available

        Returns:
            PreprocessingResult with action, confidence, and explanation
        """
        # Compute statistics
        stats = self.compute_column_statistics(column, column_name, target_available)
        stats_dict = stats.to_dict()

        # Evaluate rules in priority order
        matches = []
        for rule in self.rules:
            result = rule.evaluate(stats_dict)
            if result:
                action, confidence, explanation = result
                matches.append((rule, action, confidence, explanation))

        # Track decision count
        self.decision_count += 1

        # If no rules match, keep as-is
        if not matches:
            return PreprocessingResult(
                action=PreprocessingAction.KEEP_AS_IS,
                confidence=0.5,
                source="symbolic",
                explanation="No specific preprocessing rules matched",
                alternatives=[],
                parameters={},
                context=stats_dict
            )

        # Get best match (highest confidence)
        best_rule, best_action, best_confidence, best_explanation = max(
            matches, key=lambda x: x[2]
        )

        # Get alternatives
        alternatives = [
            (action, confidence)
            for rule, action, confidence, explanation in matches
            if action != best_action
        ][:3]  # Top 3 alternatives

        # Track high confidence decisions
        if best_confidence >= self.confidence_threshold:
            self.high_confidence_count += 1

        return PreprocessingResult(
            action=best_action,
            confidence=best_confidence,
            source="symbolic",
            explanation=best_explanation,
            alternatives=alternatives,
            parameters=best_rule.parameters or {},
            context=stats_dict
        )

    def add_rule(self, rule: Rule):
        """Add a new rule to the engine (for learned rules)."""
        self.rules.append(rule)
        # Re-sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def coverage(self) -> float:
        """
        Calculate coverage: percentage of decisions made with high confidence.

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if self.decision_count == 0:
            return 0.0
        return self.high_confidence_count / self.decision_count

    def reset_stats(self):
        """Reset decision statistics."""
        self.decision_count = 0
        self.high_confidence_count = 0
