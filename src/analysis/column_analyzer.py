"""
Column Analyzer - Semantic Type Detection for Preprocessing
=============================================================

Analyzes individual columns to detect:
- Semantic types (numeric_string, boolean_like, datetime_like, etc.)
- Pattern matching for special formats ($, %, commas, etc.)
- Column quality metrics
- Recommended preprocessing actions

Author: AURORA Team
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticTypeResult:
    """Result of semantic type detection."""
    semantic_type: str
    confidence: float
    detected_patterns: Dict[str, float] = field(default_factory=dict)
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColumnAnalyzer:
    """
    Analyze individual columns for semantic type detection.
    
    Detects semantic types beyond pandas dtypes:
    - numeric_string: "1,234.56", "$99.99", "45%"
    - boolean_like: "yes/no", "true/false", "Y/N", "1/0"
    - datetime_like: Various date/time formats
    - currency: "$1,234.56", "€100"
    - percentage: "45%", "0.45"
    - high_cardinality_categorical: IDs vs true categories
    - text: Paragraph vs category label
    - email, phone, url: Contact patterns
    """
    
    # Configuration constants
    HIGH_CARDINALITY_RATIO = 0.9  # Ratio above which is considered high cardinality
    HIGH_CARDINALITY_MIN_LENGTH = 10  # Min avg length for identifier detection
    HIGH_CARDINALITY_THRESHOLD = 50  # Cardinality threshold for high vs low
    TEXT_MIN_LENGTH = 50  # Min avg length to be considered text
    YEAR_MIN = 1900  # Minimum year for year detection
    YEAR_MAX = 2100  # Maximum year for year detection
    YEAR_MAX_UNIQUE = 200  # Max unique values for year detection
    NULL_INDICATORS = ['na', 'n/a', 'null', 'none', 'missing', '?', '-', '', 'nan', 'nil', 'undefined']
    NUMERIC_NULL_PLACEHOLDERS = [-999, -9999, 99999, 999999, -1]
    
    def __init__(
        self, 
        sample_size: int = 1000,
        high_cardinality_ratio: float = None,
        year_range: tuple = None,
        null_indicators: list = None
    ):
        """
        Initialize the column analyzer.
        
        Args:
            sample_size: Maximum number of rows to sample for analysis
            high_cardinality_ratio: Override default high cardinality ratio threshold
            year_range: Override default year range (min, max) for year detection
            null_indicators: Additional null indicator strings to detect
        """
        self.sample_size = sample_size
        
        # Allow overriding defaults
        if high_cardinality_ratio is not None:
            self.HIGH_CARDINALITY_RATIO = high_cardinality_ratio
        if year_range is not None:
            self.YEAR_MIN, self.YEAR_MAX = year_range
        if null_indicators is not None:
            self.NULL_INDICATORS = list(set(self.NULL_INDICATORS + null_indicators))
        
        # Define regex patterns for various types
        self.patterns = {
            # Numeric patterns
            'numeric_string': r'^-?\d+[,\d]*\.?\d*$',
            'currency': r'^[\$€£¥₹]\s*-?\d+[,\d]*\.?\d*$|^-?\d+[,\d]*\.?\d*\s*[\$€£¥₹]$',
            'percentage': r'^-?\d+\.?\d*\s*%$',
            'scientific': r'^-?\d+\.?\d*[eE][+-]?\d+$',
            
            # Boolean patterns
            'boolean_tf': r'^(?:true|false|t|f)$',
            'boolean_yn': r'^(?:yes|no|y|n)$',
            'boolean_01': r'^[01]$',
            'boolean_onoff': r'^(?:on|off)$',
            
            # DateTime patterns
            'iso_datetime': r'^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?',
            'us_date': r'^\d{1,2}/\d{1,2}/\d{2,4}$',
            'eu_date': r'^\d{1,2}-\d{1,2}-\d{2,4}$',
            'named_date': r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s*\d{4}$',
            
            # Contact patterns
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$|^\(\d{3}\)\s*\d{3}-\d{4}$|^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$',
            'url': r'^https?://[^\s]+$|^www\.[^\s]+$',
            
            # ID patterns
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'alphanumeric_id': r'^[A-Z0-9]{6,}$',
        }
    
    def detect_semantic_type(self, column: pd.Series, column_name: str = "") -> SemanticTypeResult:
        """
        Detect semantic type beyond pandas dtype.
        
        Args:
            column: Pandas Series to analyze
            column_name: Name of the column for context
            
        Returns:
            SemanticTypeResult with detected type and confidence
        """
        # Get sample for analysis
        sample = column.dropna().head(self.sample_size)
        
        if len(sample) == 0:
            return SemanticTypeResult(
                semantic_type="empty",
                confidence=1.0,
                suggested_action="drop_column"
            )
        
        # Already numeric
        if pd.api.types.is_numeric_dtype(column):
            return self._analyze_numeric_column(column, column_name)
        
        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(column):
            return SemanticTypeResult(
                semantic_type="datetime",
                confidence=1.0,
                suggested_action="datetime_extract_year"
            )
        
        # String/Object type - need deeper analysis
        sample_str = sample.astype(str).str.strip()
        
        # Check patterns in order of specificity
        patterns_found = {}
        
        # Check for numeric strings
        numeric_score = self._check_pattern(sample_str, self.patterns['numeric_string'])
        currency_score = self._check_pattern(sample_str, self.patterns['currency'])
        percentage_score = self._check_pattern(sample_str, self.patterns['percentage'])
        
        if currency_score > 0.8:
            return SemanticTypeResult(
                semantic_type="currency",
                confidence=currency_score,
                detected_patterns={'currency': currency_score},
                suggested_action="currency_normalize"
            )
        
        if percentage_score > 0.8:
            return SemanticTypeResult(
                semantic_type="percentage",
                confidence=percentage_score,
                detected_patterns={'percentage': percentage_score},
                suggested_action="percentage_to_decimal"
            )
        
        if numeric_score > 0.8:
            return SemanticTypeResult(
                semantic_type="numeric_string",
                confidence=numeric_score,
                detected_patterns={'numeric_string': numeric_score},
                suggested_action="parse_numeric"
            )
        
        # Check for boolean-like
        boolean_tf = self._check_pattern(sample_str, self.patterns['boolean_tf'], ignore_case=True)
        boolean_yn = self._check_pattern(sample_str, self.patterns['boolean_yn'], ignore_case=True)
        boolean_01 = self._check_pattern(sample_str, self.patterns['boolean_01'])
        boolean_onoff = self._check_pattern(sample_str, self.patterns['boolean_onoff'], ignore_case=True)
        
        max_boolean = max(boolean_tf, boolean_yn, boolean_01, boolean_onoff)
        if max_boolean > 0.8 and column.nunique() <= 3:
            return SemanticTypeResult(
                semantic_type="boolean_like",
                confidence=max_boolean,
                detected_patterns={
                    'boolean_tf': boolean_tf,
                    'boolean_yn': boolean_yn,
                    'boolean_01': boolean_01,
                    'boolean_onoff': boolean_onoff
                },
                suggested_action="parse_boolean"
            )
        
        # Check for datetime-like
        iso_score = self._check_pattern(sample_str, self.patterns['iso_datetime'])
        us_date_score = self._check_pattern(sample_str, self.patterns['us_date'])
        eu_date_score = self._check_pattern(sample_str, self.patterns['eu_date'])
        named_date_score = self._check_pattern(sample_str, self.patterns['named_date'], ignore_case=True)
        
        max_datetime = max(iso_score, us_date_score, eu_date_score, named_date_score)
        if max_datetime > 0.7:
            return SemanticTypeResult(
                semantic_type="datetime_like",
                confidence=max_datetime,
                detected_patterns={
                    'iso_datetime': iso_score,
                    'us_date': us_date_score,
                    'eu_date': eu_date_score,
                    'named_date': named_date_score
                },
                suggested_action="parse_datetime"
            )
        
        # Check for contact patterns
        email_score = self._check_pattern(sample_str, self.patterns['email'])
        phone_score = self._check_pattern(sample_str, self.patterns['phone'])
        url_score = self._check_pattern(sample_str, self.patterns['url'])
        
        if email_score > 0.8:
            return SemanticTypeResult(
                semantic_type="email",
                confidence=email_score,
                detected_patterns={'email': email_score},
                suggested_action="email_validate"
            )
        
        if phone_score > 0.8:
            return SemanticTypeResult(
                semantic_type="phone",
                confidence=phone_score,
                detected_patterns={'phone': phone_score},
                suggested_action="phone_standardize"
            )
        
        if url_score > 0.8:
            return SemanticTypeResult(
                semantic_type="url",
                confidence=url_score,
                detected_patterns={'url': url_score},
                suggested_action="url_parse"
            )
        
        # Check for ID patterns
        uuid_score = self._check_pattern(sample_str, self.patterns['uuid'], ignore_case=True)
        alphanum_id_score = self._check_pattern(sample_str, self.patterns['alphanumeric_id'])
        
        if uuid_score > 0.8:
            return SemanticTypeResult(
                semantic_type="uuid",
                confidence=uuid_score,
                detected_patterns={'uuid': uuid_score},
                suggested_action="keep_as_is",
                metadata={'is_identifier': True}
            )
        
        # Check cardinality for categorical vs text
        cardinality = column.nunique()
        cardinality_ratio = cardinality / len(column) if len(column) > 0 else 0
        avg_length = sample_str.str.len().mean()
        
        # High cardinality categorical (IDs)
        if cardinality_ratio > self.HIGH_CARDINALITY_RATIO and avg_length > self.HIGH_CARDINALITY_MIN_LENGTH:
            return SemanticTypeResult(
                semantic_type="high_cardinality_identifier",
                confidence=0.8,
                metadata={
                    'cardinality': cardinality,
                    'avg_length': avg_length,
                    'is_identifier': True
                },
                suggested_action="hash_encode"
            )
        
        # High cardinality categorical (not IDs)
        if cardinality > self.HIGH_CARDINALITY_THRESHOLD and cardinality_ratio > 0.3:
            return SemanticTypeResult(
                semantic_type="high_cardinality_categorical",
                confidence=0.75,
                metadata={
                    'cardinality': cardinality,
                    'cardinality_ratio': cardinality_ratio
                },
                suggested_action="frequency_encode"
            )
        
        # Text (long strings with high variance)
        if avg_length > self.TEXT_MIN_LENGTH:
            return SemanticTypeResult(
                semantic_type="text",
                confidence=0.8,
                metadata={'avg_length': avg_length},
                suggested_action="text_vectorize_tfidf"
            )
        
        # Regular categorical
        if cardinality < self.HIGH_CARDINALITY_THRESHOLD:
            return SemanticTypeResult(
                semantic_type="categorical",
                confidence=0.85,
                metadata={'cardinality': cardinality},
                suggested_action="onehot_encode" if cardinality < 10 else "label_encode"
            )
        
        # Unknown
        return SemanticTypeResult(
            semantic_type="unknown",
            confidence=0.5,
            suggested_action="keep_as_is"
        )
    
    def _analyze_numeric_column(self, column: pd.Series, column_name: str) -> SemanticTypeResult:
        """Analyze an already numeric column for special patterns."""
        non_null = column.dropna()
        
        if len(non_null) == 0:
            return SemanticTypeResult(
                semantic_type="empty_numeric",
                confidence=1.0,
                suggested_action="drop_column"
            )
        
        min_val = non_null.min()
        max_val = non_null.max()
        unique_count = non_null.nunique()
        
        # Year detection
        if (self.YEAR_MIN <= min_val <= self.YEAR_MAX and 
            self.YEAR_MIN <= max_val <= self.YEAR_MAX and 
            unique_count <= self.YEAR_MAX_UNIQUE):
            return SemanticTypeResult(
                semantic_type="year",
                confidence=0.85,
                metadata={'min_year': min_val, 'max_year': max_val},
                suggested_action="keep_as_is"  # Years are often meaningful as-is
            )
        
        # Percentage (0-100 or 0-1)
        if 0 <= min_val <= 1 and 0 <= max_val <= 1:
            return SemanticTypeResult(
                semantic_type="ratio",
                confidence=0.7,
                suggested_action="keep_as_is"
            )
        
        if 0 <= min_val <= 100 and 0 <= max_val <= 100:
            return SemanticTypeResult(
                semantic_type="percentage_numeric",
                confidence=0.6,
                suggested_action="keep_as_is"
            )
        
        # Binary
        if unique_count == 2 and set(non_null.unique()).issubset({0, 1}):
            return SemanticTypeResult(
                semantic_type="binary",
                confidence=0.95,
                suggested_action="keep_as_is"
            )
        
        # Regular numeric
        skewness = non_null.skew() if len(non_null) > 2 else 0
        
        if abs(skewness) > 2 and min_val > 0:
            return SemanticTypeResult(
                semantic_type="skewed_positive_numeric",
                confidence=0.8,
                metadata={'skewness': skewness},
                suggested_action="log_transform"
            )
        
        return SemanticTypeResult(
            semantic_type="numeric",
            confidence=0.9,
            metadata={'skewness': skewness if len(non_null) > 2 else None},
            suggested_action="standard_scale"
        )
    
    def _check_pattern(self, series: pd.Series, pattern: str, ignore_case: bool = False) -> float:
        """Check what fraction of values match a regex pattern."""
        flags = re.IGNORECASE if ignore_case else 0
        try:
            matches = series.str.match(pattern, flags=flags)
            return matches.sum() / len(series) if len(series) > 0 else 0.0
        except Exception:
            return 0.0
    
    def analyze_column_quality(self, column: pd.Series) -> Dict[str, Any]:
        """
        Analyze column quality metrics.
        
        Args:
            column: Pandas Series to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        total = len(column)
        null_count = column.isna().sum()
        unique_count = column.nunique()
        
        quality = {
            'total_rows': total,
            'null_count': int(null_count),
            'null_ratio': null_count / total if total > 0 else 0,
            'unique_count': int(unique_count),
            'unique_ratio': unique_count / total if total > 0 else 0,
            'duplicate_ratio': 1 - (unique_count / total) if total > 0 else 0,
            'dtype': str(column.dtype),
        }
        
        # Detect encoded nulls
        if not pd.api.types.is_numeric_dtype(column):
            sample = column.astype(str).str.lower().str.strip()
            encoded_nulls = sample.isin(self.NULL_INDICATORS).sum()
            quality['encoded_null_ratio'] = encoded_nulls / total if total > 0 else 0
        else:
            encoded_nulls = column.isin(self.NUMERIC_NULL_PLACEHOLDERS).sum()
            quality['encoded_null_ratio'] = encoded_nulls / total if total > 0 else 0
        
        return quality


# Singleton instance
_column_analyzer_instance: Optional[ColumnAnalyzer] = None


def get_column_analyzer() -> ColumnAnalyzer:
    """Get the global column analyzer instance."""
    global _column_analyzer_instance
    if _column_analyzer_instance is None:
        _column_analyzer_instance = ColumnAnalyzer()
    return _column_analyzer_instance
