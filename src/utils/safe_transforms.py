"""
Safe Transforms - Safe transformation wrappers with pre-execution validation.

Provides safe versions of all transformations that:
1. Validate inputs before execution
2. Use safe mathematical operations (log1p instead of log)
3. Check cardinality limits
4. Handle errors gracefully with fallbacks

Never crash, always produce valid output.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TransformResult(Enum):
    """Result status of a transformation."""
    SUCCESS = "success"
    FALLBACK = "fallback"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SafeTransformResult:
    """Result of a safe transformation."""
    status: TransformResult
    data: Optional[Union[pd.Series, pd.DataFrame]]
    explanation: str
    warnings: List[str]
    original_action: str
    actual_action: str


class SafeTransforms:
    """
    Safe transformation wrappers that never crash.
    
    All methods:
    - Validate inputs first
    - Use safe mathematical operations
    - Fallback gracefully on errors
    - Return detailed result objects
    """
    
    # Configuration
    MAX_ONEHOT_CARDINALITY = 50  # Maximum categories for one-hot encoding
    MIN_ROWS_FOR_TRANSFORM = 2   # Minimum rows needed
    
    def __init__(self):
        """Initialize safe transforms."""
        pass
    
    def safe_standard_scale(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely apply standard scaling (z-score normalization).
        
        Validates:
        - Column is numeric
        - Has sufficient variance
        - No all-null values
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with scaled data or fallback
        """
        warnings = []
        
        # Validation: Check if numeric
        if not pd.api.types.is_numeric_dtype(column):
            try:
                column = pd.to_numeric(column, errors='coerce')
                warnings.append(f"Converted {column_name} to numeric")
            except Exception as e:
                return SafeTransformResult(
                    status=TransformResult.SKIPPED,
                    data=column,
                    explanation=f"Cannot standard scale non-numeric column: {e}",
                    warnings=[],
                    original_action="standard_scale",
                    actual_action="keep_as_is"
                )
        
        # Validation: Check for all nulls
        non_null = column.dropna()
        if len(non_null) < self.MIN_ROWS_FOR_TRANSFORM:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation=f"Insufficient non-null values ({len(non_null)})",
                warnings=[],
                original_action="standard_scale",
                actual_action="keep_as_is"
            )
        
        # Validation: Check for zero variance
        std = non_null.std()
        if std == 0 or np.isnan(std):
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Zero variance - cannot scale",
                warnings=[],
                original_action="standard_scale",
                actual_action="keep_as_is"
            )
        
        try:
            # Apply standard scaling
            mean = non_null.mean()
            scaled = (column - mean) / std
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=scaled,
                explanation=f"Applied standard scaling (mean={mean:.4f}, std={std:.4f})",
                warnings=warnings,
                original_action="standard_scale",
                actual_action="standard_scale"
            )
        except Exception as e:
            logger.warning(f"Standard scale failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Standard scale failed: {e}",
                warnings=[str(e)],
                original_action="standard_scale",
                actual_action="keep_as_is"
            )
    
    def safe_log_transform(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely apply log transformation using log1p for safety.
        
        ALWAYS uses log1p instead of log to handle:
        - Zero values
        - Values close to zero
        
        Validates:
        - Column is numeric
        - Values are non-negative
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with log-transformed data or fallback
        """
        warnings = []
        
        # Validation: Check if numeric
        if not pd.api.types.is_numeric_dtype(column):
            try:
                column = pd.to_numeric(column, errors='coerce')
                warnings.append(f"Converted {column_name} to numeric")
            except Exception as e:
                return SafeTransformResult(
                    status=TransformResult.SKIPPED,
                    data=column,
                    explanation=f"Cannot log transform non-numeric column: {e}",
                    warnings=[],
                    original_action="log_transform",
                    actual_action="keep_as_is"
                )
        
        non_null = column.dropna()
        if len(non_null) < self.MIN_ROWS_FOR_TRANSFORM:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation=f"Insufficient non-null values ({len(non_null)})",
                warnings=[],
                original_action="log_transform",
                actual_action="keep_as_is"
            )
        
        # Check for negative values
        min_val = non_null.min()
        if min_val < 0:
            warnings.append(f"Column has negative values (min={min_val}), shifting to positive")
            shift = abs(min_val) + 1
            column = column + shift
        
        try:
            # SAFE: Use log1p (log(1+x)) which handles zeros safely
            transformed = np.log1p(column)
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=transformed,
                explanation="Applied log1p transform (safe log(1+x))",
                warnings=warnings,
                original_action="log_transform",
                actual_action="log1p_transform"
            )
        except Exception as e:
            logger.warning(f"Log transform failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Log transform failed: {e}",
                warnings=[str(e)],
                original_action="log_transform",
                actual_action="keep_as_is"
            )
    
    def safe_log1p_transform(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Apply log1p transformation (log(1+x)).
        
        This is the safe version that handles zeros.
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with transformed data
        """
        # Delegate to safe_log_transform which already uses log1p
        return self.safe_log_transform(column, column_name)
    
    def safe_onehot_encode(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely apply one-hot encoding with cardinality check.
        
        Validates:
        - Cardinality is within limits
        - No null-only columns
        
        Falls back to:
        - Frequency encoding for high cardinality
        - Keep as-is for errors
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with encoded data or fallback
        """
        warnings = []
        
        non_null = column.dropna()
        if len(non_null) == 0:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="All null values",
                warnings=[],
                original_action="onehot_encode",
                actual_action="keep_as_is"
            )
        
        # Check cardinality
        unique_count = non_null.nunique()
        
        if unique_count > self.MAX_ONEHOT_CARDINALITY:
            # Fallback to frequency encoding
            warnings.append(f"Cardinality {unique_count} exceeds limit {self.MAX_ONEHOT_CARDINALITY}, using frequency encoding")
            
            freq = column.value_counts(normalize=True)
            encoded = column.map(freq)
            
            return SafeTransformResult(
                status=TransformResult.FALLBACK,
                data=encoded,
                explanation=f"High cardinality ({unique_count}) - used frequency encoding instead",
                warnings=warnings,
                original_action="onehot_encode",
                actual_action="frequency_encode"
            )
        
        try:
            # Apply one-hot encoding
            dummies = pd.get_dummies(column, prefix=column_name or 'col', dummy_na=False)
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=dummies,
                explanation=f"Applied one-hot encoding ({unique_count} categories)",
                warnings=warnings,
                original_action="onehot_encode",
                actual_action="onehot_encode"
            )
        except Exception as e:
            logger.warning(f"One-hot encoding failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"One-hot encoding failed: {e}",
                warnings=[str(e)],
                original_action="onehot_encode",
                actual_action="keep_as_is"
            )
    
    def safe_text_vectorize(
        self,
        column: pd.Series,
        column_name: str = "",
        max_features: int = 100
    ) -> SafeTransformResult:
        """
        Safely apply text vectorization.
        
        Validates:
        - Column contains meaningful text (not URLs, IDs, etc.)
        - Has enough text content
        
        Args:
            column: Data to transform
            column_name: Name for logging
            max_features: Maximum number of features
            
        Returns:
            SafeTransformResult with vectorized data or fallback
        """
        warnings = []
        
        # Check for meaningful text
        non_null = column.dropna().astype(str)
        
        if len(non_null) == 0:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="No text content",
                warnings=[],
                original_action="text_vectorize",
                actual_action="keep_as_is"
            )
        
        # Check if text is meaningful (not URLs, IDs, etc.)
        avg_length = non_null.str.len().mean()
        has_spaces = non_null.str.contains(r'\s').mean()
        
        # Detect URLs - should NOT vectorize
        url_ratio = non_null.str.match(r'^https?://').mean()
        if url_ratio > 0.5:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Column contains URLs - not suitable for text vectorization",
                warnings=["Detected URL column"],
                original_action="text_vectorize",
                actual_action="drop_column"  # URLs should typically be dropped
            )
        
        # Detect IDs/hashes - should NOT vectorize
        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio > 0.95 and avg_length > 10:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Column appears to be identifiers - not suitable for vectorization",
                warnings=["Detected ID-like column"],
                original_action="text_vectorize",
                actual_action="drop_column"
            )
        
        # Check for meaningful text content
        if avg_length < 3 or (has_spaces < 0.1 and avg_length < 10):
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Text too short for meaningful vectorization",
                warnings=[],
                original_action="text_vectorize",
                actual_action="keep_as_is"
            )
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Fill nulls for vectorization
            text_data = column.fillna('')
            
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(text_data)
            
            # Convert to DataFrame
            feature_names = [f"{column_name}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=feature_names,
                index=column.index
            )
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=tfidf_df,
                explanation=f"Applied TF-IDF vectorization ({tfidf_matrix.shape[1]} features)",
                warnings=warnings,
                original_action="text_vectorize",
                actual_action="text_vectorize"
            )
        except ImportError:
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation="sklearn not available for vectorization",
                warnings=["sklearn import failed"],
                original_action="text_vectorize",
                actual_action="keep_as_is"
            )
        except Exception as e:
            logger.warning(f"Text vectorization failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Text vectorization failed: {e}",
                warnings=[str(e)],
                original_action="text_vectorize",
                actual_action="keep_as_is"
            )
    
    def safe_datetime_extract(
        self,
        column: pd.Series,
        column_name: str = "",
        components: Optional[List[str]] = None
    ) -> SafeTransformResult:
        """
        Safely extract datetime components.
        
        Args:
            column: Data to transform
            column_name: Name for logging
            components: List of components to extract (year, month, day, hour, etc.)
            
        Returns:
            SafeTransformResult with extracted features
        """
        if components is None:
            components = ['year', 'month', 'day', 'dayofweek']
        
        warnings = []
        
        try:
            # Try to parse as datetime
            if not pd.api.types.is_datetime64_any_dtype(column):
                dt_column = pd.to_datetime(column, errors='coerce')
                null_after = dt_column.isnull().sum()
                null_before = column.isnull().sum()
                
                if null_after > null_before + len(column) * 0.1:
                    return SafeTransformResult(
                        status=TransformResult.SKIPPED,
                        data=column,
                        explanation="Too many values failed datetime parsing",
                        warnings=["Datetime parsing failed for >10% of values"],
                        original_action="datetime_extract",
                        actual_action="keep_as_is"
                    )
            else:
                dt_column = column
            
            # Extract components
            extracted = pd.DataFrame(index=column.index)
            
            for comp in components:
                try:
                    if comp == 'year':
                        extracted[f'{column_name}_year'] = dt_column.dt.year
                    elif comp == 'month':
                        extracted[f'{column_name}_month'] = dt_column.dt.month
                    elif comp == 'day':
                        extracted[f'{column_name}_day'] = dt_column.dt.day
                    elif comp == 'dayofweek':
                        extracted[f'{column_name}_dayofweek'] = dt_column.dt.dayofweek
                    elif comp == 'hour':
                        extracted[f'{column_name}_hour'] = dt_column.dt.hour
                    elif comp == 'minute':
                        extracted[f'{column_name}_minute'] = dt_column.dt.minute
                    elif comp == 'quarter':
                        extracted[f'{column_name}_quarter'] = dt_column.dt.quarter
                except Exception as e:
                    warnings.append(f"Failed to extract {comp}: {e}")
            
            if extracted.empty:
                return SafeTransformResult(
                    status=TransformResult.ERROR,
                    data=column,
                    explanation="No datetime components could be extracted",
                    warnings=warnings,
                    original_action="datetime_extract",
                    actual_action="keep_as_is"
                )
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=extracted,
                explanation=f"Extracted datetime components: {list(extracted.columns)}",
                warnings=warnings,
                original_action="datetime_extract",
                actual_action="datetime_extract"
            )
        except Exception as e:
            logger.warning(f"Datetime extraction failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Datetime extraction failed: {e}",
                warnings=[str(e)],
                original_action="datetime_extract",
                actual_action="keep_as_is"
            )
    
    def safe_robust_scale(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely apply robust scaling using median and IQR.
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with scaled data
        """
        warnings = []
        
        if not pd.api.types.is_numeric_dtype(column):
            try:
                column = pd.to_numeric(column, errors='coerce')
                warnings.append("Converted to numeric")
            except Exception:
                return SafeTransformResult(
                    status=TransformResult.SKIPPED,
                    data=column,
                    explanation="Cannot scale non-numeric column",
                    warnings=[],
                    original_action="robust_scale",
                    actual_action="keep_as_is"
                )
        
        non_null = column.dropna()
        if len(non_null) < self.MIN_ROWS_FOR_TRANSFORM:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Insufficient values",
                warnings=[],
                original_action="robust_scale",
                actual_action="keep_as_is"
            )
        
        try:
            median = non_null.median()
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                return SafeTransformResult(
                    status=TransformResult.SKIPPED,
                    data=column,
                    explanation="Zero IQR - cannot scale",
                    warnings=[],
                    original_action="robust_scale",
                    actual_action="keep_as_is"
                )
            
            scaled = (column - median) / iqr
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=scaled,
                explanation=f"Applied robust scaling (median={median:.4f}, IQR={iqr:.4f})",
                warnings=warnings,
                original_action="robust_scale",
                actual_action="robust_scale"
            )
        except Exception as e:
            logger.warning(f"Robust scale failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Robust scale failed: {e}",
                warnings=[str(e)],
                original_action="robust_scale",
                actual_action="keep_as_is"
            )
    
    def safe_minmax_scale(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely apply min-max scaling to [0, 1].
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with scaled data
        """
        if not pd.api.types.is_numeric_dtype(column):
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Cannot scale non-numeric column",
                warnings=[],
                original_action="minmax_scale",
                actual_action="keep_as_is"
            )
        
        non_null = column.dropna()
        if len(non_null) < self.MIN_ROWS_FOR_TRANSFORM:
            return SafeTransformResult(
                status=TransformResult.SKIPPED,
                data=column,
                explanation="Insufficient values",
                warnings=[],
                original_action="minmax_scale",
                actual_action="keep_as_is"
            )
        
        try:
            min_val = non_null.min()
            max_val = non_null.max()
            range_val = max_val - min_val
            
            if range_val == 0:
                return SafeTransformResult(
                    status=TransformResult.SKIPPED,
                    data=column,
                    explanation="Zero range - cannot scale",
                    warnings=[],
                    original_action="minmax_scale",
                    actual_action="keep_as_is"
                )
            
            scaled = (column - min_val) / range_val
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=scaled,
                explanation=f"Applied min-max scaling (min={min_val:.4f}, max={max_val:.4f})",
                warnings=[],
                original_action="minmax_scale",
                actual_action="minmax_scale"
            )
        except Exception as e:
            logger.warning(f"Min-max scale failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Min-max scale failed: {e}",
                warnings=[str(e)],
                original_action="minmax_scale",
                actual_action="keep_as_is"
            )
    
    def safe_text_clean(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> SafeTransformResult:
        """
        Safely clean text data (lowercase, strip whitespace, remove special chars).
        
        Args:
            column: Data to transform
            column_name: Name for logging
            
        Returns:
            SafeTransformResult with cleaned text
        """
        try:
            # Convert to string and clean
            cleaned = column.astype(str).str.lower().str.strip()
            cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
            
            return SafeTransformResult(
                status=TransformResult.SUCCESS,
                data=cleaned,
                explanation="Cleaned text (lowercase, normalized whitespace)",
                warnings=[],
                original_action="text_clean",
                actual_action="text_clean"
            )
        except Exception as e:
            logger.warning(f"Text clean failed for {column_name}: {e}")
            return SafeTransformResult(
                status=TransformResult.ERROR,
                data=column,
                explanation=f"Text clean failed: {e}",
                warnings=[str(e)],
                original_action="text_clean",
                actual_action="keep_as_is"
            )


def get_safe_transforms() -> SafeTransforms:
    """Get a singleton instance of safe transforms."""
    return SafeTransforms()
