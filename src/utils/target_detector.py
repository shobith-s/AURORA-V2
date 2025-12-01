"""
Target Variable Detector - Detects target/label columns in datasets.

Identifies target variables using:
1. Keyword matching (price, target, label, churn, fraud, etc.)
2. Positional analysis (last column heuristic)
3. Exclusion patterns (id, date, url should not be targets)

Target columns require special protection - they should NEVER be transformed
in ways that destroy predictive value (e.g., no binning, no dropping).
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TargetDetectionResult:
    """Result of target variable detection."""
    is_target: bool
    confidence: float
    detection_method: str
    explanation: str
    details: Dict[str, Any]


class TargetDetector:
    """
    Detect target/label variables in datasets.
    
    Uses multiple heuristics:
    1. Name-based keywords
    2. Position (last column)
    3. Statistical properties
    4. Exclusion patterns
    """
    
    # Target keywords - columns with these in name are likely targets
    TARGET_KEYWORDS = [
        'target', 'label', 'class', 'output', 'y',
        'price', 'cost', 'revenue', 'sales', 'income', 'profit',
        'churn', 'fraud', 'default', 'click', 'convert', 'purchase',
        'score', 'rating', 'grade', 'rank',
        'survival', 'outcome', 'result', 'status',
        'prediction', 'predict', 'forecasted'
    ]
    
    # Common target column names (exact or near-exact matches)
    EXACT_TARGET_NAMES = [
        'target', 'label', 'class', 'y', 'outcome', 'result',
        'selling_price', 'sale_price', 'price', 'sales_price',
        'churn', 'fraud', 'default', 'clicked', 'converted',
        'is_fraud', 'is_churn', 'is_default', 'has_churned',
        'survived', 'survival'
    ]
    
    # Exclusion keywords - columns with these are NOT targets
    EXCLUSION_KEYWORDS = [
        'id', 'uuid', 'guid', 'key', 'index', 'row',
        'date', 'time', 'timestamp', 'datetime', 'created', 'updated',
        'url', 'link', 'path', 'file', 'image', 'photo',
        'email', 'phone', 'address', 'zip', 'postal',
        'name', 'title', 'description', 'comment', 'note',
        'count', 'num_', 'total_', 'sum_', 'avg_', 'mean_'
    ]
    
    # Binary target patterns (classification)
    BINARY_PATTERNS = [
        {'0', '1'}, {'true', 'false'}, {'yes', 'no'},
        {'positive', 'negative'}, {'good', 'bad'},
        {'pass', 'fail'}, {'win', 'lose'}
    ]
    
    def __init__(self):
        """Initialize the target detector."""
        pass
    
    def detect_target(
        self,
        column: pd.Series,
        column_name: str,
        position_in_df: Optional[int] = None,
        total_columns: Optional[int] = None,
        other_column_names: Optional[List[str]] = None
    ) -> TargetDetectionResult:
        """
        Detect if a column is likely a target variable.
        
        Args:
            column: The pandas Series to analyze
            column_name: Name of the column
            position_in_df: Position of column in dataframe (0-indexed)
            total_columns: Total number of columns in dataframe
            other_column_names: Names of other columns (for context)
            
        Returns:
            TargetDetectionResult with detection results
        """
        # Check exclusions first
        if self._is_excluded(column_name):
            return TargetDetectionResult(
                is_target=False,
                confidence=0.95,
                detection_method='exclusion',
                explanation=f'Column name matches exclusion pattern',
                details={'excluded_because': self._get_exclusion_reason(column_name)}
            )
        
        # Check for exact name match
        exact_match = self._check_exact_name(column_name)
        if exact_match[0]:
            return TargetDetectionResult(
                is_target=True,
                confidence=exact_match[1],
                detection_method='exact_name',
                explanation=f'Column name matches common target name: {column_name}',
                details={'matched_pattern': exact_match[2]}
            )
        
        # Check for keyword match
        keyword_match = self._check_keyword_match(column_name)
        if keyword_match[0]:
            return TargetDetectionResult(
                is_target=True,
                confidence=keyword_match[1],
                detection_method='keyword',
                explanation=f'Column name contains target keyword: {keyword_match[2]}',
                details={'matched_keyword': keyword_match[2]}
            )
        
        # Check for positional match (last column)
        if position_in_df is not None and total_columns is not None:
            if position_in_df == total_columns - 1:
                # Last column - check statistical properties
                stat_result = self._check_statistical_properties(column)
                if stat_result[0]:
                    return TargetDetectionResult(
                        is_target=True,
                        confidence=stat_result[1] * 0.8,  # Lower confidence for positional
                        detection_method='position_plus_stats',
                        explanation='Last column with target-like properties',
                        details=stat_result[2]
                    )
        
        # Check statistical properties alone
        stat_result = self._check_statistical_properties(column)
        if stat_result[0] and stat_result[1] >= 0.8:
            return TargetDetectionResult(
                is_target=True,
                confidence=stat_result[1] * 0.6,  # Even lower for stats-only
                detection_method='statistical',
                explanation='Column has target-like statistical properties',
                details=stat_result[2]
            )
        
        # Not a target
        return TargetDetectionResult(
            is_target=False,
            confidence=0.7,
            detection_method='no_match',
            explanation='No target indicators found',
            details={}
        )
    
    def detect_targets_in_dataframe(
        self,
        df: pd.DataFrame,
        known_target: Optional[str] = None
    ) -> Dict[str, TargetDetectionResult]:
        """
        Detect target variables in a dataframe.
        
        Args:
            df: The dataframe to analyze
            known_target: Known target column name (if provided)
            
        Returns:
            Dictionary mapping column names to detection results
        """
        results = {}
        column_names = list(df.columns)
        total_columns = len(column_names)
        
        for idx, col_name in enumerate(column_names):
            # If known target is provided, use it
            if known_target and col_name == known_target:
                results[col_name] = TargetDetectionResult(
                    is_target=True,
                    confidence=1.0,
                    detection_method='user_specified',
                    explanation='User specified this column as target',
                    details={}
                )
            else:
                results[col_name] = self.detect_target(
                    column=df[col_name],
                    column_name=col_name,
                    position_in_df=idx,
                    total_columns=total_columns,
                    other_column_names=column_names
                )
        
        return results
    
    def get_likely_targets(
        self,
        df: pd.DataFrame,
        min_confidence: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Get list of likely target columns sorted by confidence.
        
        Args:
            df: The dataframe to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (column_name, confidence) tuples, sorted by confidence
        """
        results = self.detect_targets_in_dataframe(df)
        
        targets = [
            (name, result.confidence)
            for name, result in results.items()
            if result.is_target and result.confidence >= min_confidence
        ]
        
        return sorted(targets, key=lambda x: x[1], reverse=True)
    
    def _is_excluded(self, column_name: str) -> bool:
        """Check if column name matches exclusion patterns."""
        name_lower = column_name.lower()
        
        # Check prefix exclusions
        for exclusion in self.EXCLUSION_KEYWORDS:
            if name_lower.startswith(exclusion) or name_lower.endswith('_' + exclusion):
                return True
            if exclusion in name_lower and exclusion not in ['name']:
                # Special handling - some exclusions should be strict
                if exclusion in ['id', 'uuid', 'guid', 'date', 'time', 'url', 'email', 'phone']:
                    return True
        
        return False
    
    def _get_exclusion_reason(self, column_name: str) -> str:
        """Get the reason for exclusion."""
        name_lower = column_name.lower()
        
        for exclusion in self.EXCLUSION_KEYWORDS:
            if name_lower.startswith(exclusion) or exclusion in name_lower:
                return f"contains '{exclusion}'"
        
        return "unknown"
    
    def _check_exact_name(self, column_name: str) -> Tuple[bool, float, str]:
        """Check for exact target name match."""
        name_lower = column_name.lower().strip()
        name_normalized = name_lower.replace('_', '').replace('-', '').replace(' ', '')
        
        for exact in self.EXACT_TARGET_NAMES:
            exact_normalized = exact.replace('_', '').replace('-', '').replace(' ', '')
            if name_lower == exact or name_normalized == exact_normalized:
                return True, 0.95, exact
        
        return False, 0.0, ""
    
    def _check_keyword_match(self, column_name: str) -> Tuple[bool, float, str]:
        """Check for target keyword in column name."""
        name_lower = column_name.lower()
        
        for keyword in self.TARGET_KEYWORDS:
            if keyword in name_lower:
                # Higher confidence for certain keywords
                if keyword in ['target', 'label', 'class', 'y', 'price', 'churn', 'fraud']:
                    return True, 0.85, keyword
                else:
                    return True, 0.7, keyword
        
        return False, 0.0, ""
    
    def _check_statistical_properties(
        self,
        column: pd.Series
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if column has target-like statistical properties.
        
        Target characteristics:
        - Classification: low cardinality (2-20 classes)
        - Regression: continuous numeric with reasonable distribution
        """
        non_null = column.dropna()
        if len(non_null) == 0:
            return False, 0.0, {}
        
        is_numeric = pd.api.types.is_numeric_dtype(non_null)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)
        
        details = {
            'is_numeric': is_numeric,
            'unique_count': unique_count,
            'unique_ratio': unique_ratio
        }
        
        # Binary classification target
        if unique_count == 2:
            # Check for binary patterns
            if not is_numeric:
                values_lower = set(str(v).lower() for v in non_null.unique())
                for pattern in self.BINARY_PATTERNS:
                    if values_lower.issubset(pattern) or values_lower == pattern:
                        details['binary_pattern'] = list(values_lower)
                        return True, 0.85, details
            else:
                values = set(non_null.unique())
                if values == {0, 1} or values == {0.0, 1.0}:
                    details['binary_pattern'] = list(values)
                    return True, 0.85, details
        
        # Multi-class classification target
        if 2 < unique_count <= 20 and unique_ratio < 0.1:
            details['classification_type'] = 'multiclass'
            return True, 0.7, details
        
        # Regression target (continuous numeric)
        if is_numeric and unique_ratio > 0.5:
            # Check for reasonable distribution
            std = non_null.std()
            mean = non_null.mean()
            
            if std > 0 and mean != 0:
                cv = std / abs(mean)  # Coefficient of variation
                details['cv'] = cv
                
                if 0.1 < cv < 5:  # Reasonable variability
                    details['regression_type'] = 'continuous'
                    return True, 0.65, details
        
        return False, 0.3, details
    
    def is_protected_column(
        self,
        column_name: str,
        detected_targets: Optional[Dict[str, TargetDetectionResult]] = None
    ) -> Tuple[bool, str]:
        """
        Check if a column should be protected from destructive transformations.
        
        Protected columns should NOT have:
        - Binning (destroys predictive value)
        - Dropping (loses target)
        - Standard scaling (may be okay but not always needed)
        
        Args:
            column_name: Name of the column
            detected_targets: Previous detection results (if available)
            
        Returns:
            Tuple of (is_protected, reason)
        """
        if detected_targets and column_name in detected_targets:
            result = detected_targets[column_name]
            if result.is_target and result.confidence >= 0.6:
                return True, f"Detected as target ({result.detection_method}, confidence: {result.confidence:.2f})"
        
        # Check by name alone
        name_lower = column_name.lower()
        
        # High-value keywords
        protected_keywords = ['price', 'target', 'label', 'y', 'churn', 'fraud', 'outcome', 'sales', 'revenue']
        for keyword in protected_keywords:
            if keyword in name_lower:
                return True, f"Name contains protected keyword: {keyword}"
        
        return False, ""


def get_target_detector() -> TargetDetector:
    """Get a singleton instance of the target detector."""
    return TargetDetector()
