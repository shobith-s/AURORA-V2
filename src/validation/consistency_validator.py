"""
Consistency Validator - Domain and logical consistency checks.

Ensures preprocessing decisions make logical sense and preserve important
data properties like ranges, correlations, and semantic meaning.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from .validation_result import ValidationResult, ValidationMetric, ValidationStatus


class ConsistencyValidator:
    """
    Validates preprocessing decisions for domain and logical consistency.
    
    This validator ensures that preprocessing doesn't break important
    data properties or violate domain constraints.
    """
    
    def __init__(
        self,
        correlation_change_threshold: float = 0.15,  # Max correlation change
        range_expansion_threshold: float = 0.20,  # Max range expansion (20%)
        overall_pass_threshold: float = 1.0,  # All checks must pass
    ):
        """
        Initialize the consistency validator.
        
        Args:
            correlation_change_threshold: Maximum allowed correlation change
            range_expansion_threshold: Maximum allowed range expansion
            overall_pass_threshold: Fraction of checks that must pass
        """
        self.correlation_change_threshold = correlation_change_threshold
        self.range_expansion_threshold = range_expansion_threshold
        self.overall_pass_threshold = overall_pass_threshold
    
    def validate(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
        action_name: str,
        column_name: str = "",
        other_columns: Optional[Dict[str, pd.Series]] = None,
    ) -> ValidationResult:
        """
        Validate preprocessing for consistency.
        
        Args:
            original: Original column data
            preprocessed: Preprocessed column data
            action_name: Name of the preprocessing action
            column_name: Name of the column
            other_columns: Other columns for correlation checks
        
        Returns:
            ValidationResult with consistency checks
        """
        start_time = time.time()
        
        result = ValidationResult(
            overall_score=0.0,
            passed=False,
            status=ValidationStatus.PASSED,
        )
        
        try:
            # Check 1: Range Preservation
            try:
                range_metric = self._check_range_preservation(
                    original, preprocessed, action_name
                )
                result.add_metric(range_metric)
            except Exception as e:
                result.add_warning(f"Range check skipped: {str(e)}")
            
            # Check 2: Correlation Preservation (if other columns provided)
            if other_columns and len(other_columns) > 0:
                try:
                    corr_metric = self._check_correlation_preservation(
                        original, preprocessed, other_columns
                    )
                    result.add_metric(corr_metric)
                except Exception as e:
                    result.add_warning(f"Correlation check skipped: {str(e)}")
            
            # Check 3: Semantic Validity
            try:
                semantic_metric = self._check_semantic_validity(
                    original, preprocessed, action_name, column_name
                )
                result.add_metric(semantic_metric)
            except Exception as e:
                result.add_warning(f"Semantic check skipped: {str(e)}")
            
            # Calculate overall score
            if result.metrics:
                passed_count = sum(1 for m in result.metrics.values() if m.passed)
                total_count = len(result.metrics)
                result.overall_score = passed_count / total_count
                result.passed = result.overall_score >= self.overall_pass_threshold
                result.status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED
            else:
                result.add_warning("No consistency checks could be performed")
                result.status = ValidationStatus.SKIPPED
            
            result.details['action'] = action_name
            result.details['column_name'] = column_name
            
        except Exception as e:
            result.add_error(f"Consistency validation failed: {str(e)}")
            result.status = ValidationStatus.FAILED
            result.passed = False
        
        result.latency_ms = (time.time() - start_time) * 1000
        return result
    
    def _check_range_preservation(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
        action_name: str,
    ) -> ValidationMetric:
        """
        Check if preprocessing preserved reasonable data range.
        
        Some expansion is acceptable (e.g., scaling), but extreme
        expansion suggests a problem.
        """
        orig_clean = original.dropna()
        prep_clean = preprocessed.dropna()
        
        if len(orig_clean) == 0 or len(prep_clean) == 0:
            return ValidationMetric(
                name="range_preservation",
                value_before=0.0,
                value_after=0.0,
                improvement=0.0,
                passed=True,
                explanation="Range check skipped (no data)",
            )
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(prep_clean):
            return ValidationMetric(
                name="range_preservation",
                value_before=0.0,
                value_after=0.0,
                improvement=0.0,
                passed=True,
                explanation="Range check skipped (non-numeric)",
            )
        
        # Calculate ranges
        orig_range = orig_clean.max() - orig_clean.min()
        prep_range = prep_clean.max() - prep_clean.min()
        
        # Calculate expansion ratio
        if orig_range > 0:
            expansion_ratio = prep_range / orig_range
        else:
            expansion_ratio = 1.0
        
        # Check if expansion is reasonable
        # Allow some expansion for scaling, but not extreme
        passed = expansion_ratio <= (1.0 + self.range_expansion_threshold)
        
        explanation = (
            f"Range: {orig_range:.2f} -> {prep_range:.2f} "
            f"(expansion: {expansion_ratio:.2f}x, "
            f"{'acceptable' if passed else 'excessive'})"
        )
        
        return ValidationMetric(
            name="range_preservation",
            value_before=orig_range,
            value_after=prep_range,
            improvement=-(expansion_ratio - 1.0),  # Negative = expansion
            passed=passed,
            explanation=explanation,
        )
    
    def _check_correlation_preservation(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
        other_columns: Dict[str, pd.Series],
    ) -> ValidationMetric:
        """
        Check if preprocessing preserved correlations with other columns.
        
        Important relationships should be maintained after preprocessing.
        """
        if not pd.api.types.is_numeric_dtype(preprocessed):
            return ValidationMetric(
                name="correlation_preservation",
                value_before=0.0,
                value_after=0.0,
                improvement=0.0,
                passed=True,
                explanation="Correlation check skipped (non-numeric)",
            )
        
        # Calculate average correlation change
        corr_changes = []
        
        for col_name, other_col in other_columns.items():
            if not pd.api.types.is_numeric_dtype(other_col):
                continue
            
            try:
                # Align indices
                orig_aligned, other_aligned = original.align(other_col, join='inner')
                prep_aligned, other_aligned2 = preprocessed.align(other_col, join='inner')
                
                # Calculate correlations
                corr_before = orig_aligned.corr(other_aligned)
                corr_after = prep_aligned.corr(other_aligned2)
                
                if not np.isnan(corr_before) and not np.isnan(corr_after):
                    corr_changes.append(abs(corr_after - corr_before))
            except:
                continue
        
        if not corr_changes:
            return ValidationMetric(
                name="correlation_preservation",
                value_before=0.0,
                value_after=0.0,
                improvement=0.0,
                passed=True,
                explanation="Correlation check skipped (no valid correlations)",
            )
        
        avg_corr_change = np.mean(corr_changes)
        max_corr_change = np.max(corr_changes)
        
        # Pass if average correlation change is small
        passed = avg_corr_change <= self.correlation_change_threshold
        
        explanation = (
            f"Correlations: avg change {avg_corr_change:.3f}, "
            f"max change {max_corr_change:.3f} "
            f"({'preserved' if passed else 'changed significantly'})"
        )
        
        return ValidationMetric(
            name="correlation_preservation",
            value_before=0.0,
            value_after=avg_corr_change,
            improvement=-avg_corr_change,  # Negative = change
            passed=passed,
            explanation=explanation,
        )
    
    def _check_semantic_validity(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
        action_name: str,
        column_name: str,
    ) -> ValidationMetric:
        """
        Check if preprocessing makes semantic sense.
        
        Examples:
        - Don't one-hot encode high cardinality columns
        - Don't log-transform negative values
        - Don't scale already-scaled data
        """
        issues = []
        
        # Check 1: One-hot encoding high cardinality
        if 'onehot' in action_name.lower():
            unique_count = original.nunique()
            if unique_count > 50:
                issues.append(f"One-hot encoding {unique_count} categories (too many)")
        
        # Check 2: Log transform with non-positive values
        if 'log' in action_name.lower():
            if pd.api.types.is_numeric_dtype(original):
                if (original <= 0).any():
                    issues.append("Log transform on non-positive values")
        
        # Check 3: Scaling already-scaled data
        if 'scale' in action_name.lower():
            if pd.api.types.is_numeric_dtype(original):
                orig_mean = original.mean()
                orig_std = original.std()
                if abs(orig_mean) < 0.1 and abs(orig_std - 1.0) < 0.2:
                    issues.append("Scaling already-normalized data")
        
        # Check 4: Dropping columns with meaningful names
        if 'drop' in action_name.lower():
            important_keywords = ['id', 'key', 'target', 'label', 'class']
            if any(keyword in column_name.lower() for keyword in important_keywords):
                issues.append(f"Dropping potentially important column: {column_name}")
        
        passed = len(issues) == 0
        
        if passed:
            explanation = "Semantic validity: PASSED (no issues detected)"
        else:
            explanation = f"Semantic validity: FAILED ({'; '.join(issues)})"
        
        return ValidationMetric(
            name="semantic_validity",
            value_before=0.0,
            value_after=float(len(issues)),
            improvement=-float(len(issues)),
            passed=passed,
            explanation=explanation,
        )
