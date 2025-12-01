"""
Preprocessing Validator - Pre-execution validation layer.

Validates transformations BEFORE execution to prevent:
1. Target variable destruction
2. Type mismatches (standard scale on text)
3. Unsafe mathematical operations
4. Cardinality explosions
5. Data leakage

7 Core Validation Rules:
1. Never transform targets destructively
2. Standard scale requires numeric
3. Log requires positive numeric
4. One-hot has cardinality limits
5. Text vectorization requires meaningful text
6. No binning targets
7. Empty columns should be dropped
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .universal_type_detector import UniversalTypeDetector, SemanticType
from .target_detector import TargetDetector

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of validation check."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    OVERRIDE = "override"


@dataclass
class ValidationCheck:
    """Single validation check result."""
    rule_name: str
    result: ValidationResult
    message: str
    suggested_action: Optional[str] = None


@dataclass
class PreprocessingValidation:
    """Complete validation result for a preprocessing decision."""
    is_valid: bool
    checks: List[ValidationCheck]
    original_action: str
    recommended_action: str
    warnings: List[str]
    blockers: List[str]


class PreprocessingValidator:
    """
    Pre-execution validation layer for preprocessing decisions.
    
    Catches errors BEFORE they happen:
    - Type mismatches
    - Target destruction
    - Unsafe operations
    - Data leakage
    """
    
    # Actions that destroy information
    DESTRUCTIVE_ACTIONS = [
        'drop_column', 'binning_equal_width', 'binning_equal_freq', 
        'binning_custom', 'remove_outliers', 'clip_outliers'
    ]
    
    # Actions that require numeric data
    NUMERIC_ONLY_ACTIONS = [
        'standard_scale', 'robust_scale', 'minmax_scale', 'maxabs_scale',
        'log_transform', 'log1p_transform', 'sqrt_transform',
        'box_cox', 'yeo_johnson', 'quantile_transform', 'power_transform',
        'clip_outliers', 'winsorize', 'remove_outliers',
        'polynomial_features'
    ]
    
    # Actions that require positive values
    POSITIVE_ONLY_ACTIONS = [
        'log_transform', 'sqrt_transform', 'box_cox'
    ]
    
    # Actions with cardinality concerns
    HIGH_CARDINALITY_UNSAFE_ACTIONS = [
        'onehot_encode'
    ]
    
    # Actions for text data
    TEXT_ACTIONS = [
        'text_vectorize', 'text_clean', 'text_lowercase', 'text_uppercase'
    ]
    
    # Max cardinality for one-hot encoding
    MAX_ONEHOT_CARDINALITY = 50
    
    def __init__(self):
        """Initialize the validator."""
        self.type_detector = UniversalTypeDetector()
        self.target_detector = TargetDetector()
    
    def validate_transformation(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str,
        is_target: bool = False,
        target_column_name: Optional[str] = None
    ) -> PreprocessingValidation:
        """
        Validate a proposed transformation.
        
        Args:
            column: Data to transform
            column_name: Name of the column
            proposed_action: Proposed preprocessing action
            is_target: Whether this column is the target variable
            target_column_name: Name of target column (if known)
            
        Returns:
            PreprocessingValidation with validation results
        """
        checks = []
        warnings = []
        blockers = []
        
        # Detect semantic type
        type_result = self.type_detector.detect_semantic_type(column, column_name)
        semantic_type = type_result.semantic_type
        
        # Auto-detect if this is a target
        if not is_target and target_column_name != column_name:
            target_result = self.target_detector.detect_target(column, column_name)
            if target_result.is_target and target_result.confidence >= 0.7:
                is_target = True
        
        # === RULE 1: Never transform targets destructively ===
        check = self._validate_target_protection(column_name, proposed_action, is_target)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 2: Standard scale requires numeric ===
        check = self._validate_numeric_requirement(column, column_name, proposed_action, semantic_type)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 3: Log requires positive numeric ===
        check = self._validate_positive_requirement(column, column_name, proposed_action, semantic_type)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 4: One-hot has cardinality limits ===
        check = self._validate_cardinality_limits(column, column_name, proposed_action)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 5: Text vectorization requires meaningful text ===
        check = self._validate_text_meaningfulness(column, column_name, proposed_action, semantic_type)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 6: No binning targets ===
        check = self._validate_no_target_binning(column_name, proposed_action, is_target)
        checks.append(check)
        if check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        elif check.result == ValidationResult.WARNING:
            warnings.append(check.message)
        
        # === RULE 7: Empty columns should be dropped ===
        check = self._validate_empty_column(column, column_name, proposed_action, semantic_type)
        checks.append(check)
        if check.result == ValidationResult.OVERRIDE:
            # Empty column should be dropped regardless of proposed action
            return PreprocessingValidation(
                is_valid=True,
                checks=checks,
                original_action=proposed_action,
                recommended_action='drop_column',
                warnings=warnings,
                blockers=[]
            )
        elif check.result == ValidationResult.INVALID:
            blockers.append(check.message)
        
        # Determine final action
        is_valid = len(blockers) == 0
        
        # If invalid, try to use suggested action from checks
        if not is_valid:
            # Find the first check with a suggested action
            suggested = None
            for check in checks:
                if check.result == ValidationResult.INVALID and check.suggested_action:
                    suggested = check.suggested_action
                    break
            recommended_action = suggested or self._get_safe_fallback(semantic_type, is_target)
        else:
            recommended_action = proposed_action
        
        return PreprocessingValidation(
            is_valid=is_valid,
            checks=checks,
            original_action=proposed_action,
            recommended_action=recommended_action,
            warnings=warnings,
            blockers=blockers
        )
    
    def _validate_target_protection(
        self,
        column_name: str,
        proposed_action: str,
        is_target: bool
    ) -> ValidationCheck:
        """Rule 1: Never transform targets destructively."""
        if not is_target:
            return ValidationCheck(
                rule_name="target_protection",
                result=ValidationResult.VALID,
                message="Not a target column"
            )
        
        if proposed_action in self.DESTRUCTIVE_ACTIONS:
            return ValidationCheck(
                rule_name="target_protection",
                result=ValidationResult.INVALID,
                message=f"Cannot apply destructive action '{proposed_action}' to target column '{column_name}'",
                suggested_action="keep_as_is"
            )
        
        # Warn about transformations that might affect target
        if proposed_action in ['standard_scale', 'log_transform', 'log1p_transform']:
            return ValidationCheck(
                rule_name="target_protection",
                result=ValidationResult.WARNING,
                message=f"Transforming target column '{column_name}' with '{proposed_action}' - ensure this is intentional"
            )
        
        return ValidationCheck(
            rule_name="target_protection",
            result=ValidationResult.VALID,
            message="Target protection validated"
        )
    
    def _validate_numeric_requirement(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str,
        semantic_type: SemanticType
    ) -> ValidationCheck:
        """Rule 2: Standard scale requires numeric."""
        if proposed_action not in self.NUMERIC_ONLY_ACTIONS:
            return ValidationCheck(
                rule_name="numeric_requirement",
                result=ValidationResult.VALID,
                message="Action does not require numeric data"
            )
        
        # Check if semantic type is numeric
        if semantic_type == SemanticType.NUMERIC:
            return ValidationCheck(
                rule_name="numeric_requirement",
                result=ValidationResult.VALID,
                message="Column is numeric"
            )
        
        # Check if dtype is numeric
        if pd.api.types.is_numeric_dtype(column):
            return ValidationCheck(
                rule_name="numeric_requirement",
                result=ValidationResult.VALID,
                message="Column dtype is numeric"
            )
        
        # Try to convert and check success rate
        try:
            numeric_converted = pd.to_numeric(column, errors='coerce')
            success_rate = numeric_converted.notna().sum() / len(column) if len(column) > 0 else 0
            
            if success_rate >= 0.8:
                return ValidationCheck(
                    rule_name="numeric_requirement",
                    result=ValidationResult.WARNING,
                    message=f"Column can be converted to numeric ({success_rate:.0%} success rate)"
                )
        except Exception:
            pass
        
        return ValidationCheck(
            rule_name="numeric_requirement",
            result=ValidationResult.INVALID,
            message=f"Cannot apply '{proposed_action}' to non-numeric column '{column_name}' (type: {semantic_type.value})",
            suggested_action="keep_as_is" if semantic_type == SemanticType.TEXT else "text_clean"
        )
    
    def _validate_positive_requirement(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str,
        semantic_type: SemanticType
    ) -> ValidationCheck:
        """Rule 3: Log requires positive numeric."""
        if proposed_action not in self.POSITIVE_ONLY_ACTIONS:
            return ValidationCheck(
                rule_name="positive_requirement",
                result=ValidationResult.VALID,
                message="Action does not require positive values"
            )
        
        if not pd.api.types.is_numeric_dtype(column):
            return ValidationCheck(
                rule_name="positive_requirement",
                result=ValidationResult.INVALID,
                message=f"Cannot apply '{proposed_action}' to non-numeric column",
                suggested_action="keep_as_is"
            )
        
        non_null = column.dropna()
        if len(non_null) == 0:
            return ValidationCheck(
                rule_name="positive_requirement",
                result=ValidationResult.INVALID,
                message="Column has no values",
                suggested_action="drop_column"
            )
        
        min_val = non_null.min()
        
        if proposed_action == 'log_transform' and min_val <= 0:
            return ValidationCheck(
                rule_name="positive_requirement",
                result=ValidationResult.OVERRIDE,
                message=f"Column has non-positive values (min={min_val}), using log1p instead",
                suggested_action="log1p_transform"
            )
        
        if min_val < 0:
            return ValidationCheck(
                rule_name="positive_requirement",
                result=ValidationResult.INVALID,
                message=f"Cannot apply '{proposed_action}' to column with negative values (min={min_val})",
                suggested_action="yeo_johnson"
            )
        
        return ValidationCheck(
            rule_name="positive_requirement",
            result=ValidationResult.VALID,
            message="Column has positive values"
        )
    
    def _validate_cardinality_limits(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str
    ) -> ValidationCheck:
        """Rule 4: One-hot has cardinality limits."""
        if proposed_action not in self.HIGH_CARDINALITY_UNSAFE_ACTIONS:
            return ValidationCheck(
                rule_name="cardinality_limits",
                result=ValidationResult.VALID,
                message="Action does not have cardinality concerns"
            )
        
        unique_count = column.nunique()
        
        if unique_count > self.MAX_ONEHOT_CARDINALITY:
            return ValidationCheck(
                rule_name="cardinality_limits",
                result=ValidationResult.INVALID,
                message=f"Cardinality {unique_count} exceeds limit {self.MAX_ONEHOT_CARDINALITY} for '{proposed_action}'",
                suggested_action="frequency_encode" if unique_count < 1000 else "hash_encode"
            )
        
        if unique_count > 20:
            return ValidationCheck(
                rule_name="cardinality_limits",
                result=ValidationResult.WARNING,
                message=f"High cardinality ({unique_count}) may create many features"
            )
        
        return ValidationCheck(
            rule_name="cardinality_limits",
            result=ValidationResult.VALID,
            message=f"Cardinality {unique_count} is within limits"
        )
    
    def _validate_text_meaningfulness(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str,
        semantic_type: SemanticType
    ) -> ValidationCheck:
        """Rule 5: Text vectorization requires meaningful text."""
        if proposed_action not in self.TEXT_ACTIONS:
            return ValidationCheck(
                rule_name="text_meaningfulness",
                result=ValidationResult.VALID,
                message="Action is not text-specific"
            )
        
        # URLs should not be vectorized
        if semantic_type == SemanticType.URL:
            return ValidationCheck(
                rule_name="text_meaningfulness",
                result=ValidationResult.INVALID,
                message=f"Cannot vectorize URL column '{column_name}' - drop or extract components instead",
                suggested_action="drop_column"
            )
        
        # Phone numbers should not be vectorized
        if semantic_type == SemanticType.PHONE:
            return ValidationCheck(
                rule_name="text_meaningfulness",
                result=ValidationResult.INVALID,
                message=f"Cannot vectorize phone number column '{column_name}'",
                suggested_action="drop_column"
            )
        
        # Email should not be vectorized (typically)
        if semantic_type == SemanticType.EMAIL:
            return ValidationCheck(
                rule_name="text_meaningfulness",
                result=ValidationResult.INVALID,
                message=f"Cannot vectorize email column '{column_name}'",
                suggested_action="drop_column"
            )
        
        # Identifiers should not be vectorized
        if semantic_type == SemanticType.IDENTIFIER:
            return ValidationCheck(
                rule_name="text_meaningfulness",
                result=ValidationResult.INVALID,
                message=f"Cannot vectorize identifier column '{column_name}' - data leakage risk",
                suggested_action="drop_column"
            )
        
        # Check for meaningful text content
        if semantic_type == SemanticType.TEXT:
            non_null = column.dropna().astype(str)
            avg_length = non_null.str.len().mean() if len(non_null) > 0 else 0
            
            if avg_length < 5:
                return ValidationCheck(
                    rule_name="text_meaningfulness",
                    result=ValidationResult.WARNING,
                    message="Text content may be too short for meaningful vectorization"
                )
        
        return ValidationCheck(
            rule_name="text_meaningfulness",
            result=ValidationResult.VALID,
            message="Text column can be vectorized"
        )
    
    def _validate_no_target_binning(
        self,
        column_name: str,
        proposed_action: str,
        is_target: bool
    ) -> ValidationCheck:
        """Rule 6: No binning targets."""
        if not is_target:
            return ValidationCheck(
                rule_name="no_target_binning",
                result=ValidationResult.VALID,
                message="Not a target column"
            )
        
        binning_actions = ['binning_equal_width', 'binning_equal_freq', 'binning_custom']
        
        if proposed_action in binning_actions:
            return ValidationCheck(
                rule_name="no_target_binning",
                result=ValidationResult.INVALID,
                message=f"Cannot bin target column '{column_name}' - destroys predictive value",
                suggested_action="keep_as_is"
            )
        
        return ValidationCheck(
            rule_name="no_target_binning",
            result=ValidationResult.VALID,
            message="No binning attempted on target"
        )
    
    def _validate_empty_column(
        self,
        column: pd.Series,
        column_name: str,
        proposed_action: str,
        semantic_type: SemanticType
    ) -> ValidationCheck:
        """Rule 7: Empty columns should be dropped."""
        if semantic_type == SemanticType.EMPTY:
            if proposed_action != 'drop_column':
                return ValidationCheck(
                    rule_name="empty_column",
                    result=ValidationResult.OVERRIDE,
                    message=f"Column '{column_name}' is empty - should be dropped",
                    suggested_action="drop_column"
                )
        
        # Check for mostly empty
        null_ratio = column.isnull().sum() / len(column) if len(column) > 0 else 1.0
        if null_ratio > 0.9:
            return ValidationCheck(
                rule_name="empty_column",
                result=ValidationResult.WARNING,
                message=f"Column '{column_name}' is {null_ratio:.0%} null - consider dropping"
            )
        
        return ValidationCheck(
            rule_name="empty_column",
            result=ValidationResult.VALID,
            message="Column has sufficient data"
        )
    
    def _get_safe_fallback(
        self,
        semantic_type: SemanticType,
        is_target: bool
    ) -> str:
        """Get a safe fallback action based on semantic type."""
        if is_target:
            return "keep_as_is"
        
        fallback_map = {
            SemanticType.EMPTY: "drop_column",
            SemanticType.URL: "drop_column",
            SemanticType.PHONE: "drop_column",
            SemanticType.EMAIL: "drop_column",
            SemanticType.IDENTIFIER: "drop_column",
            SemanticType.DATETIME: "datetime_extract",
            SemanticType.BOOLEAN: "parse_boolean",
            SemanticType.NUMERIC: "keep_as_is",
            SemanticType.CATEGORICAL: "onehot_encode",
            SemanticType.TEXT: "text_clean",
        }
        
        return fallback_map.get(semantic_type, "keep_as_is")
    
    def get_recommended_action(
        self,
        column: pd.Series,
        column_name: str,
        is_target: bool = False
    ) -> Tuple[str, str]:
        """
        Get recommended action based on semantic type and validation.
        
        Args:
            column: Data column
            column_name: Name of column
            is_target: Whether this is target variable
            
        Returns:
            Tuple of (recommended_action, explanation)
        """
        # Detect semantic type
        type_result = self.type_detector.detect_semantic_type(column, column_name)
        semantic_type = type_result.semantic_type
        
        # Target columns are protected
        if is_target:
            return "keep_as_is", "Target column - no transformation applied"
        
        # Lazy recommendations by semantic type
        if semantic_type == SemanticType.EMPTY:
            return ("drop_column", "Empty column should be removed")
        elif semantic_type == SemanticType.URL:
            return ("drop_column", "URL column - typically not useful for ML")
        elif semantic_type == SemanticType.PHONE:
            return ("drop_column", "Phone number column - unique identifier, drop to prevent leakage")
        elif semantic_type == SemanticType.EMAIL:
            return ("drop_column", "Email column - unique identifier, drop to prevent leakage")
        elif semantic_type == SemanticType.IDENTIFIER:
            return ("drop_column", "Identifier column - drop to prevent data leakage")
        elif semantic_type == SemanticType.DATETIME:
            return ("datetime_extract", "DateTime column - extract useful components")
        elif semantic_type == SemanticType.BOOLEAN:
            return ("parse_boolean", "Boolean column - convert to 0/1")
        elif semantic_type == SemanticType.NUMERIC:
            return self._get_numeric_recommendation(column)
        elif semantic_type == SemanticType.CATEGORICAL:
            return self._get_categorical_recommendation(column)
        elif semantic_type == SemanticType.TEXT:
            return ("text_clean", "Text column - clean and optionally vectorize")
        else:
            return ("keep_as_is", "No specific recommendation")
    
    def _get_numeric_recommendation(
        self,
        column: pd.Series
    ) -> Tuple[str, str]:
        """Get recommendation for numeric columns."""
        non_null = column.dropna()
        if len(non_null) == 0:
            return ("drop_column", "Numeric column with no values")
        
        # Check skewness
        skewness = non_null.skew()
        min_val = non_null.min()
        
        if abs(skewness) > 2 and min_val >= 0:
            return ("log1p_transform", f"High skewness ({skewness:.2f}) - log transform recommended")
        elif abs(skewness) < 1:
            return ("standard_scale", "Normal distribution - standard scaling recommended")
        else:
            return ("robust_scale", "Moderate skewness - robust scaling recommended")
    
    def _get_categorical_recommendation(
        self,
        column: pd.Series
    ) -> Tuple[str, str]:
        """Get recommendation for categorical columns."""
        unique_count = column.nunique()
        
        if unique_count <= 10:
            return ("onehot_encode", f"Low cardinality ({unique_count}) - one-hot encoding recommended")
        elif unique_count <= self.MAX_ONEHOT_CARDINALITY:
            return ("onehot_encode", f"Medium cardinality ({unique_count}) - one-hot encoding with caution")
        else:
            return ("frequency_encode", f"High cardinality ({unique_count}) - frequency encoding recommended")


def get_preprocessing_validator() -> PreprocessingValidator:
    """Get a singleton instance of the preprocessing validator."""
    return PreprocessingValidator()
