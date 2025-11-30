"""
Rule definitions for the Symbolic Engine.
Contains 100+ deterministic rules for preprocessing decisions.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import re
import numpy as np
import pandas as pd

from ..core.actions import PreprocessingAction


class RuleCategory(Enum):
    """Categories of rules."""
    DATA_QUALITY = "data_quality"
    TYPE_DETECTION = "type_detection"
    STATISTICAL = "statistical"
    CATEGORICAL = "categorical"
    DOMAIN_SPECIFIC = "domain_specific"


@dataclass
class Rule:
    """A preprocessing rule with confidence calculation."""
    name: str
    category: RuleCategory
    action: PreprocessingAction
    condition: Callable[[Dict[str, Any]], bool]
    confidence_fn: Callable[[Dict[str, Any]], float]
    explanation_fn: Callable[[Dict[str, Any]], str]
    priority: int = 0  # Higher priority rules are evaluated first
    parameters: Optional[Dict[str, Any]] = None

    def evaluate(self, column_stats: Dict[str, Any]) -> Optional[tuple[PreprocessingAction, float, str]]:
        """
        Evaluate the rule against column statistics.
        Returns (action, confidence, explanation) if rule applies, None otherwise.
        """
        if self.condition(column_stats):
            confidence = self.confidence_fn(column_stats)
            explanation = self.explanation_fn(column_stats)
            return (self.action, confidence, explanation)
        return None


# =============================================================================
# DATA QUALITY RULES (20 rules)
# =============================================================================

def create_data_quality_rules() -> List[Rule]:
    """Create data quality rules."""
    rules = []

    # Rule 1: Drop if mostly null
    rules.append(Rule(
        name="DROP_IF_MOSTLY_NULL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: stats.get("null_pct", 0) > 0.6,
        confidence_fn=lambda stats: min(0.95, stats.get("null_pct", 0) / 0.6 * 0.9),
        explanation_fn=lambda stats: f"Column has {stats.get('null_pct', 0):.1%} null values (> 60% threshold)",
        priority=100
    ))

    # Rule 2: Drop if constant
    rules.append(Rule(
        name="DROP_IF_CONSTANT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: stats.get("unique_count", 2) == 1 and stats.get("null_pct", 0) < 0.5,
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Column has only one unique value (constant)",
        priority=95
    ))

    # Rule 3: Drop if all unique (likely ID)
    rules.append(Rule(
        name="DROP_IF_ALL_UNIQUE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_ratio", 0) > 0.95 and
            stats.get("row_count", 0) > 100 and
            not stats.get("is_numeric", False)
        ),
        confidence_fn=lambda stats: 0.9 if stats.get("unique_ratio", 0) > 0.98 else 0.75,
        explanation_fn=lambda stats: f"Column has {stats.get('unique_ratio', 0):.1%} unique values (likely ID)",
        priority=90
    ))

    # Rule 4: Remove duplicates (CONSERVATIVE - only for extreme cases)
    # Only trigger when column has >70% duplicates AND very low cardinality
    # This avoids false positives on categorical columns with natural duplicates
    rules.append(Rule(
        name="REMOVE_DUPLICATES",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.REMOVE_DUPLICATES,
        condition=lambda stats: (
            stats.get("duplicate_ratio", 0) > 0.7 and
            stats.get("unique_ratio", 1.0) < 0.1 and
            not stats.get("is_categorical", False) and
            stats.get("row_count", 0) > 100
        ),
        confidence_fn=lambda stats: min(0.85, 0.6 + stats.get("duplicate_ratio", 0) * 0.25),
        explanation_fn=lambda stats: f"Column has excessive duplicates ({stats.get('duplicate_ratio', 0):.1%}) with very low cardinality",
        priority=85
    ))

    # KEEP_AS_IS RULES - HIGH PRIORITY (evaluated first)
    # These catch healthy columns that don't need preprocessing

    # Rule 4a: Keep numeric columns already normalized
    rules.append(Rule(
        name="KEEP_IF_ALREADY_NORMALIZED",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("null_pct", 0) < 0.05 and
            -0.5 <= stats.get("mean", 100) <= 0.5 and
            0.8 <= stats.get("std", 0) <= 1.2 and
            abs(stats.get("skewness", 0)) < 0.5
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"Already normalized: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}",
        priority=95
    ))

    # Rule 4b: Keep numeric columns with good distribution
    rules.append(Rule(
        name="KEEP_IF_GOOD_DISTRIBUTION",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("null_pct", 0) < 0.05 and
            abs(stats.get("skewness", 0)) < 1.0 and
            not stats.get("has_outliers", True) and
            stats.get("cv", 100) < 2.0  # Coefficient of variation
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Good distribution: low skewness ({stats.get('skewness', 0):.2f}), no outliers",
        priority=92
    ))

    # Rule 4c: Keep categorical columns with good quality
    rules.append(Rule(
        name="KEEP_IF_CATEGORICAL_CLEAN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("null_pct", 0) < 0.05 and
            2 <= stats.get("unique_count", 0) <= 50 and
            stats.get("unique_ratio", 0) < 0.8
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: f"Clean categorical: {stats.get('unique_count', 0)} categories, minimal nulls",
        priority=90
    ))

    # Rule 4d: Keep columns with no nulls and reasonable properties
    rules.append(Rule(
        name="KEEP_IF_NO_QUALITY_ISSUES",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("null_pct", 0) == 0 and
            stats.get("unique_ratio", 0) > 0.05 and
            stats.get("unique_ratio", 0) < 0.95 and
            not stats.get("has_outliers", True)
        ),
        confidence_fn=lambda stats: 0.82,
        explanation_fn=lambda stats: "No data quality issues: no nulls, good cardinality, no outliers",
        priority=88
    ))

    # Rule 5-9: Null filling strategies based on data type and distribution
    rules.append(Rule(
        name="FILL_NULL_MEDIAN_NUMERIC",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEDIAN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0.05 < stats.get("null_pct", 0) < 0.3 and
            stats.get("has_outliers", False)
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Numeric column with outliers: median imputation is robust",
        priority=70
    ))

    rules.append(Rule(
        name="FILL_NULL_MEAN_NUMERIC",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEAN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0.05 < stats.get("null_pct", 0) < 0.3 and
            not stats.get("has_outliers", False) and
            abs(stats.get("skewness", 0)) < 1.0
        ),
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Numeric column with normal distribution: mean imputation appropriate",
        priority=75
    ))

    rules.append(Rule(
        name="FILL_NULL_MODE_CATEGORICAL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            0.05 < stats.get("null_pct", 0) < 0.3
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Categorical column: mode imputation is standard",
        priority=75
    ))

    rules.append(Rule(
        name="FILL_NULL_FORWARD_TIMESERIES",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_FORWARD,
        condition=lambda stats: (
            stats.get("is_temporal", False) and
            stats.get("null_pct", 0) < 0.2
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Temporal data: forward fill preserves time dependencies",
        priority=80
    ))

    rules.append(Rule(
        name="FILL_NULL_INTERPOLATE_SMOOTH",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_INTERPOLATE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_smooth", False) and
            stats.get("null_pct", 0) < 0.15
        ),
        confidence_fn=lambda stats: 0.82,
        explanation_fn=lambda stats: "Smooth numeric series: interpolation preserves trends",
        priority=72
    ))

    # Rules 10-15: High null percentage scenarios
    rules.append(Rule(
        name="DROP_IF_HIGH_NULL_NO_PATTERN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("null_pct", 0) > 0.4 and
            stats.get("null_pct", 0) <= 0.6 and
            not stats.get("null_has_pattern", False)
        ),
        confidence_fn=lambda stats: 0.75,
        explanation_fn=lambda stats: f"High null percentage ({stats.get('null_pct', 0):.1%}) with no clear pattern",
        priority=80
    ))

    # Rules 16-20: Data quality indicators
    rules.append(Rule(
        name="KEEP_IF_LOW_NULL_GOOD_QUALITY",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("null_pct", 0) < 0.05 and
            stats.get("unique_ratio", 0) > 0.1 and
            stats.get("unique_ratio", 0) < 0.9
        ),
        confidence_fn=lambda stats: 0.7,
        explanation_fn=lambda stats: "High quality column with minimal nulls",
        priority=50
    ))

    return rules


# =============================================================================
# TYPE DETECTION RULES (15 rules)
# =============================================================================

def create_type_detection_rules() -> List[Rule]:
    """Create type detection and conversion rules."""
    rules = []

    # Rule 1-3: DateTime parsing
    rules.append(Rule(
        name="PARSE_DATETIME_ISO",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_DATETIME,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("matches_iso_datetime", 0) > 0.8
        ),
        confidence_fn=lambda stats: min(0.98, stats.get("matches_iso_datetime", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_iso_datetime', 0):.1%} values match ISO datetime format",
        priority=95
    ))

    rules.append(Rule(
        name="PARSE_DATETIME_COMMON",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_DATETIME,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("matches_date_pattern", 0) > 0.85
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_date_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_date_pattern', 0):.1%} values match common date patterns",
        priority=90
    ))

    # Rule 4-6: Boolean parsing
    rules.append(Rule(
        name="PARSE_BOOLEAN_TF",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_BOOLEAN,
        condition=lambda stats: (
            stats.get("unique_count", 100) <= 3 and
            stats.get("matches_boolean_tf", 0) > 0.9
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Values are T/F, True/False, or similar boolean pattern",
        priority=95
    ))

    rules.append(Rule(
        name="PARSE_BOOLEAN_YN",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_BOOLEAN,
        condition=lambda stats: (
            stats.get("unique_count", 100) <= 3 and
            stats.get("matches_boolean_yn", 0) > 0.9
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Values are Y/N, Yes/No, or similar boolean pattern",
        priority=95
    ))

    rules.append(Rule(
        name="PARSE_BOOLEAN_01",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_BOOLEAN,
        condition=lambda stats: (
            stats.get("unique_count", 100) <= 3 and
            stats.get("matches_boolean_01", 0) > 0.9 and
            not stats.get("looks_like_categorical_code", False)
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Binary 0/1 values (likely boolean)",
        priority=85
    ))

    # Rule 7-9: Numeric parsing
    rules.append(Rule(
        name="PARSE_NUMERIC_STRING",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_NUMERIC,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("matches_numeric_pattern", 0) > 0.9
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_numeric_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_numeric_pattern', 0):.1%} values are numeric strings",
        priority=90
    ))

    # Rule 10-12: JSON/structured data parsing
    rules.append(Rule(
        name="PARSE_JSON_STRUCTURE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_JSON,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("matches_json", 0) > 0.85
        ),
        confidence_fn=lambda stats: min(0.92, stats.get("matches_json", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_json', 0):.1%} values are valid JSON",
        priority=88
    ))

    # Rule 13-15: Categorical detection
    rules.append(Rule(
        name="PARSE_CATEGORICAL_LOW_CARDINALITY",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_CATEGORICAL,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("cardinality", 1000) < 50 and
            stats.get("unique_ratio", 1.0) < 0.5
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Low cardinality ({stats.get('cardinality', 0)} unique values) indicates categorical",
        priority=70
    ))

    return rules


# =============================================================================
# STATISTICAL RULES (25 rules)
# =============================================================================

def create_statistical_rules() -> List[Rule]:
    """Create statistical transformation rules."""
    rules = []

    # Rule 1-5: Log transformations for skewed data
    # NOTE: range_size > 100 check prevents applying log transform to bounded data
    # like ratings (0-5, 0-10, 0-100) and years (1900-2100)
    rules.append(Rule(
        name="LOG_TRANSFORM_HIGH_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 2.0 and
            stats.get("all_positive", False) and
            stats.get("min_value", 0) > 0 and
            stats.get("range_size", 0) > 100  # Avoid log transform on bounded data like ratings, years
        ),
        confidence_fn=lambda stats: min(0.95, 0.7 + (stats.get("skewness", 0) - 2.0) * 0.05),
        explanation_fn=lambda stats: f"High positive skewness ({stats.get('skewness', 0):.2f}) in positive data with large range ({stats.get('range_size', 0):.0f})",
        priority=85
    ))

    rules.append(Rule(
        name="LOG1P_TRANSFORM_WITH_ZEROS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 2.0 and
            stats.get("min_value", -1) >= 0 and
            stats.get("has_zeros", False) and
            stats.get("range_size", 0) > 100  # Avoid log transform on bounded data like ratings, years
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"High skewness ({stats.get('skewness', 0):.2f}) with zeros present and large range ({stats.get('range_size', 0):.0f})",
        priority=87
    ))

    # Rule 6-10: Box-Cox and power transformations
    rules.append(Rule(
        name="BOX_COX_MODERATE_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.BOX_COX,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            1.5 < stats.get("skewness", 0) < 2.0 and
            stats.get("all_positive", False) and
            stats.get("min_value", 0) > 0
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Moderate skewness ({stats.get('skewness', 0):.2f}) suitable for Box-Cox",
        priority=80
    ))

    rules.append(Rule(
        name="YEO_JOHNSON_NEGATIVE_VALUES",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.YEO_JOHNSON,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) > 1.5 and
            not stats.get("all_positive", True)
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: f"Skewed data with negative values: Yeo-Johnson handles all real numbers",
        priority=80
    ))

    # Rule 11-15: Outlier handling
    rules.append(Rule(
        name="CLIP_OUTLIERS_HIGH_PERCENTAGE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.CLIP_OUTLIERS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("outlier_pct", 0) > 0.1
        ),
        confidence_fn=lambda stats: min(0.9, 0.6 + stats.get("outlier_pct", 0) * 0.3),
        explanation_fn=lambda stats: f"{stats.get('outlier_pct', 0):.1%} outliers detected (IQR method)",
        priority=75
    ))

    rules.append(Rule(
        name="WINSORIZE_MODERATE_OUTLIERS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.WINSORIZE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0.05 < stats.get("outlier_pct", 0) <= 0.1
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: f"Moderate outliers ({stats.get('outlier_pct', 0):.1%}): winsorizing preserves distribution",
        priority=78
    ))

    # Rule 16-20: Scaling decisions
    rules.append(Rule(
        name="STANDARD_SCALE_NORMAL",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) < 0.5 and
            stats.get("outlier_pct", 0) < 0.05 and
            not stats.get("is_already_scaled", False)
        ),
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Normal distribution without outliers: standard scaling appropriate",
        priority=70
    ))

    rules.append(Rule(
        name="ROBUST_SCALE_WITH_OUTLIERS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("outlier_pct", 0) >= 0.05 and
            stats.get("outlier_pct", 0) < 0.1 and
            not stats.get("is_already_scaled", False)
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Outliers present: robust scaling using median/IQR",
        priority=75
    ))

    rules.append(Rule(
        name="MINMAX_SCALE_BOUNDED",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.MINMAX_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_natural_bounds", False) and
            stats.get("outlier_pct", 0) < 0.02
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Naturally bounded data: min-max scaling preserves bounds",
        priority=72
    ))

    # Rule 21-25: Distribution-based transformations
    rules.append(Rule(
        name="QUANTILE_TRANSFORM_BIMODAL",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.QUANTILE_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_bimodal", False)
        ),
        confidence_fn=lambda stats: 0.8,
        explanation_fn=lambda stats: "Bimodal distribution: quantile transform creates uniform distribution",
        priority=70
    ))

    return rules


# =============================================================================
# CATEGORICAL RULES (20 rules)
# =============================================================================

def create_categorical_rules() -> List[Rule]:
    """Create categorical encoding rules."""
    rules = []

    # Rule 1-5: One-hot encoding
    rules.append(Rule(
        name="ONEHOT_LOW_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 100) < 10 and
            not stats.get("is_ordinal", False)
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: f"Low cardinality ({stats.get('cardinality', 0)} categories) suitable for one-hot",
        priority=85
    ))

    rules.append(Rule(
        name="ONEHOT_VERY_LOW_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 100) <= 5
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: f"Very low cardinality ({stats.get('cardinality', 0)} categories): one-hot is standard",
        priority=90
    ))

    # Rule 6-10: Target encoding
    rules.append(Rule(
        name="TARGET_ENCODE_HIGH_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.TARGET_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 50 and
            stats.get("target_available", False)
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"High cardinality ({stats.get('cardinality', 0)} categories) with target: target encoding efficient",
        priority=80
    ))

    # Rule 11-15: Ordinal encoding
    rules.append(Rule(
        name="ORDINAL_ENCODE_DETECTED",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("is_ordinal", False)
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Detected ordinal pattern in categories",
        priority=92
    ))

    # Rule 16-20: Frequency encoding
    rules.append(Rule(
        name="FREQUENCY_ENCODE_MEDIUM_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            10 <= stats.get("cardinality", 0) <= 50 and
            not stats.get("target_available", False)
        ),
        confidence_fn=lambda stats: 0.82,
        explanation_fn=lambda stats: f"Medium cardinality ({stats.get('cardinality', 0)} categories) without target: frequency encoding",
        priority=75
    ))

    return rules


# =============================================================================
# DOMAIN-SPECIFIC RULES (20 rules)
# =============================================================================

def create_domain_specific_rules() -> List[Rule]:
    """Create domain-specific transformation rules."""
    rules = []

    # Rule 1-5: Currency and financial
    rules.append(Rule(
        name="CURRENCY_NORMALIZE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.CURRENCY_NORMALIZE,
        condition=lambda stats: (
            stats.get("has_currency_symbols", False) and
            stats.get("matches_currency_pattern", 0) > 0.8
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_currency_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_currency_pattern', 0):.1%} values match currency pattern",
        priority=95
    ))

    rules.append(Rule(
        name="PERCENTAGE_TO_DECIMAL",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.PERCENTAGE_TO_DECIMAL,
        condition=lambda stats: (
            stats.get("has_percentage_symbol", False) or
            (stats.get("is_numeric", False) and
             stats.get("min_value", -1) >= 0 and
             stats.get("max_value", 200) <= 100 and
             stats.get("looks_like_percentage", False))
        ),
        confidence_fn=lambda stats: 0.92 if stats.get("has_percentage_symbol", False) else 0.75,
        explanation_fn=lambda stats: "Values appear to be percentages",
        priority=90
    ))

    # Rule 6-10: Contact information
    rules.append(Rule(
        name="PHONE_STANDARDIZE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.PHONE_STANDARDIZE,
        condition=lambda stats: (
            stats.get("matches_phone_pattern", 0) > 0.85
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_phone_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_phone_pattern', 0):.1%} values match phone number patterns",
        priority=92
    ))

    rules.append(Rule(
        name="EMAIL_VALIDATE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.EMAIL_VALIDATE,
        condition=lambda stats: (
            stats.get("matches_email_pattern", 0) > 0.85
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_email_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_email_pattern', 0):.1%} values match email pattern",
        priority=92
    ))

    # Rule 11-15: URL and web data
    rules.append(Rule(
        name="URL_PARSE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.URL_PARSE,
        condition=lambda stats: (
            stats.get("matches_url_pattern", 0) > 0.85
        ),
        confidence_fn=lambda stats: min(0.95, stats.get("matches_url_pattern", 0)),
        explanation_fn=lambda stats: f"{stats.get('matches_url_pattern', 0):.1%} values are valid URLs",
        priority=90
    ))

    # Rule 16-20: Text normalization
    rules.append(Rule(
        name="TEXT_CLEAN_MESSY",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.TEXT_CLEAN,
        condition=lambda stats: (
            stats.get("is_text", False) and
            (stats.get("has_extra_whitespace", False) or
             stats.get("has_special_chars", False))
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Text column with inconsistent formatting",
        priority=70
    ))

    return rules


# =============================================================================
# RULE REGISTRY
# =============================================================================

def get_all_rules(include_extended: bool = True) -> List[Rule]:
    """
    Get all rules sorted by priority.

    Args:
        include_extended: Whether to include extended rules for universal coverage

    Returns:
        List of rules sorted by priority (highest first)
    """
    all_rules = []

    # HIGH PRIORITY: Simple case rules (catches obvious preprocessing needs)
    try:
        from .simple_case_rules import create_simple_case_rules
        all_rules.extend(create_simple_case_rules())
    except ImportError:
        # Simple case rules not available, continue
        pass

    # Base rules (~100 rules)
    all_rules.extend(create_data_quality_rules())
    all_rules.extend(create_type_detection_rules())
    all_rules.extend(create_statistical_rules())
    all_rules.extend(create_categorical_rules())
    all_rules.extend(create_domain_specific_rules())

    # Extended rules for universal coverage (~65 rules)
    if include_extended:
        try:
            from .extended_rules import get_extended_rules
            all_rules.extend(get_extended_rules())
        except ImportError:
            # Extended rules not available, continue with base rules
            pass

    # Sort by priority (higher first)
    all_rules.sort(key=lambda r: r.priority, reverse=True)

    return all_rules


def get_rules_by_category(category: RuleCategory) -> List[Rule]:
    """Get all rules in a specific category."""
    all_rules = get_all_rules()
    return [r for r in all_rules if r.category == category]
