"""
Enhanced rules for simple common cases that were being missed.

These rules have HIGH priority and HIGH confidence to catch obvious preprocessing needs.
"""

from typing import List
from .rules import Rule, RuleCategory
from ..core.actions import PreprocessingAction


def create_simple_case_rules() -> List[Rule]:
    """
    Create high-confidence rules for simple,obvious preprocessing cases.

    These rules catch common scenarios that should NEVER be missed:
    - Binary columns (0/1, True/False)
    - Completely null columns
    - Numeric columns that clearly need scaling
    - Categorical columns with very few categories
    - Text columns that need cleaning
    """
    rules = []

    # =========================================================================
    # SUPER HIGH PRIORITY - OBVIOUS CASES (Priority 100+)
    # =========================================================================

    # Binary numeric columns (0 and 1 only) - keep as is or convert to boolean
    rules.append(Rule(
        name="KEEP_BINARY_NUMERIC",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("unique_count", 100) == 2 and
            stats.get("min_value", -1) == 0 and
            stats.get("max_value", 2) == 1 and
            stats.get("null_pct", 1.0) < 0.1
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Binary column (0/1): already optimal for ML",
        priority=105
    ))

    # Completely null column - always drop
    rules.append(Rule(
        name="DROP_IF_ALL_NULL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: stats.get("null_pct", 0) >= 0.99,
        confidence_fn=lambda stats: 0.99,
        explanation_fn=lambda stats: "Column is entirely (or almost entirely) null - no information",
        priority=110
    ))

    # Single value column - always drop
    rules.append(Rule(
        name="DROP_SINGLE_VALUE_STRICT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 2) == 1 or
            (stats.get("unique_ratio", 1.0) < 0.01 and stats.get("row_count", 0) > 100)
        ),
        confidence_fn=lambda stats: 0.99,
        explanation_fn=lambda stats: "Column has only one unique value - no variance",
        priority=108
    ))

    # Categorical with 2-3 categories - one-hot encode
    rules.append(Rule(
        name="ONEHOT_VERY_LOW_CARDINALITY_STRICT",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            2 <= stats.get("unique_count", 0) <= 3 and
            stats.get("null_pct", 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: f"{stats.get('unique_count', 0)} categories: one-hot is standard",
        priority=103
    ))

    # Numeric column with huge range and no scaling - needs robust scaling
    rules.append(Rule(
        name="ROBUST_SCALE_HUGE_RANGE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("range_size", 0) > 10000 and
            not stats.get("is_already_scaled", False) and
            stats.get("null_pct", 1.0) < 0.3
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: f"Huge range ({stats.get('range_size', 0):.0f}): robust scaling needed",
        priority=102
    ))

    # =========================================================================
    # HIGH PRIORITY - VERY COMMON CASES (Priority 95-99)
    # =========================================================================

    # Small integer range (like 1-5 ratings) - keep as is
    rules.append(Rule(
        name="KEEP_SMALL_INTEGER_RANGE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("dtype", "").startswith("int") and
            stats.get("range_size", 1000) <= 10 and
            stats.get("min_value", -100) >= 0 and
            stats.get("null_pct", 1.0) < 0.05
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: f"Small integer range (0-{int(stats.get('max_value', 0))}): likely ordinal or count",
        priority=98
    ))

    # Percentage values (0-100 range) - normalize to 0-1
    rules.append(Rule(
        name="NORMALIZE_PERCENTAGE_RANGE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.PERCENTAGE_TO_DECIMAL,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0 <= stats.get("min_value", -1) and
            stats.get("max_value", 200) <= 100 and
            stats.get("max_value", 0) > 10 and  # Distinguish from 0-1 range
            stats.get("null_pct", 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Values in 0-100 range: likely percentages",
        priority=97
    ))

    # Already normalized (0-1 range) - keep as is
    rules.append(Rule(
        name="KEEP_ALREADY_NORMALIZED_01",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0 <= stats.get("min_value", -1) and
            stats.get("max_value", 2) <= 1.0 and
            stats.get("min_value", 1.0) < 1.0 and  # Not just 0 or 1
            stats.get("null_pct", 1.0) < 0.05
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Already in [0, 1] range: no scaling needed",
        priority=99
    ))

    # Categorical with 4-7 categories - one-hot still good
    rules.append(Rule(
        name="ONEHOT_LOW_CARDINALITY_CLEAR",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            4 <= stats.get("unique_count", 0) <= 7 and
            stats.get("null_pct", 1.0) < 0.15
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"{stats.get('unique_count', 0)} categories: one-hot won't explode dimensions",
        priority=96
    ))

    # Extreme positive skew (>3) with positive values - log transform
    rules.append(Rule(
        name="LOG_TRANSFORM_EXTREME_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 3.0 and
            stats.get("min_value", 0) > 0 and
            stats.get("null_pct", 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: f"Extreme skewness ({stats.get('skewness', 0):.2f}): log transform essential",
        priority=98
    ))

    # Extreme positive skew with zeros - log1p transform
    rules.append(Rule(
        name="LOG1P_TRANSFORM_EXTREME_SKEW_ZEROS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 3.0 and
            stats.get("min_value", -1) >= 0 and
            (stats.get("has_zeros", False) or stats.get("min_value", 1) == 0) and
            stats.get("null_pct", 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: f"Extreme skewness ({stats.get('skewness', 0):.2f}) with zeros: log1p transform",
        priority=97
    ))

    # Very low null percentage (< 2%) for numeric - fill with median
    rules.append(Rule(
        name="FILL_NULL_MEDIAN_MINIMAL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEDIAN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0.001 < stats.get("null_pct", 0) < 0.02 and
            stats.get("row_count", 0) > 100
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"Minimal nulls ({stats.get('null_pct', 0):.1%}): median imputation safe",
        priority=95
    ))

    # Very low null percentage (< 2%) for categorical - fill with mode
    rules.append(Rule(
        name="FILL_NULL_MODE_MINIMAL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            0.001 < stats.get("null_pct", 0) < 0.02 and
            stats.get("row_count", 0) > 100
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"Minimal nulls ({stats.get('null_pct', 0):.1%}): mode imputation safe",
        priority=95
    ))

    # Numeric column with std very close to 0 - constant (but not exactly)
    rules.append(Rule(
        name="DROP_IF_NEAR_CONSTANT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("std", 1.0) < 0.0001 and
            stats.get("unique_count", 2) <= 3 and
            stats.get("null_pct", 1.0) < 0.5
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: f"Near-constant: std={stats.get('std', 0):.6f}, no variance",
        priority=96
    ))

    # High cardinality categorical (>100) - hash encode
    rules.append(Rule(
        name="HASH_ENCODE_VERY_HIGH_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("unique_count", 0) > 100 and
            stats.get("unique_ratio", 1.0) < 0.8 and  # Not all unique
            stats.get("null_pct", 1.0) < 0.3
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: f"Very high cardinality ({stats.get('unique_count', 0)}): hash encoding prevents explosion",
        priority=94
    ))

    # Numeric with many outliers (>15%) - clip or winsorize
    rules.append(Rule(
        name="CLIP_OUTLIERS_MANY",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.CLIP_OUTLIERS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("outlier_pct", 0) > 0.15 and
            stats.get("null_pct", 1.0) < 0.3
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: f"Many outliers ({stats.get('outlier_pct', 0):.1%}): clipping recommended",
        priority=93
    ))

    # Normal distribution (low skew, low kurtosis) - standard scale
    rules.append(Rule(
        name="STANDARD_SCALE_NORMAL_DIST",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 10)) < 0.8 and
            abs(stats.get("kurtosis", 10)) < 2.0 and
            not stats.get("is_already_scaled", False) and
            stats.get("outlier_pct", 0.2) < 0.08 and
            stats.get("null_pct", 1.0) < 0.1
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Nearly normal distribution: standard scaling optimal",
        priority=95
    ))

    return rules
