"""
Advanced AutoML-Grade Rules for Symbolic Engine.
These rules provide enterprise-level data preprocessing capabilities comparable to
AutoML platforms like DataRobot, H2O AutoML, and Google AutoML.
"""

from typing import List
from .rules import Rule, RuleCategory
from ..core.actions import PreprocessingAction


def create_leakage_detection_rules() -> List[Rule]:
    """Create data leakage detection rules (5 rules)."""
    rules = []

    # Rule 1: Perfect correlation with target (data leakage)
    rules.append(Rule(
        name="LEAKAGE_PERFECT_CORRELATION",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("target_available", False) and
            abs(stats.get("correlation_with_target", 0)) > 0.99
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: f"Perfect correlation with target ({stats.get('correlation_with_target', 0):.3f}): likely data leakage. Dropping column.",
        priority=99
    ))

    # Rule 2: Constant column (zero variance)
    rules.append(Rule(
        name="LEAKAGE_CONSTANT_COLUMN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 2) == 1
        ),
        confidence_fn=lambda stats: 0.99,
        explanation_fn=lambda stats: "Constant column (only one unique value): provides no information. Dropping.",
        priority=99
    ))

    # Rule 3: Near-constant column (>99% same value)
    rules.append(Rule(
        name="LEAKAGE_NEAR_CONSTANT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 100) > 1 and
            stats.get("unique_ratio", 0) < 0.01  # <1% unique
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: f"Near-constant column ({stats.get('unique_ratio', 0):.2%} unique): minimal information. Dropping.",
        priority=98
    ))

    # Rule 4: ID column with perfect uniqueness
    rules.append(Rule(
        name="LEAKAGE_UNIQUE_ID",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_ratio", 0) > 0.98 and
            stats.get("row_count", 0) > 100 and
            (stats.get("is_primary_key", False) or 
             any(keyword in stats.get("column_name", "").lower() 
                 for keyword in ["id", "key", "uuid", "guid"]))
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: "Unique identifier column: no predictive value. Dropping.",
        priority=97
    ))

    # Rule 5: Duplicate columns (same values, different names)
    rules.append(Rule(
        name="LEAKAGE_DUPLICATE_COLUMN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("duplicate_ratio", 0) > 0.95 and
            stats.get("row_count", 0) > 50
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Column appears to be a duplicate: drop to avoid redundancy.",
        priority=96
    ))

    return rules


def create_advanced_categorical_rules() -> List[Rule]:
    """Create advanced categorical handling rules (10 rules)."""
    rules = []

    # Rule 1: Rare category consolidation
    rules.append(Rule(
        name="CAT_RARE_CONSOLIDATION",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 10 and
            stats.get("cardinality", 0) < 100 and
            stats.get("entropy", 1.0) > 0.7  # High entropy = many rare categories
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Categorical with many rare categories ({stats.get('cardinality', 0)} levels): frequency encoding handles rare values well.",
        priority=82
    ))

    # Rule 2: Very high cardinality categorical (>1000)
    rules.append(Rule(
        name="CAT_VERY_HIGH_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 1000
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: f"Very high cardinality ({stats.get('cardinality', 0)} categories): hash encoding for memory efficiency.",
        priority=90
    ))

    # Rule 3: Binary categorical (2 levels)
    rules.append(Rule(
        name="CAT_BINARY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.LABEL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) == 2
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Binary categorical (2 levels): label encoding is optimal.",
        priority=92
    ))

    # Rule 4: Low cardinality with target (target encoding)
    rules.append(Rule(
        name="CAT_TARGET_ENCODE",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.TARGET_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("target_available", False) and
            3 <= stats.get("cardinality", 0) <= 50
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: f"Categorical with target available ({stats.get('cardinality', 0)} levels): target encoding captures target relationship.",
        priority=85
    ))

    # Rule 5: Medium cardinality without target (one-hot)
    rules.append(Rule(
        name="CAT_ONEHOT_MEDIUM",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            not stats.get("target_available", False) and
            3 <= stats.get("cardinality", 0) <= 10
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"Categorical without target ({stats.get('cardinality', 0)} levels): one-hot encoding for interpretability.",
        priority=88
    ))

    # Rule 6: Ordinal categorical (detected ordering)
    rules.append(Rule(
        name="CAT_ORDINAL_DETECTED",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.LABEL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("is_ordinal", False)
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Ordinal categorical detected: label encoding preserves order.",
        priority=91
    ))

    # Rule 7: Imbalanced categorical (one category >80%)
    rules.append(Rule(
        name="CAT_IMBALANCED",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 1.0) < 0.5 and  # Low entropy = imbalanced
            stats.get("cardinality", 0) > 2
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "Imbalanced categorical: frequency encoding preserves class distribution information.",
        priority=83
    ))

    # Rule 8: Text-like categorical (high avg length)
    rules.append(Rule(
        name="CAT_TEXT_LIKE",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("avg_length", 0) > 20 and
            stats.get("cardinality", 0) > 50
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: f"Text-like categorical (avg length {stats.get('avg_length', 0):.0f}): hash encoding for long strings.",
        priority=84
    ))

    # Rule 9: Categorical with special characters
    rules.append(Rule(
        name="CAT_SPECIAL_CHARS",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("has_special_chars", False) and
            stats.get("cardinality", 0) > 20
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Categorical with special characters: hash encoding handles complex strings.",
        priority=81
    ))

    # Rule 10: Low cardinality (<3 categories)
    rules.append(Rule(
        name="CAT_VERY_LOW_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) < 3
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: f"Very low cardinality ({stats.get('cardinality', 0)} levels): one-hot encoding is optimal.",
        priority=89
    ))

    return rules


def create_advanced_automl_rules() -> List[Rule]:
    """Get all advanced AutoML rules."""
    rules = []
    rules.extend(create_leakage_detection_rules())
    rules.extend(create_advanced_categorical_rules())
    rules.extend(create_time_series_rules())
    rules.extend(create_imbalanced_data_rules())
    rules.extend(create_missing_data_rules())
    rules.extend(create_multicollinearity_rules())
    rules.extend(create_statistical_anomaly_rules())
    rules.extend(create_feature_selection_rules())
    rules.extend(create_dimensionality_reduction_rules())
    return rules


def create_time_series_rules() -> List[Rule]:
    """Create time series detection and handling rules (8 rules)."""
    rules = []

    # Rule 1: Temporal column with high autocorrelation
    rules.append(Rule(
        name="TS_HIGH_AUTOCORRELATION",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_temporal", False) and
            stats.get("unique_ratio", 0) > 0.5
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Temporal column detected: cyclic encoding preserves time patterns.",
        priority=87
    ))

    # Rule 2: Date column with year-month-day pattern
    rules.append(Rule(
        name="TS_DATE_PATTERN",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("matches_date_pattern", False) and
            not stats.get("is_temporal", False)
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Date pattern detected: converting to cyclic time features.",
        priority=86
    ))

    # Rule 3: Timestamp column (Unix epoch)
    rules.append(Rule(
        name="TS_UNIX_TIMESTAMP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", 0) > 1_000_000_000 and
            stats.get("max_value", 0) < 2_000_000_000
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "Unix timestamp detected: converting to time features.",
        priority=85
    ))

    # Rule 4: ISO datetime string
    rules.append(Rule(
        name="TS_ISO_DATETIME",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("matches_iso_datetime", False)
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "ISO datetime format detected: extracting time features.",
        priority=88
    ))

    # Rule 5: Sequential numeric (potential time index)
    rules.append(Rule(
        name="TS_SEQUENTIAL_INDEX",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_smooth", False) and
            stats.get("unique_ratio", 0) > 0.95
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Sequential index detected: keeping as-is for time series analysis.",
        priority=80
    ))

    # Rule 6: Month/Day/Hour columns (cyclic)
    rules.append(Rule(
        name="TS_CYCLIC_COMPONENT",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            any(keyword in stats.get("column_name", "").lower() 
                for keyword in ["month", "day", "hour", "minute", "weekday"]) and
            stats.get("min_value", 0) >= 0 and
            stats.get("max_value", 100) <= 60
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Cyclic time component detected: applying sine/cosine encoding.",
        priority=89
    ))

    # Rule 7: Year column (linear time)
    rules.append(Rule(
        name="TS_YEAR_COLUMN",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            "year" in stats.get("column_name", "").lower() and
            stats.get("min_value", 0) > 1900 and
            stats.get("max_value", 3000) < 2100
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Year column detected: standard scaling for linear trend.",
        priority=84
    ))

    # Rule 8: Time delta (duration)
    rules.append(Rule(
        name="TS_DURATION",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.LOG_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            any(keyword in stats.get("column_name", "").lower() 
                for keyword in ["duration", "elapsed", "time_diff"]) and
            stats.get("skewness", 0) > 2.0
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "Duration column with skew: log transform for normalization.",
        priority=82
    ))

    return rules


def create_imbalanced_data_rules() -> List[Rule]:
    """Create imbalanced data handling rules (5 rules)."""
    rules = []

    # Rule 1: Severe class imbalance (>95:5)
    rules.append(Rule(
        name="IMBALANCE_SEVERE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 1.0) < 0.3 and
            stats.get("cardinality", 0) == 2
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Severe class imbalance detected: consider SMOTE or class weighting.",
        priority=93
    ))

    # Rule 2: Moderate imbalance (80:20)
    rules.append(Rule(
        name="IMBALANCE_MODERATE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            0.3 <= stats.get("entropy", 1.0) < 0.6 and
            stats.get("cardinality", 0) <= 5
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Moderate class imbalance: stratified sampling recommended.",
        priority=91
    ))

    # Rule 3: Rare class (<1%)
    rules.append(Rule(
        name="IMBALANCE_RARE_CLASS",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("unique_ratio", 0) < 0.01 and
            stats.get("cardinality", 0) > 10
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "Rare classes detected: frequency encoding handles minority classes.",
        priority=84
    ))

    # Rule 4: Multi-class imbalance
    rules.append(Rule(
        name="IMBALANCE_MULTICLASS",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 5 and
            stats.get("entropy", 1.0) < 0.7
        ),
        confidence_fn=lambda stats: 0.83,
        explanation_fn=lambda stats: "Multi-class imbalance: frequency encoding preserves class distribution.",
        priority=82
    ))

    # Rule 5: Binary with extreme imbalance
    rules.append(Rule(
        name="IMBALANCE_BINARY_EXTREME",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) == 2 and
            stats.get("entropy", 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Extreme binary imbalance (>99:1): anomaly detection approach recommended.",
        priority=94
    ))

    return rules


def create_missing_data_rules() -> List[Rule]:
    """Create missing data pattern detection rules (6 rules)."""
    rules = []

    # Rule 1: High missingness (>50%)
    rules.append(Rule(
        name="MISSING_HIGH",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("null_pct", 0) > 0.5
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"High missingness ({stats.get('null_pct', 0):.1%}): dropping column.",
        priority=95
    ))

    # Rule 2: Moderate missingness with pattern
    rules.append(Rule(
        name="MISSING_PATTERN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MODE,
        condition=lambda stats: (
            0.1 < stats.get("null_pct", 0) <= 0.5 and
            stats.get("null_has_pattern", False)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: f"Patterned missingness ({stats.get('null_pct', 0):.1%}): mode imputation.",
        priority=88
    ))

    # Rule 3: Low missingness numeric (mean imputation)
    rules.append(Rule(
        name="MISSING_LOW_NUMERIC",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEAN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0 < stats.get("null_pct", 0) <= 0.1 and
            abs(stats.get("skewness", 0)) < 1.0
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: f"Low missingness ({stats.get('null_pct', 0):.1%}): mean imputation for symmetric distribution.",
        priority=86
    ))

    # Rule 4: Low missingness skewed (median imputation)
    rules.append(Rule(
        name="MISSING_LOW_SKEWED",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEDIAN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0 < stats.get("null_pct", 0) <= 0.1 and
            abs(stats.get("skewness", 0)) >= 1.0
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: f"Low missingness ({stats.get('null_pct', 0):.1%}): median imputation for skewed distribution.",
        priority=87
    ))

    # Rule 5: Low missingness categorical (mode)
    rules.append(Rule(
        name="MISSING_LOW_CATEGORICAL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            0 < stats.get("null_pct", 0) <= 0.1
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"Low missingness ({stats.get('null_pct', 0):.1%}): mode imputation for categorical.",
        priority=88
    ))

    # Rule 6: Missingness indicator feature
    rules.append(Rule(
        name="MISSING_INDICATOR",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            0.1 < stats.get("null_pct", 0) <= 0.3 and
            stats.get("target_available", False)
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Moderate missingness with target: consider missingness indicator feature.",
        priority=83
    ))

    return rules


def create_multicollinearity_rules() -> List[Rule]:
    """Create multi-collinearity detection rules (4 rules)."""
    rules = []

    # Rule 1: Perfect correlation (duplicate)
    rules.append(Rule(
        name="MULTICOL_PERFECT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("duplicate_ratio", 0) > 0.99
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: "Perfect correlation detected: dropping duplicate column.",
        priority=97
    ))

    # Rule 2: Very high correlation (>0.95)
    rules.append(Rule(
        name="MULTICOL_VERY_HIGH",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("duplicate_ratio", 0) > 0.95
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Very high correlation: consider PCA or feature selection.",
        priority=90
    ))

    # Rule 3: High correlation with low variance
    rules.append(Rule(
        name="MULTICOL_LOW_VARIANCE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("cv", 1.0) < 0.01 and
            stats.get("unique_ratio", 0) < 0.1
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Low variance with high correlation: dropping redundant feature.",
        priority=92
    ))

    # Rule 4: Constant after groupby (leakage)
    rules.append(Rule(
        name="MULTICOL_CONSTANT_GROUPBY",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 100) < 5 and
            stats.get("duplicate_ratio", 0) > 0.8
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "Near-constant within groups: potential leakage or redundancy.",
        priority=91
    ))

    return rules


def create_statistical_anomaly_rules() -> List[Rule]:
    """Create statistical anomaly detection rules (10 rules)."""
    rules = []

    # Rule 1: Extreme outliers (>10% beyond 3 IQR)
    rules.append(Rule(
        name="STAT_EXTREME_OUTLIERS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.CLIP_OUTLIERS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("outlier_pct", 0) > 0.1 and
            stats.get("has_outliers", False)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: f"Extreme outliers ({stats.get('outlier_pct', 0):.1%}): clipping recommended.",
        priority=81
    ))

    # Rule 2: Heavy tails (kurtosis >10)
    rules.append(Rule(
        name="STAT_HEAVY_TAILS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.YEO_JOHNSON,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("kurtosis", 0) > 10.0
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Heavy tails (kurtosis {stats.get('kurtosis', 0):.1f}): Yeo-Johnson transform.",
        priority=83
    ))

    # Rule 3: Extreme skew (>5)
    rules.append(Rule(
        name="STAT_EXTREME_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) > 5.0 and
            stats.get("all_positive", False)
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: f"Extreme skew ({stats.get('skewness', 0):.1f}): log transform.",
        priority=85
    ))

    # Rule 4: Bimodal with clear separation
    rules.append(Rule(
        name="STAT_BIMODAL_CLEAR",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.BINNING_EQUAL_FREQ,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_bimodal", False) and
            stats.get("unique_count", 0) > 50
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "Clear bimodal distribution: binning to separate modes.",
        priority=84
    ))

    # Rule 5: Zero-inflated (>30% zeros)
    rules.append(Rule(
        name="STAT_ZERO_INFLATED",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.YEO_JOHNSON,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_zeros", False) and
            stats.get("null_pct", 0) < 0.1
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "Zero-inflated distribution: Yeo-Johnson handles zeros.",
        priority=82
    ))

    # Rule 6: Uniform distribution (low variance)
    rules.append(Rule(
        name="STAT_UNIFORM",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) < 0.1 and
            abs(stats.get("kurtosis", 0)) < 0.1
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Uniform distribution: no transformation needed.",
        priority=80
    ))

    # Rule 7: High coefficient of variation (>2)
    rules.append(Rule(
        name="STAT_HIGH_CV",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("cv", 0) > 2.0
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: f"High variability (CV={stats.get('cv', 0):.2f}): robust scaling.",
        priority=81
    ))

    # Rule 8: Low entropy categorical
    rules.append(Rule(
        name="STAT_LOW_ENTROPY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.LABEL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 1.0) < 0.4 and
            stats.get("cardinality", 0) > 2
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Low entropy: dominant category, label encoding sufficient.",
        priority=79
    ))

    # Rule 9: High entropy categorical
    rules.append(Rule(
        name="STAT_HIGH_ENTROPY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 0) > 0.9 and
            stats.get("cardinality", 0) > 50
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "High entropy: many balanced categories, hash encoding.",
        priority=80
    ))

    # Rule 10: Smooth numeric (potential continuous)
    rules.append(Rule(
        name="STAT_SMOOTH_CONTINUOUS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_smooth", False) and
            stats.get("unique_ratio", 0) > 0.8
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Smooth continuous distribution: standard scaling.",
        priority=83
    ))

    return rules
def create_feature_selection_rules() -> List[Rule]:
    """Create automated feature selection recommendation rules (10 rules)."""
    rules = []

    # Rule 1: Too many features (curse of dimensionality)
    rules.append(Rule(
        name="FEATSEL_TOO_MANY_FEATURES",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("row_count", 1000) < 100 and
            stats.get("is_numeric", False)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "High feature-to-sample ratio: consider feature selection (RFE, LASSO).",
        priority=89
    ))

    # Rule 2: Low variance feature
    rules.append(Rule(
        name="FEATSEL_LOW_VARIANCE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("std", 1.0) < 0.001 and
            stats.get("unique_count", 10) < 3
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: f"Very low variance (std={stats.get('std', 0):.4f}): dropping uninformative feature.",
        priority=94
    ))

    # Rule 3: Zero correlation with target
    rules.append(Rule(
        name="FEATSEL_ZERO_CORRELATION",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("target_available", False) and
            abs(stats.get("correlation_with_target", 0)) < 0.01
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Zero correlation with target: consider removing if model performance doesn't improve.",
        priority=78
    ))

    # Rule 4: Redundant feature (high correlation with others)
    rules.append(Rule(
        name="FEATSEL_REDUNDANT",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("duplicate_ratio", 0) > 0.9
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "Highly correlated with other features: consider feature selection.",
        priority=85
    ))

    # Rule 5: Sparse feature (>90% zeros)
    rules.append(Rule(
        name="FEATSEL_SPARSE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_zeros", False) and
            stats.get("null_pct", 0) < 0.1
        ),
        confidence_fn=lambda stats: 0.83,
        explanation_fn=lambda stats: "Sparse feature: consider sparse encoding or removal.",
        priority=80
    ))

    # Rule 6: High cardinality with low frequency
    rules.append(Rule(
        name="FEATSEL_HIGH_CARD_LOW_FREQ",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 100 and
            stats.get("entropy", 1.0) > 0.8
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"High cardinality ({stats.get('cardinality', 0)}) with balanced distribution: hash encoding or feature selection.",
        priority=83
    ))

    # Rule 7: Constant within groups (no predictive power)
    rules.append(Rule(
        name="FEATSEL_CONSTANT_GROUPS",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 100) < 3 and
            stats.get("row_count", 100) > 100
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Constant within groups: no predictive power, dropping.",
        priority=91
    ))

    # Rule 8: All missing in subset
    rules.append(Rule(
        name="FEATSEL_SUBSET_MISSING",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("null_pct", 0) > 0.8
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: f"Mostly missing ({stats.get('null_pct', 0):.1%}): dropping feature.",
        priority=95
    ))

    # Rule 9: Single unique value after encoding
    rules.append(Rule(
        name="FEATSEL_SINGLE_VALUE_ENCODED",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 2) == 1 and
            stats.get("null_pct", 0) == 0
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Single unique value: zero information, dropping.",
        priority=99
    ))

    # Rule 10: Feature with only nulls and one value
    rules.append(Rule(
        name="FEATSEL_NULL_AND_ONE",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_count", 3) <= 2 and
            stats.get("null_pct", 0) > 0.3
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Only nulls and one value: minimal information, dropping.",
        priority=93
    ))

    return rules


def create_dimensionality_reduction_rules() -> List[Rule]:
    """Create dimensionality reduction recommendation rules (10 rules)."""
    rules = []

    # Rule 1: Many correlated features (PCA candidate)
    rules.append(Rule(
        name="DIMRED_MANY_CORRELATED",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("duplicate_ratio", 0) > 0.7
        ),
        confidence_fn=lambda stats: 0.82,
        explanation_fn=lambda stats: "Many correlated features detected: consider PCA for dimensionality reduction.",
        priority=77
    ))

    # Rule 2: Wide dataset (more features than samples)
    rules.append(Rule(
        name="DIMRED_WIDE_DATASET",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("row_count", 1000) < 50
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Wide dataset (p > n): PCA or feature selection recommended.",
        priority=84
    ))

    # Rule 3: High-dimensional categorical (embedding candidate)
    rules.append(Rule(
        name="DIMRED_HIGH_DIM_CAT",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 500
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: f"Very high cardinality ({stats.get('cardinality', 0)}): consider entity embeddings or hashing.",
        priority=86
    ))

    # Rule 4: Text with high vocabulary (dimensionality reduction needed)
    rules.append(Rule(
        name="DIMRED_TEXT_HIGH_VOCAB",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.TEXT_VECTORIZE_TFIDF,
        condition=lambda stats: (
            stats.get("is_text", False) and
            stats.get("unique_ratio", 0) > 0.8 and
            stats.get("avg_length", 0) > 50
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "High-dimensional text: TF-IDF with max_features or LSA recommended.",
        priority=82
    ))

    # Rule 5: Many sparse features
    rules.append(Rule(
        name="DIMRED_SPARSE_FEATURES",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_zeros", False) and
            stats.get("unique_ratio", 0) < 0.3
        ),
        confidence_fn=lambda stats: 0.81,
        explanation_fn=lambda stats: "Sparse features: consider truncated SVD or feature hashing.",
        priority=76
    ))

    # Rule 6: Multicollinear features (VIF > 10)
    rules.append(Rule(
        name="DIMRED_MULTICOLLINEAR",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("duplicate_ratio", 0) > 0.85
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "High multicollinearity: PCA or ridge regression recommended.",
        priority=81
    ))

    # Rule 7: Image-like data (flatten candidate)
    rules.append(Rule(
        name="DIMRED_IMAGE_LIKE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", -1) >= 0 and
            stats.get("max_value", 300) <= 255 and
            stats.get("unique_count", 0) > 50
        ),
        confidence_fn=lambda stats: 0.78,
        explanation_fn=lambda stats: "Pixel-like values detected: consider CNN or PCA for images.",
        priority=75
    ))

    # Rule 8: Time series with many lags (dimensionality reduction)
    rules.append(Rule(
        name="DIMRED_MANY_LAGS",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_smooth", False) and
            any(keyword in stats.get("column_name", "").lower() 
                for keyword in ["lag", "rolling", "window"])
        ),
        confidence_fn=lambda stats: 0.80,
        explanation_fn=lambda stats: "Many lag features: consider PCA or autoencoder for compression.",
        priority=77
    ))

    # Rule 9: One-hot encoded with many categories
    rules.append(Rule(
        name="DIMRED_ONEHOT_EXPLOSION",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 50 and
            stats.get("cardinality", 0) < 500
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: f"Medium-high cardinality ({stats.get('cardinality', 0)}): hash encoding to avoid one-hot explosion.",
        priority=83
    ))

    # Rule 10: Normalized features (ready for PCA)
    rules.append(Rule(
        name="DIMRED_READY_FOR_PCA",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_already_scaled", False) and
            abs(stats.get("mean", 100)) < 1.0 and
            abs(stats.get("std", 0) - 1.0) < 0.2
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Already normalized: ready for PCA if needed.",
        priority=79
    ))

    return rules
