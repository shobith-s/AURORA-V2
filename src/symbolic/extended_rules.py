"""
Extended Rule Definitions for Universal Coverage.
These rules complement the base rules.py to achieve 95%+ coverage.

Added for universality:
- Advanced data type detection (UUID, IP, coordinates, hashes, etc.)
- Domain-specific patterns (financial, medical, web analytics, IoT)
- Composite rules (multi-condition logic)
- Distribution-based rules (bimodal, multimodal, etc.)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import re
import numpy as np

from .rules import Rule, RuleCategory
from ..core.actions import PreprocessingAction


# =============================================================================
# ADVANCED TYPE DETECTION RULES (20 rules)
# =============================================================================

def create_advanced_type_detection_rules() -> List[Rule]:
    """Create advanced data type detection rules for universal coverage."""
    rules = []

    # Rule 1: UUID/GUID detection
    rules.append(Rule(
        name="DETECT_UUID_DROP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("unique_ratio", 0) > 0.95 and
            _check_uuid_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Column contains UUIDs (unique identifiers with no predictive value)",
        priority=93
    ))

    # Rule 2: IP Address detection
    rules.append(Rule(
        name="DETECT_IP_ADDRESS",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.HASH_ENCODE,  # Preserve privacy
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_ip_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Column contains IP addresses: hash encoding for privacy",
        priority=91
    ))

    # Rule 3: Geographic coordinates (latitude)
    rules.append(Rule(
        name="DETECT_LATITUDE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", -1000) >= -90 and
            stats.get("max_value", 1000) <= 90 and
            (_column_name_contains(stats, ["lat", "latitude"]) or
             _is_likely_coordinate(stats, -90, 90))
        ),
        confidence_fn=lambda stats: 0.94 if _column_name_contains(stats, ["lat", "latitude"]) else 0.82,
        explanation_fn=lambda stats: "Latitude coordinate: keeping as-is (already in valid range)",
        priority=92
    ))

    # Rule 4: Geographic coordinates (longitude)
    rules.append(Rule(
        name="DETECT_LONGITUDE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", -1000) >= -180 and
            stats.get("max_value", 1000) <= 180 and
            (_column_name_contains(stats, ["lon", "lng", "longitude"]) or
             _is_likely_coordinate(stats, -180, 180))
        ),
        confidence_fn=lambda stats: 0.94 if _column_name_contains(stats, ["lon", "lng", "longitude"]) else 0.82,
        explanation_fn=lambda stats: "Longitude coordinate: keeping as-is (already in valid range)",
        priority=92
    ))

    # Rule 5: Hash values (MD5, SHA)
    rules.append(Rule(
        name="DETECT_HASH_DROP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("unique_ratio", 0) > 0.95 and
            _check_hash_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Column contains hash values (no predictive value)",
        priority=91
    ))

    # Rule 6: Hexadecimal values
    rules.append(Rule(
        name="DETECT_HEXADECIMAL",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_NUMERIC,  # Convert to numeric
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_hex_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "Hexadecimal values: converting to numeric",
        priority=87
    ))

    # Rule 7: Scientific notation
    rules.append(Rule(
        name="DETECT_SCIENTIFIC_NOTATION",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_NUMERIC,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_scientific_notation(stats)
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: "Scientific notation detected: converting to numeric",
        priority=94
    ))

    # Rule 8: Epoch timestamps
    rules.append(Rule(
        name="DETECT_EPOCH_TIMESTAMP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_DATETIME,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            _is_epoch_timestamp(stats)
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "Unix epoch timestamp detected: converting to datetime",
        priority=89
    ))

    # Rule 9: Base64 encoded data
    rules.append(Rule(
        name="DETECT_BASE64_DROP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_base64_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Base64 encoded data: typically not useful for ML",
        priority=85
    ))

    # Rule 10: File paths
    rules.append(Rule(
        name="DETECT_FILE_PATH_DROP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_file_path_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "File paths detected: typically not predictive",
        priority=86
    ))

    # Rule 11: Semantic versioning (1.2.3)
    rules.append(Rule(
        name="DETECT_SEMANTIC_VERSION",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.ORDINAL_ENCODE,  # Versions have order
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_semver_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Semantic version numbers: ordinal encoding preserves order",
        priority=88
    ))

    # Rule 12: ISO country codes
    rules.append(Rule(
        name="DETECT_COUNTRY_CODE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) <= 250 and  # ~195 countries
            _check_country_code_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "ISO country codes detected: one-hot encoding",
        priority=90
    ))

    # Rule 13: ISO currency codes
    rules.append(Rule(
        name="DETECT_CURRENCY_CODE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) <= 200 and  # ~180 currencies
            _check_currency_code_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "ISO currency codes detected: one-hot encoding",
        priority=89
    ))

    # Rule 14: MAC addresses
    rules.append(Rule(
        name="DETECT_MAC_ADDRESS_HASH",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_mac_address_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "MAC addresses detected: hash encoding for categorical representation",
        priority=92
    ))

    # Rule 15: Color codes (hex/rgb)
    rules.append(Rule(
        name="DETECT_COLOR_CODE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_color_code_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Color codes detected: hash encoding",
        priority=84
    ))

    # Rule 16: Credit card numbers (be careful!)
    rules.append(Rule(
        name="DETECT_CREDIT_CARD_DROP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            (stats.get("dtype", "") in ["object", "string"] or stats.get("is_numeric", False)) and
            _check_credit_card_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.98,  # High confidence - security risk
        explanation_fn=lambda stats: "Credit card numbers detected: DROPPING for security (PCI compliance)",
        priority=98
    ))

    # Rule 17: Phone numbers (international format)
    # (Already in base rules, but we add more sophisticated detection)
    rules.append(Rule(
        name="DETECT_PHONE_HASH",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("matches_phone_pattern", 0) > 0.7 and
            _column_name_contains(stats, ["phone", "mobile", "tel", "cell"])
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Phone numbers detected: hash encoding for privacy",
        priority=90
    ))

    # Rule 18: Error/status codes
    rules.append(Rule(
        name="DETECT_ERROR_CODE_CATEGORICAL",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.ONEHOT_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) < 100 and
            _column_name_contains(stats, ["error", "status", "code", "return_code", "http_status"])
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: "Error/status codes detected: one-hot encoding",
        priority=86
    ))

    # Rule 19: Bitmap/bitflag values
    rules.append(Rule(
        name="DETECT_BITMAP",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.KEEP_AS_IS,  # Binary representation is often useful
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            _is_likely_bitmap(stats)
        ),
        confidence_fn=lambda stats: 0.81,
        explanation_fn=lambda stats: "Bitmap/bitflag detected: keeping as-is (binary encoding useful)",
        priority=81
    ))

    # Rule 20: ICD-10 medical codes
    rules.append(Rule(
        name="DETECT_ICD10_CODE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            _check_icd10_pattern(stats)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "ICD-10 medical codes detected: hash encoding",
        priority=87
    ))

    return rules


# =============================================================================
# DOMAIN-SPECIFIC PATTERN RULES (25 rules)
# =============================================================================

def create_domain_pattern_rules() -> List[Rule]:
    """Create domain-specific pattern rules."""
    rules = []

    # Business Metrics Rules (5 rules)

    # Rule 1: Rate/Ratio/Percentage columns (already normalized 0-100)
    rules.append(Rule(
        name="DOMAIN_RATE_RATIO_KEEP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("max_value") is not None and
            0 <= stats.get("min_value") <= stats.get("max_value") <= 100 and
            _column_name_contains(stats, ["rate", "ratio", "percentage", "pct", "percent", "_pct", "bounce_rate", "conversion"])
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Rate/ratio/percentage: already normalized (0-100 range)",
        priority=91
    ))

    # Rule 2: Count/Frequency columns (non-negative integers)
    rules.append(Rule(
        name="DOMAIN_COUNT_LOG_TRANSFORM",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("min_value") >= 0 and
            stats.get("skewness", 0) > 1.5 and
            _column_name_contains(stats, ["count", "frequency", "num_", "total_", "_count", "quantity"])
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: f"Count/frequency column with high skewness ({stats.get('skewness', 0):.2f}): log1p transform",
        priority=89
    ))

    # Rule 3: Revenue/Amount columns (financial)
    rules.append(Rule(
        name="DOMAIN_REVENUE_ROBUST_SCALE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("cv", 0) > 1.0 and  # High variability
            _column_name_contains(stats, ["revenue", "amount", "sales", "price", "cost", "value", "spend"])
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Financial amount with high variability (CV={stats.get('cv', 0):.2f}): robust scaling",
        priority=88
    ))

    # Rule 4: Duration columns (time-based)
    rules.append(Rule(
        name="DOMAIN_DURATION_LOG_TRANSFORM",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("min_value") >= 0 and
            stats.get("skewness", 0) > 2.0 and
            _column_name_contains(stats, ["duration", "time", "_ms", "_seconds", "latency", "elapsed", "session_time"])
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Duration/time metric with right skew: log1p transform",
        priority=90
    ))

    # Rule 5: Score columns (already bounded)
    rules.append(Rule(
        name="DOMAIN_SCORE_KEEP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) < 0.8 and  # Fairly symmetric
            stats.get("has_outliers", True) == False and
            _column_name_contains(stats, ["score", "rating", "grade", "rank"])
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "Score/rating with good distribution: keeping as-is",
        priority=87
    ))

    # Identifier Rules (5 rules)

    # Rule 6: ID columns with specific naming
    rules.append(Rule(
        name="DOMAIN_ID_COLUMN_DROP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_ratio", 0) > 0.95 and
            _column_name_matches_id(stats)
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: f"ID column with {stats.get('unique_ratio', 0):.1%} unique values: no predictive value",
        priority=94
    ))

    # Rule 7: Key columns (foreign keys with moderate cardinality)
    rules.append(Rule(
        name="DOMAIN_KEY_HASH_ENCODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("cardinality", 0) > 50 and
            stats.get("unique_ratio", 0) < 0.95 and
            _column_name_contains(stats, ["_key", "fk_", "foreign_key", "ref_id"])
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: f"Foreign key with {stats.get('cardinality', 0)} categories: hash encoding",
        priority=86
    ))

    # Web Analytics Rules (5 rules)

    # Rule 8: UTM parameters
    rules.append(Rule(
        name="DOMAIN_UTM_HASH_ENCODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            _column_name_contains(stats, ["utm_", "campaign", "source", "medium", "referrer"])
        ),
        confidence_fn=lambda stats: 0.89,
        explanation_fn=lambda stats: "UTM/marketing parameter: hash encoding",
        priority=89
    ))

    # Rule 9: User agent strings
    rules.append(Rule(
        name="DOMAIN_USER_AGENT_DROP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string"] and
            stats.get("unique_ratio", 0) > 0.5 and
            _column_name_contains(stats, ["user_agent", "useragent", "browser_string"])
        ),
        confidence_fn=lambda stats: 0.83,
        explanation_fn=lambda stats: "User agent string: high cardinality, consider feature engineering instead",
        priority=83
    ))

    # Rule 10: Session IDs
    rules.append(Rule(
        name="DOMAIN_SESSION_ID_DROP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_ratio", 0) > 0.90 and
            _column_name_contains(stats, ["session_id", "sessionid", "visit_id"])
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Session ID: high uniqueness, no predictive value",
        priority=92
    ))

    # IoT/Sensor Rules (3 rules)

    # Rule 11: Temperature readings
    rules.append(Rule(
        name="DOMAIN_TEMPERATURE_KEEP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("max_value") is not None and
            -100 < stats.get("min_value") < stats.get("max_value") < 200 and
            _column_name_contains(stats, ["temp", "temperature", "_celsius", "_fahrenheit", "_kelvin"])
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Temperature reading in valid range: keeping as-is",
        priority=85
    ))

    # Rule 12: Sensor readings with high frequency
    rules.append(Rule(
        name="DOMAIN_SENSOR_SMOOTH",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_outliers", False) and  # Sensor noise/spikes
            _column_name_contains(stats, ["sensor", "reading", "measurement", "vibration", "pressure", "voltage"])
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Sensor reading with outliers: robust scaling handles noise",
        priority=84
    ))

    # Medical/Healthcare Rules (3 rules)

    # Rule 13: Medical measurements (blood pressure, glucose, etc.)
    rules.append(Rule(
        name="DOMAIN_MEDICAL_MEASUREMENT_KEEP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("has_natural_bounds", False) and
            _column_name_contains(stats, ["blood_pressure", "glucose", "heart_rate", "bmi", "cholesterol", "hba1c"])
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Medical measurement in clinical range: keeping as-is",
        priority=88
    ))

    # Rule 14: Patient age
    rules.append(Rule(
        name="DOMAIN_AGE_KEEP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("max_value") is not None and
            0 <= stats.get("min_value") <= stats.get("max_value") <= 120 and
            abs(stats.get("skewness", 0)) < 2.0 and
            _column_name_contains(stats, ["age", "patient_age", "age_years"])
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Age in valid human range: keeping as-is",
        priority=91
    ))

    # Encoded Missing Values (3 rules)

    # Rule 15: Detect -999, 999, 9999 as coded nulls
    rules.append(Rule(
        name="DOMAIN_CODED_NULL_REPLACE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.FILL_NULL_MEDIAN,  # Replace then fill
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            _has_coded_missing_values(stats)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "Detected coded missing values (-999, 999, etc.): replacing with null then median",
        priority=87
    ))

    # Text/NLP columns (3 rules)

    # Rule 16: Free text columns (descriptions, comments)
    rules.append(Rule(
        name="DOMAIN_FREE_TEXT_DROP",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("is_text", False) and
            stats.get("unique_ratio", 0) > 0.7 and  # Mostly unique
            _column_name_contains(stats, ["description", "comment", "notes", "remarks", "text", "content", "message"])
        ),
        confidence_fn=lambda stats: 0.80,
        explanation_fn=lambda stats: "Free text field: consider NLP feature engineering instead",
        priority=80
    ))

    # Rule 17: Category names with natural ordering
    rules.append(Rule(
        name="DOMAIN_SIZE_ORDINAL",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            _has_size_ordering(stats)
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Size category with natural order: ordinal encoding",
        priority=93
    ))

    # Rule 18: Priority/Severity levels
    rules.append(Rule(
        name="DOMAIN_PRIORITY_ORDINAL",
        category=RuleCategory.DOMAIN_SPECIFIC,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            _has_priority_ordering(stats)
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Priority/severity level with natural order: ordinal encoding",
        priority=92
    ))

    return rules


# =============================================================================
# COMPOSITE RULES (multi-condition edge cases) (20 rules)
# =============================================================================

def create_composite_rules() -> List[Rule]:
    """Create composite rules for complex edge cases."""
    rules = []

    # =============================================================================
    # CLASS IMBALANCE HANDLING RULES (HIGH PRIORITY - prevent dropping)
    # =============================================================================
    
    # Rule 0a: NEVER drop target columns (highest priority)
    rules.append(Rule(
        name="PRESERVE_TARGET_COLUMN",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            _column_name_contains(stats, ["target", "label", "class", "y", "outcome", "response"]) and
            (stats.get("entropy", 1.0) < 0.5 or stats.get("unique_ratio", 1.0) < 0.1)
        ),
        confidence_fn=lambda stats: 0.99,
        explanation_fn=lambda stats: f"Target column with class imbalance: KEEPING (imbalance = {1 - stats.get('entropy', 0):.1%}). Handle with SMOTE/class weights during training.",
        priority=98  # Highest priority to override DROP rules
    ))
    
    # Rule 0b: Keep imbalanced binary indicators (flags, is_, has_)
    rules.append(Rule(
        name="PRESERVE_IMBALANCED_BINARY_FLAGS",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("unique_count", 100) <= 3 and
            stats.get("entropy", 1.0) < 0.3 and  # Very imbalanced
            _column_name_contains(stats, ["flag", "indicator", "is_", "has_", "bool", "binary"])
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: f"Imbalanced binary flag (entropy={stats.get('entropy', 0):.2f}): KEEPING as-is. Rare events may be important.",
        priority=97
    ))
    
    # Rule 0c: Handle imbalanced categorical features (use frequency encoding)
    rules.append(Rule(
        name="HANDLE_IMBALANCED_CATEGORICAL",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 1.0) < 0.3 and  # Imbalanced
            stats.get("unique_ratio", 1.0) < 0.05 and
            stats.get("cardinality", 0) >= 5 and  # At least 5 categories
            not _column_name_contains(stats, ["target", "label", "class", "y"])
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Imbalanced categorical (entropy={stats.get('entropy', 0):.2f}): using frequency encoding to preserve rare categories",
        priority=96
    ))
    
    # Rule 0d: Handle imbalanced numeric features (keep and note for SMOTE)
    rules.append(Rule(
        name="PRESERVE_IMBALANCED_NUMERIC",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("entropy", 1.0) < 0.25 and  # Very imbalanced
            stats.get("unique_ratio", 1.0) < 0.05 and
            stats.get("unique_count", 100) >= 5 and  # At least 5 unique values
            not _column_name_contains(stats, ["target", "label", "class"])
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: f"Imbalanced numeric feature (entropy={stats.get('entropy', 0):.2f}): KEEPING. Consider SMOTE for minority class augmentation.",
        priority=96
    ))

    # Rule 1: Bimodal numeric (likely mixed data types)
    rules.append(Rule(
        name="COMPOSITE_BIMODAL_MIXED_TYPE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.PARSE_CATEGORICAL,  # Treat as categorical
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("kurtosis", 0) < -1.2 and  # Strong bimodal
            stats.get("cardinality", 1000) < 20 and  # Low unique values
            stats.get("unique_ratio", 1.0) < 0.1
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: "Bimodal numeric with low cardinality: likely mis-coded categorical",
        priority=84
    ))

    # Rule 2: Near-constant with rare events (99% same value)
    rules.append(Rule(
        name="COMPOSITE_NEAR_CONSTANT_RARE_EVENTS",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("unique_ratio", 1.0) < 0.02 and
            stats.get("row_count", 0) > 100 and
            not _column_name_contains(stats, ["flag", "indicator", "is_", "has_"])  # Keep boolean flags
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: f"Near-constant column ({stats.get('unique_ratio', 0):.1%} unique): low information",
        priority=86
    ))

    # Rule 3: High CV with small range (measurement noise)
    rules.append(Rule(
        name="COMPOSITE_HIGH_CV_SMALL_RANGE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.STANDARD_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("cv", 0) > 2.0 and
            stats.get("range_size", 1000) < 10 and
            stats.get("mean", 0) != 0
        ),
        confidence_fn=lambda stats: 0.79,
        explanation_fn=lambda stats: "High CV with small range: likely measurement noise, standard scaling",
        priority=79
    ))

    # Rule 4: All positive with zeros and high skew (add-one log transform)
    rules.append(Rule(
        name="COMPOSITE_POSITIVE_ZEROS_SKEWED",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("min_value") >= 0 and
            stats.get("has_zeros", False) and
            stats.get("skewness", 0) > 2.5 and
            not stats.get("is_already_scaled", False)
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: f"Positive with zeros and high skew ({stats.get('skewness', 0):.2f}): log1p optimal",
        priority=91
    ))

    # Rule 5: Low entropy (little information content)
    rules.append(Rule(
        name="COMPOSITE_LOW_ENTROPY_DROP",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("entropy", 1.0) < 0.2 and  # Very low information
            stats.get("unique_ratio", 1.0) < 0.05 and
            stats.get("row_count", 0) > 50
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: f"Low entropy ({stats.get('entropy', 0):.2f}): little information content",
        priority=85
    ))

    # Rule 6: Numeric but only 2-3 unique values (binary/ordinal)
    rules.append(Rule(
        name="COMPOSITE_NUMERIC_FEW_UNIQUE",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            2 <= stats.get("unique_count", 100) <= 3 and
            stats.get("row_count", 0) > 50
        ),
        confidence_fn=lambda stats: 0.83,
        explanation_fn=lambda stats: f"Numeric with only {stats.get('unique_count', 0)} unique values: ordinal encoding",
        priority=83
    ))

    # Rule 7: Extreme outliers (>10% outliers by IQR method)
    rules.append(Rule(
        name="COMPOSITE_EXTREME_OUTLIERS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.WINSORIZE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("outlier_pct", 0) > 0.15 and  # >15% outliers
            stats.get("outlier_pct", 0) < 0.4  # But not too many (would be bad data)
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: f"Extreme outliers ({stats.get('outlier_pct', 0):.1%}): winsorizing",
        priority=87
    ))

    # Rule 8: Symmetric distribution with outliers (use robust scaling)
    rules.append(Rule(
        name="COMPOSITE_SYMMETRIC_WITH_OUTLIERS",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.ROBUST_SCALE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            abs(stats.get("skewness", 0)) < 0.5 and  # Symmetric
            stats.get("has_outliers", False) and  # But has outliers
            stats.get("outlier_pct", 0) > 0.05
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Symmetric distribution with outliers: robust scaling preserves shape",
        priority=90
    ))

    # Rule 9: Categorical high cardinality without target (use frequency)
    rules.append(Rule(
        name="COMPOSITE_HIGH_CARD_NO_TARGET",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            50 < stats.get("cardinality", 0) <= 1000 and  # Increased from 500 to 1000
            not stats.get("target_available", False)
        ),
        confidence_fn=lambda stats: 0.84,
        explanation_fn=lambda stats: f"High cardinality ({stats.get('cardinality', 0)}) without target: frequency encoding",
        priority=84
    ))

    # Rule 10: Very high cardinality (>1000 categories)
    rules.append(Rule(
        name="COMPOSITE_VERY_HIGH_CARDINALITY",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.HASH_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("cardinality", 0) > 1000 and  # Increased from 500 to 1000
            stats.get("unique_ratio", 1.0) < 0.95  # Not an ID
        ),
        confidence_fn=lambda stats: 0.86,
        explanation_fn=lambda stats: f"Very high cardinality ({stats.get('cardinality', 0)}): hash encoding",
        priority=86
    ))

    # Rule 11: Mixed positive/negative with high skew (Yeo-Johnson)
    rules.append(Rule(
        name="COMPOSITE_MIXED_SIGN_SKEWED",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.YEO_JOHNSON,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            not stats.get("all_positive", True) and
            abs(stats.get("skewness", 0)) > 1.8 and
            stats.get("min_value") is not None and stats.get("min_value") < 0
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Mixed sign with high skew: Yeo-Johnson handles negative values",
        priority=88
    ))

    # Rule 12: Bounded 0-1 range (already normalized probability)
    rules.append(Rule(
        name="COMPOSITE_PROBABILITY_RANGE",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("max_value") is not None and
            0 <= stats.get("min_value") <= stats.get("max_value") <= 1.0 and
            stats.get("max_value", 0) - stats.get("min_value", 0) > 0.5  # Uses significant part of range
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: "Values in [0,1] range: likely probability/normalized, keeping as-is",
        priority=90
    ))

    # Rule 13: Integer range -1 to 1 (likely standardized ordinal)
    rules.append(Rule(
        name="COMPOSITE_STANDARDIZED_ORDINAL",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("max_value") is not None and
            -1 <= stats.get("min_value") <= stats.get("max_value") <= 1 and
            stats.get("unique_count", 100) <= 5
        ),
        confidence_fn=lambda stats: 0.87,
        explanation_fn=lambda stats: "Values in {-1, 0, 1} range: already standardized ordinal",
        priority=87
    ))

    # Rule 14: Null% between 30-40% (borderline case)
    rules.append(Rule(
        name="COMPOSITE_BORDERLINE_NULL",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.FILL_NULL_MEDIAN,  # Conservative - fill instead of drop
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            0.30 < stats.get("null_pct", 0) <= 0.40 and
            stats.get("null_has_pattern", False)  # Only if no pattern (else might be informative)
        ),
        confidence_fn=lambda stats: 0.72,  # Lower confidence - borderline
        explanation_fn=lambda stats: f"Borderline null% ({stats.get('null_pct', 0):.1%}): filling with median",
        priority=72
    ))

    # Rule 15: Categorical with imbalanced distribution (>80% one class)
    rules.append(Rule(
        name="COMPOSITE_IMBALANCED_CATEGORICAL",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.FREQUENCY_ENCODE,  # Preserve frequency information
        condition=lambda stats: (
            stats.get("is_categorical", False) and
            stats.get("entropy", 1.0) < 0.5 and  # Low entropy = imbalanced
            5 < stats.get("cardinality", 0) < 50
        ),
        confidence_fn=lambda stats: 0.81,
        explanation_fn=lambda stats: "Imbalanced categorical: frequency encoding preserves class distribution",
        priority=81
    ))

    # Rule 16: Already scaled numeric (-3 to 3 range, mean~0)
    rules.append(Rule(
        name="COMPOSITE_ALREADY_Z_SCORED",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and -3 <= stats.get("min_value") and
            stats.get("max_value") is not None and stats.get("max_value") <= 3 and
            abs(stats.get("mean", 100)) < 0.3 and
            0.7 <= stats.get("std", 0) <= 1.3
        ),
        confidence_fn=lambda stats: 0.94,
        explanation_fn=lambda stats: "Already Z-scored (mean~0, std~1): keeping as-is",
        priority=94
    ))

    # Rule 17: Sparse binary (mostly 0 with rare 1s)
    rules.append(Rule(
        name="COMPOSITE_SPARSE_BINARY",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.KEEP_AS_IS,  # Binary encoding is already optimal
        condition=lambda stats: (
            stats.get("unique_count", 100) == 2 and
            stats.get("entropy", 1.0) < 0.3 and  # Very imbalanced
            (stats.get("min_value") is not None and stats.get("min_value") >= 0)
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "Sparse binary indicator: keeping as-is (optimal encoding)",
        priority=88
    ))

    # Rule 18: Timestamp in milliseconds (very large numbers)
    rules.append(Rule(
        name="COMPOSITE_TIMESTAMP_MS",
        category=RuleCategory.TYPE_DETECTION,
        action=PreprocessingAction.PARSE_DATETIME,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value") is not None and stats.get("min_value") > 1_000_000_000_000 and  # After year 2001 in ms
            stats.get("max_value") is not None and stats.get("max_value") < 2_000_000_000_000 and  # Before year 2033 in ms
            _column_name_contains(stats, ["time", "timestamp", "created", "updated", "date"])
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Millisecond timestamp detected: converting to datetime",
        priority=91
    ))

    # Rule 19: Negative values only (inverted scale)
    rules.append(Rule(
        name="COMPOSITE_NEGATIVE_ONLY",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.ROBUST_SCALE,  # Handles negative gracefully
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("max_value", 1) < 0 and  # All negative
            stats.get("row_count", 0) > 10
        ),
        confidence_fn=lambda stats: 0.83,
        explanation_fn=lambda stats: "All negative values: robust scaling preserves relative magnitudes",
        priority=83
    ))

    # Rule 20: Perfect multicollinearity indicator (correlation = 1 if target available)
    rules.append(Rule(
        name="COMPOSITE_PERFECT_CORRELATION",
        category=RuleCategory.DATA_QUALITY,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("target_available", False) and
            abs(stats.get("target_correlation", 0)) > 0.99  # Perfect correlation
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: "Perfect correlation with target: likely data leakage, dropping",
        priority=96
    ))

    return rules


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_uuid_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains UUID/GUID patterns."""
    # UUID format: 8-4-4-4-12 hex characters
    # We approximate by checking column name and high unique ratio
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["uuid", "guid", "unique_id", "identifier"])


def _check_ip_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains IP addresses."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["ip", "ip_address", "ipv4", "ipv6", "ip_addr"])


def _column_name_contains(stats: Dict[str, Any], keywords: List[str]) -> bool:
    """Check if column name contains any of the keywords."""
    column_name = str(stats.get("column_name", "")).lower()
    return any(keyword.lower() in column_name for keyword in keywords)


def _column_name_matches_id(stats: Dict[str, Any]) -> bool:
    """Check if column name matches ID patterns."""
    column_name = str(stats.get("column_name", "")).lower()
    id_patterns = [
        r'^id$', r'.*_id$', r'^.*_key$', r'^pk_.*', r'^uuid.*',
        r'^guid.*', r'.*identifier.*', r'^key$'
    ]
    import re
    return any(re.match(pattern, column_name) for pattern in id_patterns)


def _is_likely_coordinate(stats: Dict[str, Any], min_val: float, max_val: float) -> bool:
    """Check if numeric column is likely a coordinate."""
    return (min_val <= stats.get("min_value", -1000) and
            stats.get("max_value", 1000) <= max_val and
            stats.get("range_size", 0) > (max_val - min_val) * 0.1)  # Uses >10% of range


def _check_hash_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains hash values (MD5, SHA)."""
    # Hash columns typically have: all same length, high uniqueness, alphanumeric
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["hash", "md5", "sha", "checksum", "digest"])


def _check_hex_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains hexadecimal values."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["hex", "hexadecimal", "0x"])


def _check_scientific_notation(stats: Dict[str, Any]) -> bool:
    """Check if column uses scientific notation."""
    # Approximate check - real check would need sample values
    return ("scientific" in str(stats.get("column_name", "")).lower() or
            "exp" in str(stats.get("column_name", "")).lower())


def _is_epoch_timestamp(stats: Dict[str, Any]) -> bool:
    """Check if numeric column is Unix epoch timestamp."""
    # Unix timestamp: seconds since 1970-01-01
    # Reasonable range: 1970 to 2050 = 0 to ~2.5 billion
    min_val = stats.get("min_value", 0)
    max_val = stats.get("max_value", 0)
    return (1_000_000_000 < min_val < 2_000_000_000 and  # ~2001 to ~2033
            1_000_000_000 < max_val < 2_000_000_000)


def _check_base64_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains Base64 encoded data."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["base64", "encoded", "b64"])


def _check_file_path_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains file paths."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["path", "file_path", "filepath", "directory", "folder"])


def _check_semver_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains semantic version numbers."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["version", "ver", "release", "build"])


def _check_country_code_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains ISO country codes."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["country", "country_code", "iso_country", "nation"])


def _check_currency_code_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains ISO currency codes."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["currency", "currency_code", "iso_currency"])


def _check_mac_address_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains MAC addresses."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["mac", "mac_address", "hardware_addr"])


def _check_color_code_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains color codes."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["color", "colour", "rgb", "hex_color"])


def _check_credit_card_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column might contain credit card numbers (SECURITY RISK!)."""
    column_name = str(stats.get("column_name", "")).lower()
    # Check column name AND numeric with 13-19 digits (CC length range)
    has_cc_name = any(keyword in column_name for keyword in
                     ["card", "credit_card", "cc_number", "card_num", "pan"])

    # If numeric, check if values are in CC range (13-19 digits = 1e12 to 1e19)
    if stats.get("is_numeric", False):
        min_val = stats.get("min_value", 0)
        max_val = stats.get("max_value", 0)
        in_cc_range = (1_000_000_000_000 <= min_val and max_val < 10_000_000_000_000_000_000)
        return has_cc_name or in_cc_range

    return has_cc_name


def _check_icd10_pattern(stats: Dict[str, Any]) -> bool:
    """Check if column contains ICD-10 medical codes."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["icd", "icd10", "diagnosis_code", "condition_code"])


def _is_likely_bitmap(stats: Dict[str, Any]) -> bool:
    """Check if numeric column is likely a bitmap/bitflag."""
    # Bitmaps are typically: integers, powers of 2 present, limited range
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["flag", "flags", "bitmap", "bit_", "mask"])


def _has_coded_missing_values(stats: Dict[str, Any]) -> bool:
    """Check if column likely has coded missing values like -999, 999, 9999."""
    # This would require inspecting actual values, so we approximate
    min_val = stats.get("min_value", 0)
    max_val = stats.get("max_value", 0)

    # Common coded missing patterns
    has_negative_coded = min_val in [-999, -99, -9, -1]
    has_positive_coded = max_val in [999, 9999, 99999]

    return has_negative_coded or has_positive_coded


def _has_size_ordering(stats: Dict[str, Any]) -> bool:
    """Check if categorical has size ordering (S/M/L, etc.)."""
    # Would need to check actual unique values
    # For now, check column name
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["size", "sizes", "shirt_size", "clothing_size"])


def _has_priority_ordering(stats: Dict[str, Any]) -> bool:
    """Check if categorical has priority/severity ordering."""
    return any(keyword in str(stats.get("column_name", "")).lower()
              for keyword in ["priority", "severity", "urgency", "level", "tier"])


# =============================================================================
# DOMAIN-SPECIFIC PATTERN RULES (26 rules)
# =============================================================================

def create_domain_pattern_rules() -> List[Rule]:
    """Create domain-specific pattern rules for Finance, Healthcare, and E-commerce."""
    rules = []

    # -------------------------------------------------------------------------
    # FINANCE DOMAIN (10 rules)
    # -------------------------------------------------------------------------

    # Rule 1: Currency Detection
    rules.append(Rule(
        name="FINANCE_CURRENCY_DETECTION",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("currency", 0) > 0.8
        ),
        action=PreprocessingAction.TEXT_CLEAN,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Column contains currency values. Clean text to remove symbols and convert to numeric.",
        priority=50,
        parameters={"tags": ["finance", "currency", "cleaning"]}
    ))

    # Rule 2: Account Number
    rules.append(Rule(
        name="FINANCE_ACCOUNT_NUMBER",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            any(k in str(stats.get("column_name", "")).lower() for k in ["account", "acct_no", "iban"]) and
            stats.get("unique_ratio", 0) > 0.8
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Column appears to be an account number. Keep as identifier.",
        priority=45,
        parameters={"tags": ["finance", "pii", "identifier"]}
    ))

    # Rule 3: Stock Ticker
    rules.append(Rule(
        name="FINANCE_STOCK_TICKER",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("stock_ticker", 0) > 0.9 and
            stats.get("avg_length", 0) < 6 and
            any(k in str(stats.get("column_name", "")).lower() for k in ["ticker", "symbol", "stock"])
        ),
        action=PreprocessingAction.LABEL_ENCODE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Column appears to be stock ticker symbols. Treat as categorical.",
        priority=45,
        parameters={"tags": ["finance", "categorical"]}
    ))

    # Rule 4: Credit Card (Masking)
    rules.append(Rule(
        name="FINANCE_CREDIT_CARD",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("credit_card", 0) > 0.5
        ),
        action=PreprocessingAction.DROP_COLUMN,
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Column contains credit card numbers. Drop for privacy/security.",
        priority=90,
        parameters={"tags": ["finance", "pii", "security"]}
    ))

    # Rule 5: Tax ID / SSN
    rules.append(Rule(
        name="FINANCE_TAX_ID",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            (stats.get("domain_pattern_matches", {}).get("ssn", 0) > 0.5 or
             stats.get("domain_pattern_matches", {}).get("ein", 0) > 0.5)
        ),
        action=PreprocessingAction.DROP_COLUMN,
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: "Column contains Tax IDs or SSNs. Drop for privacy.",
        priority=90,
        parameters={"tags": ["finance", "pii", "security"]}
    ))

    # Rule 6: IBAN
    rules.append(Rule(
        name="FINANCE_IBAN",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("iban", 0) > 0.5
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Column contains IBANs. Keep as identifier.",
        priority=45,
        parameters={"tags": ["finance", "identifier"]}
    ))

    # Rule 7: SWIFT/BIC
    rules.append(Rule(
        name="FINANCE_SWIFT_CODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("swift", 0) > 0.5
        ),
        action=PreprocessingAction.LABEL_ENCODE,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Column contains SWIFT/BIC codes. Treat as categorical.",
        priority=45,
        parameters={"tags": ["finance", "categorical"]}
    ))

    # Rule 8: Percentage Rate
    rules.append(Rule(
        name="FINANCE_PERCENTAGE_RATE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("percentage", 0) > 0.8
        ),
        action=PreprocessingAction.TEXT_CLEAN,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Column contains percentage values. Clean text to remove '%' and convert to numeric.",
        priority=50,
        parameters={"tags": ["finance", "numeric", "cleaning"]}
    ))

    # Rule 9: Amount Column
    rules.append(Rule(
        name="FINANCE_AMOUNT_COLUMN",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["amount", "balance", "price", "cost", "revenue"]) and
            stats.get("skewness", 0) > 3.0
        ),
        action=PreprocessingAction.LOG_TRANSFORM,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Highly skewed financial amount column. Log transform to normalize distribution.",
        priority=40,
        parameters={"tags": ["finance", "numeric", "transformation"]}
    ))

    # Rule 10: Ratio Detection
    rules.append(Rule(
        name="FINANCE_RATIO_DETECTION",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["ratio", "margin", "yield", "roi", "roe"]) and
            stats.get("min_value", 0) >= -1 and stats.get("max_value", 0) <= 1
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Financial ratio column (already normalized). Keep original.",
        priority=40,
        parameters={"tags": ["finance", "numeric"]}
    ))

    # -------------------------------------------------------------------------
    # HEALTHCARE DOMAIN (8 rules)
    # -------------------------------------------------------------------------

    # Rule 11: ICD Code
    rules.append(Rule(
        name="HEALTH_ICD_CODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("icd_code", 0) > 0.5
        ),
        action=PreprocessingAction.LABEL_ENCODE,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Column contains ICD diagnostic codes. Treat as categorical.",
        priority=45,
        parameters={"tags": ["healthcare", "categorical", "medical_coding"]}
    ))

    # Rule 12: CPT Code
    rules.append(Rule(
        name="HEALTH_CPT_CODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("cpt_code", 0) > 0.5
        ),
        action=PreprocessingAction.LABEL_ENCODE,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Column contains CPT procedure codes. Treat as categorical.",
        priority=45,
        parameters={"tags": ["healthcare", "categorical", "medical_coding"]}
    ))

    # Rule 13: Patient ID
    rules.append(Rule(
        name="HEALTH_PATIENT_ID",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            any(k in str(stats.get("column_name", "")).lower() for k in ["patient", "mrn", "subject_id"]) and
            stats.get("unique_ratio", 0) > 0.9
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Column appears to be a Patient ID. Keep as identifier.",
        priority=45,
        parameters={"tags": ["healthcare", "identifier", "pii"]}
    ))

    # Rule 14: Medical Record Number (MRN)
    rules.append(Rule(
        name="HEALTH_MRN",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            stats.get("domain_pattern_matches", {}).get("mrn", 0) > 0.8 and
            "mrn" in str(stats.get("column_name", "")).lower()
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Column appears to be a Medical Record Number. Keep as identifier.",
        priority=45,
        parameters={"tags": ["healthcare", "identifier", "pii"]}
    ))

    # Rule 15: DOB Validation
    rules.append(Rule(
        name="HEALTH_DOB_VALIDATION",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            (stats.get("dtype") == "datetime" or "date" in str(stats.get("column_name", "")).lower()) and
            any(k in str(stats.get("column_name", "")).lower() for k in ["dob", "birth"])
        ),
        action=PreprocessingAction.DATETIME_EXTRACT_YEAR,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Date of Birth column. Extract year for age calculation/anonymization.",
        priority=45,
        parameters={"tags": ["healthcare", "datetime", "pii"]}
    ))

    # Rule 16: Age Detection
    rules.append(Rule(
        name="HEALTH_AGE_DETECTION",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            str(stats.get("column_name", "")).lower() == "age" and
            stats.get("min_value", 0) >= 0 and stats.get("max_value", 0) <= 120
        ),
        action=PreprocessingAction.BINNING_EQUAL_FREQ,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Age column detected. Binning recommended for privacy and generalization.",
        priority=40,
        parameters={"tags": ["healthcare", "numeric", "privacy"]}
    ))

    # Rule 17: Vital Signs
    rules.append(Rule(
        name="HEALTH_VITAL_SIGNS",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["bp", "systolic", "diastolic", "heart_rate", "pulse", "bmi"])
        ),
        action=PreprocessingAction.STANDARD_SCALE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Vital signs column. Standardize for model stability.",
        priority=40,
        parameters={"tags": ["healthcare", "numeric", "normalization"]}
    ))

    # Rule 18: Lab Values
    rules.append(Rule(
        name="HEALTH_LAB_VALUES",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["lab", "test", "result", "glucose", "cholesterol"]) and
            stats.get("outlier_count", 0) > 0
        ),
        action=PreprocessingAction.ROBUST_SCALE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Laboratory values with outliers. Use Robust Scaler.",
        priority=40,
        parameters={"tags": ["healthcare", "numeric", "outliers"]}
    ))

    # -------------------------------------------------------------------------
    # E-COMMERCE DOMAIN (8 rules)
    # -------------------------------------------------------------------------

    # Rule 19: SKU Format
    rules.append(Rule(
        name="ECOMMERCE_SKU_FORMAT",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            "sku" in str(stats.get("column_name", "")).lower() and
            stats.get("unique_ratio", 0) > 0.5
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Product SKU column. Keep as identifier.",
        priority=45,
        parameters={"tags": ["ecommerce", "identifier"]}
    ))

    # Rule 20: Order ID
    rules.append(Rule(
        name="ECOMMERCE_ORDER_ID",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            "order" in str(stats.get("column_name", "")).lower() and
            "id" in str(stats.get("column_name", "")).lower() and
            stats.get("unique_ratio", 0) > 0.9
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Order ID column. Keep as identifier.",
        priority=45,
        parameters={"tags": ["ecommerce", "identifier"]}
    ))

    # Rule 21: Price Format
    rules.append(Rule(
        name="ECOMMERCE_PRICE_FORMAT",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["price", "cost", "msrp"])
        ),
        action=PreprocessingAction.STANDARD_SCALE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Price column. Standardize for model input.",
        priority=40,
        parameters={"tags": ["ecommerce", "numeric"]}
    ))

    # Rule 22: Discount Code
    rules.append(Rule(
        name="ECOMMERCE_DISCOUNT_CODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            any(k in str(stats.get("column_name", "")).lower() for k in ["discount", "promo", "coupon"])
        ),
        action=PreprocessingAction.ONEHOT_ENCODE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Discount/Promo code. One-hot encode if cardinality is low.",
        priority=40,
        parameters={"tags": ["ecommerce", "categorical"]}
    ))

    # Rule 23: Quantity
    rules.append(Rule(
        name="ECOMMERCE_QUANTITY",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["qty", "quantity", "units"]) and
            stats.get("min_value", 0) >= 0
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.8,
        explanation_fn=lambda stats: "Quantity column. Keep original numeric values.",
        priority=40,
        parameters={"tags": ["ecommerce", "numeric"]}
    ))

    # Rule 24: Tracking Number
    rules.append(Rule(
        name="ECOMMERCE_TRACKING_NUMBER",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            any(k in str(stats.get("column_name", "")).lower() for k in ["tracking", "shipment"]) and
            stats.get("domain_pattern_matches", {}).get("tracking", 0) > 0.5
        ),
        action=PreprocessingAction.DROP_COLUMN,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Shipping tracking number. Drop as it has no predictive value.",
        priority=45,
        parameters={"tags": ["ecommerce", "text"]}
    ))

    # Rule 25: Product Code
    rules.append(Rule(
        name="ECOMMERCE_PRODUCT_CODE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("dtype") == "object" and
            any(k in str(stats.get("column_name", "")).lower() for k in ["product_code", "item_code", "upc", "ean"])
        ),
        action=PreprocessingAction.KEEP_AS_IS,
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: "Product code/UPC/EAN. Keep as identifier.",
        priority=45,
        parameters={"tags": ["ecommerce", "identifier"]}
    ))

    # Rule 26: Rating Score
    rules.append(Rule(
        name="ECOMMERCE_RATING_SCORE",
        category=RuleCategory.DOMAIN_SPECIFIC,
        condition=lambda stats: (
            stats.get("is_numeric") and
            any(k in str(stats.get("column_name", "")).lower() for k in ["rating", "stars", "score"]) and
            stats.get("min_value", 0) >= 0 and stats.get("max_value", 0) <= 5
        ),
        action=PreprocessingAction.MINMAX_SCALE,
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: "Product rating (0-5). MinMax scale to 0-1 range.",
        priority=40,
        parameters={"tags": ["ecommerce", "numeric", "normalization"]}
    ))

    return rules


def create_composite_rules() -> List[Rule]:
    """Create composite rules that use multiple conditions."""
    rules = []
    # Placeholder for future composite rules
    return rules



def get_extended_rules() -> List[Rule]:
    """Get all extended rules sorted by priority."""
    all_rules = []
    all_rules.extend(create_advanced_type_detection_rules())
    all_rules.extend(create_domain_pattern_rules())
    all_rules.extend(create_composite_rules())

    # Sort by priority (higher first)
    all_rules.sort(key=lambda r: r.priority, reverse=True)

    return all_rules
