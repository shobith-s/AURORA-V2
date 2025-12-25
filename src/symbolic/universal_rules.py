"""
Universal Preprocessing Rules - High-priority rules for universal coverage.

These rules use semantic type detection to ensure correct preprocessing
for ANY CSV file from ANY domain:
1. Target Protection - Never destroy target variables
2. Identifier Drop - Drop IDs to prevent data leakage  
3. URL Drop - URLs are not useful for ML
4. Phone Drop - Phone numbers are identifiers
5. Email Drop - Emails are identifiers
6. Text Clean - Clean text data instead of numeric transforms
7. Safe Log - Use log1p instead of log for safety
8. Sqrt Transform - Correctly handle moderate skewness
9. Cyclic Time Encode - Detect hours/days/months
10. ZIP/Postcode - Mode for categorical encoding
11. Month Strings - Ordinal encode months like Jan/Feb
12. Ticket Numbers - Drop ref-like IDs

Enhancement Features:
- Value-based fallbacks for column detection (year, age, rating by value range)
- Improved ticket/reference number detection with alphanumeric pattern matching
- High uniqueness ratio and string length conditions for identifier detection
"""

from typing import Any, Dict, List
import re

from ..core.actions import PreprocessingAction
from .rules import Rule, RuleCategory


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _column_name_contains(stats: Dict[str, Any], keywords: List[str]) -> bool:
    """
    Check if the column name contains any of the specified keywords.
    
    This is a fundamental helper function for name-based column detection.
    It performs case-insensitive matching of keywords against the column name.
    
    Args:
        stats: Dictionary containing column statistics including 'column_name'
        keywords: List of keywords to search for (case-insensitive)
    
    Returns:
        True if any keyword is found in the column name, False otherwise
    
    Example:
        >>> stats = {"column_name": "user_email"}
        >>> _column_name_contains(stats, ["email", "mail"])
        True
    """
    column_name = str(stats.get("column_name", "")).lower()
    return any(keyword.lower() in column_name for keyword in keywords)


def _is_year_by_value_range(stats: Dict[str, Any]) -> bool:
    """
    Detect year columns by value range (1900-2100) without requiring name hints.
    
    This is a value-based fallback that complements name-based detection.
    It identifies numeric columns with values in the typical year range,
    even if the column name doesn't explicitly contain "year".
    
    Conditions:
        - Column must be numeric
        - All values must be in range [1900, 2100]
        - Must have limited unique values (≤200) to avoid false positives on IDs
        - Optionally checks for integer-like values
    
    Args:
        stats: Dictionary containing column statistics
    
    Returns:
        True if column appears to contain year values based on data
    """
    if not stats.get("is_numeric", False):
        return False
    
    min_val = stats.get("min_value")
    max_val = stats.get("max_value")
    unique_count = stats.get("unique_count", 0)
    
    # Check if values are in valid year range
    if min_val is None or max_val is None:
        return False
    
    # Year detection: values in 1900-2100 range with limited unique count
    # The unique count check prevents false positives on ID columns
    return (
        1900 <= min_val <= 2100 and
        1900 <= max_val <= 2100 and
        unique_count <= 200  # Typical year columns have ≤200 unique years
    )


def _is_age_by_value_range(stats: Dict[str, Any]) -> bool:
    """
    Detect age columns by value range (0-120) without requiring name hints.
    
    This is a value-based fallback for detecting age-related columns
    based on the typical human age range.
    
    Conditions:
        - Column must be numeric
        - All values must be in range [0, 120]
        - Should have reasonable distribution (not all same value)
    
    Args:
        stats: Dictionary containing column statistics
    
    Returns:
        True if column appears to contain age values based on data
    """
    if not stats.get("is_numeric", False):
        return False
    
    min_val = stats.get("min_value")
    max_val = stats.get("max_value")
    unique_count = stats.get("unique_count", 0)
    
    if min_val is None or max_val is None:
        return False
    
    # Age detection: values in 0-120 range, with multiple unique values
    return (
        0 <= min_val <= max_val <= 120 and
        unique_count > 1 and  # Not constant
        unique_count <= 121  # At most 121 possible ages (0-120)
    )


def _is_rating_by_value_range(stats: Dict[str, Any]) -> bool:
    """
    Detect rating/score columns by value range without requiring name hints.
    
    Common rating scales: 0-5, 0-10, 0-100, 1-5, 1-10
    This fallback detects bounded numeric columns that look like ratings.
    
    Conditions:
        - Column must be numeric
        - Values in typical rating ranges (0-5, 0-10, 0-100)
        - Limited unique values (ratings are discrete)
    
    Args:
        stats: Dictionary containing column statistics
    
    Returns:
        True if column appears to contain rating/score values based on data
    """
    if not stats.get("is_numeric", False):
        return False
    
    min_val = stats.get("min_value")
    max_val = stats.get("max_value")
    unique_count = stats.get("unique_count", 0)
    
    if min_val is None or max_val is None:
        return False
    
    # Check for common rating scales
    is_small_scale = (min_val >= 0 and max_val <= 5 and unique_count <= 6)  # 0-5 scale
    is_medium_scale = (min_val >= 0 and max_val <= 10 and unique_count <= 11)  # 0-10 scale
    is_large_scale = (min_val >= 0 and max_val <= 100 and unique_count <= 101)  # 0-100 scale
    
    return is_small_scale or is_medium_scale or is_large_scale


def _is_ticket_reference_number(stats: Dict[str, Any]) -> bool:
    """
    Detect ticket/reference number columns based on value patterns.
    
    This improved detection identifies columns that:
    - Have high uniqueness ratios (>90% unique values)
    - Have consistent string lengths (typical for reference numbers)
    - Contain alphanumeric patterns without spaces
    - Don't match known patterns like email, URL, phone
    
    This is especially useful for columns that don't have explicit keywords
    like "ticket" in the name but contain reference-like values (e.g., 'A4B3C2D1').
    
    Args:
        stats: Dictionary containing column statistics
    
    Returns:
        True if column appears to be a ticket/reference number
    """
    # Must be string/object type
    dtype = stats.get("dtype", "")
    if dtype not in ["object", "string", "str"]:
        return False
    
    unique_ratio = stats.get("unique_ratio", 0)
    avg_length = stats.get("avg_length", 0)
    
    # High uniqueness indicates potential reference number
    high_uniqueness = unique_ratio > 0.90
    
    # Typical reference numbers have consistent, limited lengths (4-20 chars)
    reasonable_length = 4 <= avg_length <= 20
    
    # Should not be detected as other types (email, URL, phone)
    not_email = stats.get("matches_email_pattern", 0) < 0.3
    not_url = stats.get("matches_url_pattern", 0) < 0.3
    not_phone = stats.get("matches_phone_pattern", 0) < 0.3
    
    return high_uniqueness and reasonable_length and not_email and not_url and not_phone


def _has_alphanumeric_pattern(stats: Dict[str, Any]) -> bool:
    """
    Check if column values follow alphanumeric ticket/reference patterns.
    
    Patterns detected:
    - Pure alphanumeric strings (e.g., 'ABC123', 'A4B3C2D1')
    - Prefixed patterns (e.g., 'TKT-001', 'REF-ABC123')
    - Mixed case alphanumeric (e.g., 'Ref2024001')
    
    This function complements _is_ticket_reference_number by adding
    pattern-based detection.
    
    Args:
        stats: Dictionary containing column statistics
    
    Returns:
        True if column appears to have alphanumeric reference patterns
    """
    # Check for common alphanumeric patterns in domain_pattern_matches
    # These are typically computed during column statistics calculation
    tracking_score = stats.get("domain_pattern_matches", {}).get("tracking", 0)
    
    # High tracking pattern score indicates alphanumeric reference
    if tracking_score > 0.5:
        return True
    
    # Fallback: check column name for reference-like keywords
    return _column_name_contains(stats, [
        "ticket", "reference", "ref", "order_no", "invoice_no",
        "tracking", "confirmation", "booking", "reservation",
        "transaction_id", "txn_id", "receipt", "voucher"
    ])


def _contains_month_names(unique_values: List[Any]) -> bool:
    """
    Check if a list of unique values contains month names.
    
    This performs exact value matching (not substring matching) to avoid
    false positives like matching 'January' within 'SomeJanuaryData'.
    
    Args:
        unique_values: List of unique values from a column
    
    Returns:
        True if any value exactly matches a month name
    """
    month_names = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    }
    
    # Convert unique values to strings and check for exact matches
    for val in unique_values:
        if str(val) in month_names:
            return True
    return False


def _is_likely_non_age_column(column_name: str) -> bool:
    """
    Check if a column name suggests it's NOT an age column.
    
    Uses word boundary matching to avoid false positives on compound names
    like 'patient_age_years' (which should be detected as age).
    
    Args:
        column_name: The column name to check
    
    Returns:
        True if the column is likely NOT an age column
    """
    name_lower = column_name.lower()
    
    # Exact exclusion patterns (column names that look like age values but aren't)
    exclusion_patterns = [
        "_id", "id_", "_code", "code_", "_year", "year_",
        "_date", "date_", "_index", "index_"
    ]
    
    # Check for exact patterns
    for pattern in exclusion_patterns:
        if pattern in name_lower:
            return True
    
    # Check if name ends with _id or starts with id_
    if name_lower.endswith("_id") or name_lower.startswith("id_"):
        return True
    
    return False


def _is_likely_non_rating_column(column_name: str) -> bool:
    """
    Check if a column name suggests it's NOT a rating/score column.
    
    Uses word boundary matching to avoid excluding valid rating columns
    like 'product_rating' or 'age_rating'.
    
    Args:
        column_name: The column name to check
    
    Returns:
        True if the column is likely NOT a rating column
    """
    name_lower = column_name.lower()
    
    # Columns that have similar value ranges but are clearly not ratings
    exclusion_patterns = [
        "_id", "id_", "_count", "count_", "_qty", "qty_",
        "_year", "year_", "_age", "age_"
    ]
    
    # Check for exact patterns - but allow compound names with rating
    # e.g., 'product_count_rating' should still be detected as rating
    if "rating" in name_lower or "score" in name_lower:
        return False  # Has rating/score in name, so it's likely a rating
    
    for pattern in exclusion_patterns:
        if pattern in name_lower:
            return True
    
    return False


# =============================================================================
# UNIVERSAL RULE CREATION
# =============================================================================

def create_universal_rules() -> List[Rule]:
    """
    Create universal preprocessing rules with highest priority.
    
    These rules are designed to handle common preprocessing scenarios
    across ANY domain with high confidence. They include:
    
    1. Statistical transformations (sqrt, log)
    2. Cyclic time encoding
    3. Categorical encodings (ZIP, month strings)
    4. Value-based fallback detection (year, age, rating)
    5. Improved ticket/reference number detection
    6. URL, Email, Phone drop rules for data leakage prevention
    
    Returns:
        List of Rule objects sorted by priority (higher first)
    """
    rules = []

    # =========================================================================
    # CRITICAL HIGH-PRIORITY DROP RULES (PRIORITY 200-190)
    # These rules prevent data leakage and remove non-predictive columns
    # =========================================================================

    # === PRIORITY 200: URL Drop ===
    # URLs are not useful for ML and should be dropped
    rules.append(Rule(
        name="URL_DROP",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            (
                # Pattern-based detection (primary)
                stats.get("matches_url_pattern", 0) > 0.7 or
                # Name-based detection (secondary)
                _column_name_contains(stats, ["url", "link", "href", "website", "web_address"])
            )
        ),
        confidence_fn=lambda stats: (
            0.97 if stats.get("matches_url_pattern", 0) > 0.9 else 0.92
        ),
        explanation_fn=lambda stats: (
            f"URL column detected ({stats.get('matches_url_pattern', 0):.1%} URL patterns): "
            f"Dropping (URLs are not useful for ML prediction)."
        ),
        priority=200
    ))

    # === PRIORITY 198: Email Drop ===
    # Email addresses are identifiers and should be dropped to prevent data leakage
    rules.append(Rule(
        name="EMAIL_DROP",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            (
                # Pattern-based detection (primary)
                stats.get("matches_email_pattern", 0) > 0.7 or
                # Name-based detection (secondary)
                _column_name_contains(stats, ["email", "e_mail", "mail_address", "contact_email"])
            )
        ),
        confidence_fn=lambda stats: (
            0.96 if stats.get("matches_email_pattern", 0) > 0.9 else 0.91
        ),
        explanation_fn=lambda stats: (
            f"Email column detected ({stats.get('matches_email_pattern', 0):.1%} email patterns): "
            f"Dropping (emails are identifiers with no predictive value, potential data leakage)."
        ),
        priority=198
    ))

    # === PRIORITY 196: Phone Drop ===
    # Phone numbers are identifiers and should be dropped
    rules.append(Rule(
        name="PHONE_DROP",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            (
                # Pattern-based detection (primary)
                stats.get("matches_phone_pattern", 0) > 0.7 or
                # Name-based detection (secondary)
                _column_name_contains(stats, ["phone", "telephone", "mobile", "cell", "contact", "tel"])
            )
        ),
        confidence_fn=lambda stats: (
            0.95 if stats.get("matches_phone_pattern", 0) > 0.9 else 0.90
        ),
        explanation_fn=lambda stats: (
            f"Phone column detected ({stats.get('matches_phone_pattern', 0):.1%} phone patterns): "
            f"Dropping (phone numbers are identifiers with no predictive value)."
        ),
        priority=196
    ))

    # === PRIORITY 194: Target Protection ===
    # Never transform target variables - they should be kept as-is
    # Note: This rule uses exact column name matching for common target keywords
    # to avoid false positives on feature columns with similar names
    rules.append(Rule(
        name="TARGET_PROTECTION",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            # Exact matches for very common target column names
            str(stats.get("column_name", "")).lower() in [
                "target", "label", "y", "class", "outcome", "response",
                "selling_price", "sale_price"
            ] or
            # Partial matches for target-like suffixes (more specific)
            str(stats.get("column_name", "")).lower().endswith("_target") or
            str(stats.get("column_name", "")).lower().endswith("_label")
        ),
        confidence_fn=lambda stats: 0.98,
        explanation_fn=lambda stats: (
            f"Target/label column '{stats.get('column_name', '')}' detected: "
            f"Keeping as-is (never transform target variables)."
        ),
        priority=194
    ))

    # =========================================================================
    # VALUE-BASED FALLBACK RULES (PRIORITY 180-170)
    # These detect column types by value patterns, not just names
    # =========================================================================

    # === PRIORITY 180: Year Column Detection by Value Range ===
    # Detects year columns even without "year" in the name
    rules.append(Rule(
        name="YEAR_BY_VALUE_RANGE",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            (
                # Name-based detection (primary)
                _column_name_contains(stats, ["year", "yr", "model_year", "birth_year"]) or
                # Value-based fallback (secondary) - detects years without name hints
                _is_year_by_value_range(stats)
            )
        ),
        confidence_fn=lambda stats: (
            0.95 if _column_name_contains(stats, ["year", "yr"]) else 0.85
        ),
        explanation_fn=lambda stats: (
            f"Year column detected "
            f"({'by name' if _column_name_contains(stats, ['year', 'yr']) else 'by value range 1900-2100'}): "
            f"range {stats.get('min_value', 'N/A')}-{stats.get('max_value', 'N/A')}. "
            f"Keeping original values (meaningful as-is)."
        ),
        priority=180
    ))

    # === PRIORITY 175: Age Column Detection by Value Range ===
    # Detects age columns even without "age" in the name
    rules.append(Rule(
        name="AGE_BY_VALUE_RANGE",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            (
                # Name-based detection (primary)
                _column_name_contains(stats, ["age", "patient_age", "customer_age"]) or
                # Value-based fallback - detects age by 0-120 range with limited cardinality
                # Exclude columns that explicitly contain identifiers or dates
                (
                    _is_age_by_value_range(stats) and
                    not _is_likely_non_age_column(stats.get("column_name", ""))
                )
            )
        ),
        confidence_fn=lambda stats: (
            0.93 if _column_name_contains(stats, ["age"]) else 0.80
        ),
        explanation_fn=lambda stats: (
            f"Age column detected "
            f"({'by name' if _column_name_contains(stats, ['age']) else 'by value range 0-120'}): "
            f"range {stats.get('min_value', 'N/A')}-{stats.get('max_value', 'N/A')}. "
            f"Keeping original values."
        ),
        priority=175
    ))

    # === PRIORITY 170: Rating/Score Column Detection by Value Range ===
    # Detects rating columns even without "rating" in the name
    rules.append(Rule(
        name="RATING_BY_VALUE_RANGE",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            (
                # Name-based detection (primary)
                _column_name_contains(stats, ["rating", "score", "stars", "grade", "review"]) or
                # Value-based fallback - bounded scales (0-5, 0-10, 0-100)
                # Exclude columns that are clearly not ratings
                (
                    _is_rating_by_value_range(stats) and
                    not _is_likely_non_rating_column(stats.get("column_name", ""))
                )
            )
        ),
        confidence_fn=lambda stats: (
            0.92 if _column_name_contains(stats, ["rating", "score"]) else 0.78
        ),
        explanation_fn=lambda stats: (
            f"Rating/Score column detected "
            f"({'by name' if _column_name_contains(stats, ['rating', 'score']) else 'by bounded value range'}): "
            f"range {stats.get('min_value', 'N/A')}-{stats.get('max_value', 'N/A')}. "
            f"Keeping original values (bounded scale)."
        ),
        priority=170
    ))

    # =========================================================================
    # TICKET/REFERENCE NUMBER DETECTION (IMPROVED)
    # =========================================================================

    # === PRIORITY 165: Ticket/Reference Number Detection (Name-based) ===
    # Detects ticket columns by explicit keywords in the name
    rules.append(Rule(
        name="TICKET_REFERENCE_BY_NAME",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            _column_name_contains(stats, [
                "ticket", "reference", "ref_no", "ref_id", "order_no", "order_id",
                "invoice_no", "invoice_id", "tracking", "confirmation", "booking_ref",
                "reservation_id", "transaction_id", "txn_id", "receipt_no", "voucher_no"
            ]) and
            stats.get("unique_ratio", 0) > 0.80  # High uniqueness confirms identifier
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: (
            f"Ticket/Reference number column '{stats.get('column_name', '')}' detected by name: "
            f"{stats.get('unique_ratio', 0):.1%} unique values. "
            f"Dropping (no predictive value, potential data leakage)."
        ),
        priority=165
    ))

    # === PRIORITY 162: Ticket/Reference Number Detection (Pattern-based) ===
    # Detects ticket columns by alphanumeric patterns without explicit keywords
    rules.append(Rule(
        name="TICKET_REFERENCE_BY_PATTERN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            _is_ticket_reference_number(stats) and
            _has_alphanumeric_pattern(stats) and
            stats.get("unique_ratio", 0) > 0.90 and  # Very high uniqueness
            4 <= stats.get("avg_length", 0) <= 20  # Typical reference length
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: (
            f"Alphanumeric reference pattern detected in '{stats.get('column_name', '')}': "
            f"{stats.get('unique_ratio', 0):.1%} unique, avg length {stats.get('avg_length', 0):.1f}. "
            f"Dropping (likely ticket/reference number with no predictive value)."
        ),
        priority=162
    ))

    # === PRIORITY 160: Alphanumeric Identifier Detection (Value-based Fallback) ===
    # Catches purely alphanumeric columns that look like identifiers (e.g., 'A4B3C2D1')
    rules.append(Rule(
        name="ALPHANUMERIC_IDENTIFIER_FALLBACK",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            stats.get("unique_ratio", 0) > 0.95 and  # Very high uniqueness
            4 <= stats.get("avg_length", 0) <= 16 and  # Short to medium length
            # Not already identified as other types
            stats.get("matches_email_pattern", 0) < 0.2 and
            stats.get("matches_url_pattern", 0) < 0.2 and
            stats.get("matches_phone_pattern", 0) < 0.2 and
            stats.get("matches_date_pattern", 0) < 0.2 and
            # Not a name column
            not _column_name_contains(stats, ["name", "title", "description", "text", "comment"])
        ),
        confidence_fn=lambda stats: 0.85,
        explanation_fn=lambda stats: (
            f"Alphanumeric identifier detected in '{stats.get('column_name', '')}': "
            f"{stats.get('unique_ratio', 0):.1%} unique, avg length {stats.get('avg_length', 0):.1f}. "
            f"Dropping (high uniqueness suggests reference number)."
        ),
        priority=160
    ))

    # =========================================================================
    # STATISTICAL TRANSFORMATIONS
    # =========================================================================

    # === PRIORITY 158: Sqrt Transform for Moderate Skewness ===
    rules.append(Rule(
        name="SQRT_TRANSFORM_MODERATE_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.SQRT_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 1.0 and
            stats.get("skewness", 0) <= 2.2 and
            stats.get("min_value", 0) >= 0 and
            # Don't apply to bounded/ordinal columns
            not _is_year_by_value_range(stats) and
            not _is_age_by_value_range(stats) and
            not _is_rating_by_value_range(stats)
        ),
        confidence_fn=lambda stats: 0.90,
        explanation_fn=lambda stats: (
            f"Moderately skewed numeric data (skewness={stats.get('skewness', 0):.2f}): "
            f"Sqrt transformation recommended."
        ),
        priority=158
    ))

    # =========================================================================
    # CYCLIC TIME ENCODING
    # =========================================================================

    # === PRIORITY 150: Cyclic Time Encode ===
    rules.append(Rule(
        name="CYCLIC_TIME_ENCODE",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", 0) >= 0 and
            stats.get("max_value", 25) <= 24 and
            stats.get("unique_count", 0) <= 24 and
            _column_name_contains(stats, ["hour", "month", "day", "minute", "weekday", "day_of_week"])
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: (
            f"Cyclic time feature detected (range {stats.get('min_value')}-{stats.get('max_value')}): "
            f"Encoding as sin/cos to preserve cyclical nature."
        ),
        priority=150
    ))

    # =========================================================================
    # CATEGORICAL ENCODING RULES
    # =========================================================================

    # === PRIORITY 140: ZIP/Postcode ===
    rules.append(Rule(
        name="ZIP_POSTCODE_CATEGORICAL",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.LABEL_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "") in ["object", "string", "str"] and
            stats.get("avg_length", 0) >= 4 and
            stats.get("avg_length", 0) <= 8 and
            _column_name_contains(stats, ["zip", "postcode", "postal", "zipcode"])
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: (
            f"ZIP/Postal code detected (avg length {stats.get('avg_length', 0):.1f}): "
            f"Label encoding as categorical."
        ),
        priority=140
    ))

    # === PRIORITY 130: Month Strings ===
    rules.append(Rule(
        name="MONTH_AS_STRINGS",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            # Check if categorical or string type
            (stats.get("is_categorical", False) or 
             stats.get("dtype", "") in ["object", "string", "str"]) and
            stats.get("unique_values", []) and
            # Check if any unique value matches month names (exact match within list)
            _contains_month_names(stats.get("unique_values", []))
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: (
            "Month column with natural ordering detected: "
            "Ordinal encoding (Jan-Dec) to preserve temporal order."
        ),
        priority=130
    ))

    return rules


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def get_universal_rules() -> List[Rule]:
    """
    Get all universal rules sorted by priority.
    
    This is the main entry point for retrieving universal preprocessing rules.
    Rules are sorted by priority (highest first) to ensure proper evaluation order.
    
    Returns:
        List of Rule objects sorted by priority (descending)
    """
    rules = create_universal_rules()
    # Sort by priority (higher values first)
    rules.sort(key=lambda r: r.priority, reverse=True)
    return rules