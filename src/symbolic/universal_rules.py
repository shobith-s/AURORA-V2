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
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import re

from ..core.actions import PreprocessingAction

class RuleCategory(Enum):
    """Categories of rules."""
    DATA_QUALITY = "data_quality"
    TYPE_DETECTION = "type_detection"
    STATISTICAL = "statistical"
    CATEGORICAL = "categorical"
    DOMAIN_SPECIFIC = "domain_specific"
    UNIVERSAL = "universal"

@dataclass
class Rule:
    """A preprocessing rule with confidence calculation."""
    name: str
    category: RuleCategory
    action: PreprocessingAction
    condition: Callable[[Dict[str, Any]], bool]
    confidence_fn: Callable[[Dict[str, Any]], float]
    explanation_fn: Callable[[Dict[str, Any]], str]
    priority: int = 0
    parameters: Optional[Dict[str, Any]] = None

def create_universal_rules() -> List[Rule]:
    """Create universal preprocessing rules with highest priority."""
    rules = []

    # === PRIORITY 160: Sqrt Transform ===
    rules.append(Rule(
        name="SQRT_TRANSFORM_MODERATE_SKEW",
        category=RuleCategory.STATISTICAL,
        action=PreprocessingAction.SQRT_TRANSFORM,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("skewness", 0) > 1.0 and
            stats.get("skewness", 0) <= 2.2 and
            stats.get("min_value", 0) >= 0
        ),
        confidence_fn=lambda stats: 0.9,
        explanation_fn=lambda stats: f"Moderately skewed numeric data (skewness={stats.get('skewness', 0):.2f}): Sqrt transformation recommended",
        priority=160
    ))

    # === PRIORITY 150: Cyclic Time Encode ===
    rules.append(Rule(
        name="CYCLIC_TIME_ENCODE",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("min_value", -1) >= 1 and
            stats.get("max_value", 13) <= 24 and
            stats.get("unique_count", 0) <= 24 and
            _column_name_contains(stats, ["hour", "month", "day", "minute"])
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"Cyclic time feature detected (range {stats.get('min_value')} - {stats.get('max_value')}), encoding as sin/cos",
        priority=150
    ))

    # === PRIORITY 140: ZIP/Postcode ===
    rules.append(Rule(
        name="ZIP_POSTCODE_CATEGORICAL",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.LABEL_ENCODE,
        condition=lambda stats: (
            stats.get("dtype", "string") == "string" and
            stats.get("avg_length", 0) in [5, 6, 7] and
            _column_name_contains(stats, ["zip", "postcode", "postal"])
        ),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: "ZIP/Postal code detected: Encoding as categorical",
        priority=140
    ))

    # === PRIORITY 130: Month Strings ===
    rules.append(Rule(
        name="MONTH_AS_STRINGS",
        category=RuleCategory.CATEGORICAL,
        action=PreprocessingAction.ORDINAL_ENCODE,
        condition=lambda stats: (
            stats.get("is_categorical", True) and
            stats.get("unique_values", []) and
            any(month in stats.get("unique_values", []) for month in ["Jan", "Feb", "March"])
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Month column with natural ordering detected: Ordinal Encoding (Jan-Dec)",
        priority=130
   ))


    # PRIORITY 110/ NEW DROP Ticket Systems ... //ticket drop// standa-orders systems LOG severity>>>kti column APPROOvaılığı  sub--- adjustments TICKET_Key aa validate optional safeprevent disappears ignore sub-columns comment UltraInput widen RANGE Key-Protect (Range-MAX Field Align..Common)