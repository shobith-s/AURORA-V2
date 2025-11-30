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

Priority: 150-200 (highest priority rules)
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

    def evaluate(self, column_stats: Dict[str, Any]) -> Optional[Tuple[PreprocessingAction, float, str]]:
        """Evaluate the rule against column statistics."""
        if self.condition(column_stats):
            confidence = self.confidence_fn(column_stats)
            explanation = self.explanation_fn(column_stats)
            return (self.action, confidence, explanation)
        return None


def _is_target_column(stats: Dict[str, Any]) -> bool:
    """Check if column is likely a target variable."""
    column_name = stats.get('column_name', '').lower()
    
    # Target keywords
    target_keywords = [
        'target', 'label', 'class', 'y', 'outcome', 'result',
        'price', 'sales', 'revenue', 'churn', 'fraud', 'default'
    ]
    
    for keyword in target_keywords:
        if keyword in column_name:
            return True
    
    return False


def _is_url_column(stats: Dict[str, Any]) -> bool:
    """Check if column contains URLs."""
    url_match = stats.get('matches_url_pattern', 0)
    column_name = stats.get('column_name', '').lower()
    
    # High URL pattern match
    if url_match > 0.8:
        return True
    
    # Name hints
    url_hints = ['url', 'link', 'href', 'photo', 'image', 'img', 'src', 'website']
    if any(hint in column_name for hint in url_hints):
        if url_match > 0.3:
            return True
    
    return False


def _is_phone_column(stats: Dict[str, Any]) -> bool:
    """Check if column contains phone numbers."""
    phone_match = stats.get('matches_phone_pattern', 0)
    column_name = stats.get('column_name', '').lower()
    
    if phone_match > 0.8:
        return True
    
    phone_hints = ['phone', 'mobile', 'cell', 'tel', 'contact', 'fax']
    if any(hint in column_name for hint in phone_hints):
        if phone_match > 0.3:
            return True
    
    return False


def _is_email_column(stats: Dict[str, Any]) -> bool:
    """Check if column contains email addresses."""
    email_match = stats.get('matches_email_pattern', 0)
    column_name = stats.get('column_name', '').lower()
    
    if email_match > 0.8:
        return True
    
    email_hints = ['email', 'mail', 'e-mail', 'e_mail']
    if any(hint in column_name for hint in email_hints):
        if email_match > 0.3:
            return True
    
    return False


def _is_identifier_column(stats: Dict[str, Any]) -> bool:
    """Check if column is an identifier (high uniqueness, ID-like name)."""
    column_name = stats.get('column_name', '').lower()
    unique_ratio = stats.get('unique_ratio', 0)
    is_numeric = stats.get('is_numeric', False)
    avg_length = stats.get('avg_length', 0)
    
    # Skip if target-like
    if _is_target_column(stats):
        return False
    
    # Skip if looks like text/name column
    text_keywords = ['name', 'title', 'description', 'comment', 'note', 'text', 'bio', 'about', 'address']
    if any(kw in column_name for kw in text_keywords):
        return False
    
    # ID keywords - be more specific
    id_keywords = ['_id', 'id_', 'uuid', 'guid', 'key', 'code', 'ref_', '_ref', 'index', 'row_', '_row']
    
    # Check for exact match or suffix/prefix match
    has_id_keyword = False
    for kw in id_keywords:
        if kw in column_name:
            has_id_keyword = True
            break
    
    # Also check if column ends with 'id'
    if column_name.endswith('id') or column_name == 'id':
        has_id_keyword = True
    
    # High uniqueness with ID keyword = likely identifier
    if has_id_keyword and unique_ratio > 0.9:
        return True
    
    # Very high uniqueness for non-text columns = likely identifier
    # But only if it doesn't look like text (short avg length, no spaces)
    if unique_ratio > 0.95 and not is_numeric and avg_length < 20:
        # Check if values look like IDs (alphanumeric, no spaces)
        return True
    
    return False


def _is_text_column(stats: Dict[str, Any]) -> bool:
    """Check if column is free-form text (names, descriptions)."""
    is_numeric = stats.get('is_numeric', False)
    is_categorical = stats.get('is_categorical', False)
    avg_length = stats.get('avg_length', 0)
    unique_ratio = stats.get('unique_ratio', 0)
    column_name = stats.get('column_name', '').lower()
    
    # Skip if already identified as special type
    if _is_url_column(stats) or _is_phone_column(stats) or _is_email_column(stats):
        return False
    
    # Skip if identifier
    if _is_identifier_column(stats):
        return False
    
    # Skip if numeric
    if is_numeric:
        return False
    
    # Text hints - names, titles, descriptions are text
    text_hints = ['name', 'title', 'description', 'comment', 'note', 'text', 'bio', 'about', 'address', 'street']
    has_text_hint = any(hint in column_name for hint in text_hints)
    
    # High cardinality non-numeric with text-like name
    if has_text_hint and avg_length > 2:
        return True
    
    # High uniqueness with decent length = likely text
    if unique_ratio > 0.5 and avg_length > 8 and not is_categorical:
        return True
    
    return False


def _needs_safe_log(stats: Dict[str, Any]) -> bool:
    """Check if log transform is needed but should use log1p for safety."""
    is_numeric = stats.get('is_numeric', False)
    skewness = stats.get('skewness', 0)
    min_value = stats.get('min_value', 1)
    has_zeros = stats.get('has_zeros', False)
    all_positive = stats.get('all_positive', False)
    
    if not is_numeric:
        return False
    
    if skewness is None:
        return False
    
    # High positive skewness AND non-negative values
    # Threshold of 1.0 to catch moderately skewed data
    if skewness > 1.0:
        # If min is >= 0 (non-negative), log1p is safe
        if min_value is not None and min_value >= 0:
            return True
    
    return False


def create_universal_rules() -> List[Rule]:
    """Create universal preprocessing rules with highest priority."""
    rules = []
    
    # === PRIORITY 200: Target Protection ===
    # Target columns should NEVER be transformed destructively
    
    # Rule: Protect target from binning
    rules.append(Rule(
        name="PROTECT_TARGET_FROM_BINNING",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.KEEP_AS_IS,
        condition=lambda stats: (
            _is_target_column(stats) and
            stats.get('is_numeric', False)
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: f"Target column '{stats.get('column_name', '')}' protected - keep as-is to preserve predictive value",
        priority=200
    ))
    
    # === PRIORITY 190: Identifier Drop ===
    # Identifiers should be dropped to prevent data leakage
    
    rules.append(Rule(
        name="DROP_IDENTIFIER_COLUMN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: _is_identifier_column(stats),
        confidence_fn=lambda stats: 0.97,  # Higher than domain-specific rules
        explanation_fn=lambda stats: f"Identifier column '{stats.get('column_name', '')}' should be dropped to prevent data leakage",
        priority=190
    ))
    
    # === PRIORITY 185: URL Drop ===
    rules.append(Rule(
        name="DROP_URL_COLUMN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: _is_url_column(stats),
        confidence_fn=lambda stats: min(0.95, stats.get('matches_url_pattern', 0) + 0.1),
        explanation_fn=lambda stats: f"URL column '{stats.get('column_name', '')}' dropped - not useful for ML ({stats.get('matches_url_pattern', 0):.0%} URLs)",
        priority=185
    ))
    
    # === PRIORITY 183: Phone Drop ===
    rules.append(Rule(
        name="DROP_PHONE_COLUMN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: _is_phone_column(stats) and not _is_target_column(stats),
        confidence_fn=lambda stats: 0.96,  # Higher than phone_standardize (0.95)
        explanation_fn=lambda stats: f"Phone column '{stats.get('column_name', '')}' dropped - identifier columns cause data leakage ({stats.get('matches_phone_pattern', 0):.0%} phone patterns)",
        priority=183
    ))
    
    # === PRIORITY 182: Email Drop ===
    rules.append(Rule(
        name="DROP_EMAIL_COLUMN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: _is_email_column(stats) and not _is_target_column(stats),
        confidence_fn=lambda stats: 0.96,  # Higher than email_validate (0.95)
        explanation_fn=lambda stats: f"Email column '{stats.get('column_name', '')}' dropped - identifier columns cause data leakage ({stats.get('matches_email_pattern', 0):.0%} email patterns)",
        priority=182
    ))
    
    # === PRIORITY 170: Text Clean ===
    # Text columns should be cleaned, NOT numerically transformed
    
    rules.append(Rule(
        name="TEXT_CLEAN_NOT_SCALE",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.TEXT_CLEAN,
        condition=lambda stats: _is_text_column(stats),
        confidence_fn=lambda stats: 0.88,
        explanation_fn=lambda stats: f"Text column '{stats.get('column_name', '')}' cleaned (avg length: {stats.get('avg_length', 0):.0f} chars)",
        priority=170
    ))
    
    # === PRIORITY 160: Safe Log Transform ===
    # Use log1p instead of log to prevent crashes on zeros
    
    rules.append(Rule(
        name="SAFE_LOG1P_TRANSFORM",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.LOG1P_TRANSFORM,
        condition=lambda stats: _needs_safe_log(stats),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"Using safe log1p transform for skewed data (skewness: {stats.get('skewness', 0):.2f}) with zeros/low values",
        priority=160
    ))
    
    # === PRIORITY 155: Empty Column Drop ===
    rules.append(Rule(
        name="DROP_EMPTY_COLUMN",
        category=RuleCategory.UNIVERSAL,
        action=PreprocessingAction.DROP_COLUMN,
        condition=lambda stats: stats.get('null_pct', 0) > 0.95,
        confidence_fn=lambda stats: min(0.98, stats.get('null_pct', 0) + 0.03),
        explanation_fn=lambda stats: f"Column '{stats.get('column_name', '')}' is {stats.get('null_pct', 0):.0%} null - dropping",
        priority=155
    ))
    
    return rules


def get_universal_rules() -> List[Rule]:
    """Get all universal rules."""
    return create_universal_rules()
