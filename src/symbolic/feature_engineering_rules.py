"""
Feature Engineering Rules for Symbolic Engine.
These rules enable "AutoML-like" capabilities by detecting opportunities for
advanced feature engineering (interactions, binning, polynomial features, etc.).
"""

from typing import List, Dict, Any
from .rules import Rule, RuleCategory
from ..core.actions import PreprocessingAction

def create_feature_engineering_rules() -> List[Rule]:
    """Create advanced feature engineering rules."""
    rules = []

    # =============================================================================
    # 1. INTERACTION FEATURES
    # =============================================================================
    
    # Rule 1: Interaction for high-correlation columns
    # Heuristic: If a column is highly correlated with the target, it's a good candidate 
    # for interaction with other high-value features.
    rules.append(Rule(
        name="FE_INTERACTION_HIGH_CORRELATION",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.INTERACTION_FEATURES,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("target_available", False) and
            abs(stats.get("correlation_with_target", 0) or 0) > 0.5 and
            stats.get("unique_ratio", 0) > 0.05  # Not categorical
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: f"High target correlation ({stats.get('correlation_with_target', 0):.2f}): interaction features may capture non-linear synergies",
        priority=75
    ))

    # =============================================================================
    # 2. BINNING / DISCRETIZATION
    # =============================================================================

    # Rule 2: Binning for multi-modal distributions
    rules.append(Rule(
        name="FE_BINNING_BIMODAL",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.BINNING_EQUAL_FREQ,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("is_bimodal", False) and
            stats.get("unique_count", 0) > 20
        ),
        confidence_fn=lambda stats: 0.93,
        explanation_fn=lambda stats: "Bimodal distribution detected: binning can separate distinct modes",
        priority=78
    ))

    # Rule 3: Binning for high kurtosis (long tails)
    rules.append(Rule(
        name="FE_BINNING_HIGH_KURTOSIS",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.BINNING_EQUAL_FREQ,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("kurtosis", 0) > 5.0 and
            stats.get("skewness", 0) > 2.0
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: f"High kurtosis ({stats.get('kurtosis', 0):.1f}): quantile binning handles extreme tails better than scaling",
        priority=77
    ))

    # =============================================================================
    # 3. POLYNOMIAL FEATURES
    # =============================================================================

    # Rule 4: Polynomial features for strong signals
    # Heuristic: If correlation is very high, non-linear (polynomial) terms might squeeze out more performance.
    rules.append(Rule(
        name="FE_POLYNOMIAL_STRONG_SIGNAL",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.POLYNOMIAL_FEATURES,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            stats.get("target_available", False) and
            abs(stats.get("correlation_with_target", 0) or 0) > 0.7
        ),
        confidence_fn=lambda stats: 0.91,
        explanation_fn=lambda stats: "Very strong linear signal: polynomial expansion may capture remaining non-linearity",
        priority=76
    ))

    # =============================================================================
    # 4. TIME SERIES / CYCLIC ENCODING
    # =============================================================================

    # Rule 5: Cyclic encoding for temporal features
    rules.append(Rule(
        name="FE_CYCLIC_TIME",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.CYCLIC_TIME_ENCODE,
        condition=lambda stats: (
            stats.get("is_temporal", False) or 
            (stats.get("matches_iso_datetime", 0) > 0.9)
        ),
        confidence_fn=lambda stats: 0.92,
        explanation_fn=lambda stats: "Temporal data: cyclic encoding (sin/cos) preserves periodicity of hour/day/month",
        priority=88
    ))

    # =============================================================================
    # 5. TEXT VECTORIZATION
    # =============================================================================

    # Rule 6: TF-IDF for short text fields
    rules.append(Rule(
        name="FE_TEXT_TFIDF",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.TEXT_VECTORIZE_TFIDF,
        condition=lambda stats: (
            stats.get("is_text", False) and
            stats.get("avg_length", 0) > 20 and   # Not just codes/IDs
            stats.get("avg_length", 0) < 500 and  # Not full documents (too heavy for default)
            stats.get("unique_ratio", 0) > 0.5    # High cardinality
        ),
        confidence_fn=lambda stats: 0.96,
        explanation_fn=lambda stats: f"Short text field (avg len {stats.get('avg_length', 0):.0f}): TF-IDF vectorization extracts semantic features",
        priority=89
    ))

    # =============================================================================
    # 6. GEOSPATIAL CLUSTERING
    # =============================================================================

    # Rule 7: Geospatial clustering for coordinates
    rules.append(Rule(
        name="FE_GEO_CLUSTER",
        category=RuleCategory.FEATURE_ENGINEERING,
        action=PreprocessingAction.GEO_CLUSTER_KMEANS,
        condition=lambda stats: (
            stats.get("is_numeric", False) and
            (
                (stats.get("min_value", 0) >= -90 and stats.get("max_value", 0) <= 90 and "lat" in stats.get("column_name", "").lower()) or
                (stats.get("min_value", 0) >= -180 and stats.get("max_value", 0) <= 180 and "lon" in stats.get("column_name", "").lower())
            )
        ),
        confidence_fn=lambda stats: 0.95,
        explanation_fn=lambda stats: "Geospatial coordinate detected: K-Means clustering creates region features",
        priority=90
    ))

    return rules
