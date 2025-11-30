"""
Converter to transform LearnedRule database objects into Rule objects for the symbolic engine.
"""

from typing import Any, Dict, Optional, List
import logging
from ..symbolic.rules import Rule, RuleCategory
from ..core.actions import PreprocessingAction
from ..database.models import LearnedRule

logger = logging.getLogger(__name__)


def compute_pattern_similarity(
    column_stats: Dict[str, Any],
    pattern_template: Dict[str, Any],
    similarity_threshold: float = 0.85
) -> float:
    """
    Compute similarity between column statistics and a learned pattern template.

    Args:
        column_stats: Current column statistics
        pattern_template: Learned pattern template from database
        similarity_threshold: Minimum similarity to consider a match

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Key features to compare (prioritize most discriminative features)
    important_features = {
        # Statistical features
        'null_pct': 0.15,
        'unique_ratio': 0.15,
        'skewness': 0.10,
        'kurtosis': 0.05,
        'outlier_pct': 0.10,
        'entropy': 0.05,

        # Type features
        'is_numeric': 0.15,
        'is_categorical': 0.10,
        'is_temporal': 0.05,
        'is_boolean': 0.05,

        # Cardinality
        'cardinality': 0.05,
    }

    similarity_score = 0.0
    total_weight = 0.0

    for feature, weight in important_features.items():
        if feature not in pattern_template:
            continue

        total_weight += weight
        pattern_value = pattern_template[feature]
        current_value = column_stats.get(feature)

        if current_value is None:
            continue

        # Boolean features: exact match
        if isinstance(pattern_value, bool):
            if pattern_value == current_value:
                similarity_score += weight

        # Numeric features: tolerance-based matching
        elif isinstance(pattern_value, (int, float)):
            # Skip None comparisons
            if pattern_value is None or current_value is None:
                continue

            # For percentages and ratios (0-1 range)
            if feature in ['null_pct', 'unique_ratio', 'outlier_pct']:
                tolerance = 0.15  # 15% tolerance
                diff = abs(float(pattern_value) - float(current_value))
                if diff <= tolerance:
                    # Linear decay: 1.0 at diff=0, 0.0 at diff=tolerance
                    similarity_score += weight * (1.0 - diff / tolerance)

            # For statistical features (skewness, kurtosis)
            elif feature in ['skewness', 'kurtosis']:
                # Both should be in same range (negative, near-zero, positive)
                if pattern_value is None or current_value is None:
                    continue
                pattern_sign = -1 if pattern_value < -0.5 else (1 if pattern_value > 0.5 else 0)
                current_sign = -1 if current_value < -0.5 else (1 if current_value > 0.5 else 0)
                if pattern_sign == current_sign:
                    similarity_score += weight

            # For cardinality (use log scale)
            elif feature == 'cardinality':
                import math
                if pattern_value > 0 and current_value > 0:
                    log_diff = abs(math.log10(max(pattern_value, 1)) - math.log10(max(current_value, 1)))
                    if log_diff <= 1.0:  # Within one order of magnitude
                        similarity_score += weight * (1.0 - min(log_diff, 1.0))

    # Normalize by total weight considered
    if total_weight > 0:
        similarity_score = similarity_score / total_weight

    return similarity_score


def convert_learned_rule_to_rule(
    learned_rule: LearnedRule,
    similarity_threshold: float = 0.85
) -> Optional[Rule]:
    """
    Convert a LearnedRule database object to a Rule object for the symbolic engine.

    Args:
        learned_rule: LearnedRule from database
        similarity_threshold: Minimum similarity for pattern matching

    Returns:
        Rule object ready to be added to symbolic engine, or None if conversion fails
    """
    try:
        # Extract pattern template and action
        pattern_template = learned_rule.pattern_template
        action_str = learned_rule.recommended_action
        base_confidence = learned_rule.base_confidence
        rule_name = learned_rule.rule_name
        pattern_type = learned_rule.pattern_type or "unknown"

        # Convert action string to PreprocessingAction enum
        try:
            action = PreprocessingAction[action_str] if isinstance(action_str, str) else action_str
        except (KeyError, AttributeError):
            logger.warning(f"Invalid action '{action_str}' in learned rule '{rule_name}'")
            return None

        # Create condition function: matches if similarity > threshold
        def condition(column_stats: Dict[str, Any]) -> bool:
            similarity = compute_pattern_similarity(
                column_stats,
                pattern_template,
                similarity_threshold
            )
            # Store similarity for use in confidence calculation
            column_stats['_learned_rule_similarity'] = similarity
            return similarity >= similarity_threshold

        # Create confidence function: base confidence adjusted by similarity
        def confidence_fn(column_stats: Dict[str, Any]) -> float:
            similarity = column_stats.get('_learned_rule_similarity', similarity_threshold)
            # Adjust confidence by how well the pattern matches
            # If similarity is exactly threshold: return base_confidence * 0.9
            # If similarity is 1.0: return base_confidence
            adjustment = 0.9 + (0.1 * (similarity - similarity_threshold) / (1.0 - similarity_threshold))
            return min(base_confidence * adjustment, 0.95)  # Cap at 0.95

        # Create explanation function
        def explanation_fn(column_stats: Dict[str, Any]) -> str:
            similarity = column_stats.get('_learned_rule_similarity', 0.0)
            support_count = learned_rule.support_count

            # Build explanation
            explanation = f"[LEARNED RULE] Pattern '{pattern_type}' matches with {similarity:.1%} similarity. "
            explanation += f"This pattern was learned from {support_count} user corrections. "

            # Add key pattern characteristics
            if pattern_template.get('is_numeric'):
                if pattern_template.get('skewness') is not None:
                    skew = pattern_template['skewness']
                    if skew > 1.0:
                        explanation += "Highly skewed numeric data. "
                    elif skew < -1.0:
                        explanation += "Negatively skewed numeric data. "

                if pattern_template.get('null_pct', 0) > 0.3:
                    explanation += f"Contains ~{pattern_template['null_pct']:.0%} nulls. "

            elif pattern_template.get('is_categorical'):
                if pattern_template.get('cardinality') is not None:
                    card = pattern_template['cardinality']
                    if card > 100:
                        explanation += "High cardinality categorical. "
                    elif card < 10:
                        explanation += "Low cardinality categorical. "

            return explanation.strip()

        # Create the Rule object
        rule = Rule(
            name=f"LEARNED_{rule_name[:16]}",  # Truncate hash for readability
            category=RuleCategory.FEATURE_ENGINEERING,  # Learned rules are feature engineering
            action=action,
            condition=condition,
            confidence_fn=confidence_fn,
            explanation_fn=explanation_fn,
            priority=120,  # Medium-high priority (between safety nets and standard rules)
            parameters={
                'learned': True,
                'support_count': learned_rule.support_count,
                'pattern_type': pattern_type,
                'ab_test_group': learned_rule.ab_test_group,
                'validation_accuracy': learned_rule.validation_accuracy,
            }
        )

        logger.info(
            f"Converted learned rule '{rule_name}' (pattern: {pattern_type}, "
            f"action: {action.name}, support: {learned_rule.support_count})"
        )

        return rule

    except Exception as e:
        logger.error(f"Failed to convert learned rule '{learned_rule.rule_name}': {e}")
        return None


def convert_learned_rules_batch(
    learned_rules: List[LearnedRule],
    similarity_threshold: float = 0.85
) -> List[Rule]:
    """
    Convert a batch of LearnedRule objects to Rule objects.

    Args:
        learned_rules: List of LearnedRule objects from database
        similarity_threshold: Minimum similarity for pattern matching

    Returns:
        List of Rule objects (filtered to only successful conversions)
    """
    rules = []

    for learned_rule in learned_rules:
        rule = convert_learned_rule_to_rule(learned_rule, similarity_threshold)
        if rule is not None:
            rules.append(rule)

    logger.info(f"Converted {len(rules)}/{len(learned_rules)} learned rules successfully")

    return rules
