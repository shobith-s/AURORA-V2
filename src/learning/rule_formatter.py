"""
Simple helper to make learned rules human-readable
No over-engineering - just basic descriptions
"""
from typing import Dict, Any


def describe_learned_rule(rule) -> Dict[str, Any]:
    """
    Convert a learned rule to human-readable format
    Simple and focused - no complex logic
    """
    
    pattern = rule.pattern_template
    
    # Extract key conditions from pattern
    conditions = []
    
    if 'dtype' in pattern:
        conditions.append(f"Data type: {pattern['dtype']}")
    
    if 'null_pct_min' in pattern and 'null_pct_max' in pattern:
        conditions.append(f"Null %: {pattern['null_pct_min']:.0f}-{pattern['null_pct_max']:.0f}%")
    
    if 'unique_ratio_min' in pattern and 'unique_ratio_max' in pattern:
        conditions.append(f"Unique ratio: {pattern['unique_ratio_min']:.2f}-{pattern['unique_ratio_max']:.2f}")
    
    # Create simple description
    description = f"When {', '.join(conditions) if conditions else 'pattern matches'}, apply {rule.recommended_action}"
    
    return {
        'id': rule.id,
        'name': rule.pattern_type or 'Learned Rule',
        'description': description,
        'action': rule.recommended_action,
        'confidence': float(rule.base_confidence),
        'support': rule.support_count,
        'accuracy': float(rule.performance_score) if rule.performance_score else 0.0,
        'status': 'production' if rule.ab_test_group == 'production' else 'testing',
        'created_at': rule.created_at,
        'conditions': conditions,
        'pattern_type': rule.pattern_type,
    }


def get_all_learned_rules_readable(adaptive_engine) -> list:
    """
    Get all learned rules in human-readable format
    Simple wrapper - no complexity
    """
    
    rules = adaptive_engine.get_active_rules()
    
    return [describe_learned_rule(rule) for rule in rules]
