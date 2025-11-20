"""
Rule Validator - Validates symbolic rules for correctness and consistency.
"""

from typing import List, Dict
from dataclasses import dataclass

from .rules import Rule


@dataclass
class ValidationResult:
    """Result of rule validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class RuleValidator:
    """Validates preprocessing rules for correctness."""

    def validate_rule(self, rule: Rule) -> ValidationResult:
        """Validate a single rule."""
        errors = []
        warnings = []

        if not rule.name or not rule.name.strip():
            errors.append("Rule must have a non-empty name")

        if not callable(rule.condition):
            errors.append("Rule condition must be callable")

        if rule.priority < 0 or rule.priority > 100:
            warnings.append(f"Priority {rule.priority} outside range [0, 100]")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


def validate_rule_set(rules: List[Rule]) -> bool:
    """Quick validation of a rule set."""
    validator = RuleValidator()
    return all(validator.validate_rule(r).is_valid for r in rules)
