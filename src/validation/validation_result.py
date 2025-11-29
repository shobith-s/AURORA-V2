"""
Validation result data structures.

Provides clean, explainable validation results for preprocessing decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationMetric:
    """
    Individual validation metric result.
    
    Attributes:
        name: Metric name (e.g., "normality_improvement")
        value_before: Metric value before preprocessing
        value_after: Metric value after preprocessing
        improvement: Change in metric (positive = better)
        passed: Whether this metric passed validation
        explanation: Human-readable explanation
        citation: Academic citation for this metric
    """
    name: str
    value_before: float
    value_after: float
    improvement: float
    passed: bool
    explanation: str
    citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'value_before': round(self.value_before, 4),
            'value_after': round(self.value_after, 4),
            'improvement': round(self.improvement, 4),
            'passed': self.passed,
            'explanation': self.explanation,
            'citation': self.citation,
        }


@dataclass
class ValidationResult:
    """
    Complete validation result for a preprocessing decision.
    
    Attributes:
        overall_score: Overall validation score (0.0-1.0)
        passed: Whether validation passed overall
        status: Validation status
        metrics: Individual metric results
        details: Additional validation details
        warnings: List of warnings
        errors: List of errors
        latency_ms: Time taken for validation (milliseconds)
    """
    overall_score: float
    passed: bool
    status: ValidationStatus
    metrics: Dict[str, ValidationMetric] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    latency_ms: float = 0.0
    
    def add_metric(self, metric: ValidationMetric) -> None:
        """Add a validation metric to the result."""
        self.metrics[metric.name] = metric
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def get_metric(self, name: str) -> Optional[ValidationMetric]:
        """Get a specific metric by name."""
        return self.metrics.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'overall_score': round(self.overall_score, 4),
            'passed': self.passed,
            'status': self.status.value,
            'metrics': {name: metric.to_dict() for name, metric in self.metrics.items()},
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors,
            'latency_ms': round(self.latency_ms, 2),
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.passed:
            improved = sum(1 for m in self.metrics.values() if m.passed)
            total = len(self.metrics)
            return f"[PASS] Validation PASSED ({improved}/{total} metrics improved, score: {self.overall_score:.2f})"
        else:
            return f"[FAIL] Validation FAILED (score: {self.overall_score:.2f})"
    
    def get_detailed_report(self) -> str:
        """Get a detailed validation report."""
        lines = [self.get_summary(), ""]
        
        for metric in self.metrics.values():
            status_icon = "[PASS]" if metric.passed else "[FAIL]"
            lines.append(f"{status_icon} {metric.explanation}")
            if metric.citation:
                lines.append(f"   Citation: {metric.citation}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"[WARN] {warning}")
        
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"[ERROR] {error}")
        
        return "\n".join(lines)
