"""
Enhanced explanation system for AURORA preprocessing decisions.

Consolidated into a single explainer.py module for simplicity.
"""

# Import from consolidated explainer module
from .explainer import (
    # Data structures
    EnhancedExplanation,
    ExplanationSection,
    AlternativeExplanation,
    ImpactPrediction,
    StatisticalEvidence,
    ExplanationSeverity,
    # Template registry
    ExplanationTemplateRegistry,
    # New simple API
    Explainer
)

__all__ = [
    # Existing exports (backward compatible)
    'EnhancedExplanation',
    'ExplanationSection',
    'AlternativeExplanation',
    'ImpactPrediction',
    'StatisticalEvidence',
    'ExplanationSeverity',
    'ExplanationTemplateRegistry',
    # New export
    'Explainer'
]
