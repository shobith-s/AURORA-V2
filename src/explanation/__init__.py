"""
Enhanced explanation system for AURORA preprocessing decisions.
"""

from .enhanced_explanation import (
    EnhancedExplanation,
    ExplanationSection,
    AlternativeExplanation,
    ImpactPrediction,
    StatisticalEvidence,
    ExplanationSeverity
)
from .explanation_templates import ExplanationTemplateRegistry

__all__ = [
    'EnhancedExplanation',
    'ExplanationSection',
    'AlternativeExplanation',
    'ImpactPrediction',
    'StatisticalEvidence',
    'ExplanationSeverity',
    'ExplanationTemplateRegistry'
]
