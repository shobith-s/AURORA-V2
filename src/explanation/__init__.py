"""
Explainability Engine for AURORA - World-class preprocessing explanations.

This module transforms simple preprocessing decisions into rich, scientifically-backed
explanations that help users understand WHY decisions were made and what alternatives exist.
"""

from .enhanced_explanation import EnhancedExplanation, ExplanationSection
from .explanation_engine import ExplanationEngine
from .counterfactual_analyzer import CounterfactualAnalyzer
from .explanation_templates import ExplanationTemplateRegistry

__all__ = [
    'EnhancedExplanation',
    'ExplanationSection',
    'ExplanationEngine',
    'CounterfactualAnalyzer',
    'ExplanationTemplateRegistry'
]
