"""
Explanation Engine - Generates rich, detailed explanations for preprocessing decisions.

This engine transforms simple symbolic/neural decisions into comprehensive explanations
with scientific justification, alternatives, impact predictions, and best practices.
"""

from typing import Dict, Any, Optional
from ..core.actions import PreprocessingAction, PreprocessingResult
from .enhanced_explanation import EnhancedExplanation, ExplanationSection, ExplanationSeverity
from .explanation_templates import ExplanationTemplateRegistry


class ExplanationEngine:
    """
    Main explanation engine that generates world-class explanations for preprocessing decisions.
    """

    def __init__(self):
        self.template_registry = ExplanationTemplateRegistry()

    def generate_enhanced_explanation(
        self,
        preprocessing_result: PreprocessingResult,
        column_stats: Dict[str, Any]
    ) -> EnhancedExplanation:
        """
        Generate a rich, enhanced explanation from a preprocessing decision.

        Args:
            preprocessing_result: The original preprocessing decision
            column_stats: Statistics about the column

        Returns:
            EnhancedExplanation with rich details, alternatives, impact predictions, etc.
        """
        action = preprocessing_result.action

        # Get template-based explanation if available
        explanation_data = self._get_template_explanation(action, column_stats)

        if explanation_data:
            # Use template
            enhanced = EnhancedExplanation(
                action=action.value,
                confidence=preprocessing_result.confidence,
                why_this_action=explanation_data["why_section"],
                statistical_evidence=explanation_data["statistical_evidence"],
                alternatives_not_chosen=explanation_data["alternatives"],
                impact_prediction=explanation_data["impact_prediction"],
                risks_and_warnings=explanation_data.get("risks_warnings", []),
                best_practices=explanation_data.get("best_practices", []),
                scientific_references=explanation_data.get("scientific_references", []),
                what_if_scenarios=explanation_data.get("what_if_scenarios", {})
            )
        else:
            # Fallback: Generate generic explanation
            enhanced = self._generate_generic_explanation(
                preprocessing_result,
                column_stats
            )

        # Calculate explanation quality scores
        enhanced.completeness_score = self._calculate_completeness(enhanced)
        enhanced.stakeholder_readability_score = self._calculate_readability(enhanced)
        enhanced.audit_trail_quality = self._calculate_audit_quality(enhanced)

        return enhanced

    def _get_template_explanation(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get template-based explanation if available."""
        template_map = {
            PreprocessingAction.LOG_TRANSFORM: self.template_registry.get_log_transform_explanation,
            PreprocessingAction.LOG1P_TRANSFORM: self.template_registry.get_log_transform_explanation,
            PreprocessingAction.STANDARD_SCALE: self.template_registry.get_standard_scale_explanation,
            PreprocessingAction.DROP_COLUMN: self.template_registry.get_drop_column_explanation,
            PreprocessingAction.DROP_IF_MOSTLY_NULL: self.template_registry.get_drop_column_explanation,
            PreprocessingAction.DROP_IF_CONSTANT: self.template_registry.get_drop_column_explanation,
            PreprocessingAction.DROP_IF_ALL_UNIQUE: self.template_registry.get_drop_column_explanation,
            PreprocessingAction.ONEHOT_ENCODE: self.template_registry.get_onehot_encode_explanation,
        }

        template_fn = template_map.get(action)
        if template_fn:
            return template_fn(stats)
        return None

    def _generate_generic_explanation(
        self,
        result: PreprocessingResult,
        stats: Dict[str, Any]
    ) -> EnhancedExplanation:
        """Generate a generic explanation when no template exists."""
        from .enhanced_explanation import (
            AlternativeExplanation,
            StatisticalEvidence,
            ImpactPrediction
        )

        why_section = ExplanationSection(
            title=f"Why {result.action.value}",
            content=result.explanation or f"Applied {result.action.value} based on column statistics",
            severity=ExplanationSeverity.INFO
        )

        statistical_evidence = StatisticalEvidence(
            key_statistics={k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
            thresholds_met=["Decision based on symbolic rules or neural oracle"],
            distribution_characteristics={}
        )

        alternatives = [
            AlternativeExplanation(
                action=alt_action.value,
                confidence=alt_conf,
                reason_not_chosen=f"Lower confidence ({alt_conf:.1%}) than chosen action ({result.confidence:.1%})",
                pros=["Alternative approach"],
                cons=[f"Lower confidence score"],
                when_to_use="Consider when circumstances differ"
            )
            for alt_action, alt_conf in result.alternatives[:3]
        ]

        impact_prediction = ImpactPrediction(
            expected_accuracy_change="Impact depends on downstream model",
            feature_importance_impact="May affect feature importance",
            interpretability_impact="Moderate",
            computational_cost="Varies by action",
            reversibility="Depends on specific transformation",
            data_loss="Varies by action"
        )

        return EnhancedExplanation(
            action=result.action.value,
            confidence=result.confidence,
            why_this_action=why_section,
            statistical_evidence=statistical_evidence,
            alternatives_not_chosen=alternatives,
            impact_prediction=impact_prediction,
            best_practices=[
                "Validate preprocessing on held-out data",
                "Document all transformations",
                "Monitor for distribution drift in production"
            ]
        )

    def _calculate_completeness(self, explanation: EnhancedExplanation) -> float:
        """
        Calculate completeness score (0-1) for the explanation.
        Higher score = more complete explanation.
        """
        score = 0.0

        # Core sections (60%)
        if explanation.why_this_action and len(explanation.why_this_action.content) > 50:
            score += 0.20
        if explanation.statistical_evidence and explanation.statistical_evidence.key_statistics:
            score += 0.20
        if explanation.alternatives_not_chosen and len(explanation.alternatives_not_chosen) >= 2:
            score += 0.20

        # Additional sections (40%)
        if explanation.impact_prediction:
            score += 0.15
        if explanation.best_practices and len(explanation.best_practices) >= 3:
            score += 0.10
        if explanation.scientific_references and len(explanation.scientific_references) >= 1:
            score += 0.10
        if explanation.what_if_scenarios and len(explanation.what_if_scenarios) >= 2:
            score += 0.05

        return min(1.0, score)

    def _calculate_readability(self, explanation: EnhancedExplanation) -> float:
        """
        Calculate stakeholder readability score (0-1).
        Higher score = more accessible to non-technical stakeholders.
        """
        score = 0.5  # Base score

        # Check for plain language
        content = explanation.why_this_action.content
        if content:
            # Penalty for overly technical language
            technical_terms = ['eigenvalue', 'orthogonal', 'hessian', 'jacobian']
            has_technical = any(term in content.lower() for term in technical_terms)
            if has_technical:
                score -= 0.1

            # Bonus for clear structure
            if '•' in content or '\n' in content:
                score += 0.1

            # Bonus for examples
            if 'example' in content.lower() or 'e.g.' in content.lower():
                score += 0.1

        # Bonus for what-if scenarios (make it concrete)
        if explanation.what_if_scenarios:
            score += 0.15

        # Bonus for visual evidence (if implemented)
        if explanation.statistical_evidence and explanation.statistical_evidence.distribution_characteristics:
            score += 0.15

        return min(1.0, max(0.0, score))

    def _calculate_audit_quality(self, explanation: EnhancedExplanation) -> float:
        """
        Calculate audit trail quality score (0-1).
        Higher score = better for regulatory compliance and auditing.
        """
        score = 0.0

        # Statistical evidence (30%)
        if explanation.statistical_evidence:
            if explanation.statistical_evidence.key_statistics:
                score += 0.15
            if explanation.statistical_evidence.thresholds_met:
                score += 0.15

        # Scientific references (25%)
        if explanation.scientific_references:
            score += 0.25

        # Alternatives considered (20%)
        if explanation.alternatives_not_chosen and len(explanation.alternatives_not_chosen) >= 2:
            score += 0.20

        # Risk documentation (15%)
        if explanation.risks_and_warnings:
            score += 0.15

        # Traceability (10%)
        if explanation.impact_prediction:
            score += 0.10

        return min(1.0, score)

    def explain_decision_comparison(
        self,
        chosen_explanation: EnhancedExplanation,
        alternative_action: PreprocessingAction,
        alternative_stats: Dict[str, Any]
    ) -> str:
        """
        Generate a side-by-side comparison of chosen action vs an alternative.
        Useful for "what if" analysis.
        """
        comparison = f"# Comparison: {chosen_explanation.action} vs {alternative_action.value}\n\n"

        comparison += f"## Chosen: {chosen_explanation.action}\n"
        comparison += f"- **Confidence:** {chosen_explanation.confidence:.1%}\n"
        comparison += f"- **Expected Impact:** {chosen_explanation.impact_prediction.expected_accuracy_change}\n"
        comparison += f"- **Risks:** {len(chosen_explanation.risks_and_warnings)} warnings\n\n"

        # Find the alternative in the alternatives list
        alt_detail = next(
            (alt for alt in chosen_explanation.alternatives_not_chosen
             if alt.action == alternative_action.value),
            None
        )

        if alt_detail:
            comparison += f"## Alternative: {alternative_action.value}\n"
            comparison += f"- **Confidence:** {alt_detail.confidence:.1%}\n"
            comparison += f"- **Why not chosen:** {alt_detail.reason_not_chosen}\n"
            comparison += f"- **When to use:** {alt_detail.when_to_use}\n\n"

            comparison += "### Trade-offs\n\n"
            comparison += f"**{chosen_explanation.action} Advantages:**\n"
            for con in alt_detail.cons:
                comparison += f"- ✓ {con}\n"

            comparison += f"\n**{alternative_action.value} Advantages:**\n"
            for pro in alt_detail.pros:
                comparison += f"- ✓ {pro}\n"

        return comparison
