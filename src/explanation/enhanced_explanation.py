"""
Enhanced explanation data structures for rich, detailed preprocessing explanations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ExplanationSeverity(Enum):
    """Severity level for explanation points."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"


@dataclass
class ExplanationSection:
    """A section of the explanation (why, why not, impact, risks, etc.)"""
    title: str
    content: str
    severity: ExplanationSeverity = ExplanationSeverity.INFO
    evidence: List[str] = field(default_factory=list)  # Scientific citations or statistical evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "severity": self.severity.value,
            "evidence": self.evidence
        }


@dataclass
class AlternativeExplanation:
    """Detailed explanation of why an alternative action was NOT chosen."""
    action: str
    confidence: float
    reason_not_chosen: str
    pros: List[str]
    cons: List[str]
    when_to_use: str  # When this alternative would be better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reason_not_chosen": self.reason_not_chosen,
            "pros": self.pros,
            "cons": self.cons,
            "when_to_use": self.when_to_use
        }


@dataclass
class ImpactPrediction:
    """Predicted impact of the preprocessing action on model performance."""
    expected_accuracy_change: str  # e.g., "+5-12%", "minimal", "significant"
    feature_importance_impact: str
    interpretability_impact: str
    computational_cost: str
    reversibility: str
    data_loss: str  # e.g., "none", "outliers removed", "10% of rows"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_accuracy_change": self.expected_accuracy_change,
            "feature_importance_impact": self.feature_importance_impact,
            "interpretability_impact": self.interpretability_impact,
            "computational_cost": self.computational_cost,
            "reversibility": self.reversibility,
            "data_loss": self.data_loss
        }


@dataclass
class StatisticalEvidence:
    """Statistical evidence supporting the decision."""
    key_statistics: Dict[str, float]  # {stat_name: value}
    thresholds_met: List[str]  # Which thresholds triggered this decision
    distribution_characteristics: Dict[str, str]  # {characteristic: description}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_statistics": self.key_statistics,
            "thresholds_met": self.thresholds_met,
            "distribution_characteristics": self.distribution_characteristics
        }


@dataclass
class EnhancedExplanation:
    """
    Rich, detailed explanation of a preprocessing decision.

    This goes far beyond "applied log_transform because skewness is high"
    to provide scientific justification, alternatives, impact predictions,
    and actionable insights.
    """

    # Core decision
    action: str
    confidence: float

    # Main explanation sections
    why_this_action: ExplanationSection
    statistical_evidence: StatisticalEvidence
    alternatives_not_chosen: List[AlternativeExplanation]
    impact_prediction: ImpactPrediction

    # Additional insights
    risks_and_warnings: List[ExplanationSection] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    scientific_references: List[str] = field(default_factory=list)

    # Counterfactual reasoning
    what_if_scenarios: Dict[str, str] = field(default_factory=dict)  # {scenario: outcome}

    # Explanation quality metadata
    completeness_score: float = 0.0  # 0-1, how complete is this explanation
    stakeholder_readability_score: float = 0.0  # 0-1, how readable for non-technical
    audit_trail_quality: float = 0.0  # 0-1, how suitable for regulatory audit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "why_this_action": self.why_this_action.to_dict(),
            "statistical_evidence": self.statistical_evidence.to_dict(),
            "alternatives_not_chosen": [alt.to_dict() for alt in self.alternatives_not_chosen],
            "impact_prediction": self.impact_prediction.to_dict(),
            "risks_and_warnings": [r.to_dict() for r in self.risks_and_warnings],
            "best_practices": self.best_practices,
            "scientific_references": self.scientific_references,
            "what_if_scenarios": self.what_if_scenarios,
            "quality_scores": {
                "completeness": self.completeness_score,
                "stakeholder_readability": self.stakeholder_readability_score,
                "audit_trail_quality": self.audit_trail_quality
            }
        }

    def to_markdown(self) -> str:
        """Generate a markdown report of the explanation."""
        md = f"# Preprocessing Decision Explanation\n\n"
        md += f"**Action:** `{self.action}`  \n"
        md += f"**Confidence:** {self.confidence:.1%}\n\n"

        md += f"## {self.why_this_action.title}\n\n"
        md += f"{self.why_this_action.content}\n\n"

        if self.why_this_action.evidence:
            md += "**Evidence:**\n"
            for ev in self.why_this_action.evidence:
                md += f"- {ev}\n"
            md += "\n"

        md += "## Statistical Evidence\n\n"
        md += "**Key Statistics:**\n"
        for stat, value in self.statistical_evidence.key_statistics.items():
            md += f"- **{stat}**: {value:.4f}\n"
        md += "\n"

        if self.statistical_evidence.thresholds_met:
            md += "**Thresholds Met:**\n"
            for threshold in self.statistical_evidence.thresholds_met:
                md += f"- ✓ {threshold}\n"
            md += "\n"

        md += "## Alternatives Considered\n\n"
        for i, alt in enumerate(self.alternatives_not_chosen, 1):
            md += f"### {i}. {alt.action} (confidence: {alt.confidence:.1%})\n\n"
            md += f"**Why not chosen:** {alt.reason_not_chosen}\n\n"
            md += "**Pros:**\n"
            for pro in alt.pros:
                md += f"- ✓ {pro}\n"
            md += "\n**Cons:**\n"
            for con in alt.cons:
                md += f"- ✗ {con}\n"
            md += f"\n**When to use:** {alt.when_to_use}\n\n"

        md += "## Expected Impact\n\n"
        md += f"- **Model Accuracy:** {self.impact_prediction.expected_accuracy_change}\n"
        md += f"- **Feature Importance:** {self.impact_prediction.feature_importance_impact}\n"
        md += f"- **Interpretability:** {self.impact_prediction.interpretability_impact}\n"
        md += f"- **Computational Cost:** {self.impact_prediction.computational_cost}\n"
        md += f"- **Reversibility:** {self.impact_prediction.reversibility}\n"
        md += f"- **Data Loss:** {self.impact_prediction.data_loss}\n\n"

        if self.risks_and_warnings:
            md += "## ⚠️ Risks and Warnings\n\n"
            for risk in self.risks_and_warnings:
                md += f"### {risk.title}\n\n"
                md += f"{risk.content}\n\n"

        if self.best_practices:
            md += "## 💡 Best Practices\n\n"
            for practice in self.best_practices:
                md += f"- {practice}\n"
            md += "\n"

        if self.what_if_scenarios:
            md += "## 🤔 What-If Scenarios\n\n"
            for scenario, outcome in self.what_if_scenarios.items():
                md += f"**{scenario}**  \n{outcome}\n\n"

        if self.scientific_references:
            md += "## 📚 References\n\n"
            for i, ref in enumerate(self.scientific_references, 1):
                md += f"{i}. {ref}\n"
            md += "\n"

        md += "---\n\n"
        md += f"*Explanation Quality: Completeness={self.completeness_score:.0%}, "
        md += f"Readability={self.stakeholder_readability_score:.0%}, "
        md += f"Audit Trail={self.audit_trail_quality:.0%}*\n"

        return md

    def to_plain_text(self) -> str:
        """Generate a plain text summary for simple display."""
        text = f"Action: {self.action} (confidence: {self.confidence:.1%})\n\n"
        text += f"{self.why_this_action.content}\n\n"

        if self.alternatives_not_chosen:
            text += "Alternatives considered:\n"
            for alt in self.alternatives_not_chosen[:3]:  # Top 3
                text += f"  • {alt.action} ({alt.confidence:.0%}) - {alt.reason_not_chosen}\n"

        return text
