"""
Unified explanation system for AURORA preprocessing decisions.

This module consolidates all explanation functionality into a single file:
- Data structures for rich explanations
- Template registry for generating explanations
- Simple API for ease of use

Consolidated from:
- enhanced_explanation.py (data structures)
- explanation_templates.py (template registry)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# ============================================================================
# SECTION 1: Data Structures
# ============================================================================

class ExplanationSeverity(Enum):
    """Severity level of an explanation."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class StatisticalEvidence:
    """Statistical evidence supporting a decision."""
    
    metric: str
    value: float
    threshold: Optional[float] = None
    comparison: str = ""  # e.g., "greater than", "less than"
    
    def __str__(self) -> str:
        """Format as human-readable string."""
        if self.threshold is not None:
            return f"{self.metric}: {self.value:.2f} ({self.comparison} threshold {self.threshold:.2f})"
        return f"{self.metric}: {self.value:.2f}"


@dataclass
class ExplanationSection:
    """A section of an explanation."""
    
    title: str
    content: str
    evidence: List[StatisticalEvidence] = field(default_factory=list)
    severity: ExplanationSeverity = ExplanationSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'content': self.content,
            'evidence': [
                {
                    'metric': e.metric,
                    'value': e.value,
                    'threshold': e.threshold,
                    'comparison': e.comparison
                }
                for e in self.evidence
            ],
            'severity': self.severity.value
        }


@dataclass
class ImpactPrediction:
    """Predicted impact of an action."""
    
    metric: str
    expected_change: str
    confidence: float
    
    def __str__(self) -> str:
        """Format as human-readable string."""
        return f"{self.metric}: {self.expected_change} (confidence: {self.confidence:.0%})"


@dataclass
class AlternativeExplanation:
    """Alternative action and explanation."""
    
    action: str
    reason: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action,
            'reason': self.reason,
            'pros': self.pros,
            'cons': self.cons
        }


@dataclass
class EnhancedExplanation:
    """
    Comprehensive explanation for a preprocessing decision.
    
    Includes:
    - Why the action was chosen
    - Statistical evidence
    - Predicted impact
    - Alternative actions
    """
    
    action: str
    confidence: float
    why_section: ExplanationSection
    evidence_section: ExplanationSection
    impact_section: ExplanationSection
    alternatives: List[AlternativeExplanation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action,
            'confidence': self.confidence,
            'why_section': self.why_section.to_dict(),
            'evidence_section': self.evidence_section.to_dict(),
            'impact_section': self.impact_section.to_dict(),
            'alternatives': [alt.to_dict() for alt in self.alternatives],
            'metadata': self.metadata
        }
    
    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = [
            f"# {self.action} (Confidence: {self.confidence:.0%})",
            "",
            f"## {self.why_section.title}",
            self.why_section.content,
            ""
        ]
        
        if self.why_section.evidence:
            lines.append("**Evidence:**")
            for evidence in self.why_section.evidence:
                lines.append(f"- {evidence}")
            lines.append("")
        
        lines.extend([
            f"## {self.impact_section.title}",
            self.impact_section.content,
            ""
        ])
        
        if self.alternatives:
            lines.append("## Alternatives")
            for alt in self.alternatives:
                lines.extend([
                    f"### {alt.action}",
                    alt.reason,
                    ""
                ])
                if alt.pros:
                    lines.append("**Pros:**")
                    for pro in alt.pros:
                        lines.append(f"- {pro}")
                    lines.append("")
                if alt.cons:
                    lines.append("**Cons:**")
                    for con in alt.cons:
                        lines.append(f"- {con}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Format as HTML."""
        html_parts = [
            f"<div class='explanation'>",
            f"<h1>{self.action} <span class='confidence'>(Confidence: {self.confidence:.0%})</span></h1>",
            f"<div class='why-section'>",
            f"<h2>{self.why_section.title}</h2>",
            f"<p>{self.why_section.content}</p>",
        ]
        
        if self.why_section.evidence:
            html_parts.append("<ul class='evidence'>")
            for evidence in self.why_section.evidence:
                html_parts.append(f"<li>{evidence}</li>")
            html_parts.append("</ul>")
        
        html_parts.extend([
            "</div>",
            f"<div class='impact-section'>",
            f"<h2>{self.impact_section.title}</h2>",
            f"<p>{self.impact_section.content}</p>",
            "</div>"
        ])
        
        if self.alternatives:
            html_parts.append("<div class='alternatives'>")
            html_parts.append("<h2>Alternatives</h2>")
            for alt in self.alternatives:
                html_parts.append(f"<div class='alternative'>")
                html_parts.append(f"<h3>{alt.action}</h3>")
                html_parts.append(f"<p>{alt.reason}</p>")
                
                if alt.pros:
                    html_parts.append("<h4>Pros:</h4><ul>")
                    for pro in alt.pros:
                        html_parts.append(f"<li>{pro}</li>")
                    html_parts.append("</ul>")
                
                if alt.cons:
                    html_parts.append("<h4>Cons:</h4><ul>")
                    for con in alt.cons:
                        html_parts.append(f"<li>{con}</li>")
                    html_parts.append("</ul>")
                
                html_parts.append("</div>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        return "\n".join(html_parts)


# ============================================================================
# SECTION 2: Template Registry
# ============================================================================

class ExplanationTemplateRegistry:
    """Registry of explanation templates for different preprocessing actions."""
    
    def get_log_transform_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for log transform."""
        skewness = stats.get('skewness', 0)
        min_val = stats.get('min_value', 0)
        max_val = stats.get('max_value', 0)
        
        # Why section
        why_section = ExplanationSection(
            title="Why Log Transform?",
            content=(
                "The data is highly skewed, meaning most values cluster at one end "
                "of the distribution. Log transformation will normalize the distribution, "
                "making it more suitable for machine learning algorithms."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Skewness",
                    value=skewness,
                    threshold=1.5,
                    comparison="greater than"
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        # Evidence section
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The data ranges from {min_val:.2f} to {max_val:.2f} with high skewness ({skewness:.2f}). "
                "This indicates an exponential or power-law distribution."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        # Impact section
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "After log transformation, the data will have a more normal distribution. "
                "This will improve model performance and make relationships more linear."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        # Alternatives
        alternatives = [
            AlternativeExplanation(
                action="Square Root Transform",
                reason="Less aggressive than log transform",
                pros=["Handles moderate skewness", "Preserves zero values"],
                cons=["Less effective for high skewness"]
            ),
            AlternativeExplanation(
                action="Standard Scaling",
                reason="Simple normalization without transformation",
                pros=["Preserves distribution shape", "Fast"],
                cons=["Doesn't address skewness"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_standard_scale_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for standard scaling."""
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        min_val = stats.get('min_value', 0)
        max_val = stats.get('max_value', 0)
        
        why_section = ExplanationSection(
            title="Why Standard Scaling?",
            content=(
                "The data has a reasonable distribution but needs to be normalized "
                "to have mean=0 and standard deviation=1. This is required for many "
                "machine learning algorithms."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Mean",
                    value=mean,
                    threshold=None,
                    comparison=""
                ),
                StatisticalEvidence(
                    metric="Std Dev",
                    value=std,
                    threshold=None,
                    comparison=""
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The data ranges from {min_val:.2f} to {max_val:.2f} with mean {mean:.2f} "
                f"and standard deviation {std:.2f}."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "All features will be on the same scale, preventing features with "
                "large values from dominating the model."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        alternatives = [
            AlternativeExplanation(
                action="MinMax Scaling",
                reason="Scale to range [0, 1]",
                pros=["Bounded output", "Preserves zero values"],
                cons=["Sensitive to outliers"]
            ),
            AlternativeExplanation(
                action="Robust Scaling",
                reason="Scale using median and IQR",
                pros=["Resistant to outliers"],
                cons=["May not center at zero"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_drop_column_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for dropping a column."""
        null_pct = stats.get('null_pct', 0) * 100
        unique_count = stats.get('unique_count', 0)
        row_count = stats.get('row_count', 1)
        
        why_section = ExplanationSection(
            title="Why Drop This Column?",
            content=(
                "This column has significant data quality issues that make it unsuitable "
                "for analysis or modeling."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Null Percentage",
                    value=null_pct,
                    threshold=80.0,
                    comparison="greater than"
                )
            ],
            severity=ExplanationSeverity.WARNING
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"{null_pct:.1f}% of values are missing. "
                f"Only {unique_count} unique values out of {row_count} rows."
            ),
            evidence=[],
            severity=ExplanationSeverity.WARNING
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "Removing this column will reduce noise and improve model performance. "
                "The missing data would require imputation which could introduce bias."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        alternatives = [
            AlternativeExplanation(
                action="Fill Nulls",
                reason="Impute missing values",
                pros=["Preserves column"],
                cons=["May introduce bias", "Reduces data quality"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_onehot_encode_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for one-hot encoding."""
        cardinality = stats.get('cardinality', 0)
        
        why_section = ExplanationSection(
            title="Why One-Hot Encoding?",
            content=(
                "This is a categorical column with moderate cardinality. "
                "One-hot encoding will convert it to binary features suitable for ML models."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Cardinality",
                    value=cardinality,
                    threshold=10,
                    comparison="less than"
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The column has {cardinality} unique categories. "
                "This is suitable for one-hot encoding."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                f"This will create {cardinality} new binary columns. "
                "The categorical relationships will be preserved without imposing order."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        alternatives = [
            AlternativeExplanation(
                action="Label Encoding",
                reason="Convert to integers",
                pros=["Fewer columns", "Faster"],
                cons=["Implies ordering that may not exist"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }


# ============================================================================
# SECTION 3: Simple API (NEW)
# ============================================================================

class Explainer:
    """
    Simple unified interface for generating explanations.
    
    Usage:
        explainer = Explainer()
        explanation = explainer.explain(result, detail_level="detailed")
    """
    
    def __init__(self):
        """Initialize the explainer with template registry."""
        self.registry = ExplanationTemplateRegistry()
    
    def explain(
        self,
        result,
        detail_level: str = "basic",  # "basic" or "detailed"
        format: str = "text"  # "text", "markdown", "html", "dict"
    ) -> str:
        """
        Generate explanation for a preprocessing result.
        
        Args:
            result: PreprocessingResult object with action, confidence, explanation
            detail_level: "basic" (1-2 sentences) or "detailed" (full breakdown)
            format: Output format ("text", "markdown", "html", "dict")
        
        Returns:
            Formatted explanation string
        """
        if detail_level == "basic":
            return self._basic_explanation(result)
        else:
            return self._detailed_explanation(result, format)
    
    def _basic_explanation(self, result) -> str:
        """Generate simple 1-2 sentence explanation."""
        action = getattr(result, 'action', 'unknown')
        explanation = getattr(result, 'explanation', 'No explanation available')
        confidence = getattr(result, 'confidence', 0.0)
        return f"{action}: {explanation} (confidence: {confidence:.0%})"
    
    def _detailed_explanation(self, result, format: str):
        """
        Generate comprehensive explanation using templates.
        
        Note: This is a placeholder for future implementation.
        Full implementation would use the registry to generate EnhancedExplanation
        and format according to the requested format.
        """
        # For now, return basic explanation
        # TODO: Implement full template-based explanation generation
        return self._basic_explanation(result)
