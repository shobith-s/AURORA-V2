"""
Enhanced explanation data structures for preprocessing decisions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


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
