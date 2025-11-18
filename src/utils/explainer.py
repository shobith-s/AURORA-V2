"""
Explainability utilities for preprocessing decisions.
Provides human-readable explanations and visualizations.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ..core.actions import PreprocessingAction


@dataclass
class Explanation:
    """Structured explanation for a preprocessing decision."""

    action: PreprocessingAction
    confidence: float
    primary_reason: str
    supporting_evidence: List[str]
    alternatives: List[Tuple[PreprocessingAction, float, str]]
    statistics: Dict[str, Any]
    rule_name: Optional[str] = None
    source: str = "symbolic"  # 'symbolic', 'neural', 'learned', 'blended'


class PreprocessingExplainer:
    """Generate human-readable explanations for preprocessing decisions."""

    def __init__(self, detail_level: str = 'medium'):
        """
        Initialize explainer.

        Args:
            detail_level: 'simple', 'medium', or 'detailed'
        """
        self.detail_level = detail_level

    def explain_action(
        self,
        action: PreprocessingAction,
        confidence: float,
        column_stats: Dict[str, Any],
        rule_name: Optional[str] = None,
        alternatives: Optional[List[Tuple[PreprocessingAction, float]]] = None
    ) -> Explanation:
        """
        Generate explanation for a preprocessing action.

        Args:
            action: Recommended action
            confidence: Confidence score
            column_stats: Column statistics
            rule_name: Name of the rule that triggered
            alternatives: Alternative actions with scores

        Returns:
            Structured explanation
        """
        # Generate primary reason
        primary_reason = self._generate_primary_reason(action, column_stats)

        # Collect supporting evidence
        evidence = self._collect_evidence(action, column_stats)

        # Format alternatives
        formatted_alternatives = []
        if alternatives:
            for alt_action, alt_score in alternatives:
                reason = self._generate_primary_reason(alt_action, column_stats)
                formatted_alternatives.append((alt_action, alt_score, reason))

        # Extract relevant statistics
        relevant_stats = self._extract_relevant_stats(action, column_stats)

        return Explanation(
            action=action,
            confidence=confidence,
            primary_reason=primary_reason,
            supporting_evidence=evidence,
            alternatives=formatted_alternatives,
            statistics=relevant_stats,
            rule_name=rule_name
        )

    def _generate_primary_reason(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> str:
        """Generate the primary reason for an action."""
        null_pct = stats.get('null_pct', 0)
        skewness = stats.get('skewness', 0)
        unique_ratio = stats.get('unique_ratio', 0)
        is_numeric = stats.get('is_numeric', False)
        outlier_pct = stats.get('outlier_pct', 0)

        if action == PreprocessingAction.KEEP:
            return "Column appears clean and well-distributed"

        elif action == PreprocessingAction.REMOVE_COLUMN:
            if null_pct > 0.9:
                return f"Column is {null_pct*100:.1f}% missing values"
            elif unique_ratio > 0.99:
                return "Column contains mostly unique values (likely an ID)"
            else:
                return "Column does not contribute useful information"

        elif action == PreprocessingAction.FILL_MEAN:
            return f"Column has {null_pct*100:.1f}% missing values with normal distribution"

        elif action == PreprocessingAction.FILL_MEDIAN:
            if abs(skewness) > 1:
                return f"Column has {null_pct*100:.1f}% missing values and is skewed (skewness={skewness:.2f})"
            else:
                return f"Column has {null_pct*100:.1f}% missing values"

        elif action == PreprocessingAction.FILL_MODE:
            return f"Categorical column with {null_pct*100:.1f}% missing values"

        elif action == PreprocessingAction.FILL_FORWARD:
            return "Time series data with sequential dependencies"

        elif action == PreprocessingAction.NORMALIZE:
            return "Numerical data requires scaling for model training"

        elif action == PreprocessingAction.STANDARDIZE:
            if abs(skewness) < 0.5:
                return "Data is normally distributed and benefits from standardization"
            else:
                return "Data requires zero-mean unit-variance scaling"

        elif action == PreprocessingAction.LOG_TRANSFORM:
            return f"Data is highly skewed (skewness={skewness:.2f}) and requires transformation"

        elif action == PreprocessingAction.REMOVE_OUTLIERS:
            return f"Column contains {outlier_pct*100:.1f}% outliers that may affect model"

        elif action == PreprocessingAction.ENCODE_ONEHOT:
            if unique_ratio < 0.05:
                return f"Low cardinality categorical column ({int(stats.get('unique_count', 0))} unique values)"
            else:
                return "Categorical column suitable for one-hot encoding"

        elif action == PreprocessingAction.ENCODE_LABEL:
            return "Ordinal categorical column with natural ordering"

        else:
            return f"Action recommended: {action.value}"

    def _collect_evidence(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> List[str]:
        """Collect supporting evidence for a decision."""
        evidence = []

        null_pct = stats.get('null_pct', 0)
        skewness = stats.get('skewness', 0)
        unique_ratio = stats.get('unique_ratio', 0)
        outlier_pct = stats.get('outlier_pct', 0)
        is_numeric = stats.get('is_numeric', False)

        # Evidence for missing value handling
        if null_pct > 0:
            evidence.append(f"Missing values: {null_pct*100:.1f}%")

        # Evidence for distribution shape
        if is_numeric:
            if abs(skewness) > 1:
                evidence.append(f"Highly skewed distribution (skewness={skewness:.2f})")
            elif abs(skewness) > 0.5:
                evidence.append(f"Moderately skewed (skewness={skewness:.2f})")
            else:
                evidence.append(f"Nearly normal distribution (skewness={skewness:.2f})")

        # Evidence for cardinality
        if unique_ratio > 0.95:
            evidence.append("Very high cardinality (likely unique identifier)")
        elif unique_ratio < 0.05:
            evidence.append(f"Low cardinality ({int(stats.get('unique_count', 0))} unique values)")

        # Evidence for outliers
        if outlier_pct > 0.05:
            evidence.append(f"Contains {outlier_pct*100:.1f}% outliers")

        # Evidence for data type patterns
        if stats.get('matches_date_pattern', 0) > 0.8:
            evidence.append("Contains date/time patterns")
        if stats.get('has_currency_symbols', False):
            evidence.append("Contains currency symbols")
        if stats.get('matches_email_pattern', 0) > 0.8:
            evidence.append("Contains email addresses")

        # Statistical properties
        if is_numeric:
            if 'mean' in stats and 'std' in stats:
                evidence.append(f"Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
            if 'min' in stats and 'max' in stats:
                evidence.append(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

        return evidence[:5] if self.detail_level != 'detailed' else evidence

    def _extract_relevant_stats(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract statistics most relevant to the action."""
        relevant = {}

        # Always include basic info
        relevant['data_type'] = 'numeric' if stats.get('is_numeric') else 'categorical'
        relevant['row_count'] = stats.get('count', 0)

        # Action-specific stats
        if action in [PreprocessingAction.FILL_MEAN, PreprocessingAction.FILL_MEDIAN,
                      PreprocessingAction.FILL_MODE, PreprocessingAction.FILL_FORWARD]:
            relevant['null_pct'] = stats.get('null_pct', 0)
            relevant['null_count'] = stats.get('null_count', 0)

        if action in [PreprocessingAction.LOG_TRANSFORM, PreprocessingAction.STANDARDIZE]:
            relevant['skewness'] = stats.get('skewness', 0)
            relevant['kurtosis'] = stats.get('kurtosis', 0)

        if action == PreprocessingAction.REMOVE_OUTLIERS:
            relevant['outlier_pct'] = stats.get('outlier_pct', 0)
            relevant['outlier_count'] = stats.get('outlier_count', 0)

        if action in [PreprocessingAction.ENCODE_ONEHOT, PreprocessingAction.ENCODE_LABEL]:
            relevant['unique_count'] = stats.get('unique_count', 0)
            relevant['unique_ratio'] = stats.get('unique_ratio', 0)

        if action in [PreprocessingAction.NORMALIZE, PreprocessingAction.STANDARDIZE]:
            relevant['min'] = stats.get('min', 0)
            relevant['max'] = stats.get('max', 0)
            relevant['mean'] = stats.get('mean', 0)
            relevant['std'] = stats.get('std', 0)

        return relevant

    def format_explanation(self, explanation: Explanation) -> str:
        """
        Format explanation as human-readable text.

        Args:
            explanation: Structured explanation

        Returns:
            Formatted text
        """
        if self.detail_level == 'simple':
            return self._format_simple(explanation)
        elif self.detail_level == 'detailed':
            return self._format_detailed(explanation)
        else:
            return self._format_medium(explanation)

    def _format_simple(self, exp: Explanation) -> str:
        """Format simple explanation."""
        return f"{exp.action.value}: {exp.primary_reason}"

    def _format_medium(self, exp: Explanation) -> str:
        """Format medium-detail explanation."""
        lines = [
            f"Action: {exp.action.value}",
            f"Confidence: {exp.confidence*100:.1f}%",
            f"Reason: {exp.primary_reason}"
        ]

        if exp.supporting_evidence:
            lines.append("\nSupporting Evidence:")
            for evidence in exp.supporting_evidence[:3]:
                lines.append(f"  - {evidence}")

        return "\n".join(lines)

    def _format_detailed(self, exp: Explanation) -> str:
        """Format detailed explanation."""
        lines = [
            "=" * 60,
            f"Preprocessing Recommendation: {exp.action.value}",
            "=" * 60,
            f"Confidence: {exp.confidence*100:.1f}%",
            f"Source: {exp.source}",
        ]

        if exp.rule_name:
            lines.append(f"Rule: {exp.rule_name}")

        lines.append(f"\nPrimary Reason:\n  {exp.primary_reason}")

        if exp.supporting_evidence:
            lines.append("\nSupporting Evidence:")
            for evidence in exp.supporting_evidence:
                lines.append(f"  - {evidence}")

        if exp.alternatives:
            lines.append("\nAlternative Actions:")
            for action, score, reason in exp.alternatives[:3]:
                lines.append(f"  {action.value} ({score*100:.1f}%): {reason}")

        if exp.statistics:
            lines.append("\nRelevant Statistics:")
            for key, value in exp.statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def compare_actions(
        self,
        actions: List[Tuple[PreprocessingAction, float]],
        column_stats: Dict[str, Any]
    ) -> str:
        """
        Compare multiple action recommendations.

        Args:
            actions: List of (action, confidence) tuples
            column_stats: Column statistics

        Returns:
            Comparison text
        """
        lines = ["Action Comparison:", ""]

        sorted_actions = sorted(actions, key=lambda x: x[1], reverse=True)

        for i, (action, confidence) in enumerate(sorted_actions, 1):
            reason = self._generate_primary_reason(action, column_stats)
            lines.append(f"{i}. {action.value} ({confidence*100:.1f}%)")
            lines.append(f"   {reason}")
            lines.append("")

        return "\n".join(lines)


def generate_decision_tree_explanation(
    column_stats: Dict[str, Any],
    action: PreprocessingAction
) -> str:
    """
    Generate a decision tree-style explanation.

    Args:
        column_stats: Column statistics
        action: Chosen action

    Returns:
        Decision tree explanation
    """
    lines = ["Decision Path:"]

    # Check data type
    if column_stats.get('is_numeric'):
        lines.append("   Column is numeric")

        # Check for missing values
        null_pct = column_stats.get('null_pct', 0)
        if null_pct > 0.5:
            lines.append(f"   High missing rate ({null_pct*100:.1f}%)")
            lines.append("  ’ Recommend: REMOVE_COLUMN or FILL")
        elif null_pct > 0:
            lines.append(f"   Some missing values ({null_pct*100:.1f}%)")
            skewness = column_stats.get('skewness', 0)
            if abs(skewness) > 1:
                lines.append(f"   Skewed distribution (skew={skewness:.2f})")
                lines.append("  ’ Recommend: FILL_MEDIAN")
            else:
                lines.append(f"   Normal distribution (skew={skewness:.2f})")
                lines.append("  ’ Recommend: FILL_MEAN")
    else:
        lines.append("   Column is categorical")
        null_pct = column_stats.get('null_pct', 0)
        if null_pct > 0:
            lines.append(f"   Missing values present ({null_pct*100:.1f}%)")
            lines.append("  ’ Recommend: FILL_MODE")

    lines.append(f"\nFinal Action: {action.value}")

    return "\n".join(lines)


def create_explanation_report(
    preprocessing_results: List[Dict[str, Any]],
    detail_level: str = 'medium'
) -> str:
    """
    Create a comprehensive explanation report for all preprocessing decisions.

    Args:
        preprocessing_results: List of preprocessing results
        detail_level: Level of detail

    Returns:
        Formatted report
    """
    explainer = PreprocessingExplainer(detail_level=detail_level)

    lines = [
        "PREPROCESSING RECOMMENDATIONS REPORT",
        "=" * 70,
        ""
    ]

    for i, result in enumerate(preprocessing_results, 1):
        column_name = result.get('column_name', f'Column_{i}')
        action = result.get('action')
        confidence = result.get('confidence', 0)
        stats = result.get('statistics', {})

        lines.append(f"Column {i}: {column_name}")
        lines.append("-" * 70)

        if action:
            explanation = explainer.explain_action(action, confidence, stats)
            lines.append(explainer.format_explanation(explanation))
        else:
            lines.append("No recommendation available")

        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Total Columns Analyzed: {len(preprocessing_results)}")

    return "\n".join(lines)
