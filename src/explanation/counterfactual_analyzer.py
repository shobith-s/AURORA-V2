"""
Counterfactual Analyzer - "What if" simulator for preprocessing decisions.

Allows users to explore: "What if I used a different action? What would happen?"
This is crucial for understanding trade-offs and building trust in decisions.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
from ..core.actions import PreprocessingAction


@dataclass
class CounterfactualScenario:
    """A 'what if' scenario with predicted outcomes."""
    scenario_description: str
    alternative_action: PreprocessingAction
    predicted_confidence: float
    expected_outcomes: Dict[str, str]  # {aspect: outcome description}
    trade_offs: Dict[str, str]  # {aspect: trade-off description}
    recommendation: str  # When to consider this alternative


class CounterfactualAnalyzer:
    """
    Analyzes "what if" scenarios for preprocessing decisions.

    Examples:
    - "What if the data had more outliers?"
    - "What if I used robust_scale instead?"
    - "What if test data has different distribution?"
    """

    def __init__(self):
        pass

    def simulate_alternative_action(
        self,
        current_action: PreprocessingAction,
        alternative_action: PreprocessingAction,
        column_stats: Dict[str, Any]
    ) -> CounterfactualScenario:
        """
        Simulate what would happen if we used alternative_action instead.

        Args:
            current_action: The action we chose
            alternative_action: The action we're simulating
            column_stats: Statistics about the column

        Returns:
            CounterfactualScenario describing the outcomes
        """
        # Build scenario description
        scenario_desc = f"What if we used {alternative_action.value} instead of {current_action.value}?"

        # Predict outcomes based on statistics
        expected_outcomes = self._predict_outcomes(
            alternative_action,
            column_stats
        )

        # Analyze trade-offs
        trade_offs = self._analyze_trade_offs(
            current_action,
            alternative_action,
            column_stats
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            alternative_action,
            column_stats,
            trade_offs
        )

        # Estimate confidence (heuristic)
        predicted_confidence = self._estimate_confidence(
            alternative_action,
            column_stats
        )

        return CounterfactualScenario(
            scenario_description=scenario_desc,
            alternative_action=alternative_action,
            predicted_confidence=predicted_confidence,
            expected_outcomes=expected_outcomes,
            trade_offs=trade_offs,
            recommendation=recommendation
        )

    def simulate_data_change(
        self,
        current_action: PreprocessingAction,
        current_stats: Dict[str, Any],
        data_change_description: str,
        modified_stats: Dict[str, Any]
    ) -> CounterfactualScenario:
        """
        Simulate what would happen if the data characteristics changed.

        Examples:
        - "What if skewness increased to 3.0?"
        - "What if null percentage was 80%?"
        - "What if cardinality doubled?"
        """
        scenario_desc = f"What if {data_change_description}?"

        # Determine what action would be recommended with modified stats
        # (This would call the symbolic engine, but we'll approximate here)
        new_action = self._predict_action_for_stats(modified_stats)

        expected_outcomes = {
            "Action Change": f"Would change from {current_action.value} to {new_action.value}" if new_action != current_action else "No change",
            "Reasoning": self._explain_action_for_stats(new_action, modified_stats),
            "Impact": "Decision adapts to new data characteristics"
        }

        trade_offs = {
            "Stability": "Different action would be used" if new_action != current_action else "Same action maintained",
            "Robustness": self._assess_robustness(current_action, new_action, data_change_description)
        }

        recommendation = (
            f"If data changes as described, consider using {new_action.value}. "
            f"Monitor your data for this type of drift in production."
        )

        return CounterfactualScenario(
            scenario_description=scenario_desc,
            alternative_action=new_action,
            predicted_confidence=self._estimate_confidence(new_action, modified_stats),
            expected_outcomes=expected_outcomes,
            trade_offs=trade_offs,
            recommendation=recommendation
        )

    def generate_sensitivity_analysis(
        self,
        action: PreprocessingAction,
        column_stats: Dict[str, Any]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Analyze sensitivity: How would decision change if key statistics changed?

        Returns:
        {
            "skewness": [(value, predicted_action), ...],
            "null_pct": [(value, predicted_action), ...],
            ...
        }
        """
        sensitivity = {}

        # Numeric data: test skewness sensitivity
        if column_stats.get('is_numeric'):
            skewness_tests = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            sensitivity['skewness'] = []
            for skew in skewness_tests:
                modified_stats = column_stats.copy()
                modified_stats['skewness'] = skew
                predicted_action = self._predict_action_for_stats(modified_stats)
                sensitivity['skewness'].append((
                    f"skewness={skew}",
                    predicted_action.value
                ))

            # Test outlier sensitivity
            outlier_tests = [0.0, 0.05, 0.1, 0.2, 0.4]
            sensitivity['outlier_pct'] = []
            for outlier_pct in outlier_tests:
                modified_stats = column_stats.copy()
                modified_stats['outlier_pct'] = outlier_pct
                modified_stats['has_outliers'] = outlier_pct > 0.05
                predicted_action = self._predict_action_for_stats(modified_stats)
                sensitivity['outlier_pct'].append((
                    f"outliers={outlier_pct:.0%}",
                    predicted_action.value
                ))

        # Categorical data: test cardinality sensitivity
        if column_stats.get('is_categorical'):
            cardinality_tests = [5, 10, 20, 50, 100, 500]
            sensitivity['cardinality'] = []
            for card in cardinality_tests:
                modified_stats = column_stats.copy()
                modified_stats['cardinality'] = card
                modified_stats['unique_count'] = card
                predicted_action = self._predict_action_for_stats(modified_stats)
                sensitivity['cardinality'].append((
                    f"cardinality={card}",
                    predicted_action.value
                ))

        # All data: test null sensitivity
        null_tests = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        sensitivity['null_pct'] = []
        for null_pct in null_tests:
            modified_stats = column_stats.copy()
            modified_stats['null_pct'] = null_pct
            predicted_action = self._predict_action_for_stats(modified_stats)
            sensitivity['null_pct'].append((
                f"nulls={null_pct:.0%}",
                predicted_action.value
            ))

        return sensitivity

    # --- Helper Methods ---

    def _predict_outcomes(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """Predict outcomes of using this action."""
        outcomes = {}

        # Distribution impact
        if action in [PreprocessingAction.LOG_TRANSFORM, PreprocessingAction.LOG1P_TRANSFORM]:
            outcomes["Distribution"] = "Reduces skewness, normalizes right-tailed distribution"
            outcomes["Scale"] = "Compresses large values, expands small values"
            outcomes["Interpretability"] = "Log scale - natural for multiplicative phenomena"
        elif action == PreprocessingAction.STANDARD_SCALE:
            outcomes["Distribution"] = "Preserves shape, centers and scales"
            outcomes["Scale"] = "Mean=0, Std=1"
            outcomes["Interpretability"] = "Z-scores - standard deviations from mean"
        elif action == PreprocessingAction.ROBUST_SCALE:
            outcomes["Distribution"] = "Preserves shape, uses median/IQR"
            outcomes["Scale"] = "Median=0, IQR-based"
            outcomes["Interpretability"] = "Robust to outliers"
        elif action in [PreprocessingAction.ONEHOT_ENCODE, PreprocessingAction.LABEL_ENCODE]:
            card = stats.get('cardinality', 0)
            if action == PreprocessingAction.ONEHOT_ENCODE:
                outcomes["Features"] = f"Creates {card} binary features"
                outcomes["Sparsity"] = "Sparse representation (mostly zeros)"
                outcomes["Interpretability"] = "Each feature = presence of category"
            else:
                outcomes["Features"] = "Creates 1 numeric feature"
                outcomes["Ordering"] = "Imposes artificial ordering on categories"
                outcomes["Interpretability"] = "Integer encoding, order may be misleading"

        # Model compatibility
        outcomes["Model Compatibility"] = self._assess_model_compatibility(action)

        return outcomes

    def _analyze_trade_offs(
        self,
        current: PreprocessingAction,
        alternative: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """Analyze trade-offs between current and alternative."""
        trade_offs = {}

        # Accuracy trade-off
        trade_offs["Accuracy"] = self._compare_accuracy_impact(current, alternative, stats)

        # Interpretability trade-off
        trade_offs["Interpretability"] = self._compare_interpretability(current, alternative)

        # Computational cost trade-off
        trade_offs["Computational Cost"] = self._compare_cost(current, alternative, stats)

        # Robustness trade-off
        trade_offs["Robustness"] = self._compare_robustness(current, alternative)

        return trade_offs

    def _predict_action_for_stats(self, stats: Dict[str, Any]) -> PreprocessingAction:
        """Predict what action would be recommended for given statistics (heuristic)."""

        # Check for drop conditions first
        if stats.get('null_pct', 0) > 0.6:
            return PreprocessingAction.DROP_COLUMN
        if stats.get('unique_count', 2) == 1:
            return PreprocessingAction.DROP_COLUMN
        if stats.get('unique_ratio', 0) > 0.95:
            return PreprocessingAction.DROP_COLUMN

        # Numeric data
        if stats.get('is_numeric', False):
            skewness = abs(stats.get('skewness', 0) or 0)
            has_outliers = stats.get('has_outliers', False)

            if skewness > 1.5:
                return PreprocessingAction.LOG_TRANSFORM
            elif has_outliers:
                return PreprocessingAction.ROBUST_SCALE
            else:
                return PreprocessingAction.STANDARD_SCALE

        # Categorical data
        if stats.get('is_categorical', False):
            cardinality = stats.get('cardinality', 0)
            if cardinality < 50:
                return PreprocessingAction.ONEHOT_ENCODE
            else:
                return PreprocessingAction.FREQUENCY_ENCODE

        # Default
        return PreprocessingAction.KEEP_AS_IS

    def _explain_action_for_stats(self, action: PreprocessingAction, stats: Dict[str, Any]) -> str:
        """Generate brief explanation for why action was predicted."""
        if action == PreprocessingAction.LOG_TRANSFORM:
            return f"High skewness ({abs(stats.get('skewness', 0)):.2f}) indicates need for log transform"
        elif action == PreprocessingAction.ROBUST_SCALE:
            return f"Outliers present ({{stats.get('outlier_pct', 0):.0%}}) suggests robust scaling"
        elif action == PreprocessingAction.STANDARD_SCALE:
            return "Normal-ish distribution suitable for standard scaling"
        elif action == PreprocessingAction.ONEHOT_ENCODE:
            return f"Moderate cardinality ({stats.get('cardinality', 0)}) fits one-hot encoding"
        elif action == PreprocessingAction.DROP_COLUMN:
            if stats.get('null_pct', 0) > 0.6:
                return f"High null percentage ({stats.get('null_pct', 0):.0%})"
            elif stats.get('unique_count', 2) == 1:
                return "Constant value"
            else:
                return "Low information content"
        return "Based on statistical characteristics"

    def _estimate_confidence(self, action: PreprocessingAction, stats: Dict[str, Any]) -> float:
        """Estimate confidence for this action given statistics (heuristic)."""
        # This is a simplified heuristic - real implementation would call symbolic engine
        if action == PreprocessingAction.DROP_COLUMN:
            if stats.get('null_pct', 0) > 0.8:
                return 0.95
            return 0.85

        if action == PreprocessingAction.LOG_TRANSFORM:
            skewness = abs(stats.get('skewness', 0) or 0)
            if skewness > 2.0:
                return 0.90
            elif skewness > 1.5:
                return 0.80
            return 0.70

        return 0.75  # Default moderate confidence

    def _generate_recommendation(
        self,
        action: PreprocessingAction,
        stats: Dict[str, Any],
        trade_offs: Dict[str, str]
    ) -> str:
        """Generate recommendation for when to use this alternative."""
        # Parse trade-offs to determine recommendation
        if "better" in trade_offs.get("Accuracy", "").lower():
            return f"Consider {action.value} if maximizing accuracy is top priority"
        elif "simpler" in trade_offs.get("Interpretability", "").lower():
            return f"Consider {action.value} if interpretability is more important than accuracy"
        elif "faster" in trade_offs.get("Computational Cost", "").lower():
            return f"Consider {action.value} if computational efficiency is critical"
        else:
            return f"Consider {action.value} based on your specific requirements and constraints"

    def _assess_robustness(
        self,
        current_action: PreprocessingAction,
        new_action: PreprocessingAction,
        change_desc: str
    ) -> str:
        """Assess robustness implications of data change."""
        if current_action == new_action:
            return f"Decision is robust to {change_desc}"
        else:
            return f"Decision is sensitive to {change_desc} - would switch to {new_action.value}"

    def _assess_model_compatibility(self, action: PreprocessingAction) -> str:
        """Describe which models work well with this preprocessing."""
        compatibility = {
            PreprocessingAction.LOG_TRANSFORM: "Best for: Linear models, neural networks. Good for: Trees (slight improvement)",
            PreprocessingAction.STANDARD_SCALE: "Best for: Linear models, SVM, neural networks, KNN. Neutral for: Trees",
            PreprocessingAction.ROBUST_SCALE: "Best for: Outlier-heavy data. Good for: Linear models, SVM. Neutral for: Trees",
            PreprocessingAction.MINMAX_SCALE: "Best for: Neural networks (bounded inputs). Fair for: Linear models. Neutral for: Trees",
            PreprocessingAction.ONEHOT_ENCODE: "Best for: Linear models, neural networks, SVM. Neutral for: Trees",
            PreprocessingAction.LABEL_ENCODE: "Best for: Tree-based models. Poor for: Linear models, SVM (implies false ordering)",
        }
        return compatibility.get(action, "Compatible with most models")

    def _compare_accuracy_impact(
        self,
        current: PreprocessingAction,
        alternative: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> str:
        """Compare expected accuracy impact."""
        # Simplified heuristic comparison
        if current == PreprocessingAction.LOG_TRANSFORM and alternative == PreprocessingAction.STANDARD_SCALE:
            skewness = abs(stats.get('skewness', 0) or 0)
            if skewness > 2.0:
                return "LOG_TRANSFORM likely 5-15% better accuracy due to high skewness"
            return "LOG_TRANSFORM likely 2-5% better for slightly skewed data"
        elif current == PreprocessingAction.ROBUST_SCALE and alternative == PreprocessingAction.STANDARD_SCALE:
            if stats.get('has_outliers'):
                return "ROBUST_SCALE likely 3-8% better with outliers present"
            return "STANDARD_SCALE likely equivalent or slightly better without outliers"
        return "Accuracy impact depends on downstream model and data characteristics"

    def _compare_interpretability(
        self,
        current: PreprocessingAction,
        alternative: PreprocessingAction
    ) -> str:
        """Compare interpretability."""
        interpretability_scores = {
            PreprocessingAction.KEEP_AS_IS: 10,
            PreprocessingAction.STANDARD_SCALE: 8,
            PreprocessingAction.LOG_TRANSFORM: 8,
            PreprocessingAction.ROBUST_SCALE: 7,
            PreprocessingAction.MINMAX_SCALE: 7,
            PreprocessingAction.ONEHOT_ENCODE: 9,
            PreprocessingAction.LABEL_ENCODE: 5,
            PreprocessingAction.QUANTILE_TRANSFORM: 3,
        }

        current_score = interpretability_scores.get(current, 5)
        alt_score = interpretability_scores.get(alternative, 5)

        if current_score > alt_score:
            return f"{current.value} more interpretable than {alternative.value}"
        elif alt_score > current_score:
            return f"{alternative.value} more interpretable than {current.value}"
        return "Similar interpretability"

    def _compare_cost(
        self,
        current: PreprocessingAction,
        alternative: PreprocessingAction,
        stats: Dict[str, Any]
    ) -> str:
        """Compare computational cost."""
        # All these are O(n), but some have different constants
        expensive_actions = [PreprocessingAction.BOX_COX, PreprocessingAction.YEO_JOHNSON, PreprocessingAction.QUANTILE_TRANSFORM]
        fast_actions = [PreprocessingAction.STANDARD_SCALE, PreprocessingAction.MINMAX_SCALE, PreprocessingAction.LOG_TRANSFORM]

        current_expensive = current in expensive_actions
        alt_expensive = alternative in expensive_actions
        current_fast = current in fast_actions
        alt_fast = alternative in fast_actions

        if current_fast and alt_expensive:
            return f"{current.value} significantly faster (O(n) single-pass vs complex computation)"
        elif current_expensive and alt_fast:
            return f"{alternative.value} significantly faster (O(n) single-pass vs complex computation)"
        return "Similar computational cost (both O(n))"

    def _compare_robustness(
        self,
        current: PreprocessingAction,
        alternative: PreprocessingAction
    ) -> str:
        """Compare robustness to distribution changes."""
        robust_actions = [PreprocessingAction.ROBUST_SCALE, PreprocessingAction.QUANTILE_TRANSFORM]

        if current in robust_actions and alternative not in robust_actions:
            return f"{current.value} more robust to outliers and distribution shift"
        elif alternative in robust_actions and current not in robust_actions:
            return f"{alternative.value} more robust to outliers and distribution shift"
        return "Similar robustness characteristics"
