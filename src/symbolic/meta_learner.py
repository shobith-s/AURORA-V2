"""
Meta-Learner: Universal preprocessing decisions based on statistical theory.

Unlike NeuralOracle (which learns from specific datasets), MetaLearner uses
universal mathematical and statistical principles that apply to ALL data.

This achieves universality without dataset-specific training.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.actions import PreprocessingAction, PreprocessingResult


@dataclass
class StatisticalHeuristic:
    """A statistical heuristic based on mathematical theory."""
    name: str
    condition: callable  # Takes stats dict, returns bool
    action: PreprocessingAction
    confidence: float
    reasoning: str
    priority: int = 50  # Lower than symbolic rules


class MetaLearner:
    """
    Makes preprocessing decisions based on universal statistical principles.

    Key difference from NeuralOracle:
    - NeuralOracle: Learns "column X in dataset Y needs action Z" (dataset-specific)
    - MetaLearner: Knows "skewness > 2 always benefits from log transform" (universal)

    This is why it achieves universality without training on specific datasets.
    """

    def __init__(self):
        """Initialize the meta-learner with statistical heuristics."""
        self.heuristics = self._build_statistical_knowledge_base()
        self.stats = {
            'total_decisions': 0,
            'heuristic_applications': {}
        }

    def _build_statistical_knowledge_base(self) -> List[StatisticalHeuristic]:
        """
        Build knowledge base from statistical theory.
        These are universal mathematical principles, not learned from data.
        """
        heuristics = []

        # =============================================================================
        # DISTRIBUTION-BASED HEURISTICS (from probability theory)
        # =============================================================================

        # Heuristic 1: High positive skew → log transform
        # Theory: Log transform reduces right skew (mathematical property)
        heuristics.append(StatisticalHeuristic(
            name="HIGH_SKEW_LOG_TRANSFORM",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('skewness', 0) > 1.5 and
                s.get('all_positive', False) and
                s.get('min_value', 0) > 0
            ),
            action=PreprocessingAction.LOG_TRANSFORM,
            confidence=0.88,
            reasoning="High right skew (>1.5) in positive data: log transform reduces skewness (mathematical property)",
            priority=75
        ))

        # Heuristic 2: High skew with zeros → log1p
        # Theory: log1p handles zero values gracefully
        heuristics.append(StatisticalHeuristic(
            name="HIGH_SKEW_LOG1P",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('skewness', 0) > 1.5 and
                s.get('min_value', -1) >= 0 and
                s.get('has_zeros', False)
            ),
            action=PreprocessingAction.LOG1P_TRANSFORM,
            confidence=0.90,
            reasoning="High skew with zeros: log1p(x) = log(1+x) handles zeros mathematically",
            priority=76
        ))

        # Heuristic 3: Symmetric skew → Yeo-Johnson
        # Theory: Yeo-Johnson works with any real numbers
        heuristics.append(StatisticalHeuristic(
            name="GENERAL_SKEW_YEO_JOHNSON",
            condition=lambda s: (
                s.get('is_numeric', False) and
                abs(s.get('skewness', 0)) > 1.5 and
                not s.get('all_positive', True)  # Has negative values
            ),
            action=PreprocessingAction.YEO_JOHNSON,
            confidence=0.85,
            reasoning="High skew with negative values: Yeo-Johnson handles all real numbers",
            priority=74
        ))

        # =============================================================================
        # VARIANCE-BASED HEURISTICS (from statistical theory)
        # =============================================================================

        # Heuristic 4: High CV → robust scaling
        # Theory: High coefficient of variation indicates outliers; robust methods less affected
        heuristics.append(StatisticalHeuristic(
            name="HIGH_CV_ROBUST_SCALE",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('cv', 0) > 2.0 and
                not s.get('is_already_scaled', False)
            ),
            action=PreprocessingAction.ROBUST_SCALE,
            confidence=0.87,
            reasoning=f"High coefficient of variation (CV={s.get('cv', 0):.2f}): indicates outliers, robust scaling optimal" if 's' in locals() else "High CV: robust scaling handles outliers",
            priority=73
        ))

        # Heuristic 5: Low CV and normal distribution → standard scaling
        # Theory: Standard scaling optimal for Gaussian data
        heuristics.append(StatisticalHeuristic(
            name="LOW_CV_STANDARD_SCALE",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('cv', 0) < 0.5 and
                abs(s.get('skewness', 0)) < 0.5 and  # Approximately normal
                not s.get('is_already_scaled', False)
            ),
            action=PreprocessingAction.STANDARD_SCALE,
            confidence=0.90,
            reasoning="Low CV + symmetric distribution: standard scaling is statistically optimal for Gaussian data",
            priority=72
        ))

        # =============================================================================
        # INFORMATION THEORY HEURISTICS (from Shannon entropy)
        # =============================================================================

        # Heuristic 6: Very low entropy → drop
        # Theory: Shannon entropy measures information content; low entropy = low information
        heuristics.append(StatisticalHeuristic(
            name="LOW_ENTROPY_DROP",
            condition=lambda s: (
                s.get('entropy', 1.0) < 0.15 and  # <15% of max entropy
                s.get('unique_ratio', 1.0) < 0.05 and
                s.get('row_count', 0) > 50
            ),
            action=PreprocessingAction.DROP_COLUMN,
            confidence=0.86,
            reasoning=f"Very low entropy ({s.get('entropy', 0):.2f}): Shannon information theory indicates minimal information content" if 's' in locals() else "Low entropy: minimal information",
            priority=78
        ))

        # Heuristic 7: Medium entropy categorical → preserve frequency
        # Theory: Entropy reflects class balance; moderate entropy needs frequency preservation
        heuristics.append(StatisticalHeuristic(
            name="MEDIUM_ENTROPY_FREQUENCY_ENCODE",
            condition=lambda s: (
                s.get('is_categorical', False) and
                0.4 < s.get('entropy', 0) < 0.8 and  # Medium entropy = moderate imbalance
                10 < s.get('cardinality', 0) <= 50
            ),
            action=PreprocessingAction.FREQUENCY_ENCODE,
            confidence=0.82,
            reasoning="Medium entropy: information theory suggests preserving frequency distribution",
            priority=70
        ))

        # =============================================================================
        # CARDINALITY-BASED HEURISTICS (from combinatorics)
        # =============================================================================

        # Heuristic 8: Very low cardinality → one-hot
        # Theory: Small categorical space can be fully represented
        heuristics.append(StatisticalHeuristic(
            name="VERY_LOW_CARD_ONEHOT",
            condition=lambda s: (
                s.get('is_categorical', False) and
                s.get('cardinality', 100) <= 5 and
                not s.get('is_ordinal', False)
            ),
            action=PreprocessingAction.ONEHOT_ENCODE,
            confidence=0.93,
            reasoning=f"Very low cardinality ({s.get('cardinality', 0)}): one-hot encoding is computationally efficient" if 's' in locals() else "Low cardinality: one-hot optimal",
            priority=77
        ))

        # Heuristic 9: High cardinality without target → hash
        # Theory: Dimensionality curse; hashing reduces feature space
        heuristics.append(StatisticalHeuristic(
            name="HIGH_CARD_HASH",
            condition=lambda s: (
                s.get('is_categorical', False) and
                s.get('cardinality', 0) > 500 and
                s.get('unique_ratio', 1.0) < 0.95  # Not an ID
            ),
            action=PreprocessingAction.HASH_ENCODE,
            confidence=0.84,
            reasoning=f"High cardinality ({s.get('cardinality', 0)}): hash encoding prevents dimensionality explosion",
            priority=71
        ))

        # =============================================================================
        # OUTLIER-BASED HEURISTICS (from robust statistics)
        # =============================================================================

        # Heuristic 10: Many outliers → winsorize
        # Theory: Winsorization caps extremes while preserving distribution shape
        heuristics.append(StatisticalHeuristic(
            name="MANY_OUTLIERS_WINSORIZE",
            condition=lambda s: (
                s.get('is_numeric', False) and
                0.10 < s.get('outlier_pct', 0) < 0.25  # 10-25% outliers
            ),
            action=PreprocessingAction.WINSORIZE,
            confidence=0.85,
            reasoning=f"Significant outliers ({s.get('outlier_pct', 0):.1%}): winsorization preserves distribution while capping extremes" if 's' in locals() else "Many outliers: winsorize",
            priority=74
        ))

        # Heuristic 11: Few outliers but present → clip
        # Theory: Clipping removes outliers while keeping valid range
        heuristics.append(StatisticalHeuristic(
            name="FEW_OUTLIERS_CLIP",
            condition=lambda s: (
                s.get('is_numeric', False) and
                0.05 < s.get('outlier_pct', 0) <= 0.10  # 5-10% outliers
            ),
            action=PreprocessingAction.CLIP_OUTLIERS,
            confidence=0.83,
            reasoning="Moderate outliers (5-10%): clipping at IQR boundaries is statistically sound",
            priority=72
        ))

        # =============================================================================
        # RANGE-BASED HEURISTICS (from normalization theory)
        # =============================================================================

        # Heuristic 12: Already in [0,1] range → keep
        # Theory: Already normalized probability range
        heuristics.append(StatisticalHeuristic(
            name="PROBABILITY_RANGE_KEEP",
            condition=lambda s: (
                s.get('is_numeric', False) and
                0 <= s.get('min_value', -1) and
                s.get('max_value', 2) <= 1.0 and
                (s.get('max_value', 0) - s.get('min_value', 1)) > 0.5  # Uses significant range
            ),
            action=PreprocessingAction.KEEP_AS_IS,
            confidence=0.91,
            reasoning="Values in [0,1] range: already normalized (likely probabilities)",
            priority=80
        ))

        # Heuristic 13: Large range → scaling needed
        # Theory: Large ranges cause numerical instability
        heuristics.append(StatisticalHeuristic(
            name="LARGE_RANGE_ROBUST_SCALE",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('range_size', 0) > 1000 and
                not s.get('is_already_scaled', False)
            ),
            action=PreprocessingAction.ROBUST_SCALE,
            confidence=0.84,
            reasoning=f"Large range ({s.get('range_size', 0):.0f}): scaling prevents numerical instability" if 's' in locals() else "Large range: scaling needed",
            priority=71
        ))

        # =============================================================================
        # NULL-HANDLING HEURISTICS (from imputation theory)
        # =============================================================================

        # Heuristic 14: Moderate nulls in numeric → median
        # Theory: Median is robust to outliers and skewness
        heuristics.append(StatisticalHeuristic(
            name="MODERATE_NULL_MEDIAN",
            condition=lambda s: (
                s.get('is_numeric', False) and
                0.10 < s.get('null_pct', 0) < 0.30 and
                s.get('has_outliers', False)  # Has outliers
            ),
            action=PreprocessingAction.FILL_NULL_MEDIAN,
            confidence=0.86,
            reasoning=f"Moderate nulls ({s.get('null_pct', 0):.1%}) with outliers: median imputation is robust" if 's' in locals() else "Moderate nulls: median robust",
            priority=69
        ))

        # Heuristic 15: Moderate nulls in categorical → mode
        # Theory: Mode is the maximum likelihood estimator for categorical data
        heuristics.append(StatisticalHeuristic(
            name="MODERATE_NULL_MODE_CATEGORICAL",
            condition=lambda s: (
                s.get('is_categorical', False) and
                0.10 < s.get('null_pct', 0) < 0.30
            ),
            action=PreprocessingAction.FILL_NULL_MODE,
            confidence=0.87,
            reasoning="Categorical with moderate nulls: mode is maximum likelihood estimator",
            priority=69
        ))

        # =============================================================================
        # CORRELATION-BASED HEURISTICS (when target available)
        # =============================================================================

        # Heuristic 16: Zero correlation with target → drop
        # Theory: Uncorrelated features have no linear predictive power
        heuristics.append(StatisticalHeuristic(
            name="ZERO_CORRELATION_DROP",
            condition=lambda s: (
                s.get('target_available', False) and
                abs(s.get('target_correlation', 0)) < 0.01 and
                s.get('row_count', 0) > 100  # Enough samples for reliable correlation
            ),
            action=PreprocessingAction.DROP_COLUMN,
            confidence=0.82,
            reasoning="Zero correlation with target: no linear predictive value (Pearson correlation theory)",
            priority=73
        ))

        # Heuristic 17: Perfect correlation → data leakage
        # Theory: Perfect correlation indicates target leakage
        heuristics.append(StatisticalHeuristic(
            name="PERFECT_CORRELATION_DROP",
            condition=lambda s: (
                s.get('target_available', False) and
                abs(s.get('target_correlation', 0)) > 0.99
            ),
            action=PreprocessingAction.DROP_COLUMN,
            confidence=0.97,
            reasoning="Perfect correlation (>0.99): likely data leakage, must drop",
            priority=85
        ))

        # =============================================================================
        # IQR-BASED HEURISTICS (from robust statistics)
        # =============================================================================

        # Heuristic 18: Very small IQR → near-constant
        # Theory: Small IQR indicates low variability
        heuristics.append(StatisticalHeuristic(
            name="SMALL_IQR_DROP",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('iqr', 1.0) < 0.01 and
                s.get('std', 1.0) < 0.1 and
                s.get('row_count', 0) > 50
            ),
            action=PreprocessingAction.DROP_COLUMN,
            confidence=0.84,
            reasoning="Very small IQR: quasi-constant column with minimal variability",
            priority=76
        ))

        # =============================================================================
        # BIMODALITY HEURISTICS (from mixture models)
        # =============================================================================

        # Heuristic 19: Strong bimodality → quantile transform
        # Theory: Quantile transform creates uniform distribution regardless of shape
        heuristics.append(StatisticalHeuristic(
            name="BIMODAL_QUANTILE_TRANSFORM",
            condition=lambda s: (
                s.get('is_numeric', False) and
                s.get('kurtosis', 0) < -1.0 and  # Negative kurtosis indicates bimodality
                abs(s.get('skewness', 0)) < 0.5  # But not skewed
            ),
            action=PreprocessingAction.QUANTILE_TRANSFORM,
            confidence=0.79,
            reasoning="Bimodal distribution (negative kurtosis): quantile transform to uniform",
            priority=68
        ))

        # =============================================================================
        # SPECIAL NUMERIC RANGES (domain knowledge from math)
        # =============================================================================

        # Heuristic 20: -1 to 1 range → already scaled
        # Theory: Common standardization range
        heuristics.append(StatisticalHeuristic(
            name="MINUS_ONE_TO_ONE_KEEP",
            condition=lambda s: (
                s.get('is_numeric', False) and
                -1.5 < s.get('min_value', -10) <= -0.5 and
                0.5 <= s.get('max_value', 10) < 1.5 and
                abs(s.get('mean', 100)) < 0.3
            ),
            action=PreprocessingAction.KEEP_AS_IS,
            confidence=0.89,
            reasoning="Values in [-1, 1] range with mean~0: already standardized",
            priority=79
        ))

        # Sort by priority (higher first)
        heuristics.sort(key=lambda h: h.priority, reverse=True)

        return heuristics

    def decide(
        self,
        column_stats: Dict[str, Any],
        column_name: str = ""
    ) -> Optional[PreprocessingResult]:
        """
        Make a preprocessing decision based on statistical heuristics.

        Args:
            column_stats: Column statistics dictionary
            column_name: Name of the column (for context)

        Returns:
            PreprocessingResult if a heuristic applies, None otherwise
        """
        # Add column name to stats for pattern matching
        stats_with_name = {**column_stats, 'column_name': column_name}

        # Find all applicable heuristics
        applicable = []
        for heuristic in self.heuristics:
            try:
                if heuristic.condition(stats_with_name):
                    applicable.append(heuristic)
            except Exception as e:
                # Skip heuristic if condition evaluation fails
                continue

        if not applicable:
            return None

        # Get the highest priority heuristic
        best = applicable[0]  # Already sorted by priority

        # Track statistics
        self.stats['total_decisions'] += 1
        if best.name not in self.stats['heuristic_applications']:
            self.stats['heuristic_applications'][best.name] = 0
        self.stats['heuristic_applications'][best.name] += 1

        # Get alternatives (other applicable heuristics)
        alternatives = [
            (h.action, h.confidence)
            for h in applicable[1:4]  # Top 3 alternatives
            if h.action != best.action
        ]

        return PreprocessingResult(
            action=best.action,
            confidence=best.confidence,
            source='meta_learning',
            explanation=best.reasoning,
            alternatives=alternatives,
            parameters={},
            context=column_stats
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learner statistics."""
        return {
            **self.stats,
            'total_heuristics': len(self.heuristics),
            'heuristics_by_priority': [
                {'name': h.name, 'priority': h.priority, 'confidence': h.confidence}
                for h in self.heuristics[:10]  # Top 10
            ]
        }

    def reset_statistics(self):
        """Reset decision statistics."""
        self.stats = {
            'total_decisions': 0,
            'heuristic_applications': {}
        }


# Singleton instance
_meta_learner_instance: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    """Get the global meta-learner instance."""
    global _meta_learner_instance
    if _meta_learner_instance is None:
        _meta_learner_instance = MetaLearner()
    return _meta_learner_instance
