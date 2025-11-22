"""
Adaptive Symbolic Rules - Learn from corrections to improve symbolic engine.

Instead of creating separate learned patterns that override symbolic rules,
this module uses corrections to fine-tune symbolic rule parameters, thresholds,
and confidence scores. This prevents overgeneralization while still learning
from user feedback.

Architecture:
    User Corrections
          ↓
    Extract Patterns
          ↓
    Update Symbolic Rule Parameters
          ↓
    Symbolic Engine (now domain-adapted)
          ↓
    Better Decisions

Benefits over direct learned layer:
- No overgeneralization from limited data
- Symbolic rules remain primary (reliable)
- Corrections fine-tune existing good logic
- Domain adaptation without replacement
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from ..core.actions import PreprocessingAction


@dataclass
class RuleAdjustment:
    """Adjustment to a symbolic rule based on corrections."""
    rule_category: str  # e.g., "high_skewness", "high_nulls"
    action: PreprocessingAction
    confidence_delta: float  # How much to adjust confidence (-0.2 to +0.2)
    threshold_adjustments: Dict[str, float]  # e.g., {"skewness": 1.5 → 2.0}
    correction_count: int  # How many corrections support this


class AdaptiveSymbolicRules:
    """
    Learn from corrections to adapt symbolic rule parameters.

    This is superior to creating separate learned patterns because:
    1. No overgeneralization - symbolic rules already have good logic
    2. Fine-tuning instead of replacement
    3. Maintains symbolic reliability while learning preferences
    """

    def __init__(
        self,
        min_corrections_for_adjustment: int = 5,
        max_confidence_delta: float = 0.15,
        min_corrections_for_production: int = 10,
        persistence_file: Optional[Path] = None
    ):
        """
        Initialize adaptive rules system.

        Args:
            min_corrections_for_adjustment: Minimum corrections to compute adjustments (training phase)
            max_confidence_delta: Maximum confidence adjustment (+/-)
            min_corrections_for_production: Minimum corrections before using adjustments in decisions (production phase)
            persistence_file: Where to save/load adjustments
        """
        self.min_corrections_for_adjustment = min_corrections_for_adjustment
        self.max_confidence_delta = max_confidence_delta
        self.min_corrections_for_production = min_corrections_for_production
        self.persistence_file = persistence_file

        # Track corrections by pattern
        self.correction_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Computed adjustments for symbolic rules
        self.rule_adjustments: Dict[str, RuleAdjustment] = {}

        # Load existing adjustments if available
        if persistence_file and persistence_file.exists():
            self.load()

    def record_correction(
        self,
        column_stats: Dict[str, Any],
        wrong_action: PreprocessingAction,
        correct_action: PreprocessingAction
    ):
        """
        Record a correction and update rule adjustments.

        Args:
            column_stats: Statistics about the column
            wrong_action: What symbolic engine recommended
            correct_action: What user corrected it to
        """
        # Identify which symbolic rule category this correction affects
        pattern_key = self._identify_pattern(column_stats)

        correction = {
            'stats': column_stats,
            'wrong_action': wrong_action.value,
            'correct_action': correct_action.value
        }

        self.correction_patterns[pattern_key].append(correction)

        # Recompute adjustments for this pattern
        self._update_adjustments(pattern_key)

        # Save if persistence enabled
        if self.persistence_file:
            self.save()

    def _identify_pattern(self, stats: Dict[str, Any]) -> str:
        """
        Identify which symbolic rule category this column matches.

        Returns pattern key like: "numeric_high_skewness", "categorical_high_cardinality"
        """
        # Handle both old and new field names for compatibility
        dtype = stats.get('dtype', stats.get('detected_dtype', 'unknown'))
        is_numeric = stats.get('is_numeric', False)

        # Numeric patterns
        if is_numeric or dtype in ['numeric', 'integer', 'float', 'int64', 'float64']:
            skewness = abs(stats.get('skewness', 0) or 0)
            null_pct = stats.get('null_pct', stats.get('null_percentage', 0))
            outlier_pct = stats.get('outlier_pct', stats.get('outlier_percentage', 0))

            if null_pct > 0.5:
                return 'numeric_high_nulls'
            elif null_pct > 0.1:
                return 'numeric_medium_nulls'
            elif skewness > 2.0:
                return 'numeric_high_skewness'
            elif skewness > 1.0:
                return 'numeric_medium_skewness'
            elif outlier_pct > 0.1:
                return 'numeric_many_outliers'
            else:
                return 'numeric_normal'

        # Categorical patterns
        elif stats.get('is_categorical') or dtype in ['categorical', 'object']:
            unique_ratio = stats.get('unique_ratio', 0)
            cardinality = stats.get('unique_count', 0)

            if unique_ratio > 0.9:
                return 'categorical_high_uniqueness'
            elif cardinality > 50:
                return 'categorical_high_cardinality'
            elif cardinality < 10:
                return 'categorical_low_cardinality'
            else:
                return 'categorical_medium_cardinality'

        return 'unknown'

    def _update_adjustments(self, pattern_key: str):
        """
        Recompute rule adjustments based on corrections for a pattern.

        Args:
            pattern_key: Pattern identifier (e.g., "numeric_high_skewness")
        """
        corrections = self.correction_patterns[pattern_key]

        if len(corrections) < self.min_corrections_for_adjustment:
            # Not enough data to adjust yet
            return

        # Analyze which actions are preferred
        action_preferences = defaultdict(int)
        for correction in corrections:
            # Count how many times each action was the correct one
            action_preferences[correction['correct_action']] += 1

        # Find most preferred action
        if not action_preferences:
            return

        most_preferred = max(action_preferences.items(), key=lambda x: x[1])
        preferred_action = PreprocessingAction(most_preferred[0])
        preference_count = most_preferred[1]

        # Calculate confidence boost
        # More corrections = higher confidence in this preference
        support_ratio = preference_count / len(corrections)
        confidence_delta = min(
            self.max_confidence_delta,
            support_ratio * 0.1  # Up to +0.1 for 100% support
        )

        # Create adjustment
        adjustment = RuleAdjustment(
            rule_category=pattern_key,
            action=preferred_action,
            confidence_delta=confidence_delta,
            threshold_adjustments={},  # TODO: Implement threshold learning
            correction_count=len(corrections)
        )

        self.rule_adjustments[pattern_key] = adjustment

    def is_production_ready(self, column_stats: Dict[str, Any]) -> bool:
        """
        Check if a pattern has enough corrections to use in production decisions.

        Args:
            column_stats: Column statistics

        Returns:
            True if pattern has >= min_corrections_for_production, False otherwise
        """
        pattern_key = self._identify_pattern(column_stats)
        corrections = self.correction_patterns.get(pattern_key, [])
        # Explicit bool conversion to avoid numpy.bool_ issues
        return bool(len(corrections) >= self.min_corrections_for_production)

    def get_adjustment(self, column_stats: Dict[str, Any]) -> Optional[RuleAdjustment]:
        """
        Get rule adjustment for a column if one exists.

        Args:
            column_stats: Column statistics

        Returns:
            RuleAdjustment if pattern has learned preferences, None otherwise
        """
        pattern_key = self._identify_pattern(column_stats)
        return self.rule_adjustments.get(pattern_key)

    def adjust_confidence(
        self,
        action: PreprocessingAction,
        original_confidence: float,
        column_stats: Dict[str, Any]
    ) -> float:
        """
        Adjust confidence of a symbolic rule decision based on corrections.

        Only applies adjustments if pattern is production-ready (has enough corrections).
        This implements training/production phase separation.

        Args:
            action: The action being recommended
            original_confidence: Original confidence from symbolic rule
            column_stats: Column statistics

        Returns:
            Adjusted confidence (or original if not production-ready)
        """
        # Check if pattern has enough corrections for production use
        if not self.is_production_ready(column_stats):
            # TRAINING PHASE: Don't affect decisions yet
            return original_confidence

        # PRODUCTION PHASE: Apply learned adjustments
        adjustment = self.get_adjustment(column_stats)

        if not adjustment:
            return original_confidence

        # If this action matches the learned preference, boost confidence
        if action == adjustment.action:
            boosted = original_confidence + adjustment.confidence_delta
            return min(0.98, boosted)  # Cap at 0.98 (leave room for uncertainty)

        # If different action, slightly reduce confidence
        else:
            reduced = original_confidence - (adjustment.confidence_delta * 0.5)
            return max(0.3, reduced)  # Floor at 0.3 (never completely dismiss)

    def get_alternative_action(self, column_stats: Dict[str, Any]) -> Optional[PreprocessingAction]:
        """
        Get learned preferred action for a pattern if it exists.

        This can be used to suggest alternatives when symbolic confidence is low.

        Args:
            column_stats: Column statistics

        Returns:
            Preferred action if learned, None otherwise
        """
        adjustment = self.get_adjustment(column_stats)
        return adjustment.action if adjustment else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about corrections and adjustments."""
        total_corrections = sum(len(c) for c in self.correction_patterns.values())

        return {
            'total_corrections': total_corrections,
            'patterns_tracked': len(self.correction_patterns),
            'active_adjustments': len(self.rule_adjustments),
            'adjustments': {
                pattern: {
                    'action': adj.action.value,
                    'confidence_delta': f"+{adj.confidence_delta:.3f}",
                    'corrections': adj.correction_count
                }
                for pattern, adj in self.rule_adjustments.items()
            }
        }

    def save(self):
        """Save adjustments to disk."""
        if not self.persistence_file:
            return

        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'correction_patterns': {
                pattern: corrections
                for pattern, corrections in self.correction_patterns.items()
            },
            'rule_adjustments': {
                pattern: {
                    'rule_category': adj.rule_category,
                    'action': adj.action.value,
                    'confidence_delta': adj.confidence_delta,
                    'threshold_adjustments': adj.threshold_adjustments,
                    'correction_count': adj.correction_count
                }
                for pattern, adj in self.rule_adjustments.items()
            }
        }

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            import numpy as np
            import pandas as pd

            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif pd.isna(obj):
                return None
            else:
                return obj

        data = convert_numpy_types(data)

        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load adjustments from disk."""
        if not self.persistence_file or not self.persistence_file.exists():
            return

        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)

            # Restore correction patterns
            self.correction_patterns = defaultdict(list)
            for pattern, corrections in data.get('correction_patterns', {}).items():
                self.correction_patterns[pattern] = corrections

            # Restore adjustments
            self.rule_adjustments = {}
            for pattern, adj_data in data.get('rule_adjustments', {}).items():
                self.rule_adjustments[pattern] = RuleAdjustment(
                    rule_category=adj_data['rule_category'],
                    action=PreprocessingAction(adj_data['action']),
                    confidence_delta=adj_data['confidence_delta'],
                    threshold_adjustments=adj_data.get('threshold_adjustments', {}),
                    correction_count=adj_data['correction_count']
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Corrupted file - backup and start fresh
            import shutil
            import logging
            logger = logging.getLogger(__name__)

            backup_file = self.persistence_file.with_suffix('.json.corrupted')
            shutil.move(str(self.persistence_file), str(backup_file))

            logger.warning(
                f"Corrupted adaptive rules file detected. "
                f"Backed up to {backup_file} and starting fresh. Error: {e}"
            )

            # Initialize with empty state
            self.correction_patterns = defaultdict(list)
            self.rule_adjustments = {}
