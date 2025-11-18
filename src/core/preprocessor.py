"""
Main Preprocessing Pipeline - Integrates all three layers.
Symbolic Engine -> NeuralOracle -> Pattern Learner
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import time

from ..symbolic.engine import SymbolicEngine
from ..neural.oracle import NeuralOracle, get_neural_oracle
from ..learning.pattern_learner import LocalPatternLearner
from ..features.minimal_extractor import MinimalFeatureExtractor, get_feature_extractor
from ..features.intelligent_cache import get_cache
from .actions import PreprocessingAction, PreprocessingResult


class IntelligentPreprocessor:
    """
    Main preprocessing pipeline with three-layer architecture:
    1. Learned patterns (fastest)
    2. Symbolic engine (fast)
    3. NeuralOracle (for edge cases)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        use_neural_oracle: bool = True,
        enable_learning: bool = True,
        neural_model_path: Optional[Path] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the intelligent preprocessor.

        Args:
            confidence_threshold: Minimum confidence for symbolic engine
            use_neural_oracle: Whether to use neural oracle for low-confidence cases
            enable_learning: Whether to enable pattern learning
            neural_model_path: Path to neural oracle model
            enable_cache: Whether to enable intelligent caching
        """
        self.confidence_threshold = confidence_threshold
        self.use_neural_oracle = use_neural_oracle
        self.enable_learning = enable_learning
        self.enable_cache = enable_cache

        # Initialize components
        self.symbolic_engine = SymbolicEngine(confidence_threshold=confidence_threshold)
        self.pattern_learner = LocalPatternLearner() if enable_learning else None
        self.feature_extractor = get_feature_extractor()
        self.cache = get_cache() if enable_cache else None

        # Initialize neural oracle (lazy loading)
        self._neural_oracle: Optional[NeuralOracle] = None
        self.neural_model_path = neural_model_path

        # Statistics
        self.stats = {
            'total_decisions': 0,
            'learned_decisions': 0,
            'symbolic_decisions': 0,
            'neural_decisions': 0,
            'high_confidence_decisions': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0
        }

    @property
    def neural_oracle(self) -> Optional[NeuralOracle]:
        """Lazy load neural oracle."""
        if self.use_neural_oracle and self._neural_oracle is None:
            try:
                self._neural_oracle = get_neural_oracle(self.neural_model_path)
            except (ImportError, FileNotFoundError) as e:
                print(f"Warning: Could not load neural oracle: {e}")
                self.use_neural_oracle = False
        return self._neural_oracle

    def preprocess_column(
        self,
        column: Union[pd.Series, List, np.ndarray],
        column_name: str = "",
        target_available: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PreprocessingResult:
        """
        Preprocess a single column using the three-layer architecture.

        Args:
            column: Column data
            column_name: Name of the column
            target_available: Whether target variable is available
            metadata: Additional metadata

        Returns:
            PreprocessingResult with action, confidence, and explanation
        """
        start_time = time.time()

        # Convert to pandas Series if needed
        if isinstance(column, (list, np.ndarray)):
            column = pd.Series(column, name=column_name)
        elif not isinstance(column, pd.Series):
            raise TypeError(f"Column must be pd.Series, list, or np.ndarray, got {type(column)}")

        # Update statistics
        self.stats['total_decisions'] += 1

        # Generate decision ID
        decision_id = str(uuid.uuid4())

        # LAYER 0: Check intelligent cache (ultra-fast - <0.1ms for L1 hits)
        if self.enable_cache and self.cache:
            # Compute stats for cache lookup
            stats = self.symbolic_engine.compute_column_statistics(
                column, column_name, target_available
            )
            stats_dict = stats.to_dict()

            # Check cache
            cached_decision, cache_level = self.cache.get(stats_dict, column_name)
            if cached_decision:
                self.stats['cache_hits'] += 1
                self.stats['high_confidence_decisions'] += 1

                elapsed_ms = (time.time() - start_time) * 1000
                self.stats['total_time_ms'] += elapsed_ms

                return PreprocessingResult(
                    action=cached_decision,
                    confidence=0.98,  # Very high confidence for cached decisions
                    source='learned',
                    explanation=f"Cached decision from {cache_level} (previously seen similar column)",
                    alternatives=[],
                    parameters={},
                    context=stats_dict,
                    decision_id=decision_id
                )

        # LAYER 1: Check learned patterns first (fastest - <1ms)
        if self.enable_learning and self.pattern_learner:
            # We need column stats for pattern matching
            stats = self.symbolic_engine.compute_column_statistics(
                column, column_name, target_available
            )
            stats_dict = stats.to_dict()

            learned_action = self.pattern_learner.check_patterns(stats_dict)
            if learned_action:
                self.stats['learned_decisions'] += 1
                self.stats['high_confidence_decisions'] += 1

                elapsed_ms = (time.time() - start_time) * 1000
                self.stats['total_time_ms'] += elapsed_ms

                # Update cache with learned decision
                if self.enable_cache and self.cache:
                    self.cache.set(stats_dict, learned_action, column_name)

                return PreprocessingResult(
                    action=learned_action,
                    confidence=0.95,  # High confidence for learned patterns
                    source='learned',
                    explanation=f"Matched learned pattern from user corrections",
                    alternatives=[],
                    parameters={},
                    context=stats_dict,
                    decision_id=decision_id
                )

        # LAYER 2: Try symbolic engine (fast - <100us)
        symbolic_result = self.symbolic_engine.evaluate(
            column, column_name, target_available
        )

        # If symbolic engine has high confidence, use it
        if symbolic_result.confidence >= self.confidence_threshold:
            self.stats['symbolic_decisions'] += 1
            self.stats['high_confidence_decisions'] += 1

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed_ms

            # Update cache with symbolic decision
            if self.enable_cache and self.cache and symbolic_result.context:
                self.cache.set(symbolic_result.context, symbolic_result.action, column_name)

            symbolic_result.decision_id = decision_id
            return symbolic_result

        # LAYER 3: Use NeuralOracle for ambiguous cases (<5ms)
        if self.use_neural_oracle and self.neural_oracle:
            # Extract minimal features
            features = self.feature_extractor.extract(column, column_name)

            # Get neural prediction
            try:
                neural_pred = self.neural_oracle.predict(
                    features,
                    return_probabilities=True,
                    return_feature_importance=False
                )

                # Blend symbolic and neural if both have medium confidence
                if symbolic_result.confidence > 0.5:
                    action, confidence, explanation = self._blend_decisions(
                        symbolic_result, neural_pred
                    )
                else:
                    action = neural_pred.action
                    confidence = neural_pred.confidence
                    explanation = f"Neural oracle prediction (symbolic confidence too low: {symbolic_result.confidence:.2f})"

                self.stats['neural_decisions'] += 1
                if confidence >= self.confidence_threshold:
                    self.stats['high_confidence_decisions'] += 1

                elapsed_ms = (time.time() - start_time) * 1000
                self.stats['total_time_ms'] += elapsed_ms

                # Get alternatives from neural probabilities
                alternatives = sorted(
                    [(a, p) for a, p in neural_pred.action_probabilities.items() if a != action],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                # Update cache with neural decision
                if self.enable_cache and self.cache and symbolic_result.context:
                    self.cache.set(symbolic_result.context, action, column_name)

                return PreprocessingResult(
                    action=action,
                    confidence=confidence,
                    source='neural',
                    explanation=explanation,
                    alternatives=alternatives,
                    parameters={},
                    context=symbolic_result.context,
                    decision_id=decision_id
                )

            except Exception as e:
                print(f"Warning: Neural oracle failed: {e}")
                # Fall back to symbolic result
                pass

        # Fallback: Use symbolic result even if confidence is low
        self.stats['symbolic_decisions'] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        symbolic_result.decision_id = decision_id
        return symbolic_result

    def _blend_decisions(
        self,
        symbolic_result: PreprocessingResult,
        neural_pred
    ) -> tuple:
        """
        Blend symbolic and neural decisions when both have medium confidence.

        Args:
            symbolic_result: Result from symbolic engine
            neural_pred: Prediction from neural oracle

        Returns:
            (action, confidence, explanation) tuple
        """
        # If both agree, high confidence
        if symbolic_result.action == neural_pred.action:
            confidence = min(0.95, (symbolic_result.confidence + neural_pred.confidence) / 2 + 0.1)
            explanation = f"Both symbolic and neural agree on {symbolic_result.action.value}"
            return symbolic_result.action, confidence, explanation

        # If they disagree, use the one with higher confidence
        if symbolic_result.confidence > neural_pred.confidence:
            return (
                symbolic_result.action,
                symbolic_result.confidence * 0.9,  # Slight penalty for disagreement
                f"Symbolic engine ({symbolic_result.confidence:.2f}) vs neural ({neural_pred.confidence:.2f})"
            )
        else:
            return (
                neural_pred.action,
                neural_pred.confidence * 0.9,
                f"Neural oracle ({neural_pred.confidence:.2f}) vs symbolic ({symbolic_result.confidence:.2f})"
            )

    def process_correction(
        self,
        column: Union[pd.Series, List, np.ndarray],
        column_name: str,
        wrong_action: Union[str, PreprocessingAction],
        correct_action: Union[str, PreprocessingAction],
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process a user correction to learn from it (privacy-preserving).

        Args:
            column: Column data (only used to extract pattern, not stored)
            column_name: Column name
            wrong_action: Action that was incorrect
            correct_action: Correct action
            confidence: Confidence of the wrong prediction

        Returns:
            Learning result
        """
        if not self.enable_learning or not self.pattern_learner:
            return {'learned': False, 'reason': 'Learning disabled'}

        # Convert to pandas Series if needed
        if isinstance(column, (list, np.ndarray)):
            column = pd.Series(column, name=column_name)

        # Convert actions to enum if strings
        if isinstance(wrong_action, str):
            wrong_action = PreprocessingAction(wrong_action)
        if isinstance(correct_action, str):
            correct_action = PreprocessingAction(correct_action)

        # Extract privacy-preserving pattern
        stats = self.symbolic_engine.compute_column_statistics(column, column_name)
        stats_dict = stats.to_dict()

        pattern = self.pattern_learner.extract_pattern(stats_dict, column_name)

        # Learn from correction
        new_rule = self.pattern_learner.learn_correction(
            pattern=pattern,
            wrong_action=wrong_action,
            correct_action=correct_action,
            confidence=confidence
        )

        result = {
            'learned': True,
            'pattern_recorded': True,
            'new_rule_created': new_rule is not None
        }

        # If a new rule was created, add it to symbolic engine
        if new_rule:
            self.symbolic_engine.add_rule(new_rule)
            result['rule_name'] = new_rule.name
            result['rule_confidence'] = new_rule.confidence_fn({})

        # Get learning statistics
        if new_rule:
            similar = self.pattern_learner.find_similar_patterns(
                pattern,
                self.pattern_learner.similarity_threshold
            )
            result['similar_patterns_count'] = len(similar)
            result['applicable_to'] = f"~{len(similar)} similar cases"

        return result

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, PreprocessingResult]:
        """
        Preprocess all columns in a dataframe.

        Args:
            df: Input dataframe
            target_column: Name of target column (if any)

        Returns:
            Dictionary mapping column names to PreprocessingResults
        """
        results = {}

        for column_name in df.columns:
            if column_name == target_column:
                continue  # Skip target column

            target_available = target_column is not None
            result = self.preprocess_column(
                df[column_name],
                column_name,
                target_available
            )
            results[column_name] = result

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        total = self.stats['total_decisions']

        stats = {
            **self.stats,
            'learned_pct': self.stats['learned_decisions'] / total * 100 if total > 0 else 0,
            'symbolic_pct': self.stats['symbolic_decisions'] / total * 100 if total > 0 else 0,
            'neural_pct': self.stats['neural_decisions'] / total * 100 if total > 0 else 0,
            'high_confidence_pct': self.stats['high_confidence_decisions'] / total * 100 if total > 0 else 0,
            'avg_time_ms': self.stats['total_time_ms'] / total if total > 0 else 0,
            'symbolic_coverage': self.symbolic_engine.coverage()
        }

        # Add learning statistics if available
        if self.pattern_learner:
            stats['learning'] = self.pattern_learner.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            'total_decisions': 0,
            'learned_decisions': 0,
            'symbolic_decisions': 0,
            'neural_decisions': 0,
            'high_confidence_decisions': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0
        }
        self.symbolic_engine.reset_stats()

    def save_learned_patterns(self, path: Path):
        """
        Save learned patterns to disk.

        Args:
            path: Path to save patterns
        """
        if self.pattern_learner:
            self.pattern_learner.save(path)

    def load_learned_patterns(self, path: Path):
        """
        Load learned patterns from disk.

        Args:
            path: Path to load patterns from
        """
        if self.pattern_learner:
            self.pattern_learner.load(path)

            # Add learned rules to symbolic engine
            for rule in self.pattern_learner.learned_rules:
                self.symbolic_engine.add_rule(rule)


# Singleton instance
_preprocessor_instance: Optional[IntelligentPreprocessor] = None


def get_preprocessor(**kwargs) -> IntelligentPreprocessor:
    """Get the global preprocessor instance."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = IntelligentPreprocessor(**kwargs)
    return _preprocessor_instance
