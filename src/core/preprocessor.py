"""
Main Preprocessing Pipeline - Integrates all layers.
Symbolic Engine (with adaptive learning) -> Meta-Learning -> NeuralOracle
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import time

from ..symbolic.engine import SymbolicEngine
from ..symbolic.meta_learner import get_meta_learner, MetaLearner
from ..neural.oracle import NeuralOracle, get_neural_oracle
from ..learning.pattern_learner import LocalPatternLearner
from ..learning.adaptive_rules import AdaptiveSymbolicRules
from ..features.minimal_extractor import MinimalFeatureExtractor, get_feature_extractor
from ..features.intelligent_cache import get_cache
from .actions import PreprocessingAction, PreprocessingResult
from ..utils.layer_metrics import LayerMetrics

# Confidence thresholds for decision quality
CONFIDENCE_HIGH = 0.9      # Auto-apply decision (highly confident)
CONFIDENCE_MEDIUM = 0.7    # Show warning (moderate confidence)
CONFIDENCE_LOW = 0.5       # Require manual review (low confidence)


class IntelligentPreprocessor:
    """
    Main preprocessing pipeline with adaptive learning architecture:

    0. Cache (validated decisions) - Instant lookup
    1. Symbolic rules (165+ rules) - Primary decision layer
       └─ Enhanced by adaptive learning from corrections
    2. Meta-learning (statistical heuristics) - Bridge layer
    3. NeuralOracle (ML predictions) - Ambiguous cases

    Learning Approach:
    - Corrections fine-tune symbolic rule parameters (NOT direct decisions)
    - Prevents overgeneralization from limited data
    - Maintains symbolic reliability while learning preferences
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        use_neural_oracle: bool = True,
        enable_learning: bool = True,
        neural_model_path: Optional[Path] = None,
        enable_cache: bool = True,
        enable_meta_learning: bool = True
    ):
        """
        Initialize the intelligent preprocessor.

        Args:
            confidence_threshold: Minimum confidence for symbolic engine
            use_neural_oracle: Whether to use neural oracle for low-confidence cases
            enable_learning: Whether to enable pattern learning
            neural_model_path: Path to neural oracle model
            enable_cache: Whether to enable intelligent caching
            enable_meta_learning: Whether to enable meta-learning (statistical heuristics)
        """
        import logging
        logger = logging.getLogger(__name__)

        self.confidence_threshold = confidence_threshold
        self.use_neural_oracle = use_neural_oracle
        self.enable_learning = enable_learning
        self.enable_cache = enable_cache
        self.enable_meta_learning = enable_meta_learning

        # Initialize components with error handling
        try:
            self.symbolic_engine = SymbolicEngine(confidence_threshold=confidence_threshold)
        except Exception as e:
            logger.error(f"Failed to initialize symbolic engine: {e}")
            raise RuntimeError(f"Critical component failed: symbolic engine - {e}")

        # Legacy pattern learner (being phased out)
        try:
            self.pattern_learner = LocalPatternLearner() if enable_learning else None
        except Exception as e:
            logger.warning(f"Pattern learner initialization failed, continuing without it: {e}")
            self.pattern_learner = None

        # Adaptive rules (with graceful degradation)
        try:
            self.adaptive_rules = AdaptiveSymbolicRules(
                min_corrections_for_adjustment=2,  # REDUCED from 5 to 2 for faster learning
                max_confidence_delta=0.20,  # INCREASED from 0.15 to 0.20 for stronger adjustments
                persistence_file=Path("data/adaptive_rules.json")
            ) if enable_learning else None
        except Exception as e:
            logger.warning(f"Adaptive rules initialization failed, continuing without it: {e}")
            self.adaptive_rules = None
            self.enable_learning = False  # Disable learning if adaptive rules fail

        # Meta learner
        try:
            self.meta_learner = get_meta_learner() if enable_meta_learning else None
        except Exception as e:
            logger.warning(f"Meta learner initialization failed, continuing without it: {e}")
            self.meta_learner = None

        # Feature extractor (required for neural oracle)
        try:
            self.feature_extractor = get_feature_extractor()
        except Exception as e:
            logger.warning(f"Feature extractor initialization failed: {e}")
            self.feature_extractor = None
            self.use_neural_oracle = False  # Disable neural oracle if feature extractor fails

        # Cache
        try:
            self.cache = get_cache() if enable_cache else None
        except Exception as e:
            logger.warning(f"Cache initialization failed, continuing without it: {e}")
            self.cache = None

        # Initialize neural oracle (lazy loading)
        self._neural_oracle: Optional[NeuralOracle] = None
        self.neural_model_path = neural_model_path

        # Initialize layer metrics tracker
        try:
            self.layer_metrics = LayerMetrics(
                persistence_file=Path("data/layer_metrics.json")
            )
        except Exception as e:
            logger.warning(f"Layer metrics initialization failed, continuing without it: {e}")
            self.layer_metrics = None

        # Statistics
        self.stats = {
            'total_decisions': 0,
            'learned_decisions': 0,
            'symbolic_decisions': 0,
            'meta_learning_decisions': 0,
            'neural_decisions': 0,
            'high_confidence_decisions': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0
        }

    @property
    def neural_oracle(self) -> Optional[NeuralOracle]:
        """Lazy load neural oracle with graceful degradation."""
        import logging
        logger = logging.getLogger(__name__)

        if self.use_neural_oracle and self._neural_oracle is None:
            try:
                self._neural_oracle = get_neural_oracle(self.neural_model_path)
                logger.info("Neural oracle loaded successfully")
            except FileNotFoundError as e:
                logger.warning(f"Neural oracle model file not found: {e}. Continuing with symbolic rules only.")
                self.use_neural_oracle = False
            except ImportError as e:
                logger.warning(f"Neural oracle dependencies missing: {e}. Install with: pip install xgboost shap")
                self.use_neural_oracle = False
            except Exception as e:
                logger.error(f"Unexpected error loading neural oracle: {e}. Falling back to symbolic rules.")
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
        # NOTE: Cache confidence reduced and validated to prevent overconfident incorrect decisions
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

                elapsed_ms = (time.time() - start_time) * 1000
                self.stats['total_time_ms'] += elapsed_ms

                # Get validation-adjusted confidence
                validation_adj = self.cache.get_validation_confidence(stats_dict)

                # Base confidence depends on cache level
                if cache_level == 'l1':
                    base_confidence = 0.85  # Exact match - high but not overconfident
                elif cache_level == 'l2':
                    base_confidence = 0.75  # Similar (98% cosine) - moderate confidence
                else:  # l3
                    base_confidence = 0.65  # Pattern match - lower confidence

                # Apply validation adjustment
                final_confidence = max(0.4, min(0.95, base_confidence + validation_adj))

                # Only count as high confidence if >= threshold
                if final_confidence >= self.confidence_threshold:
                    self.stats['high_confidence_decisions'] += 1

                result = PreprocessingResult(
                    action=cached_decision,
                    confidence=final_confidence,
                    source='learned',
                    explanation=f"Cached decision from {cache_level} (validated: {validation_adj:+.2f} confidence adjustment)",
                    alternatives=[],
                    parameters={},
                    context=stats_dict,
                    decision_id=decision_id
                )
                return self._add_confidence_warnings(result)

        # LAYER 1: Symbolic engine with adaptive learning
        # Corrections fine-tune symbolic rules instead of creating separate patterns
        symbolic_result = self.symbolic_engine.evaluate(
            column, column_name, target_available
        )

        # Apply adaptive learning adjustments to confidence (if enabled)
        original_confidence = symbolic_result.confidence
        if self.enable_learning and self.adaptive_rules and symbolic_result.context:
            adjusted_confidence = self.adaptive_rules.adjust_confidence(
                symbolic_result.action,
                original_confidence,
                symbolic_result.context
            )

            # Update explanation if confidence was adjusted
            if abs(adjusted_confidence - original_confidence) > 0.01:
                adjustment = self.adaptive_rules.get_adjustment(symbolic_result.context)
                if adjustment:
                    delta = adjusted_confidence - original_confidence
                    symbolic_result.explanation += f" [Adapted: {delta:+.2f} from {adjustment.correction_count} corrections]"

            symbolic_result.confidence = adjusted_confidence

        # If symbolic engine has high confidence (possibly boosted by adaptive learning), use it
        if symbolic_result.confidence >= self.confidence_threshold:
            self.stats['symbolic_decisions'] += 1
            self.stats['high_confidence_decisions'] += 1

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed_ms

            # Update cache with symbolic decision
            if self.enable_cache and self.cache and symbolic_result.context:
                self.cache.set(symbolic_result.context, symbolic_result.action, column_name)

            symbolic_result.decision_id = decision_id
            return self._add_confidence_warnings(symbolic_result)

        # LAYER 2.5: Try meta-learning (statistical heuristics) for universal coverage
        # This bridges the gap between symbolic rules and neural oracle
        if self.enable_meta_learning and self.meta_learner:
            # Use the same stats computed for symbolic engine
            stats_dict = symbolic_result.context if symbolic_result.context else {}

            meta_result = self.meta_learner.decide(stats_dict, column_name)
            if meta_result and meta_result.confidence >= (self.confidence_threshold - 0.1):
                # Accept meta-learning if confidence is close to threshold
                # (e.g., threshold=0.9, accept >=0.8)
                self.stats['meta_learning_decisions'] += 1

                if meta_result.confidence >= self.confidence_threshold:
                    self.stats['high_confidence_decisions'] += 1

                elapsed_ms = (time.time() - start_time) * 1000
                self.stats['total_time_ms'] += elapsed_ms

                # Update cache with meta-learning decision
                if self.enable_cache and self.cache:
                    self.cache.set(stats_dict, meta_result.action, column_name)

                meta_result.decision_id = decision_id
                return self._add_confidence_warnings(meta_result)

        # LAYER 3: Use NeuralOracle for ambiguous cases (<5ms)
        if self.use_neural_oracle and self.neural_oracle:
            # Extract minimal features
            features = self.feature_extractor.extract(column, column_name)

            # Get neural prediction with SHAP explanation
            try:
                # Try SHAP-enabled prediction first
                try:
                    neural_shap_result = self.neural_oracle.predict_with_shap(features, top_k=3)

                    # Blend symbolic and neural if both have medium confidence
                    if symbolic_result.confidence > 0.5:
                        action, confidence, base_explanation = self._blend_decisions_shap(
                            symbolic_result, neural_shap_result
                        )
                    else:
                        action = neural_shap_result['action']
                        confidence = neural_shap_result['confidence']
                        base_explanation = f"Neural oracle prediction (symbolic confidence too low: {symbolic_result.confidence:.2f})"

                    # Build enhanced explanation with SHAP insights
                    shap_explanation = "\n".join(f"  • {exp}" for exp in neural_shap_result['explanation'])
                    explanation = f"{base_explanation}\n\nKey factors:\n{shap_explanation}"

                    self.stats['neural_decisions'] += 1
                    if confidence >= self.confidence_threshold:
                        self.stats['high_confidence_decisions'] += 1

                    elapsed_ms = (time.time() - start_time) * 1000
                    self.stats['total_time_ms'] += elapsed_ms

                    # Get alternatives from neural probabilities
                    alternatives = sorted(
                        [(a, p) for a, p in neural_shap_result['action_probabilities'].items() if a != action],
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    # Update cache with neural decision
                    if self.enable_cache and self.cache and symbolic_result.context:
                        self.cache.set(symbolic_result.context, action, column_name)

                    # Build context with SHAP values
                    enhanced_context = symbolic_result.context.copy() if symbolic_result.context else {}
                    enhanced_context['shap_values'] = neural_shap_result['shap_values']
                    enhanced_context['top_features'] = neural_shap_result['top_features']
                    if symbolic_result.confidence > 0.5:
                        enhanced_context['symbolic_fallback'] = {
                            'action': symbolic_result.action.value,
                            'confidence': symbolic_result.confidence,
                            'reasoning': symbolic_result.explanation
                        }

                    result = PreprocessingResult(
                        action=action,
                        confidence=confidence,
                        source='neural',
                        explanation=explanation,
                        alternatives=alternatives,
                        parameters={},
                        context=enhanced_context,
                        decision_id=decision_id
                    )
                    return self._add_confidence_warnings(result)

                except ImportError:
                    # SHAP not available, fall back to regular prediction
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

                    result = PreprocessingResult(
                        action=action,
                        confidence=confidence,
                        source='neural',
                        explanation=explanation,
                        alternatives=alternatives,
                        parameters={},
                        context=symbolic_result.context,
                        decision_id=decision_id
                    )
                    return self._add_confidence_warnings(result)

            except Exception as e:
                print(f"Warning: Neural oracle failed: {e}")
                # Fall back to symbolic result
                pass

        # LAYER 4: Ultra-conservative fallback
        # When all layers are uncertain, make the safest possible decision
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        # If symbolic had some confidence, use it with a warning
        if symbolic_result.confidence > 0.5:
            self.stats['symbolic_decisions'] += 1
            symbolic_result.decision_id = decision_id
            symbolic_result.explanation = f"[LOW CONFIDENCE] {symbolic_result.explanation}"
            return self._add_confidence_warnings(symbolic_result)

        # Otherwise, use ultra-conservative fallback
        stats_dict = symbolic_result.context if symbolic_result.context else {}
        fallback_result = self._ultra_conservative_fallback(stats_dict, column_name)
        fallback_result.decision_id = decision_id

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        return self._add_confidence_warnings(fallback_result)

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

    def _blend_decisions_shap(
        self,
        symbolic_result: PreprocessingResult,
        neural_shap_result: Dict[str, Any]
    ) -> tuple:
        """
        Blend symbolic and neural decisions with SHAP explanations.

        Args:
            symbolic_result: Result from symbolic engine
            neural_shap_result: SHAP-enabled prediction from neural oracle

        Returns:
            (action, confidence, explanation) tuple
        """
        neural_action = neural_shap_result['action']
        neural_confidence = neural_shap_result['confidence']

        # If both agree, high confidence
        if symbolic_result.action == neural_action:
            confidence = min(0.95, (symbolic_result.confidence + neural_confidence) / 2 + 0.1)
            explanation = f"Both symbolic and neural agree on {symbolic_result.action.value}"
            return symbolic_result.action, confidence, explanation

        # If they disagree, use the one with higher confidence
        if symbolic_result.confidence > neural_confidence:
            return (
                symbolic_result.action,
                symbolic_result.confidence * 0.9,  # Slight penalty for disagreement
                f"Symbolic engine ({symbolic_result.confidence:.2f}) vs neural ({neural_confidence:.2f})"
            )
        else:
            return (
                neural_action,
                neural_confidence * 0.9,
                f"Neural oracle ({neural_confidence:.2f}) vs symbolic ({symbolic_result.confidence:.2f})"
            )

    def _add_confidence_warnings(self, result: PreprocessingResult) -> PreprocessingResult:
        """
        Add warnings based on confidence level and record metrics.

        Args:
            result: Preprocessing result to enhance with warnings

        Returns:
            Enhanced result with appropriate warnings
        """
        # Add confidence warnings
        if result.confidence < CONFIDENCE_LOW:
            result.warning = "⚠️ Very low confidence - manual review strongly recommended"
            result.require_manual_review = True
        elif result.confidence < CONFIDENCE_MEDIUM:
            result.warning = "⚠️ Low confidence - consider reviewing this decision"
        # No warning needed for confidence >= CONFIDENCE_MEDIUM

        # Record layer metrics (Phase 4)
        if hasattr(self, 'layer_metrics') and self.layer_metrics is not None:
            try:
                self.layer_metrics.record_decision(
                    layer=result.source,
                    confidence=result.confidence
                )

                # Save metrics periodically (every 100 decisions)
                if self.layer_metrics.stats.get(result.source):
                    total = self.layer_metrics.stats[result.source].total_decisions
                    if total % 100 == 0:
                        self.layer_metrics.save()
            except Exception as e:
                # Don't fail preprocessing if metrics recording fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to record layer metrics: {e}")

        return result

    def _ultra_conservative_fallback(
        self,
        column_stats: Dict[str, Any],
        column_name: str
    ) -> PreprocessingResult:
        """
        Ultra-conservative fallback for truly ambiguous cases.
        Prioritizes safety: preserves data, doesn't introduce artifacts, reversible.

        Args:
            column_stats: Column statistics
            column_name: Column name

        Returns:
            PreprocessingResult with safe default action
        """
        # Determine safe action based on basic statistics
        is_numeric = column_stats.get('is_numeric', False)
        is_categorical = column_stats.get('is_categorical', False)
        null_pct = column_stats.get('null_pct', 0)
        range_size = column_stats.get('range_size', 0)
        cardinality = column_stats.get('cardinality', 0)

        # High nulls → keep but flag for review
        if null_pct > 0.5:
            return PreprocessingResult(
                action=PreprocessingAction.KEEP_AS_IS,
                confidence=0.60,
                source='conservative_fallback',
                explanation=f"[REVIEW NEEDED] High null percentage ({null_pct:.1%}): keeping as-is for safety, manual review recommended",
                alternatives=[],
                parameters={},
                context=column_stats
            )

        # Numeric data
        if is_numeric:
            # Large range → scale with robust method (handles outliers)
            if range_size > 1000:
                return PreprocessingResult(
                    action=PreprocessingAction.ROBUST_SCALE,
                    confidence=0.65,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] Large numeric range ({range_size:.0f}): using robust scaling as safe default",
                    alternatives=[
                        (PreprocessingAction.STANDARD_SCALE, 0.60),
                        (PreprocessingAction.KEEP_AS_IS, 0.55)
                    ],
                    parameters={},
                    context=column_stats
                )
            # Reasonable range → keep as-is
            else:
                return PreprocessingResult(
                    action=PreprocessingAction.KEEP_AS_IS,
                    confidence=0.70,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] Numeric with reasonable range: keeping as-is to preserve information",
                    alternatives=[
                        (PreprocessingAction.STANDARD_SCALE, 0.65)
                    ],
                    parameters={},
                    context=column_stats
                )

        # Categorical data
        elif is_categorical:
            # Very low cardinality → one-hot (safe and interpretable)
            if cardinality <= 10:
                return PreprocessingResult(
                    action=PreprocessingAction.ONEHOT_ENCODE,
                    confidence=0.68,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] Categorical with {cardinality} categories: one-hot encoding as safe default",
                    alternatives=[
                        (PreprocessingAction.LABEL_ENCODE, 0.60)
                    ],
                    parameters={},
                    context=column_stats
                )
            # Medium cardinality → frequency encoding
            elif cardinality <= 50:
                return PreprocessingResult(
                    action=PreprocessingAction.FREQUENCY_ENCODE,
                    confidence=0.65,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] Categorical with {cardinality} categories: frequency encoding balances information and dimensionality",
                    alternatives=[
                        (PreprocessingAction.ONEHOT_ENCODE, 0.55),
                        (PreprocessingAction.HASH_ENCODE, 0.60)
                    ],
                    parameters={},
                    context=column_stats
                )
            # High cardinality → hash encoding
            else:
                return PreprocessingResult(
                    action=PreprocessingAction.HASH_ENCODE,
                    confidence=0.68,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] High cardinality ({cardinality}): hash encoding prevents dimensionality explosion",
                    alternatives=[
                        (PreprocessingAction.FREQUENCY_ENCODE, 0.62)
                    ],
                    parameters={},
                    context=column_stats
                )

        # Unknown type → absolute safest option is keep as-is
        else:
            return PreprocessingResult(
                action=PreprocessingAction.KEEP_AS_IS,
                confidence=0.60,
                source='conservative_fallback',
                explanation=f"[REVIEW NEEDED] Ambiguous data type: keeping as-is to preserve information, manual review recommended",
                alternatives=[],
                parameters={},
                context=column_stats
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
        Process a user correction to adapt symbolic rules (privacy-preserving).

        NEW APPROACH (V2.1):
        Instead of creating separate learned patterns, corrections are used to
        fine-tune symbolic rule parameters and confidence scores. This prevents
        overgeneralization while still learning from user feedback.

        Args:
            column: Column data (only used to extract stats, not stored)
            column_name: Column name
            wrong_action: Action that was incorrect
            correct_action: Correct action
            confidence: Confidence of the wrong prediction

        Returns:
            Learning result with adaptive rule information
        """
        if not self.enable_learning or not self.adaptive_rules:
            return {'learned': False, 'reason': 'Learning disabled'}

        # Convert to pandas Series if needed
        if isinstance(column, (list, np.ndarray)):
            column = pd.Series(column, name=column_name)

        # Convert actions to enum if strings
        if isinstance(wrong_action, str):
            wrong_action = PreprocessingAction(wrong_action)
        if isinstance(correct_action, str):
            correct_action = PreprocessingAction(correct_action)

        # Extract column statistics (privacy-preserving)
        stats = self.symbolic_engine.compute_column_statistics(column, column_name)
        stats_dict = stats.to_dict()

        # IMPORTANT: Invalidate cache if the decision was cached
        # This prevents reusing incorrect decisions
        if self.enable_cache and self.cache:
            self.cache.invalidate_decision(stats_dict, column_name)

        # Record correction to adapt symbolic rules
        self.adaptive_rules.record_correction(
            column_stats=stats_dict,
            wrong_action=wrong_action,
            correct_action=correct_action
        )

        # Get adjustment information
        adjustment = self.adaptive_rules.get_adjustment(stats_dict)
        stats = self.adaptive_rules.get_statistics()

        # Get pattern-specific correction count
        pattern_key = self.adaptive_rules._identify_pattern(stats_dict)
        pattern_corrections = self.adaptive_rules.correction_patterns.get(pattern_key, [])
        correction_count = len(pattern_corrections)

        result = {
            'learned': True,
            'approach': 'adaptive_rules',  # NEW: Fine-tuning instead of separate patterns
            'pattern_category': pattern_key,
            'cache_invalidated': True,
            'adjustment_active': adjustment is not None,
            'total_corrections': stats['total_corrections'],
            'patterns_tracked': stats['patterns_tracked'],
            'pattern_corrections': correction_count,  # NEW: Show user their progress
            'corrections_needed': max(0, self.adaptive_rules.min_corrections_for_adjustment - correction_count)  # NEW
        }

        if adjustment:
            result['confidence_boost'] = f"+{adjustment.confidence_delta:.3f}"
            result['preferred_action'] = adjustment.action.value
            result['correction_support'] = adjustment.correction_count
            result['applicable_to'] = f"Similar columns matching '{adjustment.rule_category}' pattern"
        else:
            # Provide feedback even if no adjustment yet
            if correction_count > 0:
                result['message'] = f"Correction recorded! {correction_count}/{self.adaptive_rules.min_corrections_for_adjustment} corrections for this pattern. " + \
                                  f"{max(1, self.adaptive_rules.min_corrections_for_adjustment - correction_count)} more needed to activate adjustment."

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
            'meta_learning_pct': self.stats['meta_learning_decisions'] / total * 100 if total > 0 else 0,
            'neural_pct': self.stats['neural_decisions'] / total * 100 if total > 0 else 0,
            'high_confidence_pct': self.stats['high_confidence_decisions'] / total * 100 if total > 0 else 0,
            'avg_time_ms': self.stats['total_time_ms'] / total if total > 0 else 0,
            'symbolic_coverage': self.symbolic_engine.coverage()
        }

        # Add learning statistics if available
        if self.pattern_learner:
            stats['learning'] = self.pattern_learner.get_statistics()

        # Add meta-learning statistics if available
        if self.meta_learner:
            stats['meta_learning'] = self.meta_learner.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            'total_decisions': 0,
            'learned_decisions': 0,
            'symbolic_decisions': 0,
            'meta_learning_decisions': 0,
            'neural_decisions': 0,
            'high_confidence_decisions': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0
        }
        self.symbolic_engine.reset_stats()
        if self.meta_learner:
            self.meta_learner.reset_statistics()

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
