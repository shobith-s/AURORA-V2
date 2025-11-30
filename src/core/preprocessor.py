"""
Main Preprocessing Pipeline - Integrates all layers.
Symbolic Engine (with adaptive learning) -> NeuralOracle
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import time

from ..symbolic.engine import SymbolicEngine
from ..neural.oracle import NeuralOracle, get_neural_oracle
from ..features.minimal_extractor import MinimalFeatureExtractor, get_feature_extractor
from .actions import PreprocessingAction, PreprocessingResult
from .actions import PreprocessingAction, PreprocessingResult
from .explainer import get_explainer
# DISABLED: DatasetAnalyzer not used (results ignored)
# from ..analysis.dataset_analyzer import DatasetAnalyzer

# Confidence thresholds for decision quality
CONFIDENCE_HIGH = 0.9      # Auto-apply decision (highly confident)
CONFIDENCE_MEDIUM = 0.7    # Show warning (moderate confidence)
CONFIDENCE_LOW = 0.5       # Require manual review (low confidence)


class IntelligentPreprocessor:
    """
    Main preprocessing pipeline with adaptive learning architecture (V3):

    0. Cache (validated decisions) - Instant lookup
    1. Symbolic rules (185+ rules, including learned) - PRIMARY & ONLY DECISION LAYER
       └─ Dynamically enhanced: Learner creates NEW symbolic rules from corrections
    2. NeuralOracle (ML predictions) - Ambiguous cases only

    NEW Learning Architecture (V3):
    - Learner NEVER makes direct decisions (prevents overgeneralization)
    - Training phase (2-9 corrections): Analyze patterns, compute adjustments
    - Production phase (10+ corrections): CREATE new symbolic Rule objects
    - New rules are INJECTED into symbolic engine automatically
    - Symbolic engine remains the ONLY decision-maker at all times

    Benefits:
    - No learner decision path = no overgeneralization from limited data
    - Learned rules are inspectable, maintainable symbolic logic
    - All decisions traceable to explicit rules
    - Domain adaptation through rule creation, not override
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,  # CHANGED: 0.9 → 0.75 for more neural participation
        use_neural_oracle: bool = True,
        enable_learning: bool = True,
        neural_model_path: Optional[Path] = None,
        db_url: Optional[str] = None  # NEW: Database URL for learning engine
    ):
        """
        Initialize the intelligent preprocessor.

        Args:
            confidence_threshold: Minimum confidence for symbolic engine
            use_neural_oracle: Whether to use neural oracle for low-confidence cases
            enable_learning: Whether to enable pattern learning
            neural_model_path: Path to neural oracle model
        """
        import logging
        logger = logging.getLogger(__name__)

        self.confidence_threshold = confidence_threshold
        self.use_neural_oracle = use_neural_oracle
        self.enable_learning = enable_learning
        self.db_url = db_url or "sqlite:///./aurora.db"

        # Initialize components with error handling
        try:
            self.symbolic_engine = SymbolicEngine(confidence_threshold=confidence_threshold)
        except Exception as e:
            logger.error(f"Failed to initialize symbolic engine: {e}")
            raise RuntimeError(f"Critical component failed: symbolic engine - {e}")

        # Adaptive Learning Engine (database-backed with validation and A/B testing)
        self.learning_engine = None
        if enable_learning:
            try:
                from ..learning.adaptive_engine import AdaptiveLearningEngine
                self.learning_engine = AdaptiveLearningEngine(
                    db_url=self.db_url,
                    min_support=10,  # Require 10+ corrections per pattern
                    similarity_threshold=0.85,
                    validation_sample_size=20,
                    ab_test_min_decisions=100,
                    ab_test_success_threshold=0.80
                )
                
                # Load active production rules into symbolic engine
                active_rules = self.learning_engine.get_active_rules()
                if active_rules:
                    logger.info(f"Loaded {len(active_rules)} validated production rules")

                    # Convert LearnedRule database objects to Rule objects and inject
                    from ..learning.rule_converter import convert_learned_rules_batch
                    converted_rules = convert_learned_rules_batch(
                        active_rules,
                        similarity_threshold=0.85
                    )

                    # Inject converted rules into symbolic engine
                    for rule in converted_rules:
                        self.symbolic_engine.add_rule(rule)

                    logger.info(f"Successfully injected {len(converted_rules)} learned rules into symbolic engine")

            except Exception as e:
                logger.warning(f"Learning engine initialization failed, continuing without it: {e}")
                self.learning_engine = None
                self.enable_learning = False


        # Feature extractor (required for neural oracle)
        try:
            self.feature_extractor = get_feature_extractor()
        except Exception as e:
            logger.warning(f"Feature extractor initialization failed: {e}")
            self.feature_extractor = None
            self.use_neural_oracle = False  # Disable neural oracle if feature extractor fails

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
        metadata: Optional[Dict[str, Any]] = None,
        context: str = "general"
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

        # CRITICAL FIX: Infer numeric types from object dtype (handles JSON data)
        # When data comes from frontend as JSON, pandas treats everything as object dtype
        # We need to attempt numeric conversion to get proper type detection
        if column.dtype == 'object' or column.dtype == 'O':
            try:
                # Try to convert to numeric
                numeric_column = pd.to_numeric(column, errors='coerce')
                # FIXED: Increased threshold from 50% to 90% to avoid incorrect conversions
                # (e.g., dates that partially convert to numbers)
                conversion_rate = numeric_column.notna().sum() / len(column) if len(column) > 0 else 0
                if conversion_rate > 0.9:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Type inference: '{column_name}' converted from object to numeric (success rate: {conversion_rate:.2%})")
                    column = numeric_column
                    # Update the series name to maintain consistency
                    column.name = column_name
            except Exception as e:
                # If conversion fails completely, keep as object dtype
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Type inference failed for '{column_name}': {e}")

        # Update statistics
        self.stats['total_decisions'] += 1

        # Generate decision ID
        decision_id = str(uuid.uuid4())

        # LAYER 1: Symbolic engine (THE ONLY DECISION-MAKER)
        # Note: Symbolic engine now includes validated production rules from learning engine
        # NEW: Pass metadata as context to symbolic engine
        symbolic_result = self.symbolic_engine.evaluate(
            column, column_name, target_available, context=metadata
        )

        # If symbolic engine has high confidence, use it
        if symbolic_result.confidence >= self.confidence_threshold:
            self.stats['symbolic_decisions'] += 1
            self.stats['high_confidence_decisions'] += 1

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_time_ms'] += elapsed_ms

            symbolic_result.decision_id = decision_id
            return self._add_confidence_warnings(symbolic_result, context, column_name)

        # LAYER 2.5: Meta-learning removed to simplify architecture and enable Neural Oracle
        # (Meta-learner code removed)

        # LAYER 3: Use NeuralOracle for ambiguous cases (<5ms)
        if self.use_neural_oracle and self.neural_oracle:
            # Extract minimal features
            try:
                features = self.feature_extractor.extract(column, column_name)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Feature extraction failed for '{column_name}': {e}")
                # Fall through to conservative fallback
                features = None

            if features is not None:
                # Get neural prediction with SHAP explanation
                try:
                    # Try SHAP-enabled prediction first
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
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Neural oracle prediction failed for '{column_name}': {e}")
                    # Fall back to symbolic result or conservative fallback
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
            return self._add_confidence_warnings(symbolic_result, context, column_name)

        # Otherwise, use ultra-conservative fallback
        stats_dict = symbolic_result.context if symbolic_result.context else {}
        fallback_result = self._ultra_conservative_fallback(stats_dict, column_name)
        fallback_result.decision_id = decision_id

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        return self._add_confidence_warnings(fallback_result, context, column_name)

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

    def _apply_context_bias(
        self,
        result: PreprocessingResult,
        context: str,
        column_name: str
    ) -> PreprocessingResult:
        """Apply context-specific bias to the decision."""
        if context == "general":
            return result

        explanation_prefix = f"[CONTEXT: {context.upper()}]"
        
        # REGRESSION CONTEXT (Predicting Numbers)
        if context == "regression":
            # Prefer scaling for numerics
            if result.action == PreprocessingAction.KEEP_AS_IS and "numeric" in result.explanation.lower():
                # If it's numeric but kept as is, suggest scaling
                result.action = PreprocessingAction.STANDARD_SCALE
                result.confidence = 0.85
                result.explanation = f"{explanation_prefix} Regression models require scaled features. Changed from KEEP_AS_IS to STANDARD_SCALE."
                result.source = "context_bias"
            
            # Prefer One-Hot over Label Encoding for low cardinality (to avoid ordinal assumption)
            elif result.action == PreprocessingAction.LABEL_ENCODE:
                # Check if we can switch to One-Hot
                for alt_action, alt_conf in result.alternatives:
                    if alt_action == PreprocessingAction.ONEHOT_ENCODE.value:
                        result.action = PreprocessingAction.ONEHOT_ENCODE
                        result.confidence = max(result.confidence, alt_conf)
                        result.explanation = f"{explanation_prefix} Regression models prefer One-Hot Encoding to avoid ordinal assumptions."
                        result.source = "context_bias"
                        break

        # CLASSIFICATION CONTEXT (Predicting Categories)
        elif context == "classification":
            # If target column (heuristic check on name), prefer Label Encoding
            if column_name.lower() in ['target', 'class', 'label', 'y']:
                result.action = PreprocessingAction.LABEL_ENCODE
                result.confidence = 0.95
                result.explanation = f"{explanation_prefix} Likely target column for classification. Enforcing Label Encoding."
                result.source = "context_bias"

        return result

    def _add_confidence_warnings(
        self,
        result: PreprocessingResult,
        context: str = "general",
        column_name: str = ""
    ) -> PreprocessingResult:
        """
        Add warnings based on confidence level and record metrics.
        Also applies context-specific bias and enhances explanation.
        """
        # Apply context bias first
        if context != "general":
            result = self._apply_context_bias(result, context, column_name)

        # Enhance explanation with detailed reasoning
        explainer = get_explainer()
        try:
            enhanced_explanation = explainer.generate_explanation(
                action=result.action,
                confidence=result.confidence,
                source=result.source,
                context=result.context or {},
                column_name=column_name or "column"
            )
            result.explanation = enhanced_explanation
        except Exception as e:
            # If explanation enhancement fails, keep original but ensure it's not empty
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to enhance explanation: {e}")
            
            # Fallback explanation if original is missing or empty
            if not result.explanation:
                result.explanation = f"Recommended action: {result.action.value} (confidence: {result.confidence:.1%})"

        # Add confidence warnings
        if result.confidence < CONFIDENCE_LOW:
            result.warning = "⚠️ Very low confidence - manual review strongly recommended"
            result.require_manual_review = True
        elif result.confidence < CONFIDENCE_MEDIUM:
            result.warning = "⚠️ Low confidence - consider reviewing this decision"
        # No warning needed for confidence >= CONFIDENCE_MEDIUM

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
            # Medium/High cardinality → frequency encoding (safer than hash)
            elif cardinality <= 1000:
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
            # Very high cardinality → hash encoding
            else:
                return PreprocessingResult(
                    action=PreprocessingAction.HASH_ENCODE,
                    confidence=0.68,
                    source='conservative_fallback',
                    explanation=f"[CONSERVATIVE] Very high cardinality ({cardinality}): hash encoding prevents dimensionality explosion",
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
        confidence: float = 0.0,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Process a user correction with validation and A/B testing (Option B).

        NEW APPROACH (V4 - Option B):
        - Records correction in database (privacy-preserving)
        - Requires 10+ corrections per pattern type before rule creation
        - Validates rules before activation (80% accuracy threshold)
        - A/B tests new rules before promoting to production
        - No cache invalidation (cache removed)

        Args:
            column: Column data (only used to extract stats, not stored)
            column_name: Column name
            wrong_action: Action that was incorrect
            correct_action: Correct action
            confidence: Confidence of the wrong prediction
            user_id: User ID for tracking (from JWT in production)

        Returns:
            Learning result with validation and A/B test information
        """
        if not self.enable_learning or not self.learning_engine:
            return {'learned': False, 'reason': 'Learning disabled'}

        # Convert to pandas Series if needed
        if isinstance(column, (list, np.ndarray)):
            column = pd.Series(column, name=column_name)

        # Convert actions to strings if enums
        if isinstance(wrong_action, PreprocessingAction):
            wrong_action = wrong_action.value
        if isinstance(correct_action, PreprocessingAction):
            correct_action = correct_action.value

        # Extract column statistics (privacy-preserving)
        stats = self.symbolic_engine.compute_column_statistics(column, column_name)
        stats_dict = stats.to_dict()

        # Record correction in database
        result = self.learning_engine.record_correction(
            user_id=user_id,
            column_stats=stats_dict,
            wrong_action=wrong_action,
            correct_action=correct_action,
            confidence=confidence
        )

        if not result.get('recorded'):
            return {
                'learned': False,
                'error': result.get('error', 'Failed to record correction')
            }

        # Build response with learning progress
        response = {
            'learned': True,
            'approach': 'database_with_validation_and_ab_testing',
            'pattern_hash': result.get('pattern_hash'),
            'pattern_type': result.get('pattern_type', 'unknown'),
            'correction_support': result.get('correction_support', 0),
            'corrections_needed': result.get('corrections_needed_for_production', 0),
        }

        # If rule was created, add rule information
        if result.get('new_rule_created'):
            response.update({
                'rule_created': True,
                'rule_name': result.get('rule_name'),
                'rule_confidence': result.get('rule_confidence'),
                'rule_support': result.get('rule_support'),
                'ab_test_started': result.get('ab_test_started', False),
                'validation_required': result.get('validation_required', False),
                'message': f"🎯 NEW RULE CREATED: Rule '{result.get('rule_name')}' created for pattern '{result.get('pattern_type')}' " +
                          f"with {result.get('rule_support')} corrections. Starting A/B test and validation."
            })
        else:
            # Still collecting corrections
            corrections_left = result.get('corrections_needed_for_production', 0)
            current_count = result.get('correction_support', 0)
            pattern_type = result.get('pattern_type', 'unknown')
            
            response.update({
                'rule_created': False,
                'message': f"📝 Correction recorded for pattern '{pattern_type}'. " +
                          f"Progress: {current_count}/10 corrections. " +
                          f"{corrections_left} more needed to create validated rule."
            })

        return response

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        context: str = "general"
    ) -> Dict[str, PreprocessingResult]:
        """
        Preprocess all columns in a dataframe.

        Args:
            df: Input dataframe
            target_column: Name of target column (if any)

        Returns:
            Dictionary mapping column names to PreprocessingResults
        """
        # DISABLED: Inter-column analysis (unused - results ignored)
        # analyzer = DatasetAnalyzer()
        # primary_keys = analyzer.detect_primary_keys(df)
        # correlations = analyzer.find_numeric_correlations(df)
        # foreign_keys = analyzer.analyze_foreign_key_candidates(df)

        results = {}

        for column_name in df.columns:
            if column_name == target_column:
                continue  # Skip target column

            target_available = target_column is not None

            # Use default context values (DatasetAnalyzer disabled as results weren't used)
            col_context = {
                "is_primary_key": False,
                "is_foreign_key": False,
                "correlation_with_target": 0.0
            }

            # DISABLED: Correlation computation (DatasetAnalyzer removed)
            # if target_available and column_name in correlations:
            #     for other_col, corr_value in correlations[column_name]:
            #         if other_col == target_column:
            #             col_context["correlation_with_target"] = corr_value
            #             break

            result = self.preprocess_column(
                df[column_name],
                column_name,
                target_available,
                context=context,
                metadata=col_context  # Pass analysis results as metadata
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

        # Add adaptive learning statistics if available
        if self.adaptive_rules:
            stats['adaptive_learning'] = self.adaptive_rules.get_statistics()

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

    def save_learned_rules(self, path: Path):
        """
        Save learned rules to disk (via adaptive_rules persistence).

        Args:
            path: Path to save rules
        """
        if self.adaptive_rules:
            self.adaptive_rules.save()

    def load_learned_rules(self, path: Path):
        """
        Load learned rules from disk and inject into symbolic engine.

        Args:
            path: Path to load rules from
        """
        if self.adaptive_rules:
            self.adaptive_rules.load()
            # Inject loaded rules into symbolic engine
            learned_rules = self.adaptive_rules.get_all_learned_rules()
            for rule in learned_rules:
                self.symbolic_engine.add_rule(rule)


# Singleton instance
_preprocessor_instance: Optional[IntelligentPreprocessor] = None


def get_preprocessor(**kwargs) -> IntelligentPreprocessor:
    """Get the global preprocessor instance."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = IntelligentPreprocessor(**kwargs)
    return _preprocessor_instance
