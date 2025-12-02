"""
HybridPreprocessingOracle - Combines ML predictions with rule-based fallbacks.

This oracle uses:
1. XGBoost + LightGBM ensemble for ML predictions
2. Rule-based fallbacks for edge cases
3. 40 meta-features for comprehensive column analysis

Trained to predict 10 preprocessing actions:
- clip_outliers, drop_column, frequency_encode, keep_as_is, log1p_transform,
- log_transform, minmax_scale, robust_scale, sqrt_transform, standard_scale
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..core.actions import PreprocessingAction
from ..features.meta_extractor import MetaFeatureExtractor, MetaFeatures


@dataclass
class HybridPrediction:
    """Prediction from the HybridPreprocessingOracle."""
    action: PreprocessingAction
    confidence: float
    source: str  # 'ml' or 'rule'
    reason: str
    ml_probabilities: Optional[Dict[PreprocessingAction, float]] = None


class HybridPreprocessingOracle:
    """
    Hybrid preprocessing oracle combining ML and rules.
    
    Uses XGBoost + LightGBM ensemble with rule-based fallbacks for edge cases.
    74.7% ML accuracy with rules handling special cases.
    """
    
    def __init__(
        self,
        xgb_model=None,
        lgb_model=None,
        label_encoder=None,
        feature_extractor: Optional[MetaFeatureExtractor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hybrid oracle.
        
        Args:
            xgb_model: Trained XGBoost model
            lgb_model: Trained LightGBM model
            label_encoder: Label encoder for actions
            feature_extractor: Meta-feature extractor
            config: Configuration dictionary
        """
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.label_encoder = label_encoder
        self.feature_extractor = feature_extractor or MetaFeatureExtractor()
        self.config = config or {}
        
        # Thresholds for rule-based overrides
        self.ml_confidence_threshold = 0.6  # Use rules if ML confidence < this
        self.rule_confidence_threshold = 0.85  # Use rules if rule confidence > this
    
    def predict_column(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> HybridPrediction:
        """
        Predict preprocessing action for a column using hybrid approach.
        
        Args:
            column: Column data
            column_name: Name of the column
            
        Returns:
            HybridPrediction with action, confidence, source, and reason
        """
        # 1. Extract 40 meta-features
        features = self.feature_extractor.extract(column, column_name)
        
        # 2. Get ML ensemble prediction
        ml_action, ml_confidence, ml_probs = self._get_ml_prediction(features)
        
        # 3. Get rule-based prediction
        rule_action, rule_confidence, rule_reason = self._apply_rules(column, column_name, features)
        
        # 4. Decide which to use
        if rule_action is not None and (
            ml_confidence < self.ml_confidence_threshold or 
            rule_confidence > self.rule_confidence_threshold
        ):
            # Use rule-based prediction
            return HybridPrediction(
                action=rule_action,
                confidence=rule_confidence,
                source='rule',
                reason=rule_reason,
                ml_probabilities=ml_probs
            )
        else:
            # Use ML prediction
            reason = f"ML ensemble prediction (XGBoost + LightGBM)"
            return HybridPrediction(
                action=ml_action,
                confidence=ml_confidence,
                source='ml',
                reason=reason,
                ml_probabilities=ml_probs
            )
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Predict preprocessing actions for all columns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column to exclude (if any)
            
        Returns:
            DataFrame with columns: column_name, action, confidence, source, reason
        """
        results = []
        
        for col in df.columns:
            if col == target_column:
                continue
            
            prediction = self.predict_column(df[col], col)
            
            results.append({
                'column_name': col,
                'action': prediction.action.value,
                'confidence': prediction.confidence,
                'source': prediction.source,
                'reason': prediction.reason
            })
        
        return pd.DataFrame(results)
    
    def _get_ml_prediction(
        self,
        features: MetaFeatures
    ) -> Tuple[PreprocessingAction, float, Dict[PreprocessingAction, float]]:
        """
        Get ML ensemble prediction (XGBoost + LightGBM average).
        
        Args:
            features: Extracted meta-features
            
        Returns:
            Tuple of (action, confidence, probabilities_dict)
        """
        if self.xgb_model is None or self.lgb_model is None or self.label_encoder is None:
            # No ML models available, return default
            return PreprocessingAction.KEEP_AS_IS, 0.0, {}
        
        # Convert features to array
        X = features.to_array().reshape(1, -1)
        
        # Get predictions from both models
        xgb_probs = self.xgb_model.predict_proba(X)[0]
        lgb_probs = self.lgb_model.predict_proba(X)[0]
        
        # Average ensemble
        ensemble_probs = (xgb_probs + lgb_probs) / 2.0
        
        # Get top prediction
        top_idx = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[top_idx])
        
        # Get action name from label encoder
        action_name = self.label_encoder.inverse_transform([top_idx])[0]
        
        # Map to PreprocessingAction enum
        action = self._map_action_name(action_name)
        
        # Build probabilities dictionary
        ml_probs = {}
        for idx, prob in enumerate(ensemble_probs):
            action_name = self.label_encoder.inverse_transform([idx])[0]
            mapped_action = self._map_action_name(action_name)
            ml_probs[mapped_action] = float(prob)
        
        return action, confidence, ml_probs
    
    def _apply_rules(
        self,
        column: pd.Series,
        column_name: str,
        features: MetaFeatures
    ) -> Tuple[Optional[PreprocessingAction], float, str]:
        """
        Apply rule-based fallback logic.
        
        Returns:
            Tuple of (action, confidence, reason) or (None, 0.0, "") if no rule applies
        """
        # Rule 1: Constant columns → drop_column
        # If has no variance and very low unique ratio, it's constant
        if features.has_variance == 0.0 and features.std == 0.0:
            return PreprocessingAction.DROP_COLUMN, 0.95, "Constant column with no variance"
        
        # Rule 2: ID-like columns → drop_column
        if features.has_id > 0.5 and features.unique_ratio > 0.95:
            return PreprocessingAction.DROP_COLUMN, 0.90, "ID-like column with high cardinality"
        
        # Rule 3: High cardinality categorical → drop_column or frequency_encode
        if features.is_categorical > 0.5 and features.unique_ratio > 0.7:
            if features.unique_ratio > 0.95:
                return PreprocessingAction.DROP_COLUMN, 0.85, "High cardinality categorical (>95% unique)"
            else:
                return PreprocessingAction.FREQUENCY_ENCODE, 0.80, "High cardinality categorical (70-95% unique)"
        
        # Rule 4: Highly skewed positive data → log_transform or log1p_transform
        if features.is_numeric > 0.5 and features.can_log > 0.5:
            if abs(features.skewness) > 2.0:
                if features.zero_ratio > 0.1:
                    return PreprocessingAction.LOG1P_TRANSFORM, 0.87, "Highly skewed with zeros, use log1p"
                else:
                    return PreprocessingAction.LOG_TRANSFORM, 0.88, "Highly skewed positive data"
        
        # Rule 5: Data with many outliers → clip_outliers
        if features.is_numeric > 0.5 and features.outlier_ratio > 0.15:
            return PreprocessingAction.CLIP_OUTLIERS, 0.83, f"High outlier ratio ({features.outlier_ratio:.2%})"
        
        # Rule 6: Large range numeric data → standard_scale
        if features.is_numeric > 0.5 and features.has_range > 0.5:
            if features.std > 100 or (features.max - features.min) > 1000:
                return PreprocessingAction.STANDARD_SCALE, 0.82, "Large range numeric data"
        
        # Rule 7: Boolean-like data
        if features.is_bool > 0.5 or (features.unique_ratio < 0.05 and features.cardinality_low > 0.5):
            return PreprocessingAction.KEEP_AS_IS, 0.75, "Boolean or binary data"
        
        # Rule 8: Missing-heavy columns
        if features.missing_ratio > 0.6:
            return PreprocessingAction.DROP_COLUMN, 0.80, f"High missing ratio ({features.missing_ratio:.2%})"
        
        # No rule applies
        return None, 0.0, ""
    
    def _map_action_name(self, action_name: str) -> PreprocessingAction:
        """
        Map action name from model to PreprocessingAction enum.
        
        Args:
            action_name: Action name from model
            
        Returns:
            PreprocessingAction enum
        """
        # Mapping from model action names to enum
        mapping = {
            'clip_outliers': PreprocessingAction.CLIP_OUTLIERS,
            'drop_column': PreprocessingAction.DROP_COLUMN,
            'frequency_encode': PreprocessingAction.FREQUENCY_ENCODE,
            'keep_as_is': PreprocessingAction.KEEP_AS_IS,
            'log1p_transform': PreprocessingAction.LOG1P_TRANSFORM,
            'log_transform': PreprocessingAction.LOG_TRANSFORM,
            'minmax_scale': PreprocessingAction.MINMAX_SCALE,
            'robust_scale': PreprocessingAction.ROBUST_SCALE,
            'sqrt_transform': PreprocessingAction.SQRT_TRANSFORM,
            'standard_scale': PreprocessingAction.STANDARD_SCALE,
            # Also support enum values directly
            'CLIP_OUTLIERS': PreprocessingAction.CLIP_OUTLIERS,
            'DROP_COLUMN': PreprocessingAction.DROP_COLUMN,
            'FREQUENCY_ENCODE': PreprocessingAction.FREQUENCY_ENCODE,
            'KEEP_AS_IS': PreprocessingAction.KEEP_AS_IS,
            'LOG1P_TRANSFORM': PreprocessingAction.LOG1P_TRANSFORM,
            'LOG_TRANSFORM': PreprocessingAction.LOG_TRANSFORM,
            'MINMAX_SCALE': PreprocessingAction.MINMAX_SCALE,
            'ROBUST_SCALE': PreprocessingAction.ROBUST_SCALE,
            'SQRT_TRANSFORM': PreprocessingAction.SQRT_TRANSFORM,
            'STANDARD_SCALE': PreprocessingAction.STANDARD_SCALE,
        }
        
        if action_name in mapping:
            return mapping[action_name]
        
        # Try to match by enum value
        for action in PreprocessingAction:
            if action.value == action_name:
                return action
        
        # Default fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown action '{action_name}', defaulting to KEEP_AS_IS")
        return PreprocessingAction.KEEP_AS_IS
