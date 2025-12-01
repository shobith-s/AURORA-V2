"""
NeuralOracle - Pre-trained ensemble for ambiguous preprocessing decisions.

INFERENCE ONLY - This module loads and uses a pre-trained ensemble model.
NO training occurs during runtime.

Model Details:
- Architecture: XGBoost + LightGBM ensemble (VotingClassifier, soft voting)
- Accuracy: 89.4% on test data
- Trained: November 2025 on 40 diverse OpenML datasets
- Model File: models/neural_oracle_v2_improved_20251129_150244.pkl
- Use Case: Handles cases where symbolic engine confidence < 0.75

Training:
- To retrain the model, use: validator/scripts/train_neural_oracle_v2.py
- Training should be done offline, not in production
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from dataclasses import dataclass

try:
    import xgboost as xgb
    from sklearn.ensemble import VotingClassifier
    ENSEMBLE_AVAILABLE = True
except ImportError:
    xgb = None
    ENSEMBLE_AVAILABLE = False

from ..core.actions import PreprocessingAction
from ..features.minimal_extractor import MinimalFeatures


@dataclass
class OraclePrediction:
    """Prediction from the NeuralOracle."""
    action: PreprocessingAction
    confidence: float
    action_probabilities: Dict[PreprocessingAction, float]
    feature_importance: Optional[Dict[str, float]] = None


class NeuralOracle:
    """
    Pre-trained ensemble oracle for ambiguous preprocessing decisions.

    INFERENCE ONLY - Uses pre-trained XGBoost + LightGBM ensemble (89.4% accuracy).
    No training occurs during runtime.
    Designed for <5ms inference time.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the NeuralOracle.
        
        INFERENCE ONLY - Loads pre-trained model (single XGBoost or ensemble).
        NO training happens during runtime.

        Args:
            model_path: Path to pre-trained model file
        """
        if xgb is None:
            raise ImportError(
                "XGBoost is required for NeuralOracle. "
                "Install with: pip install xgboost lightgbm"
            )

        self.model = None  # Will be sklearn VotingClassifier or xgb.Booster
        self.action_encoder = {}
        self.action_decoder = {}
        self.feature_names = [
            'null_ratio', 'unique_ratio', 'numeric_ratio',
            'mean_length', 'std_length', 'has_special_chars',
            'has_mixed_types', 'cardinality', 'entropy',
            'is_sequential', 'date_ratio', 'email_ratio',
            'url_ratio', 'phone_ratio', 'outlier_ratio',
            'skewness', 'kurtosis', 'cv', 'iqr_ratio', 'range_ratio'
        ]

        # Try to load pre-trained ensemble model by default
        if model_path is None:
            # First try the new ensemble model
            ensemble_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v2_improved_20251129_150244.pkl"
            if ensemble_path.exists():
                model_path = ensemble_path
            else:
                # Fallback to old model if ensemble not found
                default_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v1.pkl"
                if default_path.exists():
                    model_path = default_path
        
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        features: List[MinimalFeatures],
        labels: List[PreprocessingAction],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        ⚠️ DEPRECATED: Training should be done offline using validator scripts.

        This method is kept for backwards compatibility but should NOT be used
        in production. Use validator/scripts/train_neural_oracle_v2.py instead.

        Raises:
            RuntimeError: Always raises to prevent accidental training
        """
        raise RuntimeError(
            "On-the-fly training is disabled. Use pre-trained ensemble model.\n"
            "To retrain, use: python validator/scripts/train_neural_oracle_v2.py"
        )

    def predict(
        self,
        features: MinimalFeatures,
        return_probabilities: bool = True,
        return_feature_importance: bool = False
    ) -> OraclePrediction:
        """
        Predict using PRE-TRAINED model (inference only).

        Works with both:
        - Pre-trained ensemble (VotingClassifier with XGBoost + LightGBM)
        - Pre-trained single XGBoost (backwards compatibility)

        NO TRAINING occurs in this method.

        Args:
            features: Minimal features extracted from column
            return_probabilities: Return probabilities for all actions
            return_feature_importance: Return feature importance (slower)

        Returns:
            OraclePrediction with action and confidence
        """
        if self.model is None:
            raise ValueError("No pre-trained model loaded. Cannot make predictions.")

        # Convert features to array
        X = features.to_array().reshape(1, -1)
        
        # Check if model is sklearn or XGBoost
        is_sklearn = hasattr(self.model, 'predict_proba')
        
        if is_sklearn:
            # Sklearn model (VotingClassifier, etc.)
            # Get feature names from model if available to avoid warnings
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            else:
                # Fallback to generic column names matching model training
                feature_names = [f'Column_{i}' for i in range(X.shape[1])]
            
            # Convert to DataFrame with proper feature names to match training
            X_df = pd.DataFrame(X, columns=feature_names)
            probs = self.model.predict_proba(X_df)[0]
        else:
            # XGBoost model
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            probs = self.model.predict(dmatrix)[0]

        # Map v2 model actions to system PreprocessingAction enums.
        # These mappings cover:
        # 1. V2 model class labels (from the trained ensemble)
        # 2. Semantic aliases (e.g., 'retain_column' -> KEEP_AS_IS)
        # 3. Legacy model class labels for backwards compatibility
        mapping = {
            # V2 model class labels (from model.classes_)
            'drop_column': PreprocessingAction.DROP_COLUMN,
            'encode_categorical': PreprocessingAction.LABEL_ENCODE,
            'keep_as_is': PreprocessingAction.KEEP_AS_IS,
            'log_transform': PreprocessingAction.LOG_TRANSFORM,
            'onehot_encode': PreprocessingAction.ONEHOT_ENCODE,
            'parse_boolean': PreprocessingAction.PARSE_BOOLEAN,
            'parse_datetime': PreprocessingAction.PARSE_DATETIME,
            'retain_column': PreprocessingAction.KEEP_AS_IS,
            'scale': PreprocessingAction.STANDARD_SCALE,
            'scale_or_normalize': PreprocessingAction.STANDARD_SCALE,
            'standard_scale': PreprocessingAction.STANDARD_SCALE,
            # Legacy mappings for backwards compatibility
            'drop': PreprocessingAction.DROP_COLUMN,
            'fill_zero': PreprocessingAction.FILL_NULL_MODE,
            'fill_forward': PreprocessingAction.FILL_NULL_FORWARD,
            'fill_backward': PreprocessingAction.FILL_NULL_BACKWARD,
            'fill_mean': PreprocessingAction.FILL_NULL_MEAN,
            'fill_median': PreprocessingAction.FILL_NULL_MEDIAN,
            'fill_mode': PreprocessingAction.FILL_NULL_MODE,
            'minmax_scale': PreprocessingAction.MINMAX_SCALE,
            'robust_scale': PreprocessingAction.ROBUST_SCALE,
            'label_encode': PreprocessingAction.LABEL_ENCODE,
            'ordinal_encode': PreprocessingAction.ORDINAL_ENCODE,
        }

        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_action = self.action_encoder[top_idx]
        
        # Ensure action is an Enum (handle string actions from loaded models)
        if isinstance(top_action, str):
            # 1. Try direct mapping for known mismatches
            
            if top_action in mapping:
                top_action = mapping[top_action]
            else:
                # 2. Try exact value match
                found = False
                for action_enum in PreprocessingAction:
                    if action_enum.value == top_action:
                        top_action = action_enum
                        found = True
                        break
                
                # 3. Fallback to KEEP_AS_IS if unknown
                if not found:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Unknown action '{top_action}' from model. Defaulting to KEEP_AS_IS.")
                    top_action = PreprocessingAction.KEEP_AS_IS

        confidence = float(probs[top_idx])

        # Get probabilities for all actions
        action_probs = {}
        if return_probabilities:
            for idx, prob in enumerate(probs):
                if idx in self.action_encoder:
                    action_name = self.action_encoder[idx]
                    
                    # Apply same mapping logic
                    if isinstance(action_name, str):
                        if action_name in mapping:
                            action_enum = mapping[action_name]
                            action_probs[action_enum] = float(prob)
                        else:
                            # Try exact match
                            for ae in PreprocessingAction:
                                if ae.value == action_name:
                                    action_probs[ae] = float(prob)
                                    break
                    else:
                        # Already an Enum (if model was saved correctly)
                        action_probs[action_name] = float(prob)

        # Get feature importance (only for XGBoost)
        feature_importance = None
        if return_feature_importance and not is_sklearn:
            importance_dict = self.model.get_score(importance_type='gain')
            feature_importance = {
                name: importance_dict.get(name, 0.0)
                for name in self.feature_names
            }

        return OraclePrediction(
            action=top_action,
            confidence=confidence,
            action_probabilities=action_probs,
            feature_importance=feature_importance
        )

    def predict_with_shap(
        self,
        features: MinimalFeatures,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict with SHAP explanation.

        SHAP (SHapley Additive exPlanations) shows which features
        contributed to this specific prediction and by how much.

        Args:
            features: Extracted features from column
            top_k: Number of top contributing features to return

        Returns:
            Dictionary with:
            - action: Predicted preprocessing action
            - confidence: Prediction confidence (0-1)
            - explanation: Human-readable explanation
            - shap_values: Feature contributions
            - top_features: Top K contributing features
            - action_probabilities: Probabilities for all actions
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for explainable predictions. "
                "Install with: pip install shap"
            )

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Get base prediction
        prediction = self.predict(features, return_probabilities=True)

        # Calculate SHAP values
        X = features.to_array().reshape(1, -1)
        explainer = shap.TreeExplainer(self.model)

        # Get SHAP values for the predicted class
        shap_values = explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Get SHAP values for predicted class
            predicted_idx = np.argmax(self.model.predict(
                xgb.DMatrix(X, feature_names=self.feature_names)
            )[0])
            class_shap_values = shap_values[predicted_idx][0]
        else:
            class_shap_values = shap_values[0]

        # Create feature contribution dictionary
        contributions = {
            name: float(class_shap_values[idx])
            for idx, name in enumerate(self.feature_names)
        }

        # Get top contributing features
        top_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        # Build human-readable explanation
        explanation_parts = []
        for feature, impact in top_features:
            direction = "increases" if impact > 0 else "decreases"
            explanation_parts.append(
                f"{feature.replace('_', ' ')} {direction} confidence "
                f"(impact: {impact:+.2f})"
            )

        return {
            'action': prediction.action,
            'confidence': prediction.confidence,
            'explanation': explanation_parts,
            'shap_values': contributions,
            'top_features': [
                {'feature': name, 'impact': impact}
                for name, impact in top_features
            ],
            'action_probabilities': prediction.action_probabilities
        }

    def predict_batch(
        self,
        features_list: List[MinimalFeatures]
    ) -> List[OraclePrediction]:
        """
        Predict for a batch of feature sets (faster than individual predictions).

        Args:
            features_list: List of minimal features

        Returns:
            List of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        # Convert to numpy array
        X = np.vstack([f.to_array() for f in features_list])
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        # Batch prediction
        probs = self.model.predict(dmatrix)

        # Convert to predictions
        predictions = []
        for prob_vector in probs:
            top_idx = int(np.argmax(prob_vector))
            top_action = self.action_encoder[top_idx]
            confidence = float(prob_vector[top_idx])

            action_probs = {
                self.action_encoder[idx]: float(p)
                for idx, p in enumerate(prob_vector)
                if idx in self.action_encoder
            }

            predictions.append(OraclePrediction(
                action=top_action,
                confidence=confidence,
                action_probabilities=action_probs
            ))

        return predictions

    def save(self, path: Path):
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and encoders
        save_dict = {
            'model': self.model,
            'action_encoder': self.action_encoder,
            'action_decoder': self.action_decoder,
            'feature_names': self.feature_names
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, path: Path):
        """
        Load pre-trained model from disk.
        
        Supports both:
        - Ensemble models (VotingClassifier saved directly)
        - Legacy models (dictionary with model, encoders, feature_names)

        Args:
            path: Path to the saved model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        # Check if it's a direct sklearn model (ensemble) or a dictionary (legacy)
        if isinstance(loaded, dict):
            # Legacy format: dictionary with model and encoders
            self.model = loaded['model']
            self.action_encoder = loaded['action_encoder']
            self.action_decoder = loaded['action_decoder']
            self.feature_names = loaded.get('feature_names', self.feature_names)
        else:
            # Ensemble format: direct VotingClassifier or sklearn model
            self.model = loaded
            # Build action encoder/decoder from model's classes
            if hasattr(loaded, 'classes_'):
                classes = loaded.classes_
                self.action_encoder = {i: cls for i, cls in enumerate(classes)}
                self.action_decoder = {cls: i for i, cls in enumerate(classes)}

    def get_model_size(self) -> int:
        """
        Get approximate model size in bytes.

        Returns:
            Model size in bytes
        """
        if self.model is None:
            return 0

        # Use temporary file to get size (XGBoost doesn't support BytesIO in all versions)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.model.save_model(tmp_path)
            size = os.path.getsize(tmp_path)
            return size
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def benchmark_inference(
        self,
        features: MinimalFeatures,
        num_iterations: int = 1000
    ) -> float:
        """
        Benchmark inference time.

        Args:
            features: Sample features
            num_iterations: Number of iterations to run

        Returns:
            Average inference time in milliseconds
        """
        import time

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Warmup
        for _ in range(10):
            self.predict(features, return_probabilities=False, return_feature_importance=False)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(features, return_probabilities=False, return_feature_importance=False)
        end = time.time()

        avg_time_ms = (end - start) / num_iterations * 1000
        return avg_time_ms

    def get_top_features(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top K most important features.

        Args:
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        importance_dict = self.model.get_score(importance_type='gain')

        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:top_k]

    def evaluate(
        self,
        features: List[MinimalFeatures],
        labels: List[PreprocessingAction]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: List of features
            labels: True labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Get predictions
        predictions = self.predict_batch(features)

        # Compute metrics
        correct = sum(
            1 for pred, true_label in zip(predictions, labels)
            if pred.action == true_label
        )

        accuracy = correct / len(labels) if len(labels) > 0 else 0.0

        # Compute average confidence
        avg_confidence = np.mean([p.confidence for p in predictions])

        # Compute top-3 accuracy
        top3_correct = 0
        for pred, true_label in zip(predictions, labels):
            top3_actions = sorted(
                pred.action_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            top3_actions = [a for a, _ in top3_actions]
            if true_label in top3_actions:
                top3_correct += 1

        top3_accuracy = top3_correct / len(labels) if len(labels) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'avg_confidence': float(avg_confidence),
            'num_samples': len(labels)
        }


# Singleton instance for global use
_oracle_instance: Optional[NeuralOracle] = None


def get_neural_oracle(model_path: Optional[Path] = None) -> NeuralOracle:
    """
    Get the global NeuralOracle instance with pre-trained model.

    INFERENCE ONLY - Loads pre-trained ensemble (89.4% accuracy).
    NO training occurs at runtime.

    Priority:
    1. Ensemble model (neural_oracle_v2_improved_20251129_150244.pkl) - 89.4% accuracy
    2. Fallback to neural_oracle_v1.pkl if ensemble not found
    """
    global _oracle_instance
    if _oracle_instance is None:
        if model_path is None:
            # Try ensemble first
            ensemble_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v2_improved_20251129_150244.pkl"
            if ensemble_path.exists():
                model_path = ensemble_path
            else:
                # Fallback to old model
                fallback_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v1.pkl"
                if fallback_path.exists():
                    model_path = fallback_path

        _oracle_instance = NeuralOracle(model_path)
    return _oracle_instance
