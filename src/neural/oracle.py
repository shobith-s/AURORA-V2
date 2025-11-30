"""
NeuralOracle - Lightweight ML model for ambiguous preprocessing decisions.
Only used when symbolic engine confidence < 0.9.
XGBoost with 50 trees, <5MB model size, <5ms inference.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from ..core.actions import PreprocessingAction, PreprocessingResult
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
    Lightweight neural preprocessing oracle for edge cases.

    Uses XGBoost with 50 trees, trained only on ambiguous cases.
    Designed for <5ms inference time and <5MB model size.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the NeuralOracle.

        Args:
            model_path: Path to pre-trained model file (optional)
        """
        if xgb is None:
            raise ImportError(
                "XGBoost is required for NeuralOracle. "
                "Install with: pip install xgboost"
            )

        self.model = None
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

        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        features: List[MinimalFeatures],
        labels: List[PreprocessingAction],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the NeuralOracle on edge cases.

        Args:
            features: List of minimal features
            labels: List of correct preprocessing actions
            validation_split: Fraction of data for validation

        Returns:
            Training metrics
        """
        # Convert features to numpy array
        X = np.vstack([f.to_array() for f in features])

        # Encode labels
        unique_actions = sorted(set(labels), key=lambda a: a.value)
        self.action_encoder = {i: action for i, action in enumerate(unique_actions)}
        self.action_decoder = {action: i for i, action in enumerate(unique_actions)}
        y = np.array([self.action_decoder[label] for label in labels])

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
        y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        # Training parameters (optimized for speed and size)
        params = {
            'objective': 'multi:softprob',
            'num_class': len(unique_actions),
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'tree_method': 'hist',  # Faster training
            'eval_metric': 'mlogloss',
            'seed': 42
        }

        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=50,  # Lightweight: only 50 trees
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
            evals_result=evals_result
        )

        # Compute metrics
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)

        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'num_samples': n_samples,
            'num_features': X.shape[1],
            'num_classes': len(unique_actions),
            'num_trees': self.model.num_boosted_rounds()
        }

    def predict(
        self,
        features: MinimalFeatures,
        return_probabilities: bool = True,
        return_feature_importance: bool = False
    ) -> OraclePrediction:
        """
        Predict preprocessing action for edge case.

        Args:
            features: Minimal features extracted from column
            return_probabilities: Return probabilities for all actions
            return_feature_importance: Return feature importance (slower)

        Returns:
            OraclePrediction with action and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        # Convert features to array
        X = features.to_array().reshape(1, -1)
        
        # Check if model is sklearn or XGBoost
        is_sklearn = hasattr(self.model, 'predict_proba')
        
        if is_sklearn:
            # Sklearn model (VotingClassifier, etc.)
            probs = self.model.predict_proba(X)[0]
        else:
            # XGBoost model
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            probs = self.model.predict(dmatrix)[0]

        # Map v2 model actions to system actions
        mapping = {
            'drop': PreprocessingAction.DROP_COLUMN,
            'fill_zero': PreprocessingAction.FILL_NULL_MODE,  # Best approximation
            'fill_forward': PreprocessingAction.FILL_NULL_FORWARD,
            'fill_backward': PreprocessingAction.FILL_NULL_BACKWARD,
            'fill_mean': PreprocessingAction.FILL_NULL_MEAN,
            'fill_median': PreprocessingAction.FILL_NULL_MEDIAN,
            'fill_mode': PreprocessingAction.FILL_NULL_MODE,
            'keep_as_is': PreprocessingAction.KEEP_AS_IS,
            'standard_scale': PreprocessingAction.STANDARD_SCALE,
            'minmax_scale': PreprocessingAction.MINMAX_SCALE,
            'robust_scale': PreprocessingAction.ROBUST_SCALE,
            'log_transform': PreprocessingAction.LOG_TRANSFORM,
            'onehot_encode': PreprocessingAction.ONEHOT_ENCODE,
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
        Load model from disk.

        Args:
            path: Path to the saved model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model = save_dict['model']
        self.action_encoder = save_dict['action_encoder']
        self.action_decoder = save_dict['action_decoder']
        self.feature_names = save_dict['feature_names']

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
    """Get the global NeuralOracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        default_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v1.pkl"
        if model_path is None and default_path.exists():
            model_path = default_path
        _oracle_instance = NeuralOracle(model_path)
    return _oracle_instance
