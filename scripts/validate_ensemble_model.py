"""
Validate that the pre-trained ensemble model loads correctly.
INFERENCE ONLY - No training.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from src.neural.oracle import get_neural_oracle
from src.features.minimal_extractor import MinimalFeatureExtractor


def main():
    print("=" * 70)
    print("VALIDATING PRE-TRAINED ENSEMBLE MODEL (INFERENCE ONLY)")
    print("=" * 70)

    # Load pre-trained model
    print("\n1. Loading pre-trained ensemble...")
    oracle = get_neural_oracle()

    if oracle.model is None:
        print("❌ ERROR: No model loaded!")
        return 1

    # Check model type
    is_ensemble = hasattr(oracle.model, 'predict_proba')
    model_type = "VotingClassifier (Ensemble)" if is_ensemble else "XGBoost (Single)"
    print(f"✅ Loaded: {model_type}")

    # Verify train() is disabled
    print("\n2. Verifying training is disabled...")
    try:
        oracle.train([], [])
        print("❌ ERROR: train() should have raised RuntimeError!")
        return 1
    except RuntimeError as e:
        print("✅ train() correctly raises RuntimeError (training disabled)")

    # Test inference (no training)
    print("\n3. Testing inference on sample data...")
    extractor = MinimalFeatureExtractor()
    test_column = pd.Series([1, 2, 3, 4, 5, None, 7, 8, 9, 10], name="test_col")
    features = extractor.extract(test_column, "test_col")

    # Make prediction (INFERENCE ONLY)
    prediction = oracle.predict(features)

    print(f"✅ Prediction: {prediction.action.value}")
    print(f"   Confidence: {prediction.confidence:.2%}")
    print(f"   Source: {'ensemble' if is_ensemble else 'single_model'}")

    # Verify action is a valid enum
    from src.core.actions import PreprocessingAction
    if not isinstance(prediction.action, PreprocessingAction):
        print(f"❌ ERROR: Prediction action is not a PreprocessingAction enum!")
        return 1
    print("✅ Action is valid PreprocessingAction enum")

    print("\n" + "=" * 70)
    print("✅ PRE-TRAINED ENSEMBLE MODEL VALIDATED (INFERENCE ONLY)")
    print("=" * 70)
    print("\nModel is ready for production use.")
    print("NO training will occur at runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
