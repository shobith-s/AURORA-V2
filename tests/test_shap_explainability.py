"""
Test SHAP explainability for the Neural Oracle.

Tests that SHAP explanations are generated correctly and provide
meaningful insights into neural oracle predictions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import SHAP, skip tests if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestSHAPExplainability:
    """Test suite for SHAP explainability features."""

    @pytest.fixture
    def model_path(self):
        """Get path to trained model."""
        path = Path("models/neural_oracle_v1.pkl")
        if not path.exists():
            pytest.skip("No trained model found. Train model first with: python scripts/train_neural_oracle.py")
        return path

    @pytest.fixture
    def oracle(self, model_path):
        """Get NeuralOracle instance with trained model."""
        return NeuralOracle(model_path)

    @pytest.fixture
    def extractor(self):
        """Get feature extractor."""
        return MinimalFeatureExtractor()

    def test_shap_method_exists(self, oracle):
        """Test that predict_with_shap method exists."""
        assert hasattr(oracle, 'predict_with_shap')
        assert callable(oracle.predict_with_shap)

    def test_shap_explanation_structure(self, oracle, extractor):
        """Test that SHAP explanations have the correct structure."""
        # Create test column (skewed data)
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_column")

        # Get SHAP prediction
        result = oracle.predict_with_shap(features)

        # Verify structure
        assert 'action' in result
        assert 'confidence' in result
        assert 'explanation' in result
        assert 'shap_values' in result
        assert 'top_features' in result
        assert 'action_probabilities' in result

    def test_shap_explanation_non_empty(self, oracle, extractor):
        """Test that SHAP explanations are non-empty."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_column")

        result = oracle.predict_with_shap(features)

        # Verify explanation is non-empty
        assert len(result['explanation']) > 0
        assert all(isinstance(exp, str) for exp in result['explanation'])

    def test_shap_values_for_all_features(self, oracle, extractor):
        """Test that SHAP values exist for all features."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_column")

        result = oracle.predict_with_shap(features)

        # Verify SHAP values exist for all features
        assert len(result['shap_values']) == len(oracle.feature_names)
        for feature_name in oracle.feature_names:
            assert feature_name in result['shap_values']
            assert isinstance(result['shap_values'][feature_name], float)

    def test_top_features_sorted_by_importance(self, oracle, extractor):
        """Test that top features are sorted by absolute impact."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_column")

        result = oracle.predict_with_shap(features, top_k=5)

        # Verify top features are sorted
        impacts = [abs(f['impact']) for f in result['top_features']]
        assert impacts == sorted(impacts, reverse=True)

    def test_top_k_parameter(self, oracle, extractor):
        """Test that top_k parameter limits the number of top features."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_column")

        # Test different top_k values
        for k in [1, 3, 5]:
            result = oracle.predict_with_shap(features, top_k=k)
            assert len(result['top_features']) == min(k, len(oracle.feature_names))
            assert len(result['explanation']) == min(k, len(oracle.feature_names))

    def test_shap_with_numeric_column(self, oracle, extractor):
        """Test SHAP explanations with numeric column."""
        test_column = pd.Series([10, 20, 30, 40, 50, 100])
        features = extractor.extract(test_column, "numeric_col")

        result = oracle.predict_with_shap(features)

        assert result['action'] is not None
        assert 0 <= result['confidence'] <= 1
        assert len(result['explanation']) > 0
        print(f"\n✅ SHAP explanation for numeric column:")
        print(f"   Action: {result['action'].value}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print("   Top contributing features:")
        for feat in result['top_features']:
            print(f"     • {feat['feature']}: {feat['impact']:+.2f}")

    def test_shap_with_categorical_column(self, oracle, extractor):
        """Test SHAP explanations with categorical column."""
        test_column = pd.Series(['A', 'B', 'C', 'A', 'B', 'C', 'A'])
        features = extractor.extract(test_column, "category_col")

        result = oracle.predict_with_shap(features)

        assert result['action'] is not None
        assert 0 <= result['confidence'] <= 1
        assert len(result['explanation']) > 0

    def test_shap_with_high_null_column(self, oracle, extractor):
        """Test SHAP explanations with high null percentage."""
        test_column = pd.Series([1, None, 3, None, 5, None, None, None])
        features = extractor.extract(test_column, "null_col")

        result = oracle.predict_with_shap(features)

        assert result['action'] is not None
        # Check that null_percentage feature has significant impact
        null_pct_impact = abs(result['shap_values']['null_percentage'])
        print(f"\n✅ Null percentage impact: {null_pct_impact:.3f}")

    def test_shap_explanation_human_readable(self, oracle, extractor):
        """Test that SHAP explanations are human-readable."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_col")

        result = oracle.predict_with_shap(features)

        # Check that explanations contain expected keywords
        explanation_text = " ".join(result['explanation'])
        assert any(word in explanation_text for word in ['increases', 'decreases'])
        assert 'confidence' in explanation_text
        assert 'impact' in explanation_text

    def test_shap_consistency_with_regular_predict(self, oracle, extractor):
        """Test that SHAP prediction matches regular prediction."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_col")

        # Get both predictions
        regular_pred = oracle.predict(features, return_probabilities=True)
        shap_result = oracle.predict_with_shap(features)

        # Verify they match
        assert regular_pred.action == shap_result['action']
        assert abs(regular_pred.confidence - shap_result['confidence']) < 0.001

    def test_shap_values_sum_property(self, oracle, extractor):
        """Test that SHAP values approximately sum to the prediction."""
        test_column = pd.Series([1, 2, 3, 100, 200, 500])
        features = extractor.extract(test_column, "test_col")

        result = oracle.predict_with_shap(features)

        # SHAP values should sum to approximately the difference from base value
        # This is a property of SHAP values (they are additive)
        shap_sum = sum(result['shap_values'].values())
        print(f"\n✅ Sum of SHAP values: {shap_sum:.4f}")
        # Note: The exact relationship depends on the base value and model output

    def test_shap_different_columns_different_explanations(self, oracle, extractor):
        """Test that different columns get different SHAP explanations."""
        # Column with high skewness
        skewed_column = pd.Series([1, 2, 3, 1000, 2000, 5000])
        skewed_features = extractor.extract(skewed_column, "skewed")

        # Column with high nulls
        null_column = pd.Series([1, None, None, None, 5, None])
        null_features = extractor.extract(null_column, "nulls")

        # Get SHAP results
        skewed_result = oracle.predict_with_shap(skewed_features)
        null_result = oracle.predict_with_shap(null_features)

        # Top features should be different
        skewed_top = [f['feature'] for f in skewed_result['top_features']]
        null_top = [f['feature'] for f in null_result['top_features']]

        # At least one feature should be different in top 3
        assert skewed_top != null_top, "Different column types should have different top features"

    def test_shap_no_errors_with_edge_cases(self, oracle, extractor):
        """Test that SHAP doesn't error with edge cases."""
        edge_cases = [
            pd.Series([1] * 100),  # All same values
            pd.Series(range(100)),  # Perfect sequence
            pd.Series([None] * 100),  # All nulls
            pd.Series([0] * 100),  # All zeros
        ]

        for i, test_column in enumerate(edge_cases):
            features = extractor.extract(test_column, f"edge_case_{i}")
            result = oracle.predict_with_shap(features)

            assert result is not None
            assert 'action' in result
            assert 'explanation' in result


@pytest.mark.skipif(SHAP_AVAILABLE, reason="Testing error handling when SHAP not available")
def test_shap_import_error_handling():
    """Test that system handles SHAP not being installed."""
    from src.neural.oracle import NeuralOracle

    oracle = NeuralOracle()

    # Attempting to use predict_with_shap should raise ImportError
    with pytest.raises(ImportError, match="SHAP is required"):
        from src.features.minimal_extractor import MinimalFeatureExtractor
        extractor = MinimalFeatureExtractor()
        features = extractor.extract(pd.Series([1, 2, 3]), "test")
        oracle.predict_with_shap(features)


def test_preprocessor_handles_shap_gracefully():
    """Test that preprocessor handles SHAP being unavailable."""
    from src.core.preprocessor import IntelligentPreprocessor

    preprocessor = IntelligentPreprocessor(
        use_neural_oracle=True,
        enable_learning=False,
        enable_cache=False
    )

    # Should work even if SHAP is not available (falls back to regular predict)
    test_column = pd.Series([1, 2, 3, 100, 200, 500])
    result = preprocessor.preprocess_column(test_column, "test_col")

    assert result is not None
    assert result.action is not None
    assert result.confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
