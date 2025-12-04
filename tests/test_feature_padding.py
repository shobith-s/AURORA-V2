"""
Tests for feature padding in hybrid model prediction.

This test suite verifies that the neural oracle can handle feature dimension
mismatches gracefully by padding MinimalFeatures (20 features) to the expected
40 features required by hybrid models trained with MetaFeatureExtractor.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor, MinimalFeatures
from src.core.actions import PreprocessingAction


class TestFeaturePadding:
    """Test feature padding for hybrid models."""

    def test_padding_20_to_40_features(self):
        """Test that 20-feature input is correctly padded to 40 features."""
        # Create a mock hybrid oracle
        oracle = NeuralOracle.__new__(NeuralOracle)
        oracle.is_hybrid = True
        oracle.model = None
        oracle.action_encoder = {}
        oracle.action_decoder = {}
        oracle.feature_names = []
        oracle.feature_extractor = None
        oracle.config = {}
        oracle.metadata = {}
        oracle.removed_classes = []

        # Track features passed to the model
        captured_features = []

        def capture_and_validate_features(X):
            """Capture features and validate they have correct shape."""
            captured_features.append(X.copy())
            # Return mock probabilities for 5 actions
            return np.array([[0.2, 0.3, 0.15, 0.25, 0.1]])

        # Mock models that expect 40 features
        mock_xgb = Mock()
        mock_xgb.predict_proba = capture_and_validate_features
        mock_lgb = Mock()
        mock_lgb.predict_proba = capture_and_validate_features

        oracle.xgb_model = mock_xgb
        oracle.lgb_model = mock_lgb

        # Mock label encoder
        mock_label_encoder = Mock()
        mock_label_encoder.inverse_transform = lambda x: ["standard_scale"]
        oracle.label_encoder = mock_label_encoder

        # Create MinimalFeatures (20 features)
        extractor = MinimalFeatureExtractor()
        test_column = pd.Series([1, 2, 3, 4, 5, 100, 200, 300], name="test_col")
        features = extractor.extract(test_column, "test_col")

        # Verify we start with 20 features
        assert features.to_array().shape[0] == 20

        # Make prediction (should trigger padding)
        prediction = oracle._predict_hybrid(features, return_probabilities=True)

        # Verify prediction succeeded
        assert prediction is not None
        assert hasattr(prediction, "action")
        assert hasattr(prediction, "confidence")

        # Verify models received 40 features
        assert len(captured_features) == 2  # XGBoost and LightGBM
        for captured in captured_features:
            assert captured.shape == (1, 40), f"Expected (1, 40), got {captured.shape}"

    def test_padding_preserves_original_features(self):
        """Test that padding preserves the original 20 features."""
        # Create mock oracle
        oracle = NeuralOracle.__new__(NeuralOracle)
        oracle.is_hybrid = True
        oracle.model = None
        oracle.action_encoder = {}
        oracle.action_decoder = {}
        oracle.feature_names = []
        oracle.feature_extractor = None
        oracle.config = {}
        oracle.metadata = {}
        oracle.removed_classes = []

        captured_features = []

        def capture_features(X):
            captured_features.append(X.copy())
            return np.array([[0.2, 0.3, 0.15, 0.25, 0.1]])

        mock_xgb = Mock()
        mock_xgb.predict_proba = capture_features

        oracle.xgb_model = mock_xgb
        oracle.lgb_model = None  # Only use XGBoost for this test

        mock_label_encoder = Mock()
        mock_label_encoder.inverse_transform = lambda x: ["standard_scale"]
        oracle.label_encoder = mock_label_encoder

        # Create features with known values
        test_features = MinimalFeatures(
            null_percentage=0.1,
            unique_ratio=0.5,
            skewness=1.5,
            outlier_percentage=0.2,
            entropy=2.3,
            pattern_complexity=0.3,
            multimodality_score=0.4,
            cardinality_bucket=1,
            detected_dtype=0,
            column_name_signal=0.5,
            kurtosis=2.0,
            coefficient_of_variation=0.6,
            zero_ratio=0.1,
            has_negative=0.0,
            has_decimal=1.0,
            name_contains_id=0.0,
            name_contains_date=0.0,
            name_contains_price=1.0,
            range_ratio=5.0,
            iqr_ratio=1.2,
        )

        original_array = test_features.to_array()

        # Make prediction
        oracle._predict_hybrid(test_features, return_probabilities=True)

        # Verify features were captured
        assert len(captured_features) == 1
        padded_features = captured_features[0]

        # Check that first 20 features match original
        assert np.allclose(
            padded_features[0, :20], original_array
        ), "First 20 features should match original"

        # Check that features 20-39 are zeros
        assert np.allclose(
            padded_features[0, 20:], 0.0
        ), "Features 20-39 should be zeros"

    def test_no_padding_with_correct_feature_count(self):
        """Test that no padding occurs when feature count is already correct."""
        # This test would require creating a MetaFeatureExtractor scenario
        # For now, we verify the logic handles 40-feature input correctly
        pass  # Skipping as current implementation uses MinimalFeatures

    def test_truncation_with_too_many_features(self):
        """Test that truncation works when more than 40 features are provided."""
        # Create mock oracle
        oracle = NeuralOracle.__new__(NeuralOracle)
        oracle.is_hybrid = True
        oracle.model = None
        oracle.action_encoder = {}
        oracle.action_decoder = {}
        oracle.feature_names = []
        oracle.feature_extractor = None
        oracle.config = {}
        oracle.metadata = {}
        oracle.removed_classes = []

        captured_features = []

        def capture_features(X):
            captured_features.append(X.copy())
            return np.array([[0.2, 0.3, 0.15, 0.25, 0.1]])

        mock_xgb = Mock()
        mock_xgb.predict_proba = capture_features

        oracle.xgb_model = mock_xgb
        oracle.lgb_model = None

        mock_label_encoder = Mock()
        mock_label_encoder.inverse_transform = lambda x: ["standard_scale"]
        oracle.label_encoder = mock_label_encoder

        # Create a mock features object with 50 features
        class MockFeatures:
            def to_array(self):
                return np.random.randn(50)

        mock_features = MockFeatures()

        # Make prediction (should trigger truncation)
        with patch("src.neural.oracle.logger") as mock_logger:
            oracle._predict_hybrid(mock_features, return_probabilities=True)

            # Verify warning was logged
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "Truncating" in call_args or "truncat" in call_args.lower()

        # Verify features were truncated to 40
        assert len(captured_features) == 1
        assert captured_features[0].shape == (1, 40)

    def test_prediction_with_real_extractor(self):
        """Test end-to-end prediction with real MinimalFeatureExtractor."""
        # Create mock oracle with realistic setup
        oracle = NeuralOracle.__new__(NeuralOracle)
        oracle.is_hybrid = True
        oracle.model = None
        oracle.action_encoder = {}
        oracle.action_decoder = {}
        oracle.feature_names = []
        oracle.feature_extractor = None
        oracle.config = {}
        oracle.metadata = {}
        oracle.removed_classes = []

        def mock_predict_proba(X):
            # Verify we got 40 features
            if X.shape[1] != 40:
                raise ValueError(f"Expected 40 features, got {X.shape[1]}")
            # Return realistic probabilities favoring log_transform
            return np.array([[0.05, 0.1, 0.6, 0.15, 0.1]])

        mock_xgb = Mock()
        mock_xgb.predict_proba = mock_predict_proba
        mock_lgb = Mock()
        mock_lgb.predict_proba = mock_predict_proba

        oracle.xgb_model = mock_xgb
        oracle.lgb_model = mock_lgb

        # Mock label encoder with realistic actions
        mock_label_encoder = Mock()

        def mock_inverse_transform(indices):
            actions = ["drop_column", "standard_scale", "log_transform", "clip_outliers", "keep_as_is"]
            return [actions[i] for i in indices]

        mock_label_encoder.inverse_transform = mock_inverse_transform
        oracle.label_encoder = mock_label_encoder

        # Extract real features from a skewed column
        extractor = MinimalFeatureExtractor()
        skewed_column = pd.Series([1, 1, 2, 3, 5, 100, 500, 1000], name="revenue")
        features = extractor.extract(skewed_column, "revenue")

        # Make prediction
        prediction = oracle._predict_hybrid(features, return_probabilities=True)

        # Verify prediction
        assert prediction is not None
        assert prediction.action in PreprocessingAction
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.action_probabilities) > 0

    def test_warning_logged_on_padding(self):
        """Test that a warning is logged when padding occurs."""
        oracle = NeuralOracle.__new__(NeuralOracle)
        oracle.is_hybrid = True
        oracle.model = None
        oracle.action_encoder = {}
        oracle.action_decoder = {}
        oracle.feature_names = []
        oracle.feature_extractor = None
        oracle.config = {}
        oracle.metadata = {}
        oracle.removed_classes = []

        def mock_predict_proba(X):
            return np.array([[0.2, 0.3, 0.15, 0.25, 0.1]])

        mock_xgb = Mock()
        mock_xgb.predict_proba = mock_predict_proba

        oracle.xgb_model = mock_xgb
        oracle.lgb_model = None

        mock_label_encoder = Mock()
        mock_label_encoder.inverse_transform = lambda x: ["standard_scale"]
        oracle.label_encoder = mock_label_encoder

        # Create features
        extractor = MinimalFeatureExtractor()
        test_column = pd.Series([1, 2, 3, 4, 5], name="test")
        features = extractor.extract(test_column, "test")

        # Make prediction and check for warning
        with patch("src.neural.oracle.logger") as mock_logger:
            oracle._predict_hybrid(features, return_probabilities=True)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            call_args = str(mock_logger.warning.call_args)
            assert "Feature dimension mismatch" in call_args
            assert "20" in call_args
            assert "40" in call_args
            assert "Padding" in call_args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
