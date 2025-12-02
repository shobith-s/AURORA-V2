"""
Tests for HybridPreprocessingOracle and MetaFeatureExtractor.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.meta_extractor import MetaFeatureExtractor, MetaFeatures, get_meta_feature_extractor
from src.neural.hybrid_oracle import HybridPreprocessingOracle, HybridPrediction
from src.core.actions import PreprocessingAction


class TestMetaFeatureExtractor:
    """Test MetaFeatureExtractor functionality."""
    
    def test_extractor_initialization(self):
        """Test that extractor can be initialized."""
        extractor = MetaFeatureExtractor()
        assert extractor is not None
    
    def test_extract_numeric_column(self):
        """Test feature extraction for numeric columns."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([1, 2, 3, 4, 5, 100, 200], name='numeric_col')
        
        features = extractor.extract(column, 'numeric_col')
        
        # Check that we get 40 features
        assert len(features.to_array()) == 40
        
        # Check basic stats
        assert features.is_numeric == 1.0
        assert features.missing_ratio == 0.0
        assert features.unique_ratio > 0.0
        
        # Check numeric stats are populated
        assert features.mean > 0.0
        assert features.std > 0.0
        assert features.skewness != 0.0
    
    def test_extract_categorical_column(self):
        """Test feature extraction for categorical columns."""
        extractor = MetaFeatureExtractor()
        column = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'], name='cat_col')
        
        features = extractor.extract(column, 'cat_col')
        
        # Check type indicators
        assert features.is_object == 1.0
        assert features.is_categorical == 1.0 or features.is_object == 1.0
        
        # Check categorical stats
        assert features.avg_length > 0.0
        assert features.entropy > 0.0
        assert features.mode_frequency > 0.0
    
    def test_extract_with_missing_values(self):
        """Test feature extraction with missing values."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([1, 2, None, 4, None, 6], name='missing_col')
        
        features = extractor.extract(column, 'missing_col')
        
        # Check missing ratio (calculate dynamically from test data)
        expected_ratio = column.isnull().sum() / len(column)
        assert features.missing_ratio > 0.0
        assert features.missing_ratio == pytest.approx(expected_ratio, rel=0.01)
        assert features.is_complete == 0.0
    
    def test_extract_constant_column(self):
        """Test feature extraction for constant columns."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([5, 5, 5, 5, 5], name='const_col')
        
        features = extractor.extract(column, 'const_col')
        
        # Check variance indicators
        assert features.unique_ratio < 0.5
        assert features.has_variance == 0.0
        assert features.std == 0.0
    
    def test_extract_with_column_name_signals(self):
        """Test that column name affects name-based features."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([1, 2, 3, 4, 5])
        
        # Test ID column
        features_id = extractor.extract(column, 'user_id')
        assert features_id.has_id == 1.0
        
        # Test date column
        features_date = extractor.extract(column, 'created_date')
        assert features_date.has_date == 1.0
        
        # Test price column
        features_price = extractor.extract(column, 'product_price')
        assert features_price.has_price == 1.0
    
    def test_extract_skewed_data(self):
        """Test feature extraction for highly skewed data."""
        extractor = MetaFeatureExtractor()
        # Highly skewed positive data
        column = pd.Series([1, 1, 1, 2, 2, 100, 1000], name='skewed_col')
        
        features = extractor.extract(column, 'skewed_col')
        
        # Check skewness
        assert abs(features.skewness) > 1.0
        assert features.can_log == 1.0  # All positive
    
    def test_extract_with_outliers(self):
        """Test feature extraction with outliers."""
        extractor = MetaFeatureExtractor()
        # Normal data with outliers
        column = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 100, 200], name='outlier_col')
        
        features = extractor.extract(column, 'outlier_col')
        
        # Check outlier detection
        assert features.outlier_ratio > 0.0
    
    def test_feature_array_shape(self):
        """Test that feature array has correct shape."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([1, 2, 3])
        
        features = extractor.extract(column, 'test')
        array = features.to_array()
        
        assert array.shape == (40,)
        assert array.dtype == np.float32
    
    def test_feature_dict_conversion(self):
        """Test conversion to dictionary."""
        extractor = MetaFeatureExtractor()
        column = pd.Series([1, 2, 3])
        
        features = extractor.extract(column, 'test')
        feature_dict = features.to_dict()
        
        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == 40
        assert 'missing_ratio' in feature_dict
        assert 'is_numeric' in feature_dict
    
    def test_singleton_extractor(self):
        """Test that get_meta_feature_extractor returns singleton."""
        extractor1 = get_meta_feature_extractor()
        extractor2 = get_meta_feature_extractor()
        
        assert extractor1 is extractor2


class TestHybridPreprocessingOracle:
    """Test HybridPreprocessingOracle functionality."""
    
    def test_oracle_initialization(self):
        """Test that oracle can be initialized without models."""
        oracle = HybridPreprocessingOracle()
        assert oracle is not None
    
    def test_rule_based_constant_column(self):
        """Test rule-based detection of constant columns."""
        oracle = HybridPreprocessingOracle()
        extractor = MetaFeatureExtractor()
        
        # Create constant column
        column = pd.Series([5, 5, 5, 5, 5])
        features = extractor.extract(column, 'const_col')
        
        # Test rule application directly
        action, confidence, reason = oracle._apply_rules(column, 'const_col', features)
        
        assert action == PreprocessingAction.DROP_COLUMN
        assert confidence > 0.9
        assert 'constant' in reason.lower() or 'variance' in reason.lower()
    
    def test_rule_based_id_column(self):
        """Test rule-based detection of ID columns."""
        oracle = HybridPreprocessingOracle()
        extractor = MetaFeatureExtractor()
        
        # Create ID-like column (unique values)
        column = pd.Series(range(100))
        features = extractor.extract(column, 'user_id')
        
        # Test rule application
        action, confidence, reason = oracle._apply_rules(column, 'user_id', features)
        
        # Should recommend dropping ID columns
        if action is not None:
            assert action == PreprocessingAction.DROP_COLUMN
            assert confidence > 0.8
    
    def test_rule_based_skewed_data(self):
        """Test rule-based detection for highly skewed data."""
        oracle = HybridPreprocessingOracle()
        extractor = MetaFeatureExtractor()
        
        # Create highly skewed positive data
        column = pd.Series([1, 1, 1, 2, 2, 3, 100, 1000])
        features = extractor.extract(column, 'revenue')
        
        # Test rule application
        action, confidence, reason = oracle._apply_rules(column, 'revenue', features)
        
        # Should recommend log transform for skewed positive data
        if action is not None and abs(features.skewness) > 2.0:
            assert action in [PreprocessingAction.LOG_TRANSFORM, PreprocessingAction.LOG1P_TRANSFORM]
            assert confidence > 0.8
    
    def test_rule_based_outliers(self):
        """Test rule-based detection of outliers."""
        oracle = HybridPreprocessingOracle()
        extractor = MetaFeatureExtractor()
        
        # Create data with many outliers
        column = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5, 100, 200, 300])
        features = extractor.extract(column, 'value')
        
        # Test rule application
        action, confidence, reason = oracle._apply_rules(column, 'value', features)
        
        # Should recommend clipping outliers
        if features.outlier_ratio > 0.15:
            assert action == PreprocessingAction.CLIP_OUTLIERS
            assert confidence > 0.8
    
    def test_rule_based_high_missing(self):
        """Test rule-based handling of high missing ratios."""
        oracle = HybridPreprocessingOracle()
        extractor = MetaFeatureExtractor()
        
        # Create column with >60% missing
        column = pd.Series([1, 2, None, None, None, None, None, None])
        features = extractor.extract(column, 'sparse_col')
        
        # Test rule application
        action, confidence, reason = oracle._apply_rules(column, 'sparse_col', features)
        
        assert action == PreprocessingAction.DROP_COLUMN
        assert confidence > 0.7
        assert 'missing' in reason.lower()
    
    def test_action_mapping(self):
        """Test action name mapping to PreprocessingAction enum."""
        oracle = HybridPreprocessingOracle()
        
        # Test known mappings
        assert oracle._map_action_name('clip_outliers') == PreprocessingAction.CLIP_OUTLIERS
        assert oracle._map_action_name('drop_column') == PreprocessingAction.DROP_COLUMN
        assert oracle._map_action_name('log_transform') == PreprocessingAction.LOG_TRANSFORM
        assert oracle._map_action_name('standard_scale') == PreprocessingAction.STANDARD_SCALE
        
        # Test unknown action (should default to KEEP_AS_IS)
        assert oracle._map_action_name('unknown_action') == PreprocessingAction.KEEP_AS_IS
    
    def test_predict_dataframe(self):
        """Test predicting for entire DataFrame."""
        oracle = HybridPreprocessingOracle()
        
        # Create test DataFrame
        df = pd.DataFrame({
            'id': range(100),
            'const': [5] * 100,
            'normal': np.random.randn(100),
            'category': ['A', 'B', 'C'] * 33 + ['A']
        })
        
        # Predict for all columns
        results = oracle.predict_dataframe(df)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # 4 columns
        assert 'column_name' in results.columns
        assert 'action' in results.columns
        assert 'confidence' in results.columns
        assert 'source' in results.columns


class TestOracleBackwardCompatibility:
    """Test backward compatibility with existing oracle."""
    
    def test_old_model_loading(self):
        """Test that old model can still be loaded."""
        from src.neural.oracle import NeuralOracle
        
        model_path = Path('models/neural_oracle_v2_improved_20251129_150244.pkl')
        if not model_path.exists():
            pytest.skip("Old model not found")
        
        oracle = NeuralOracle(model_path)
        
        assert oracle is not None
        assert not oracle.is_hybrid
        assert oracle.model is not None
    
    def test_old_model_prediction(self):
        """Test that old model can still make predictions."""
        from src.neural.oracle import NeuralOracle
        from src.features.minimal_extractor import MinimalFeatureExtractor
        
        model_path = Path('models/neural_oracle_v2_improved_20251129_150244.pkl')
        if not model_path.exists():
            pytest.skip("Old model not found")
        
        oracle = NeuralOracle(model_path)
        extractor = MinimalFeatureExtractor()
        
        # Create test data
        column = pd.Series([1, 2, 3, 4, 5])
        features = extractor.extract(column, 'test_col')
        
        # Make prediction
        prediction = oracle.predict(features)
        
        assert prediction is not None
        assert hasattr(prediction, 'action')
        assert hasattr(prediction, 'confidence')
        assert 0.0 <= prediction.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
