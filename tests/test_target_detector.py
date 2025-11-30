"""
Tests for TargetDetector - Target variable detection and protection.

Tests:
1. Keyword detection (price, target, label, churn, fraud, etc.)
2. Positional detection (last column)
3. Exclusion patterns (id, date, url should not be targets)
4. False positive prevention
5. Target protection validation
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.target_detector import (
    TargetDetector,
    TargetDetectionResult,
    get_target_detector,
)


@pytest.fixture
def detector():
    """Create a fresh target detector for each test."""
    return TargetDetector()


class TestKeywordDetection:
    """Test target detection via keyword matching."""
    
    def test_target_keyword(self, detector):
        """Column named 'target' should be detected as target."""
        col = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_target(col, 'target')
        assert result.is_target is True
        assert result.confidence >= 0.8
        assert 'target' in result.detection_method.lower() or 'exact' in result.detection_method.lower()
    
    def test_label_keyword(self, detector):
        """Column named 'label' should be detected as target."""
        col = pd.Series(['cat', 'dog', 'cat', 'bird'])
        result = detector.detect_target(col, 'label')
        assert result.is_target is True
        assert result.confidence >= 0.8
    
    def test_class_keyword(self, detector):
        """Column named 'class' should be detected as target."""
        col = pd.Series([0, 1, 2, 0, 1])
        result = detector.detect_target(col, 'class')
        assert result.is_target is True
    
    def test_price_keyword(self, detector):
        """Column named 'price' should be detected as target."""
        col = pd.Series([100, 200, 150, 300, 250])
        result = detector.detect_target(col, 'price')
        assert result.is_target is True
        assert result.confidence >= 0.7
    
    def test_selling_price_keyword(self, detector):
        """Column named 'selling_price' should be detected as target."""
        col = pd.Series([450000, 350000, 550000, 275000])
        result = detector.detect_target(col, 'selling_price')
        assert result.is_target is True
        assert result.confidence >= 0.9
    
    def test_churn_keyword(self, detector):
        """Column named 'churn' should be detected as target."""
        col = pd.Series([0, 1, 0, 0, 1])
        result = detector.detect_target(col, 'churn')
        assert result.is_target is True
    
    def test_fraud_keyword(self, detector):
        """Column named 'fraud' should be detected as target."""
        col = pd.Series([0, 0, 1, 0, 0])
        result = detector.detect_target(col, 'fraud')
        assert result.is_target is True
    
    def test_outcome_keyword(self, detector):
        """Column named 'outcome' should be detected as target."""
        col = pd.Series(['success', 'failure', 'success'])
        result = detector.detect_target(col, 'outcome')
        assert result.is_target is True
    
    def test_y_keyword(self, detector):
        """Column named 'y' should be detected as target."""
        col = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_target(col, 'y')
        assert result.is_target is True
    
    def test_is_fraud_keyword(self, detector):
        """Column named 'is_fraud' should be detected as target."""
        col = pd.Series([True, False, True, False])
        result = detector.detect_target(col, 'is_fraud')
        assert result.is_target is True


class TestExclusionPatterns:
    """Test exclusion patterns - these should NOT be targets."""
    
    def test_id_excluded(self, detector):
        """Column named 'id' should NOT be detected as target."""
        col = pd.Series([1, 2, 3, 4, 5])
        result = detector.detect_target(col, 'id')
        assert result.is_target is False
        assert 'exclusion' in result.detection_method.lower()
    
    def test_customer_id_excluded(self, detector):
        """Column named 'customer_id' should NOT be detected as target."""
        col = pd.Series(['c001', 'c002', 'c003', 'c004'])
        result = detector.detect_target(col, 'customer_id')
        assert result.is_target is False
    
    def test_uuid_excluded(self, detector):
        """Column named 'uuid' should NOT be detected as target."""
        col = pd.Series(['550e8400-e29b-41d4', '6ba7b810-9dad-11d1'])
        result = detector.detect_target(col, 'uuid')
        assert result.is_target is False
    
    def test_date_excluded(self, detector):
        """Column named 'date' should NOT be detected as target."""
        col = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        result = detector.detect_target(col, 'date')
        assert result.is_target is False
    
    def test_timestamp_excluded(self, detector):
        """Column named 'timestamp' should NOT be detected as target."""
        col = pd.Series([1699999999, 1700000000, 1700000001])
        result = detector.detect_target(col, 'timestamp')
        assert result.is_target is False
    
    def test_url_excluded(self, detector):
        """Column named 'url' should NOT be detected as target."""
        col = pd.Series(['https://example.com', 'https://google.com'])
        result = detector.detect_target(col, 'url')
        assert result.is_target is False
    
    def test_email_excluded(self, detector):
        """Column named 'email' should NOT be detected as target."""
        col = pd.Series(['a@test.com', 'b@test.com'])
        result = detector.detect_target(col, 'email')
        assert result.is_target is False
    
    def test_name_excluded(self, detector):
        """Column named 'name' should NOT be detected as target."""
        col = pd.Series(['John Smith', 'Jane Doe', 'Bob Wilson'])
        result = detector.detect_target(col, 'name')
        assert result.is_target is False


class TestPositionalDetection:
    """Test target detection based on position."""
    
    def test_last_column_with_binary(self, detector):
        """Last column with binary values should be detected as target."""
        col = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
        result = detector.detect_target(
            col, 'unknown_col',
            position_in_df=4,
            total_columns=5
        )
        # Should be detected with moderate confidence
        assert result.is_target is True or result.confidence >= 0.4
    
    def test_non_last_column(self, detector):
        """Non-last column without keywords should not be auto-detected."""
        col = pd.Series([0, 1, 0, 1, 1])
        result = detector.detect_target(
            col, 'random_col',
            position_in_df=2,
            total_columns=5
        )
        # May not be detected as target
        assert result.confidence <= 0.7 or result.is_target is False


class TestStatisticalProperties:
    """Test target detection based on statistical properties."""
    
    def test_binary_classification(self, detector):
        """Binary column with 0/1 values has target properties."""
        col = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
        result = detector.detect_target(col, 'binary_col')
        # Should recognize as potential target even without keyword
        if result.is_target:
            assert 'binary' in result.details.get('binary_pattern', []) or result.confidence >= 0.5
    
    def test_multiclass_target(self, detector):
        """Low cardinality multiclass column may be target."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'] * 10)
        result = detector.detect_target(col, 'class_label')
        assert result.is_target is True  # Has 'class' keyword
    
    def test_continuous_regression_target(self, detector):
        """Continuous numeric column named 'target' should be detected."""
        col = pd.Series([1.5, 2.3, 3.7, 4.1, 5.8, 6.2, 7.0, 8.9])
        result = detector.detect_target(col, 'target_value')
        assert result.is_target is True  # Has 'target' keyword


class TestDataFrameDetection:
    """Test detection of targets in dataframes."""
    
    def test_detect_targets_in_dataframe(self, detector):
        """Should detect target columns in a dataframe."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'feature1': [10, 20, 30, 40],
            'feature2': [100, 200, 300, 400],
            'target': [0, 1, 0, 1]
        })
        results = detector.detect_targets_in_dataframe(df)
        
        assert results['target'].is_target is True
        assert results['id'].is_target is False
        assert results['feature1'].is_target is False
    
    def test_detect_targets_with_known_target(self, detector):
        """User-specified target should have 100% confidence."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        results = detector.detect_targets_in_dataframe(df, known_target='col1')
        
        assert results['col1'].is_target is True
        assert results['col1'].confidence == 1.0
        assert results['col1'].detection_method == 'user_specified'
    
    def test_get_likely_targets(self, detector):
        """Should return sorted list of likely targets."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'feature': [10, 20, 30, 40],
            'price': [100, 200, 300, 400],
            'label': [0, 1, 0, 1]
        })
        targets = detector.get_likely_targets(df)
        
        # Should find price and label as targets
        target_names = [t[0] for t in targets]
        assert 'price' in target_names or 'label' in target_names


class TestTargetProtection:
    """Test target protection logic."""
    
    def test_price_is_protected(self, detector):
        """Column named 'price' should be protected."""
        is_protected, reason = detector.is_protected_column('price')
        assert is_protected is True
        assert 'price' in reason.lower()
    
    def test_selling_price_is_protected(self, detector):
        """Column named 'selling_price' should be protected."""
        is_protected, reason = detector.is_protected_column('selling_price')
        assert is_protected is True
    
    def test_target_is_protected(self, detector):
        """Column named 'target' should be protected."""
        is_protected, reason = detector.is_protected_column('target')
        assert is_protected is True
    
    def test_label_is_protected(self, detector):
        """Column named 'label' should be protected."""
        is_protected, reason = detector.is_protected_column('label')
        assert is_protected is True
    
    def test_feature_not_protected(self, detector):
        """Column named 'feature' should NOT be protected."""
        is_protected, reason = detector.is_protected_column('feature')
        assert is_protected is False
    
    def test_id_not_protected(self, detector):
        """Column named 'id' should NOT be protected."""
        is_protected, reason = detector.is_protected_column('id')
        assert is_protected is False


class TestFalsePositivePrevention:
    """Test prevention of false positives."""
    
    def test_total_count_not_target(self, detector):
        """Column named 'total_count' should NOT be target (has 'count' exclusion)."""
        col = pd.Series([10, 20, 30, 40, 50])
        result = detector.detect_target(col, 'total_count')
        # 'count' is in exclusion list
        assert result.is_target is False or result.confidence < 0.7
    
    def test_created_at_not_target(self, detector):
        """Column named 'created_at' should NOT be target."""
        col = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        result = detector.detect_target(col, 'created_at')
        assert result.is_target is False
    
    def test_high_cardinality_not_target(self, detector):
        """High cardinality column without keyword should not be target."""
        col = pd.Series([f'value_{i}' for i in range(100)])
        result = detector.detect_target(col, 'random_column')
        # High cardinality without target keywords - not a target
        assert result.is_target is False or result.confidence < 0.6


class TestGetTargetDetector:
    """Test the singleton getter function."""
    
    def test_get_target_detector(self):
        """get_target_detector should return a detector instance."""
        detector = get_target_detector()
        assert isinstance(detector, TargetDetector)
    
    def test_detector_works(self):
        """Detector from getter should work correctly."""
        detector = get_target_detector()
        col = pd.Series([0, 1, 0, 1])
        result = detector.detect_target(col, 'target')
        assert result.is_target is True


class TestRealWorldScenarios:
    """Test with real-world dataset scenarios."""
    
    def test_car_dataset_selling_price(self, detector):
        """selling_price from car dataset should be detected as target."""
        col = pd.Series([450000, 350000, 550000, 275000, 650000, 800000])
        result = detector.detect_target(col, 'selling_price')
        assert result.is_target is True
        assert result.confidence >= 0.9
    
    def test_churn_dataset_churn(self, detector):
        """Churn column from churn dataset should be detected."""
        col = pd.Series(['Yes', 'No', 'Yes', 'No', 'No', 'Yes'])
        result = detector.detect_target(col, 'Churn')
        assert result.is_target is True
    
    def test_fraud_dataset_fraud(self, detector):
        """isFraud column from fraud dataset should be detected."""
        col = pd.Series([0, 0, 1, 0, 0, 0, 1, 0])
        result = detector.detect_target(col, 'isFraud')
        assert result.is_target is True
    
    def test_titanic_survived(self, detector):
        """Survived column from Titanic should be detected."""
        col = pd.Series([0, 1, 1, 0, 0, 1, 0])
        result = detector.detect_target(col, 'Survived')
        assert result.is_target is True
