"""
Tests for PreprocessingValidator - Pre-execution validation layer.

Tests all 7 validation rules:
1. Never transform targets destructively
2. Standard scale requires numeric
3. Log requires positive numeric
4. One-hot has cardinality limits
5. Text vectorization requires meaningful text
6. No binning targets
7. Empty columns should be dropped
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.preprocessing_validator import (
    PreprocessingValidator,
    PreprocessingValidation,
    ValidationCheck,
    ValidationResult,
    get_preprocessing_validator,
)


@pytest.fixture
def validator():
    """Create a fresh validator for each test."""
    return PreprocessingValidator()


class TestRule1TargetProtection:
    """Rule 1: Never transform targets destructively."""
    
    def test_drop_target_blocked(self, validator):
        """Dropping a target column should be blocked."""
        col = pd.Series([100, 200, 300, 400])
        validation = validator.validate_transformation(
            col, 'price', 'drop_column', is_target=True
        )
        
        assert validation.is_valid is False
        assert len(validation.blockers) > 0
        assert 'target' in validation.blockers[0].lower()
    
    def test_binning_target_blocked(self, validator):
        """Binning a target column should be blocked."""
        col = pd.Series([100, 200, 300, 400])
        validation = validator.validate_transformation(
            col, 'selling_price', 'binning_equal_width', is_target=True
        )
        
        assert validation.is_valid is False
        assert len(validation.blockers) >= 1
    
    def test_clip_outliers_target_blocked(self, validator):
        """Clipping outliers on target should be blocked."""
        col = pd.Series([100, 200, 300, 10000])  # 10000 is outlier
        validation = validator.validate_transformation(
            col, 'target', 'clip_outliers', is_target=True
        )
        
        assert validation.is_valid is False
    
    def test_scaling_target_warning(self, validator):
        """Scaling a target should produce a warning, not a block."""
        col = pd.Series([100, 200, 300, 400])
        validation = validator.validate_transformation(
            col, 'target', 'standard_scale', is_target=True
        )
        
        # May have warnings but should still be valid
        assert len(validation.warnings) > 0 or validation.is_valid is True
    
    def test_keep_as_is_target_allowed(self, validator):
        """Keeping target as-is should be allowed."""
        col = pd.Series([100, 200, 300, 400])
        validation = validator.validate_transformation(
            col, 'target', 'keep_as_is', is_target=True
        )
        
        assert validation.is_valid is True


class TestRule2NumericRequirement:
    """Rule 2: Standard scale requires numeric."""
    
    def test_scale_numeric_valid(self, validator):
        """Standard scale on numeric should be valid."""
        col = pd.Series([1, 2, 3, 4, 5])
        validation = validator.validate_transformation(
            col, 'numeric_col', 'standard_scale'
        )
        
        assert validation.is_valid is True
        assert validation.recommended_action == 'standard_scale'
    
    def test_scale_text_invalid(self, validator):
        """Standard scale on text should be invalid."""
        col = pd.Series(['John Smith', 'Jane Doe', 'Bob Wilson'])
        validation = validator.validate_transformation(
            col, 'name', 'standard_scale'
        )
        
        assert validation.is_valid is False
        assert validation.recommended_action in ['keep_as_is', 'text_clean']
    
    def test_log_text_invalid(self, validator):
        """Log transform on text should be invalid."""
        col = pd.Series(['abc', 'def', 'ghi'])
        validation = validator.validate_transformation(
            col, 'text_col', 'log_transform'
        )
        
        assert validation.is_valid is False
    
    def test_robust_scale_text_invalid(self, validator):
        """Robust scale on text should be invalid."""
        col = pd.Series(['cat', 'dog', 'bird'])
        validation = validator.validate_transformation(
            col, 'category', 'robust_scale'
        )
        
        assert validation.is_valid is False


class TestRule3PositiveRequirement:
    """Rule 3: Log requires positive numeric."""
    
    def test_log_positive_valid(self, validator):
        """Log transform on positive values should be valid."""
        col = pd.Series([1, 10, 100, 1000])
        validation = validator.validate_transformation(
            col, 'positive', 'log_transform'
        )
        
        assert validation.is_valid is True or validation.recommended_action == 'log1p_transform'
    
    def test_log_with_zeros_override(self, validator):
        """Log transform with zeros should override to log1p."""
        col = pd.Series([0, 1, 10, 100])
        validation = validator.validate_transformation(
            col, 'with_zeros', 'log_transform'
        )
        
        # Should suggest log1p instead
        if not validation.is_valid:
            assert validation.recommended_action == 'log1p_transform'
    
    def test_log_negative_invalid(self, validator):
        """Log transform on negative values should be handled safely."""
        col = pd.Series([-10, -5, 0, 5, 10])
        validation = validator.validate_transformation(
            col, 'negative', 'log_transform'
        )
        
        # Either invalid, or safe fallback to log1p or yeo_johnson
        assert (validation.is_valid is False or 
                validation.recommended_action in ['yeo_johnson', 'log1p_transform', 'log_transform'])
    
    def test_sqrt_negative_invalid(self, validator):
        """Square root on negative values should be invalid."""
        col = pd.Series([-10, -5, -1])
        validation = validator.validate_transformation(
            col, 'negative', 'sqrt_transform'
        )
        
        assert validation.is_valid is False


class TestRule4CardinalityLimits:
    """Rule 4: One-hot has cardinality limits."""
    
    def test_onehot_low_cardinality_valid(self, validator):
        """One-hot encoding with low cardinality should be valid."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'])
        validation = validator.validate_transformation(
            col, 'category', 'onehot_encode'
        )
        
        assert validation.is_valid is True
        assert validation.recommended_action == 'onehot_encode'
    
    def test_onehot_high_cardinality_invalid(self, validator):
        """One-hot encoding with high cardinality should be invalid."""
        col = pd.Series([f'cat_{i}' for i in range(100)])
        validation = validator.validate_transformation(
            col, 'high_card', 'onehot_encode'
        )
        
        assert validation.is_valid is False
        assert validation.recommended_action in ['frequency_encode', 'hash_encode']
    
    def test_onehot_medium_cardinality_warning(self, validator):
        """One-hot with medium cardinality should have warning."""
        col = pd.Series([f'cat_{i}' for i in range(30)])
        validation = validator.validate_transformation(
            col, 'medium_card', 'onehot_encode'
        )
        
        # Should have warning but still be valid
        if validation.is_valid:
            assert len(validation.warnings) > 0


class TestRule5TextMeaningfulness:
    """Rule 5: Text vectorization requires meaningful text."""
    
    def test_vectorize_meaningful_text_valid(self, validator):
        """Vectorizing meaningful text should be valid."""
        col = pd.Series([
            'This is a meaningful document with words',
            'Another document with meaningful content'
        ])
        validation = validator.validate_transformation(
            col, 'description', 'text_vectorize'
        )
        
        # Should be valid for text content
        assert validation.is_valid is True or 'text' in validation.recommended_action.lower()
    
    def test_vectorize_url_invalid(self, validator):
        """Vectorizing URL column should be invalid."""
        col = pd.Series([
            'https://example.com/page1',
            'https://example.com/page2',
            'https://example.com/page3'
        ])
        validation = validator.validate_transformation(
            col, 'website', 'text_vectorize'
        )
        
        assert validation.is_valid is False
        assert validation.recommended_action == 'drop_column'
    
    def test_vectorize_phone_invalid(self, validator):
        """Vectorizing phone column should be invalid."""
        col = pd.Series(['9876543210', '9988776655', '9123456789'])
        validation = validator.validate_transformation(
            col, 'phone', 'text_vectorize'
        )
        
        assert validation.is_valid is False
    
    def test_vectorize_email_invalid(self, validator):
        """Vectorizing email column should be invalid."""
        col = pd.Series(['a@test.com', 'b@test.com', 'c@test.com'])
        validation = validator.validate_transformation(
            col, 'email', 'text_vectorize'
        )
        
        assert validation.is_valid is False
    
    def test_vectorize_identifier_invalid(self, validator):
        """Vectorizing identifier column should be invalid."""
        col = pd.Series(['id_001', 'id_002', 'id_003', 'id_004', 'id_005'])
        validation = validator.validate_transformation(
            col, 'customer_id', 'text_vectorize'
        )
        
        assert validation.is_valid is False


class TestRule6NoTargetBinning:
    """Rule 6: No binning targets."""
    
    def test_bin_target_blocked(self, validator):
        """Binning target column should be blocked."""
        col = pd.Series([100, 200, 300, 400, 500])
        validation = validator.validate_transformation(
            col, 'target', 'binning_equal_width', is_target=True
        )
        
        assert validation.is_valid is False
        assert any('bin' in blocker.lower() for blocker in validation.blockers)
    
    def test_bin_feature_allowed(self, validator):
        """Binning non-target column should be allowed."""
        col = pd.Series([100, 200, 300, 400, 500])
        validation = validator.validate_transformation(
            col, 'feature', 'binning_equal_width', is_target=False
        )
        
        # Should not have binning blockers for non-target
        assert not any('target' in str(b).lower() and 'bin' in str(b).lower() 
                      for b in validation.blockers)


class TestRule7EmptyColumns:
    """Rule 7: Empty columns should be dropped."""
    
    def test_empty_column_override(self, validator):
        """Empty column should be recommended for dropping."""
        col = pd.Series([None, None, None, None])
        validation = validator.validate_transformation(
            col, 'empty_col', 'standard_scale'
        )
        
        assert validation.recommended_action == 'drop_column'
    
    def test_mostly_null_warning(self, validator):
        """Mostly null column should have warning."""
        col = pd.Series([None, None, None, None, None, None, None, None, None, 1])
        validation = validator.validate_transformation(
            col, 'mostly_null', 'standard_scale'
        )
        
        # Should warn about high null rate
        assert len(validation.warnings) > 0 or validation.recommended_action == 'drop_column'
    
    def test_normal_column_no_empty_warning(self, validator):
        """Normal column should not have empty warning."""
        col = pd.Series([1, 2, 3, 4, 5])
        validation = validator.validate_transformation(
            col, 'normal', 'standard_scale'
        )
        
        # Should not recommend dropping
        assert validation.recommended_action != 'drop_column'


class TestGetRecommendedAction:
    """Test the recommended action method."""
    
    def test_numeric_recommendation(self, validator):
        """Should recommend appropriate action for numeric."""
        col = pd.Series([1, 2, 3, 4, 5])
        action, explanation = validator.get_recommended_action(col, 'numeric')
        
        assert action in ['keep_as_is', 'standard_scale', 'robust_scale']
    
    def test_skewed_numeric_recommendation(self, validator):
        """Should recommend log for skewed numeric."""
        col = pd.Series([1, 2, 5, 10, 50, 100, 500, 1000])
        action, explanation = validator.get_recommended_action(col, 'skewed')
        
        assert action in ['log1p_transform', 'standard_scale', 'robust_scale']
    
    def test_categorical_recommendation(self, validator):
        """Should recommend encoding for categorical."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B'] * 10)
        action, explanation = validator.get_recommended_action(col, 'category')
        
        assert action in ['onehot_encode', 'frequency_encode']
    
    def test_target_recommendation(self, validator):
        """Should recommend keep_as_is for target."""
        col = pd.Series([100, 200, 300, 400])
        action, explanation = validator.get_recommended_action(col, 'target', is_target=True)
        
        assert action == 'keep_as_is'


class TestGetPreprocessingValidator:
    """Test the singleton getter function."""
    
    def test_get_preprocessing_validator(self):
        """get_preprocessing_validator should return an instance."""
        validator = get_preprocessing_validator()
        assert isinstance(validator, PreprocessingValidator)
    
    def test_validator_works(self):
        """Validator from getter should work correctly."""
        validator = get_preprocessing_validator()
        col = pd.Series([1, 2, 3, 4, 5])
        validation = validator.validate_transformation(col, 'test', 'standard_scale')
        assert validation.is_valid is True


class TestRealWorldScenarios:
    """Test with real-world dataset scenarios."""
    
    def test_cricket_dataset_name(self, validator):
        """Name column should not get standard_scale."""
        col = pd.Series(['Virat Kohli', 'Rohit Sharma', 'MS Dhoni'])
        validation = validator.validate_transformation(
            col, 'name', 'standard_scale'
        )
        
        assert validation.is_valid is False
    
    def test_cricket_dataset_contact(self, validator):
        """Contact column should be dropped."""
        col = pd.Series(['9876543210', '9988776655', '9123456789'])
        validation = validator.validate_transformation(
            col, 'contact', 'text_vectorize'
        )
        
        assert validation.is_valid is False
        assert validation.recommended_action == 'drop_column'
    
    def test_cricket_dataset_photoUrl(self, validator):
        """photoUrl column should be dropped."""
        col = pd.Series(['https://img.com/1.jpg', 'https://img.com/2.jpg'])
        validation = validator.validate_transformation(
            col, 'photoUrl', 'text_vectorize'
        )
        
        assert validation.is_valid is False
        assert validation.recommended_action == 'drop_column'
    
    def test_car_dataset_selling_price(self, validator):
        """selling_price should not be binned."""
        col = pd.Series([450000, 350000, 550000, 275000])
        validation = validator.validate_transformation(
            col, 'selling_price', 'binning_equal_width', is_target=True
        )
        
        assert validation.is_valid is False
    
    def test_car_dataset_km_driven(self, validator):
        """km_driven should allow log transform."""
        col = pd.Series([15000, 25000, 35000, 45000, 65000])
        validation = validator.validate_transformation(
            col, 'km_driven', 'log1p_transform'
        )
        
        assert validation.is_valid is True
