"""
Tests for SafeTransforms - Safe transformation wrappers.

Tests all safe transformation methods:
- safe_standard_scale
- safe_log_transform (uses log1p)
- safe_onehot_encode
- safe_text_vectorize
- safe_datetime_extract
- safe_robust_scale
- safe_minmax_scale
- safe_text_clean
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.safe_transforms import (
    SafeTransforms,
    SafeTransformResult,
    TransformResult,
    get_safe_transforms,
)


@pytest.fixture
def transforms():
    """Create a fresh SafeTransforms instance for each test."""
    return SafeTransforms()


class TestSafeStandardScale:
    """Test safe standard scaling."""
    
    def test_normal_numeric_column(self, transforms):
        """Standard scale should work on normal numeric column."""
        col = pd.Series([1, 2, 3, 4, 5])
        result = transforms.safe_standard_scale(col, 'numeric')
        
        assert result.status == TransformResult.SUCCESS
        assert result.data is not None
        assert abs(result.data.mean()) < 0.1  # Mean should be ~0
        assert abs(result.data.std() - 1.0) < 0.2  # Std should be ~1
    
    def test_non_numeric_column_skipped(self, transforms):
        """Standard scale should skip non-numeric columns."""
        col = pd.Series(['a', 'b', 'c', 'd', 'e'])
        result = transforms.safe_standard_scale(col, 'text')
        
        assert result.status in [TransformResult.SKIPPED, TransformResult.SUCCESS]
        assert result.actual_action in ['keep_as_is', 'standard_scale']
    
    def test_all_null_column(self, transforms):
        """Standard scale should skip all-null columns."""
        col = pd.Series([None, None, None])
        result = transforms.safe_standard_scale(col, 'null_col')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'keep_as_is'
    
    def test_zero_variance_column(self, transforms):
        """Standard scale should skip zero-variance columns."""
        col = pd.Series([5, 5, 5, 5, 5])
        result = transforms.safe_standard_scale(col, 'constant')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'keep_as_is'
        assert 'zero variance' in result.explanation.lower()
    
    def test_with_nulls(self, transforms):
        """Standard scale should handle columns with some nulls."""
        col = pd.Series([1, 2, None, 4, 5])
        result = transforms.safe_standard_scale(col, 'partial_null')
        
        assert result.status == TransformResult.SUCCESS
        assert pd.isna(result.data[2])  # Null should remain null


class TestSafeLogTransform:
    """Test safe log transformation (uses log1p)."""
    
    def test_positive_values(self, transforms):
        """Log transform should work on positive values."""
        col = pd.Series([1, 10, 100, 1000])
        result = transforms.safe_log_transform(col, 'positive')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'log1p_transform'
        # log1p(1) ≈ 0.693, log1p(10) ≈ 2.398, etc.
        assert result.data[0] > 0
    
    def test_with_zeros(self, transforms):
        """Log transform should handle zeros safely with log1p."""
        col = pd.Series([0, 1, 10, 100])
        result = transforms.safe_log_transform(col, 'with_zeros')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'log1p_transform'
        assert result.data[0] == 0  # log1p(0) = 0
        assert not np.isinf(result.data).any()  # No infinities
    
    def test_negative_values_shifted(self, transforms):
        """Log transform should shift negative values."""
        col = pd.Series([-10, -5, 0, 5, 10])
        result = transforms.safe_log_transform(col, 'negative')
        
        assert result.status == TransformResult.SUCCESS
        assert len(result.warnings) > 0  # Should warn about shifting
        assert not np.isinf(result.data.dropna()).any()
    
    def test_non_numeric_skipped(self, transforms):
        """Log transform should skip non-numeric columns."""
        col = pd.Series(['a', 'b', 'c'])
        result = transforms.safe_log_transform(col, 'text')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'keep_as_is'


class TestSafeOnehotEncode:
    """Test safe one-hot encoding."""
    
    def test_low_cardinality(self, transforms):
        """One-hot should work on low cardinality columns."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B'])
        result = transforms.safe_onehot_encode(col, 'category')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'onehot_encode'
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data.columns) == 3  # A, B, C
    
    def test_high_cardinality_fallback(self, transforms):
        """One-hot should fallback to frequency encoding for high cardinality."""
        col = pd.Series([f'cat_{i}' for i in range(100)])
        result = transforms.safe_onehot_encode(col, 'high_card')
        
        assert result.status == TransformResult.FALLBACK
        assert result.actual_action == 'frequency_encode'
        assert len(result.warnings) > 0
    
    def test_all_null_column(self, transforms):
        """One-hot should skip all-null columns."""
        col = pd.Series([None, None, None])
        result = transforms.safe_onehot_encode(col, 'null_col')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'keep_as_is'
    
    def test_binary_column(self, transforms):
        """One-hot should work on binary columns."""
        col = pd.Series(['Yes', 'No', 'Yes', 'No'])
        result = transforms.safe_onehot_encode(col, 'binary')
        
        assert result.status == TransformResult.SUCCESS
        assert len(result.data.columns) == 2


class TestSafeTextVectorize:
    """Test safe text vectorization."""
    
    def test_meaningful_text(self, transforms):
        """Text vectorization should work on meaningful text."""
        col = pd.Series([
            'This is a sample text document',
            'Another document with more words',
            'Yet another text sample here'
        ])
        result = transforms.safe_text_vectorize(col, 'text_col')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'text_vectorize'
        assert isinstance(result.data, pd.DataFrame)
    
    def test_url_column_rejected(self, transforms):
        """Text vectorization should reject URL columns."""
        col = pd.Series([
            'https://example.com/page1',
            'https://example.com/page2',
            'https://example.com/page3'
        ])
        result = transforms.safe_text_vectorize(col, 'url_col')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'drop_column'
        assert 'url' in result.explanation.lower()
    
    def test_id_column_rejected(self, transforms):
        """Text vectorization should reject ID-like columns."""
        col = pd.Series([
            'abc123def456ghi',
            'xyz789uvw012rst',
            'qwe345rty678uio',
            'asd234fgh567jkl',
            'zxc890vbn123mnb'
        ])
        result = transforms.safe_text_vectorize(col, 'id_col')
        
        assert result.status == TransformResult.SKIPPED
    
    def test_short_text_skipped(self, transforms):
        """Text vectorization should skip very short text."""
        col = pd.Series(['a', 'b', 'c', 'd'])
        result = transforms.safe_text_vectorize(col, 'short')
        
        assert result.status == TransformResult.SKIPPED
    
    def test_empty_column(self, transforms):
        """Text vectorization should handle empty columns."""
        col = pd.Series([None, None, None])
        result = transforms.safe_text_vectorize(col, 'empty')
        
        assert result.status == TransformResult.SKIPPED


class TestSafeDatetimeExtract:
    """Test safe datetime extraction."""
    
    def test_iso_datetime(self, transforms):
        """Datetime extraction should work on ISO datetime."""
        col = pd.Series(['2023-01-15', '2023-06-20', '2023-12-31'])
        result = transforms.safe_datetime_extract(col, 'date')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'datetime_extract'
        assert isinstance(result.data, pd.DataFrame)
        assert 'date_year' in result.data.columns
        assert 'date_month' in result.data.columns
    
    def test_datetime_with_time(self, transforms):
        """Datetime extraction should work on full datetime."""
        col = pd.Series(['2023-01-15 10:30:00', '2023-06-20 14:45:30'])
        result = transforms.safe_datetime_extract(col, 'timestamp')
        
        assert result.status == TransformResult.SUCCESS
    
    def test_native_datetime(self, transforms):
        """Datetime extraction should work on native datetime."""
        col = pd.Series(pd.to_datetime(['2023-01-15', '2023-06-20']))
        result = transforms.safe_datetime_extract(col, 'created_at')
        
        assert result.status == TransformResult.SUCCESS
    
    def test_invalid_datetime_skipped(self, transforms):
        """Datetime extraction should skip invalid datetime columns."""
        col = pd.Series(['not a date', 'also not a date', 'random text'])
        result = transforms.safe_datetime_extract(col, 'text')
        
        # Most values fail parsing
        assert result.status == TransformResult.SKIPPED
    
    def test_custom_components(self, transforms):
        """Datetime extraction should support custom components."""
        col = pd.Series(['2023-01-15', '2023-06-20', '2023-12-31'])
        result = transforms.safe_datetime_extract(col, 'date', components=['year', 'month'])
        
        assert result.status == TransformResult.SUCCESS
        assert 'date_year' in result.data.columns
        assert 'date_month' in result.data.columns
        assert 'date_day' not in result.data.columns


class TestSafeRobustScale:
    """Test safe robust scaling."""
    
    def test_normal_column(self, transforms):
        """Robust scale should work on normal numeric columns."""
        col = pd.Series([1, 2, 3, 4, 5, 100])  # With outlier
        result = transforms.safe_robust_scale(col, 'with_outlier')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'robust_scale'
    
    def test_non_numeric_skipped(self, transforms):
        """Robust scale should skip non-numeric columns."""
        col = pd.Series(['a', 'b', 'c'])
        result = transforms.safe_robust_scale(col, 'text')
        
        assert result.status == TransformResult.SKIPPED
        assert result.actual_action == 'keep_as_is'
    
    def test_zero_iqr(self, transforms):
        """Robust scale should handle zero IQR."""
        col = pd.Series([5, 5, 5, 5, 5])
        result = transforms.safe_robust_scale(col, 'constant')
        
        assert result.status == TransformResult.SKIPPED
        assert 'iqr' in result.explanation.lower()


class TestSafeMinmaxScale:
    """Test safe min-max scaling."""
    
    def test_normal_column(self, transforms):
        """Min-max scale should work on normal numeric columns."""
        col = pd.Series([0, 25, 50, 75, 100])
        result = transforms.safe_minmax_scale(col, 'percentage')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'minmax_scale'
        assert result.data.min() >= 0 - 0.01  # Allow small epsilon
        assert result.data.max() <= 1 + 0.01
    
    def test_non_numeric_skipped(self, transforms):
        """Min-max scale should skip non-numeric columns."""
        col = pd.Series(['a', 'b', 'c'])
        result = transforms.safe_minmax_scale(col, 'text')
        
        assert result.status == TransformResult.SKIPPED
    
    def test_zero_range(self, transforms):
        """Min-max scale should handle zero range."""
        col = pd.Series([5, 5, 5, 5])
        result = transforms.safe_minmax_scale(col, 'constant')
        
        assert result.status == TransformResult.SKIPPED


class TestSafeTextClean:
    """Test safe text cleaning."""
    
    def test_basic_cleaning(self, transforms):
        """Text clean should normalize text."""
        col = pd.Series(['  Hello World  ', 'UPPERCASE', 'multiple   spaces'])
        result = transforms.safe_text_clean(col, 'text')
        
        assert result.status == TransformResult.SUCCESS
        assert result.actual_action == 'text_clean'
        assert result.data[0] == 'hello world'
        assert result.data[1] == 'uppercase'
        assert result.data[2] == 'multiple spaces'
    
    def test_numeric_to_string(self, transforms):
        """Text clean should convert numeric to string."""
        col = pd.Series([123, 456, 789])
        result = transforms.safe_text_clean(col, 'numbers')
        
        assert result.status == TransformResult.SUCCESS
        assert result.data[0] == '123'


class TestGetSafeTransforms:
    """Test the singleton getter function."""
    
    def test_get_safe_transforms(self):
        """get_safe_transforms should return an instance."""
        transforms = get_safe_transforms()
        assert isinstance(transforms, SafeTransforms)
    
    def test_transforms_work(self):
        """Transforms from getter should work correctly."""
        transforms = get_safe_transforms()
        col = pd.Series([1, 2, 3, 4, 5])
        result = transforms.safe_standard_scale(col, 'test')
        assert result.status == TransformResult.SUCCESS


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series(self, transforms):
        """Should handle empty series gracefully."""
        col = pd.Series([], dtype=float)
        result = transforms.safe_standard_scale(col, 'empty')
        assert result.status == TransformResult.SKIPPED
    
    def test_single_value(self, transforms):
        """Should handle single value series."""
        col = pd.Series([5])
        result = transforms.safe_standard_scale(col, 'single')
        # Single value = zero variance
        assert result.status == TransformResult.SKIPPED
    
    def test_mixed_types(self, transforms):
        """Should handle mixed type columns."""
        col = pd.Series([1, 'two', 3, None, 5])
        result = transforms.safe_standard_scale(col, 'mixed')
        # Should try to convert to numeric or skip
        assert result.status in [TransformResult.SUCCESS, TransformResult.SKIPPED]
    
    def test_preserves_index(self, transforms):
        """Transformed data should preserve original index."""
        col = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
        result = transforms.safe_standard_scale(col, 'indexed')
        
        assert result.status == TransformResult.SUCCESS
        assert list(result.data.index) == ['a', 'b', 'c', 'd', 'e']
