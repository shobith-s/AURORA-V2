"""
Test Safety Validator - Tests that safety checks prevent crashes.

This module tests the SafetyValidator to ensure it properly prevents
dangerous operations like scaling text columns or parsing numeric years as dates.
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.safety_validator import SafetyValidator, validate_action


class TestSafetyValidatorCrashPrevention:
    """Test that safety checks prevent crashes from type mismatches."""
    
    def test_prevents_scaling_text_fuel_type(self):
        """Must prevent standard_scale on 'fuel_type' (text)."""
        col = pd.Series(['Gas', 'Diesel', 'Electric', 'Hybrid'])
        is_safe, msg = SafetyValidator.can_apply(col, 'fuel_type', 'standard_scale')
        
        assert not is_safe, "Should prevent scaling text"
        assert 'object' in msg.lower() or 'text' in msg.lower(), \
            f"Error message should mention object/text dtype: {msg}"
    
    def test_prevents_scaling_text_int_col(self):
        """Must prevent standard_scale on 'int_col' (text interior colors)."""
        col = pd.Series(['Black', 'Beige', 'Gray', 'Brown'])
        is_safe, msg = SafetyValidator.can_apply(col, 'int_col', 'standard_scale')
        
        assert not is_safe, "Should prevent scaling text"
        assert 'object' in msg.lower() or 'text' in msg.lower()
    
    def test_prevents_datetime_on_numeric_year(self):
        """Must prevent parse_datetime on 'model_year' (numeric)."""
        col = pd.Series([2015, 2020, 2018, 2019, 2021])
        is_safe, msg = SafetyValidator.can_apply(col, 'model_year', 'parse_datetime')
        
        assert not is_safe, "Should prevent datetime parsing on numeric year"
        assert 'year' in msg.lower() or 'numeric' in msg.lower(), \
            f"Error message should mention year/numeric: {msg}"
    
    def test_prevents_hash_encode_continuous(self):
        """Must prevent hash_encode on 'milage' (continuous numeric)."""
        col = pd.Series(range(10000, 200000, 1000))  # 190 unique values
        is_safe, msg = SafetyValidator.can_apply(col, 'milage', 'hash_encode')
        
        assert not is_safe, "Should prevent hash encoding continuous numeric"
        assert 'continuous' in msg.lower() or 'unique' in msg.lower()
    
    def test_prevents_onehot_high_cardinality(self):
        """Must prevent onehot_encode on high cardinality columns."""
        # 100 unique values - too many for one-hot
        col = pd.Series([f'category_{i}' for i in range(100)])
        is_safe, msg = SafetyValidator.can_apply(col, 'model', 'onehot_encode')
        
        assert not is_safe, "Should prevent one-hot encoding high cardinality"
        assert '100' in msg or 'unique' in msg.lower()
    
    def test_prevents_log_transform_negative(self):
        """Must prevent log1p_transform on columns with negative values."""
        col = pd.Series([-10, -5, 0, 5, 10, 15])
        is_safe, msg = SafetyValidator.can_apply(col, 'delta', 'log1p_transform')
        
        assert not is_safe, "Should prevent log transform on negative values"
        assert 'negative' in msg.lower()
    
    def test_prevents_robust_scale_text(self):
        """Must prevent robust_scale on text columns."""
        col = pd.Series(['Low', 'Medium', 'High'])
        is_safe, msg = SafetyValidator.can_apply(col, 'priority', 'robust_scale')
        
        assert not is_safe, "Should prevent robust scaling text"
    
    def test_prevents_minmax_scale_text(self):
        """Must prevent minmax_scale on text columns."""
        col = pd.Series(['A', 'B', 'C', 'D'])
        is_safe, msg = SafetyValidator.can_apply(col, 'grade', 'minmax_scale')
        
        assert not is_safe, "Should prevent minmax scaling text"


class TestSafetyValidatorAllows:
    """Test that valid operations are allowed."""
    
    def test_allows_scale_numeric(self):
        """Should allow standard_scale on numeric columns."""
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        is_safe, msg = SafetyValidator.can_apply(col, 'value', 'standard_scale')
        
        assert is_safe, f"Should allow scaling numeric: {msg}"
        assert msg == ""
    
    def test_allows_log_positive(self):
        """Should allow log1p_transform on positive numeric columns."""
        col = pd.Series([1, 10, 100, 1000, 10000])
        is_safe, msg = SafetyValidator.can_apply(col, 'revenue', 'log1p_transform')
        
        assert is_safe, f"Should allow log transform on positive: {msg}"
    
    def test_allows_onehot_low_cardinality(self):
        """Should allow onehot_encode on low cardinality columns."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B'])
        is_safe, msg = SafetyValidator.can_apply(col, 'category', 'onehot_encode')
        
        assert is_safe, f"Should allow one-hot on low cardinality: {msg}"
    
    def test_allows_datetime_on_date_strings(self):
        """Should allow parse_datetime on date-formatted strings."""
        col = pd.Series(['2020-01-01', '2020-06-15', '2021-12-31'])
        is_safe, msg = SafetyValidator.can_apply(col, 'date', 'parse_datetime')
        
        assert is_safe, f"Should allow datetime parsing on date strings: {msg}"
    
    def test_allows_keep_as_is(self):
        """Should always allow keep_as_is."""
        cols = [
            pd.Series(['A', 'B', 'C'], name='text'),
            pd.Series([1, 2, 3], name='numeric'),
            pd.Series([None, None, None], name='nulls'),
        ]
        
        for col in cols:
            is_safe, msg = SafetyValidator.can_apply(col, col.name or 'col', 'keep_as_is')
            assert is_safe, f"keep_as_is should always be allowed"
    
    def test_allows_drop_column(self):
        """Should always allow drop_column."""
        col = pd.Series(['A', 'B', 'C'])
        is_safe, msg = SafetyValidator.can_apply(col, 'col', 'drop_column')
        
        assert is_safe, "drop_column should always be allowed"


class TestSafetyValidatorFallback:
    """Test the validate_action method with fallback suggestions."""
    
    def test_fallback_for_text_scaling(self):
        """Should suggest encoding fallback when scaling text fails."""
        col = pd.Series(['A', 'B', 'C'])
        is_safe, error, fallback = SafetyValidator.validate_action(col, 'col', 'standard_scale')
        
        assert not is_safe
        assert fallback in ['onehot_encode', 'label_encode', 'keep_as_is']
    
    def test_fallback_for_high_cardinality_onehot(self):
        """Should suggest label_encode when one-hot fails on high cardinality."""
        col = pd.Series([f'cat_{i}' for i in range(100)])
        is_safe, error, fallback = SafetyValidator.validate_action(col, 'category', 'onehot_encode')
        
        assert not is_safe
        assert fallback == 'keep_as_is'  # Very high cardinality
    
    def test_fallback_for_negative_log(self):
        """Should suggest standard_scale when log fails on negative values."""
        col = pd.Series([-5, -2, 0, 2, 5])
        is_safe, error, fallback = SafetyValidator.validate_action(col, 'delta', 'log1p_transform')
        
        assert not is_safe
        assert fallback == 'standard_scale'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_column(self):
        """Handle empty columns gracefully."""
        col = pd.Series([], dtype=float)
        is_safe, msg = SafetyValidator.can_apply(col, 'empty', 'standard_scale')
        # Empty numeric column should be safe for scaling
        assert is_safe
    
    def test_all_null_column(self):
        """Handle all-null columns."""
        col = pd.Series([None, None, None])
        is_safe, msg = SafetyValidator.can_apply(col, 'nulls', 'keep_as_is')
        assert is_safe
    
    def test_mixed_types_column(self):
        """Handle mixed type columns (treated as object)."""
        col = pd.Series([1, 'two', 3.0, 'four'])
        is_safe, msg = SafetyValidator.can_apply(col, 'mixed', 'standard_scale')
        # Mixed types become object dtype, should fail scaling
        assert not is_safe
    
    def test_datetime_column(self):
        """Handle datetime columns."""
        col = pd.Series(pd.to_datetime(['2020-01-01', '2020-06-15', '2021-12-31']))
        is_safe, msg = SafetyValidator.can_apply(col, 'date', 'standard_scale')
        # Datetime is not numeric, should fail
        assert not is_safe
    
    def test_year_boundary_1900(self):
        """Year columns at boundary 1900 should not be parsed as datetime."""
        col = pd.Series([1900, 1950, 2000])
        is_safe, msg = SafetyValidator.can_apply(col, 'year', 'parse_datetime')
        assert not is_safe
    
    def test_year_boundary_2100(self):
        """Year columns at boundary 2100 should not be parsed as datetime."""
        col = pd.Series([2050, 2080, 2100])
        is_safe, msg = SafetyValidator.can_apply(col, 'year', 'parse_datetime')
        assert not is_safe
    
    def test_non_year_numeric_can_parse_datetime(self):
        """Non-year numeric columns should still allow datetime parsing."""
        # Unix timestamps
        col = pd.Series([1609459200, 1612137600, 1614556800])
        is_safe, msg = SafetyValidator.can_apply(col, 'timestamp', 'parse_datetime')
        # These are outside 1900-2100 range, so should be allowed
        assert is_safe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
