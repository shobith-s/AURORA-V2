"""
Test Smart Classifier on Car Dataset - Tests on actual problematic columns from user's car dataset.

This test validates that the SmartColumnClassifier correctly handles all 12 columns
from the car dataset that were previously failing (58% error rate).
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.smart_classifier import SmartColumnClassifier, classify_column


class TestCarDatasetColumns:
    """Test on actual problematic columns from user's car dataset."""
    
    def test_car_dataset_all_columns(self):
        """
        Test on actual problematic columns from user's dataset.
        Must achieve 0% error rate (12/12 correct decisions).
        """
        test_cases = [
            # (column_name, sample_data, expected_action, reason)
            ('brand', ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW'], 'onehot_encode', 'Categorical brand'),
            ('model', ['Civic', 'Accord', 'F-150', 'Camry', 'Corolla', 'Mustang', 'Fusion', 
                      'Explorer', 'Escape', 'Focus', 'Edge', 'Ranger'], 'label_encode', 'Medium cardinality model'),
            ('model_year', [2015, 2020, 2018, 2019, 2021], 'standard_scale', 'Numeric year'),
            ('milage', [50000, 100000, 75000, 25000, 150000], 'log1p_transform', 'Distance measure'),
            ('fuel_type', ['Gas', 'Diesel', 'Electric', 'Hybrid'], 'onehot_encode', 'Categorical fuel type'),
            ('engine', ['V6', 'V8', 'I4', 'V6 Twin Turbo', 'I4 Turbo'], 'onehot_encode', 'Categorical engine'),
            ('transmission', ['Manual', 'Automatic'], 'onehot_encode', 'Binary transmission'),
            ('ext_col', ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green', 
                        'Brown', 'Orange', 'Yellow', 'Purple'], 'drop_column', 'High cardinality color'),
            ('int_col', ['Black', 'Beige', 'Gray', 'Brown', 'Tan'], 'onehot_encode', 'Interior categorical'),
            ('accident', ['Yes', 'No', 'Yes'], 'keep_as_is', 'Critical binary accident'),
            ('clean_title', ['Yes', 'Yes', 'No'], 'keep_as_is', 'Critical binary title'),
            ('price', [10000, 20000, 30000, 15000, 25000], 'keep_as_is', 'TARGET VARIABLE'),
        ]
        
        errors = []
        successes = []
        
        for col_name, data, expected, reason in test_cases:
            col = pd.Series(data, name=col_name)
            result = SmartColumnClassifier.classify(col_name, col)
            
            if result['action'] != expected:
                errors.append({
                    'column': col_name,
                    'expected': expected,
                    'got': result['action'],
                    'reason': reason,
                    'classifier_reason': result['reason']
                })
            else:
                successes.append({
                    'column': col_name,
                    'action': expected,
                    'reason': reason
                })
        
        # Report results
        total = len(test_cases)
        passed = len(successes)
        failed = len(errors)
        
        print(f"\n{'='*60}")
        print(f"CAR DATASET TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if errors:
            print(f"\nFailed columns:")
            for err in errors:
                print(f"  - {err['column']}: expected '{err['expected']}', got '{err['got']}'")
                print(f"    Reason: {err['reason']}")
        
        # MUST have 0 errors for critical columns
        assert len(errors) == 0, f"Failed on {len(errors)}/{total} columns: {errors}"
    
    def test_price_column_never_dropped(self):
        """Price column must NEVER be dropped - it's the target variable."""
        test_prices = [
            pd.Series([10000, 20000, 30000], name='price'),
            pd.Series([5000.50, 15000.99, 25000.00], name='selling_price'),
            pd.Series([100, 200, 300, 400, 500], name='sale_price'),
            pd.Series([1000000, 2000000], name='cost'),
        ]
        
        for col in test_prices:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] == 'keep_as_is', \
                f"Target column '{col.name}' should be kept, got '{result['action']}'"
            assert result['confidence'] >= 0.95, \
                f"Target column should have high confidence, got {result['confidence']}"
    
    def test_model_year_not_parsed_as_datetime(self):
        """Model year (2015, 2020, etc.) should NOT be parsed as datetime."""
        year_columns = [
            pd.Series([2015, 2020, 2018, 2019], name='model_year'),
            pd.Series([2010, 2015, 2020, 2022], name='year'),
            pd.Series([1990, 2000, 2010, 2020], name='year_built'),
        ]
        
        for col in year_columns:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] != 'parse_datetime', \
                f"Year column '{col.name}' should NOT use parse_datetime, got '{result['action']}'"
            assert result['action'] == 'standard_scale', \
                f"Year column '{col.name}' should use standard_scale, got '{result['action']}'"
    
    def test_mileage_uses_log_transform(self):
        """Mileage/distance columns should use log1p_transform."""
        distance_columns = [
            pd.Series([50000, 100000, 75000, 25000], name='milage'),
            pd.Series([10000, 50000, 100000, 200000], name='mileage'),
            pd.Series([100, 500, 1000, 5000], name='odometer'),
            pd.Series([10, 50, 100, 500, 1000], name='miles'),
        ]
        
        for col in distance_columns:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] == 'log1p_transform', \
                f"Distance column '{col.name}' should use log1p_transform, got '{result['action']}'"
    
    def test_text_columns_not_scaled(self):
        """Text/categorical columns should NEVER be standard_scale."""
        text_columns = [
            pd.Series(['Gas', 'Diesel', 'Electric'], name='fuel_type'),
            pd.Series(['Black', 'White', 'Gray'], name='int_col'),
            pd.Series(['Automatic', 'Manual'], name='transmission'),
        ]
        
        for col in text_columns:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] != 'standard_scale', \
                f"Text column '{col.name}' should NOT use standard_scale, got '{result['action']}'"
            assert result['action'] in ['onehot_encode', 'label_encode', 'keep_as_is'], \
                f"Text column '{col.name}' should use encoding, got '{result['action']}'"
    
    def test_binary_accident_columns_preserved(self):
        """Binary indicator columns (accident, clean_title) should be preserved."""
        binary_columns = [
            pd.Series(['Yes', 'No', 'Yes'], name='accident'),
            pd.Series(['Yes', 'Yes', 'No'], name='clean_title'),
            pd.Series(['Yes', 'No'], name='warranty'),
            pd.Series(['Yes', 'No', 'No'], name='certified'),
        ]
        
        for col in binary_columns:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] == 'keep_as_is', \
                f"Binary column '{col.name}' should be kept as-is, got '{result['action']}'"


class TestSmartClassifierRules:
    """Test individual classification rules."""
    
    def test_high_cardinality_dropped(self):
        """High cardinality ID-like columns should be dropped."""
        # VIN numbers - all unique
        vins = [f'1HGBH41JXMN{i:06d}' for i in range(100)]
        col = pd.Series(vins, name='vin')
        result = SmartColumnClassifier.classify('vin', col)
        assert result['action'] == 'drop_column'
    
    def test_id_columns_dropped(self):
        """ID columns should be dropped."""
        id_columns = [
            pd.Series(range(100), name='id'),
            pd.Series([f'STK{i}' for i in range(100)], name='stock_number'),
            pd.Series([f'LST{i}' for i in range(100)], name='listing_id'),
        ]
        
        for col in id_columns:
            result = SmartColumnClassifier.classify(col.name, col)
            assert result['action'] == 'drop_column', \
                f"ID column '{col.name}' should be dropped, got '{result['action']}'"
    
    def test_low_cardinality_categorical(self):
        """Low cardinality categorical columns should be one-hot encoded."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B'], name='category')
        result = SmartColumnClassifier.classify('category', col)
        assert result['action'] == 'onehot_encode'
    
    def test_medium_cardinality_categorical(self):
        """Medium cardinality categorical should use label encoding."""
        # 15 unique values
        values = [f'Category_{i}' for i in range(15)] * 10
        col = pd.Series(values, name='category')
        result = SmartColumnClassifier.classify('category', col)
        assert result['action'] == 'label_encode'
    
    def test_numeric_skewed_uses_log(self):
        """Highly skewed numeric data should use log transform."""
        # Highly right-skewed data (like income, prices)
        np.random.seed(42)
        data = np.random.exponential(scale=10000, size=1000)
        col = pd.Series(data, name='revenue')
        result = SmartColumnClassifier.classify('revenue', col)
        assert result['action'] == 'log1p_transform'
    
    def test_numeric_normal_uses_standard_scale(self):
        """Normal numeric data should use standard scaling."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        col = pd.Series(data, name='score')
        result = SmartColumnClassifier.classify('score', col)
        assert result['action'] == 'standard_scale'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
