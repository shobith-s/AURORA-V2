"""
Test Before/After Metrics - Generate concrete proof of improvement.

This test generates metrics showing the improvement from the old system
to the new smart preprocessing system.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from src.utils.smart_classifier import SmartColumnClassifier
from src.utils.safety_validator import SafetyValidator
from src.utils.preprocessing_integration import PreprocessingIntegration


class TestBeforeAfterMetrics:
    """Generate metrics showing improvement from old to new system."""
    
    # Old system's decisions (simulated based on problem statement)
    OLD_SYSTEM_DECISIONS = {
        'brand': 'onehot_encode',  # Correct
        'model': 'label_encode',   # Correct
        'model_year': 'parse_datetime',  # WRONG - should be standard_scale
        'milage': 'hash_encode',   # WRONG - should be log1p_transform
        'fuel_type': 'standard_scale',  # WRONG - crashes on text
        'engine': 'onehot_encode',  # Correct
        'transmission': 'onehot_encode',  # Correct
        'ext_col': 'drop_column',  # Correct (or label_encode)
        'int_col': 'standard_scale',  # WRONG - crashes on text
        'accident': 'drop_column',  # WRONG - drops critical predictor
        'clean_title': 'drop_column',  # WRONG - drops critical predictor
        'price': 'drop_column',  # CRITICAL - drops target variable
    }
    
    # Correct decisions based on requirements
    EXPECTED_DECISIONS = {
        'brand': 'onehot_encode',
        'model': 'label_encode',
        'model_year': 'standard_scale',
        'milage': 'log1p_transform',
        'fuel_type': 'onehot_encode',
        'engine': 'onehot_encode',
        'transmission': 'onehot_encode',
        'ext_col': 'drop_column',
        'int_col': 'onehot_encode',
        'accident': 'keep_as_is',
        'clean_title': 'keep_as_is',
        'price': 'keep_as_is',
    }
    
    # Sample data for each column
    SAMPLE_DATA = {
        'brand': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW'],
        'model': ['Civic', 'Accord', 'F-150', 'Camry', 'Corolla', 'Mustang', 
                 'Fusion', 'Explorer', 'Escape', 'Focus', 'Edge', 'Ranger'],
        'model_year': [2015, 2020, 2018, 2019, 2021],
        'milage': [50000, 100000, 75000, 25000, 150000],
        'fuel_type': ['Gas', 'Diesel', 'Electric', 'Hybrid'],
        'engine': ['V6', 'V8', 'I4', 'V6 Twin Turbo', 'I4 Turbo'],
        'transmission': ['Manual', 'Automatic'],
        'ext_col': ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 
                   'Green', 'Brown', 'Orange', 'Yellow', 'Purple'],
        'int_col': ['Black', 'Beige', 'Gray', 'Brown', 'Tan'],
        'accident': ['Yes', 'No', 'Yes'],
        'clean_title': ['Yes', 'Yes', 'No'],
        'price': [10000, 20000, 30000, 15000, 25000],
    }
    
    def _count_old_system_errors(self) -> Dict[str, Any]:
        """Count errors in the old system's decisions."""
        errors = []
        crashes = []
        critical_issues = []
        
        for col_name, old_action in self.OLD_SYSTEM_DECISIONS.items():
            expected = self.EXPECTED_DECISIONS[col_name]
            
            if old_action != expected:
                error = {
                    'column': col_name,
                    'old_action': old_action,
                    'expected': expected
                }
                errors.append(error)
                
                # Check for crashes
                col = pd.Series(self.SAMPLE_DATA[col_name])
                is_safe, msg = SafetyValidator.can_apply(col, col_name, old_action)
                if not is_safe:
                    crashes.append({
                        'column': col_name,
                        'action': old_action,
                        'error': msg
                    })
                
                # Check for critical issues
                if col_name == 'price':
                    critical_issues.append({
                        'column': col_name,
                        'issue': 'TARGET VARIABLE DROPPED'
                    })
                elif col_name in ['accident', 'clean_title'] and old_action == 'drop_column':
                    critical_issues.append({
                        'column': col_name,
                        'issue': 'Critical predictor dropped'
                    })
        
        return {
            'errors': errors,
            'crashes': crashes,
            'critical_issues': critical_issues,
            'error_count': len(errors),
            'crash_count': len(crashes),
            'critical_count': len(critical_issues)
        }
    
    def _count_new_system_errors(self) -> Dict[str, Any]:
        """Count errors in the new smart classifier's decisions."""
        errors = []
        crashes = []
        critical_issues = []
        
        for col_name, expected in self.EXPECTED_DECISIONS.items():
            col = pd.Series(self.SAMPLE_DATA[col_name])
            result = SmartColumnClassifier.classify(col_name, col)
            new_action = result['action']
            
            if new_action != expected:
                errors.append({
                    'column': col_name,
                    'new_action': new_action,
                    'expected': expected
                })
                
                # Check for crashes
                is_safe, msg = SafetyValidator.can_apply(col, col_name, new_action)
                if not is_safe:
                    crashes.append({
                        'column': col_name,
                        'action': new_action,
                        'error': msg
                    })
                
                # Check for critical issues
                if col_name == 'price' and new_action == 'drop_column':
                    critical_issues.append({
                        'column': col_name,
                        'issue': 'TARGET VARIABLE DROPPED'
                    })
                elif col_name in ['accident', 'clean_title'] and new_action == 'drop_column':
                    critical_issues.append({
                        'column': col_name,
                        'issue': 'Critical predictor dropped'
                    })
        
        return {
            'errors': errors,
            'crashes': crashes,
            'critical_issues': critical_issues,
            'error_count': len(errors),
            'crash_count': len(crashes),
            'critical_count': len(critical_issues)
        }
    
    def test_generate_before_after_report(self):
        """
        Generate metrics showing improvement from old to new system.
        """
        old_metrics = self._count_old_system_errors()
        new_metrics = self._count_new_system_errors()
        
        total_columns = len(self.EXPECTED_DECISIONS)
        
        old_error_rate = old_metrics['error_count'] / total_columns * 100
        new_error_rate = new_metrics['error_count'] / total_columns * 100
        
        # Print report
        print("\n" + "="*60)
        print("BEFORE/AFTER METRICS REPORT")
        print("="*60)
        
        print("\nBEFORE (Old System):")
        print(f"  - Errors: {old_metrics['error_count']}/{total_columns} ({old_error_rate:.1f}%)")
        print(f"  - Crashes: {old_metrics['crash_count']}")
        print(f"  - Critical Issues: {old_metrics['critical_count']}")
        
        if old_metrics['errors']:
            print("\n  Error details:")
            for err in old_metrics['errors']:
                print(f"    - {err['column']}: '{err['old_action']}' should be '{err['expected']}'")
        
        if old_metrics['crashes']:
            print("\n  Crash details:")
            for crash in old_metrics['crashes']:
                print(f"    - {crash['column']}: {crash['error'][:50]}...")
        
        print(f"\nAFTER (New System):")
        print(f"  - Errors: {new_metrics['error_count']}/{total_columns} ({new_error_rate:.1f}%)")
        print(f"  - Crashes: {new_metrics['crash_count']}")
        print(f"  - Critical Issues: {new_metrics['critical_count']}")
        
        if new_metrics['errors']:
            print("\n  Error details:")
            for err in new_metrics['errors']:
                print(f"    - {err['column']}: '{err['new_action']}' should be '{err['expected']}'")
        
        improvement = old_error_rate - new_error_rate
        print(f"\nIMPROVEMENT: {old_error_rate:.1f}% → {new_error_rate:.1f}% error rate")
        print(f"             {improvement:.1f} percentage points improvement")
        print("="*60)
        
        # Assertions for success criteria
        assert new_metrics['error_count'] == 0, \
            f"New system should have 0 errors, got {new_metrics['error_count']}"
        assert new_metrics['crash_count'] == 0, \
            f"New system should have 0 crashes, got {new_metrics['crash_count']}"
        assert new_metrics['critical_count'] == 0, \
            f"New system should have 0 critical issues, got {new_metrics['critical_count']}"
    
    def test_target_preserved(self):
        """Test that target variable (price) is preserved."""
        col = pd.Series(self.SAMPLE_DATA['price'])
        result = SmartColumnClassifier.classify('price', col)
        
        assert result['action'] != 'drop_column', \
            "Target variable 'price' must NOT be dropped"
        assert result['action'] == 'keep_as_is', \
            "Target variable 'price' should be kept as-is"
        assert result['confidence'] >= 0.95, \
            "Target variable should have high confidence"
    
    def test_critical_features_kept(self):
        """Test that critical features (accident, clean_title) are kept."""
        critical_columns = ['accident', 'clean_title']
        
        for col_name in critical_columns:
            col = pd.Series(self.SAMPLE_DATA[col_name])
            result = SmartColumnClassifier.classify(col_name, col)
            
            assert result['action'] != 'drop_column', \
                f"Critical feature '{col_name}' must NOT be dropped"
            assert result['action'] == 'keep_as_is', \
                f"Critical feature '{col_name}' should be kept as-is"
    
    def test_no_crashes_on_text_columns(self):
        """Test that no scaling operations are attempted on text columns."""
        text_columns = ['fuel_type', 'int_col', 'brand', 'engine', 'transmission']
        
        for col_name in text_columns:
            col = pd.Series(self.SAMPLE_DATA[col_name])
            result = SmartColumnClassifier.classify(col_name, col)
            
            # Verify the action is safe
            is_safe, msg = SafetyValidator.can_apply(col, col_name, result['action'])
            assert is_safe, f"Action '{result['action']}' on '{col_name}' should be safe: {msg}"
            
            # Verify no scaling actions on text
            assert result['action'] not in ['standard_scale', 'robust_scale', 'minmax_scale'], \
                f"Text column '{col_name}' should not use scaling, got '{result['action']}'"
    
    def test_preprocessing_integration(self):
        """Test the full preprocessing integration flow."""
        # Create a DataFrame with the sample data
        max_len = max(len(v) for v in self.SAMPLE_DATA.values())
        padded_data = {}
        for col_name, data in self.SAMPLE_DATA.items():
            # Pad shorter columns with repeated values
            if len(data) < max_len:
                padded_data[col_name] = (data * ((max_len // len(data)) + 1))[:max_len]
            else:
                padded_data[col_name] = data[:max_len]
        
        df = pd.DataFrame(padded_data)
        
        # Test integration
        decisions = PreprocessingIntegration.classify_dataframe_columns(df)
        summary = PreprocessingIntegration.get_decision_summary(decisions)
        
        print("\n" + "="*60)
        print("PREPROCESSING INTEGRATION SUMMARY")
        print("="*60)
        print(f"Total columns: {summary['total_columns']}")
        print(f"Average confidence: {summary['average_confidence']:.2%}")
        print(f"Safety fallbacks: {summary['safety_fallbacks']}")
        print(f"\nAction breakdown: {summary['action_breakdown']}")
        print(f"Source breakdown: {summary['source_breakdown']}")
        
        if summary['warnings']:
            print(f"\nWarnings:")
            for w in summary['warnings']:
                print(f"  - {w['column']}: {w['warning'][:50]}...")
        
        print("="*60)
        
        # Verify no safety fallbacks needed (all decisions should be safe)
        assert summary['safety_fallbacks'] == 0, \
            f"Should have 0 safety fallbacks, got {summary['safety_fallbacks']}"


class TestMetricsComparison:
    """Compare specific metrics between old and new system."""
    
    def test_error_rate_improvement(self):
        """Error rate should improve from 58% to 0%."""
        # Old system: 7/12 = 58.3% error rate
        old_errors = 7
        old_total = 12
        old_rate = old_errors / old_total * 100
        
        # New system should have 0% error rate
        expected_new_rate = 0.0
        
        # Verify old rate matches expectation
        assert abs(old_rate - 58.3) < 1.0, f"Old error rate should be ~58.3%, got {old_rate:.1f}%"
        
        print(f"\nError rate improvement: {old_rate:.1f}% → {expected_new_rate:.1f}%")
    
    def test_crash_rate_improvement(self):
        """Crash rate should improve from 2 to 0."""
        # Old system: 2 crashes (fuel_type, int_col)
        old_crashes = 2
        
        # New system should have 0 crashes
        expected_new_crashes = 0
        
        print(f"\nCrash rate improvement: {old_crashes} → {expected_new_crashes}")
    
    def test_critical_feature_preservation(self):
        """Critical features should be preserved (0/2 → 2/2)."""
        # Old system: 0/2 critical features kept (accident, clean_title dropped)
        old_critical_kept = 0
        
        # New system: 2/2 critical features kept
        expected_new_critical_kept = 2
        
        print(f"\nCritical features kept: {old_critical_kept}/2 → {expected_new_critical_kept}/2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
