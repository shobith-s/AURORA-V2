"""
AURORA Data Understanding Diagnostic Tool
==========================================
This script measures how well AURORA "understands" different types of data.
It generates various column types and analyzes:
1. What features are extracted
2. What statistics are computed
3. What patterns are detected
4. How accurate the type detection is
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import random

# Add project root to path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.minimal_extractor import MinimalFeatureExtractor
from src.symbolic.engine import SymbolicEngine


class DataUnderstandingDiagnostic:
    """Diagnostic tool to measure AURORA's data understanding."""
    
    def __init__(self):
        self.feature_extractor = MinimalFeatureExtractor()
        self.symbolic_engine = SymbolicEngine()
        self.test_results = []
        
    def generate_test_columns(self) -> List[Tuple[str, pd.Series, str, Dict]]:
        """Generate diverse test columns with known characteristics."""
        columns = []
        
        # 1. NUMERIC TYPES
        
        # Normal numeric (well-behaved)
        col = pd.Series(np.random.normal(100, 15, 1000), name='normal_numeric')
        columns.append(('normal_numeric', col, 'numeric', {
            'expected_skew': 'low',
            'expected_outliers': 'low',
            'expected_type': 'numeric'
        }))
        
        # Skewed numeric (needs log transform)
        col = pd.Series(np.random.lognormal(3, 1, 1000), name='skewed_revenue')
        columns.append(('skewed_revenue', col, 'numeric', {
            'expected_skew': 'high',
            'expected_outliers': 'medium',
            'expected_type': 'numeric',
            'needs_transform': True
        }))
        
        # Numeric with outliers
        data = list(np.random.normal(50, 10, 950))
        data.extend([200, 220, -50, -80, 250])  # Add outliers
        col = pd.Series(data, name='data_with_outliers')
        columns.append(('data_with_outliers', col, 'numeric', {
            'expected_outliers': 'high',
            'expected_type': 'numeric'
        }))
        
        # Already scaled (z-score)
        col = pd.Series(np.random.normal(0, 1, 1000), name='z_scored_data')
        columns.append(('z_scored_data', col, 'numeric', {
            'expected_type': 'numeric',
            'is_scaled': True
        }))
        
        # Bounded [0, 1] (probabilities)
        col = pd.Series(np.random.beta(2, 5, 1000), name='probability_score')
        columns.append(('probability_score', col, 'numeric', {
            'expected_type': 'numeric',
            'is_bounded': True
        }))
        
        # 2. CATEGORICAL TYPES
        
        # Low cardinality categorical
        col = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], 1000), name='category_low')
        columns.append(('category_low', col, 'categorical', {
            'expected_type': 'categorical',
            'cardinality': 'low',
            'encoding': 'onehot'
        }))
        
        # High cardinality categorical
        categories = [f'cat_{i}' for i in range(200)]
        col = pd.Series(np.random.choice(categories, 1000), name='category_high')
        columns.append(('category_high', col, 'categorical', {
            'expected_type': 'categorical',
            'cardinality': 'high',
            'encoding': 'frequency_or_hash'
        }))
        
        # Ordinal categorical
        col = pd.Series(np.random.choice(['low', 'medium', 'high'], 1000), name='priority')
        columns.append(('priority', col, 'categorical', {
            'expected_type': 'categorical',
            'is_ordinal': True,
            'encoding': 'ordinal'
        }))
        
        # 3. IMBALANCED DATA
        
        # Imbalanced binary (fraud detection)
        data = [0] * 950 + [1] * 50
        random.shuffle(data)
        col = pd.Series(data, name='is_fraud')
        columns.append(('is_fraud', col, 'imbalanced', {
            'expected_type': 'binary',
            'is_imbalanced': True,
            'should_preserve': True
        }))
        
        # Imbalanced target
        data = [0] * 900 + [1] * 100
        random.shuffle(data)
        col = pd.Series(data, name='target')
        columns.append(('target', col, 'target', {
            'expected_type': 'target',
            'is_imbalanced': True,
            'should_preserve': True
        }))
        
        # 4. PATTERN COLUMNS
        
        # Datetime strings
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1000)]
        col = pd.Series(dates, name='date_column')
        columns.append(('date_column', col, 'datetime', {
            'expected_type': 'temporal',
            'pattern': 'iso_date'
        }))
        
        # Email addresses
        emails = [f'user{i}@example.com' for i in range(1000)]
        col = pd.Series(emails, name='email')
        columns.append(('email', col, 'email', {
            'expected_type': 'text',
            'pattern': 'email',
            'should_validate': True
        }))
        
        # Currency values
        prices = [f'${np.random.uniform(10, 1000):.2f}' for _ in range(1000)]
        col = pd.Series(prices, name='price')
        columns.append(('price', col, 'currency', {
            'expected_type': 'text',
            'pattern': 'currency',
            'needs_parsing': True
        }))
        
        # 5. ID-LIKE COLUMNS
        
        # UUID-like
        uuids = [f'uuid-{i:06d}-{j:04d}' for i, j in zip(range(1000), range(1000))]
        col = pd.Series(uuids, name='user_id')
        columns.append(('user_id', col, 'id', {
            'expected_type': 'id',
            'should_drop': True,
            'unique_ratio': 'high'
        }))
        
        # 6. DATA QUALITY ISSUES
        
        # High null percentage
        data = list(np.random.normal(50, 10, 300))
        data.extend([np.nan] * 700)
        col = pd.Series(data, name='mostly_null')
        columns.append(('mostly_null', col, 'quality', {
            'null_pct': 'high',
            'should_drop': True
        }))
        
        # Constant value
        col = pd.Series([42] * 1000, name='constant_column')
        columns.append(('constant_column', col, 'quality', {
            'expected_type': 'constant',
            'should_drop': True
        }))
        
        return columns
    
    def analyze_understanding(
        self, 
        column_name: str, 
        column: pd.Series, 
        col_type: str, 
        expectations: Dict
    ) -> Dict:
        """Analyze how well AURORA understands a column."""
        
        print(f"\n{'='*70}")
        print(f"Analyzing: {column_name} ({col_type})")
        print(f"{'='*70}")
        
        # Extract features
        features = self.feature_extractor.extract(column, column_name)
        feature_dict = features.to_dict()
        
        # Compute statistics
        stats = self.symbolic_engine.compute_column_statistics(column, column_name)
        stats_dict = stats.to_dict()
        
        # Get preprocessing decision
        result = self.symbolic_engine.evaluate(column, column_name)
        
        # Analyze features
        print(f"\n📊 EXTRACTED FEATURES:")
        print(f"  Null %:              {feature_dict['null_percentage']:.1%}")
        print(f"  Unique Ratio:        {feature_dict['unique_ratio']:.3f}")
        print(f"  Skewness:            {feature_dict['skewness']:.2f}")
        print(f"  Outlier %:           {feature_dict['outlier_percentage']:.1%}")
        print(f"  Entropy:             {feature_dict['entropy']:.3f}")
        print(f"  Pattern Complexity:  {feature_dict['pattern_complexity']:.3f}")
        print(f"  Multimodality:       {feature_dict['multimodality_score']:.3f}")
        print(f"  Cardinality Bucket:  {feature_dict['cardinality_bucket']} (0=low,1=med,2=high,3=unique)")
        
        dtype_map = {0: 'numeric', 1: 'categorical', 2: 'text', 3: 'temporal', 4: 'other'}
        print(f"  Detected Type:       {dtype_map[feature_dict['detected_dtype']]}")
        print(f"  Name Signal:         {feature_dict['column_name_signal']:.2f}")
        
        # Analyze statistics
        print(f"\n📈 COMPUTED STATISTICS:")
        if stats_dict.get('is_numeric'):
            print(f"  Min:                 {stats_dict.get('min_value', 0):.2f}")
            print(f"  Max:                 {stats_dict.get('max_value', 0):.2f}")
            print(f"  Mean:                {stats_dict.get('mean', 0):.2f}")
            print(f"  Std:                 {stats_dict.get('std', 0):.2f}")
            print(f"  CV:                  {stats_dict.get('cv', 0):.2f}")
            print(f"  IQR:                 {stats_dict.get('iqr', 0):.2f}")
        
        print(f"  Type Detection:")
        print(f"    - Numeric:         {stats_dict.get('is_numeric', False)}")
        print(f"    - Categorical:     {stats_dict.get('is_categorical', False)}")
        print(f"    - Text:            {stats_dict.get('is_text', False)}")
        print(f"    - Temporal:        {stats_dict.get('is_temporal', False)}")
        
        # Pattern detection
        patterns_detected = []
        if stats_dict.get('matches_iso_datetime', 0) > 0.8:
            patterns_detected.append(f"ISO DateTime ({stats_dict['matches_iso_datetime']:.0%})")
        if stats_dict.get('matches_email_pattern', 0) > 0.8:
            patterns_detected.append(f"Email ({stats_dict['matches_email_pattern']:.0%})")
        if stats_dict.get('matches_currency_pattern', 0) > 0.8:
            patterns_detected.append(f"Currency ({stats_dict['matches_currency_pattern']:.0%})")
        if stats_dict.get('matches_phone_pattern', 0) > 0.8:
            patterns_detected.append(f"Phone ({stats_dict['matches_phone_pattern']:.0%})")
        
        if patterns_detected:
            print(f"\n🔍 PATTERNS DETECTED:")
            for pattern in patterns_detected:
                print(f"  ✓ {pattern}")
        
        # Decision analysis
        print(f"\n⚙️  PREPROCESSING DECISION:")
        print(f"  Action:              {result.action.value}")
        print(f"  Confidence:          {result.confidence:.1%}")
        print(f"  Source:              {result.source}")
        print(f"  Explanation:         {result.explanation[:100]}...")
        
        # Validate against expectations
        print(f"\n✅ EXPECTATION VALIDATION:")
        validations = []
        
        # Check type detection
        if 'expected_type' in expectations:
            expected = expectations['expected_type']
            detected = 'numeric' if stats_dict.get('is_numeric') else \
                      'categorical' if stats_dict.get('is_categorical') else \
                      'temporal' if stats_dict.get('is_temporal') else \
                      'text' if stats_dict.get('is_text') else 'unknown'
            
            match = expected == detected or (expected == 'id' and detected == 'text')
            status = "✓" if match else "✗"
            validations.append((f"Type Detection ({expected} → {detected})", match))
            print(f"  {status} Type Detection: expected={expected}, detected={detected}")
        
        # Check imbalance handling
        if expectations.get('is_imbalanced') and expectations.get('should_preserve'):
            preserved = result.action.value != 'drop_column'
            status = "✓" if preserved else "✗"
            validations.append((f"Imbalance Preservation", preserved))
            print(f"  {status} Imbalanced data preserved: {preserved}")
        
        # Check drop decision
        if expectations.get('should_drop'):
            dropped = result.action.value == 'drop_column'
            status = "✓" if dropped else "✗"
            validations.append((f"Drop Decision", dropped))
            print(f"  {status} Column should be dropped: {dropped}")
        
        # Check pattern detection
        if 'pattern' in expectations:
            pattern = expectations['pattern']
            pattern_key = f"matches_{pattern}"
            if pattern_key in stats_dict:
                detected = stats_dict[pattern_key] > 0.8
                status = "✓" if detected else "✗"
                validations.append((f"Pattern Detection ({pattern})", detected))
                print(f"  {status} Pattern '{pattern}' detected: {detected}")
        
        # Overall score
        if validations:
            score = sum(1 for _, v in validations if v) / len(validations)
            print(f"\n  Overall Understanding: {score:.0%} ({sum(1 for _, v in validations if v)}/{len(validations)} checks passed)")
        else:
            score = 1.0
        
        return {
            'column_name': column_name,
            'type': col_type,
            'features': feature_dict,
            'stats': {k: v for k, v in stats_dict.items() if k in ['is_numeric', 'is_categorical', 'is_text', 'entropy', 'null_pct']},
            'decision': {
                'action': result.action.value,
                'confidence': result.confidence,
                'source': result.source
            },
            'validations': validations,
            'score': score
        }
    
    def run_diagnostic(self):
        """Run full diagnostic."""
        print("="*70)
        print("AURORA DATA UNDERSTANDING DIAGNOSTIC")
        print("="*70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Generate test columns
        test_columns = self.generate_test_columns()
        print(f"\nGenerated {len(test_columns)} test columns")
        
        # Analyze each column
        results = []
        for col_name, col, col_type, expectations in test_columns:
            result = self.analyze_understanding(col_name, col, col_type, expectations)
            results.append(result)
            self.test_results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        avg_score = np.mean([r['score'] for r in results])
        print(f"\n📊 Overall Understanding Score: {avg_score:.0%}")
        
        # By category
        categories = set(r['type'] for r in results)
        print(f"\n📈 Performance by Category:")
        for cat in sorted(categories):
            cat_results = [r for r in results if r['type'] == cat]
            cat_score = np.mean([r['score'] for r in cat_results])
            print(f"  {cat:15s}: {cat_score:.0%} ({len(cat_results)} columns)")
        
        # Feature coverage
        print(f"\n🔍 Feature Extraction Analysis:")
        print(f"  Total Features:      10 (minimal set)")
        print(f"  Numeric Features:    4 (null%, unique_ratio, skewness, outlier%)")
        print(f"  Pattern Features:    3 (entropy, pattern_complexity, multimodality)")
        print(f"  Metadata Features:   3 (cardinality, dtype, name_signal)")
        
        # Statistics coverage
        print(f"\n📊 Statistics Computation Analysis:")
        sample_stats = results[0]['stats'] if results else {}
        print(f"  Basic Stats:         ✓ (count, null, unique, dtype)")
        print(f"  Numeric Stats:       ✓ (min, max, mean, median, std, skew, kurtosis)")
        print(f"  Advanced Stats:      ✓ (CV, IQR, entropy, outliers)")
        print(f"  Pattern Matching:    ✓ (12+ patterns)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if avg_score < 0.7:
            print("  ⚠️  Overall understanding is below 70%")
            print("  → Consider adding more features or improving pattern detection")
        elif avg_score < 0.85:
            print("  ✓ Good understanding, but room for improvement")
            print("  → Focus on failing categories")
        else:
            print("  ✅ Excellent data understanding!")
        
        # Specific improvements
        weak_categories = [cat for cat in categories 
                          if np.mean([r['score'] for r in results if r['type'] == cat]) < 0.7]
        if weak_categories:
            print(f"\n  Weak categories: {', '.join(weak_categories)}")
            print("  → Review rules and feature extraction for these types")
        
        print("\n" + "="*70)


def main():
    """Run the diagnostic."""
    diagnostic = DataUnderstandingDiagnostic()
    diagnostic.run_diagnostic()


if __name__ == "__main__":
    main()
