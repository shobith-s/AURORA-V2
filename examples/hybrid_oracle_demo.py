#!/usr/bin/env python
"""
Demo script for Hybrid Preprocessing Oracle.

This script demonstrates how to use the hybrid oracle for automatic
preprocessing recommendations on sample data.
"""

import pandas as pd
import numpy as np
from src.neural.hybrid_oracle import HybridPreprocessingOracle
from src.features.meta_extractor import MetaFeatureExtractor


def demo_single_column():
    """Demonstrate analyzing a single column."""
    print("=" * 70)
    print("DEMO 1: Analyzing Single Columns")
    print("=" * 70)
    
    oracle = HybridPreprocessingOracle()
    extractor = MetaFeatureExtractor()
    
    # Test different column types
    test_cases = [
        ('ID Column', pd.Series(range(1000), name='user_id')),
        ('Constant Column', pd.Series([5] * 100, name='constant_val')),
        ('Skewed Revenue', pd.Series([10, 20, 30, 40, 50, 500, 1000, 5000], name='revenue')),
        ('Normal Data', pd.Series(np.random.randn(100), name='feature_x')),
        ('Categorical', pd.Series(['A', 'B', 'C'] * 30 + ['A'], name='category')),
        ('High Missing', pd.Series([1, None, None, None, 5, None, None], name='sparse')),
        ('With Outliers', pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 100, 200], name='values')),
    ]
    
    for name, column in test_cases:
        print(f"\n{name}:")
        print(f"  Data: {column.head(5).tolist()}...")
        
        features = extractor.extract(column, column.name)
        prediction = oracle.predict_column(column, column.name)
        
        print(f"  → Action: {prediction.action.value}")
        print(f"  → Confidence: {prediction.confidence:.2%}")
        print(f"  → Source: {prediction.source}")
        print(f"  → Reason: {prediction.reason}")


def demo_dataframe():
    """Demonstrate analyzing an entire DataFrame."""
    print("\n" + "=" * 70)
    print("DEMO 2: Analyzing Complete DataFrame")
    print("=" * 70)
    
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': range(100),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.lognormal(10, 1, 100),  # Skewed
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'status': [1, 0] * 50,  # Binary
        'constant': [5] * 100,
        'sparse': [1 if i < 10 else None for i in range(100)],
        'outliers': [1, 2, 3] * 33 + [1000],
    })
    
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Analyze all columns
    oracle = HybridPreprocessingOracle()
    results = oracle.predict_dataframe(df)
    
    print("\nRecommendations:")
    print("-" * 70)
    for _, row in results.iterrows():
        print(f"  {row['column_name']:15s} → {row['action']:20s} ({row['confidence']:.0%}) [{row['source']}]")
        print(f"                     Reason: {row['reason']}")


def demo_feature_extraction():
    """Demonstrate feature extraction details."""
    print("\n" + "=" * 70)
    print("DEMO 3: Feature Extraction Details")
    print("=" * 70)
    
    extractor = MetaFeatureExtractor()
    
    # Numeric column
    column = pd.Series([1, 2, 3, 100, 200, 300], name='revenue')
    features = extractor.extract(column, 'revenue')
    
    print("\nExtracted Features (40 total):")
    print("-" * 70)
    
    feature_dict = features.to_dict()
    
    # Group features by category
    categories = {
        'Basic Stats': ['missing_ratio', 'unique_ratio', 'unique_count_norm', 'row_count_norm', 'is_complete'],
        'Type Indicators': ['is_numeric', 'is_bool', 'is_datetime', 'is_object', 'is_categorical'],
        'Numeric Stats': ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis', 'outlier_ratio'],
        'Name Features': ['has_id', 'has_name', 'has_date', 'has_price', 'has_count'],
    }
    
    for category, feature_names in categories.items():
        print(f"\n{category}:")
        for fname in feature_names:
            if fname in feature_dict:
                value = feature_dict[fname]
                print(f"  {fname:25s}: {value:.4f}")


def demo_rule_based():
    """Demonstrate rule-based fallback logic."""
    print("\n" + "=" * 70)
    print("DEMO 4: Rule-Based Fallback Logic")
    print("=" * 70)
    
    oracle = HybridPreprocessingOracle()
    extractor = MetaFeatureExtractor()
    
    print("\nDemonstrating rules (ML models not loaded, using rules only):")
    
    # Test each rule type
    rule_tests = [
        ('Constant', pd.Series([5] * 50)),
        ('ID-like', pd.Series(range(100))),
        ('Skewed', pd.Series([1, 1, 1, 2, 2, 100, 1000])),
        ('Outliers', pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 100, 200])),
        ('High Missing', pd.Series([1, None, None, None, None, None, None, None])),
    ]
    
    for name, column in rule_tests:
        features = extractor.extract(column, name)
        action, confidence, reason = oracle._apply_rules(column, name, features)
        
        print(f"\n{name}:")
        if action:
            print(f"  → Action: {action.value}")
            print(f"  → Confidence: {confidence:.2%}")
            print(f"  → Reason: {reason}")
        else:
            print(f"  → No rule matched (would use ML)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" HYBRID PREPROCESSING ORACLE DEMO")
    print("=" * 70)
    print("\nThis demo shows how the hybrid oracle analyzes columns and")
    print("recommends preprocessing actions using ML + rule-based logic.")
    
    try:
        demo_single_column()
        demo_dataframe()
        demo_feature_extraction()
        demo_rule_based()
        
        print("\n" + "=" * 70)
        print(" DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nFor more details, see: docs/HYBRID_ORACLE.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
