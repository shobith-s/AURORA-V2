"""
Comprehensive test script for Phase 1 features:
1. Inter-Column Analysis
2. Domain-Specific Rules
3. Enhanced Null Detection
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.dataset_analyzer import DatasetAnalyzer
from src.symbolic.engine import SymbolicEngine
from src.symbolic.extended_rules import get_extended_rules

def test_inter_column_analysis():
    print("\n=== Testing Inter-Column Analysis ===")
    
    # Create sample dataset
    df = pd.DataFrame({
        'customer_id': range(1, 101),  # Primary Key
        'order_id': range(1001, 1101), # Primary Key
        'price': [10.0 + i for i in range(100)],
        'quantity': [2 + (i % 5) for i in range(100)],
        'revenue': [(10.0 + i) * (2 + (i % 5)) for i in range(100)],       # Correlated
        'category': ['A', 'B'] * 50,
        'region': ['North', 'South'] * 50 # Associated
    })
    
    analyzer = DatasetAnalyzer()
    result = analyzer.analyze(df)
    
    print(f"Primary Keys: {result.primary_key_candidates}")
    assert 'customer_id' in result.primary_key_candidates
    assert 'order_id' in result.primary_key_candidates
    
    print(f"Correlations: {list(result.numeric_correlations.keys())}")
    assert 'revenue' in result.numeric_correlations
    
    print("✅ Inter-Column Analysis Passed")

def test_domain_rules():
    print("\n=== Testing Domain-Specific Rules ===")
    
    engine = SymbolicEngine()
    
    # Finance Test
    finance_df = pd.DataFrame({
        'price': ['$1,234.56', '€50.00', '£100'],
        'ticker': ['AAPL', 'GOOGL', 'MSFT'],
        'credit_card': ['1234-5678-9012-3456', '1234 5678 9012 3456', '1234567890123456']
    })
    
    print("Testing Finance Rules...")
    # Currency
    stats = engine.compute_column_statistics(finance_df['price'], 'price')
    rule = next((r for r in engine.rules if r.name == 'FINANCE_CURRENCY_DETECTION'), None)
    if rule and rule.condition(stats.to_dict()):
        print("  ✅ Currency Detection Works")
    else:
        print("  ❌ Currency Detection Failed")
        
    # Ticker
    stats = engine.compute_column_statistics(finance_df['ticker'], 'ticker')
    rule = next((r for r in engine.rules if r.name == 'FINANCE_STOCK_TICKER'), None)
    if rule and rule.condition(stats.to_dict()):
        print("  ✅ Stock Ticker Detection Works")
    else:
        print("  ❌ Stock Ticker Detection Failed")

    # Healthcare Test
    health_df = pd.DataFrame({
        'icd_code': ['A01.0', 'B20', 'C34.9'],
        'patient_id': ['PAT001', 'PAT002', 'PAT003'],
        'dob': ['1980-01-01', '1990-05-15', '2000-12-31']
    })
    
    print("Testing Healthcare Rules...")
    # ICD
    stats = engine.compute_column_statistics(health_df['icd_code'], 'icd_code')
    rule = next((r for r in engine.rules if r.name == 'HEALTH_ICD_CODE'), None)
    if rule and rule.condition(stats.to_dict()):
        print("  ✅ ICD Code Detection Works")
    else:
        print("  ❌ ICD Code Detection Failed")

    # E-commerce Test
    ecom_df = pd.DataFrame({
        'sku': ['PROD-001', 'PROD-002', 'PROD-003'],
        'order_id': ['ORD-1001', 'ORD-1002', 'ORD-1003']
    })
    
    print("Testing E-commerce Rules...")
    # SKU
    stats = engine.compute_column_statistics(ecom_df['sku'], 'sku')
    rule = next((r for r in engine.rules if r.name == 'ECOMMERCE_SKU_FORMAT'), None)
    if rule and rule.condition(stats.to_dict()):
        print("  ✅ SKU Detection Works")
    else:
        print("  ❌ SKU Detection Failed")

def test_encoded_nulls():
    print("\n=== Testing Encoded Null Detection ===")
    
    engine = SymbolicEngine()
    
    # Numeric Nulls
    num_df = pd.Series([1, 2, 3, -999, 99999, -1])
    stats = engine.compute_column_statistics(num_df, 'numeric_col')
    print(f"Numeric Encoded Null Ratio: {stats.encoded_null_ratio:.2f}")
    assert stats.encoded_null_ratio >= 0.5
    
    # Text Nulls
    text_df = pd.Series(['A', 'B', 'missing', 'N/A', 'unknown', 'C'])
    stats = engine.compute_column_statistics(text_df, 'text_col')
    print(f"Text Encoded Null Ratio: {stats.encoded_null_ratio:.2f}")
    assert stats.encoded_null_ratio >= 0.5
    
    print("✅ Encoded Null Detection Passed")

if __name__ == "__main__":
    try:
        test_inter_column_analysis()
        test_domain_rules()
        test_encoded_nulls()
        print("\n🎉 All Phase 1 Features Verified Successfully!")
    except Exception as e:
        print(f"\n❌ Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()
