"""
Quick comparison script: Minimal (10) vs Enhanced (30) features.
Shows side-by-side what each extractor captures.
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from src.features.minimal_extractor import MinimalFeatureExtractor
from src.features.enhanced_extractor import EnhancedFeatureExtractor

print("="*70)
print("MINIMAL (10) vs ENHANCED (30) FEATURE EXTRACTION COMPARISON")
print("="*70)

minimal_ext = MinimalFeatureExtractor()
enhanced_ext = EnhancedFeatureExtractor()

# Test columns
test_cases = [
    ("customer_id", pd.Series(range(1000), name='customer_id')),
    ("total_revenue", pd.Series(np.random.lognormal(5, 2, 1000), name='total_revenue')),
    ("email", pd.Series([f'user{i}@example.com' for i in range(100)], name='email')),
    ("target", pd.Series([0]*90 + [1]*10, name='target')),
]

for col_name, column in test_cases:
    print(f"\n{'='*70}")
    print(f"Column: {col_name}")
    print(f"{'='*70}")
    
    # Minimal features
    minimal = minimal_ext.extract(column, col_name)
    print("\n📦 MINIMAL (10 features):")
    print(f"  null_percentage:      {minimal.null_percentage:.3f}")
    print(f"  unique_ratio:         {minimal.unique_ratio:.3f}")
    print(f"  entropy:              {minimal.entropy:.3f}")
    print(f"  cardinality_bucket:   {minimal.cardinality_bucket}")
    print(f"  detected_dtype:       {minimal.detected_dtype} (0=num,1=cat,2=text,3=temp)")
    print(f"  column_name_signal:   {minimal.column_name_signal:.2f}")
    
    # Enhanced features
    enhanced = enhanced_ext.extract_enhanced(column, col_name)
    print("\n🚀 ENHANCED (30 features) - NEW INSIGHTS:")
    
    # Semantic
    sem_types = ['ID', 'Metric', 'Attribute', 'Temporal', 'Target', 'Other']
    domains = ['Finance', 'Customer', 'Product', 'Temporal', 'Geo', 'Other']
    print(f"  semantic_type:        {enhanced.semantic_type} ({sem_types[enhanced.semantic_type]})")
    print(f"  domain_category:      {enhanced.domain_category} ({domains[enhanced.domain_category]})")
    print(f"  is_pk_candidate:      {enhanced.is_primary_key_candidate:.2f}")
    print(f"  is_fk_candidate:      {enhanced.is_foreign_key_candidate:.2f}")
    
    # Distribution (if numeric)
    if enhanced.detected_dtype == 0:
        print(f"  quantile_25:          {enhanced.quantile_25:.2f}")
        print(f"  quantile_75:          {enhanced.quantile_75:.2f}")
        print(f"  coefficient_dispersion: {enhanced.coefficient_dispersion:.2f}")
    
    # Text (if text)
    if enhanced.detected_dtype in [1, 2]:
        print(f"  avg_string_length:    {enhanced.avg_string_length:.1f}")
        print(f"  char_diversity:       {enhanced.char_diversity:.3f}")
        print(f"  numeric_string_ratio: {enhanced.numeric_string_ratio:.1%}")
    
    # Advanced patterns
    if enhanced.has_embedded_nulls > 0:
        print(f"  has_embedded_nulls:   {enhanced.has_embedded_nulls:.1%}")
    if enhanced.email_domain_diversity > 0:
        print(f"  email_domain_diversity: {enhanced.email_domain_diversity:.2f}")

print("\n" + "="*70)
print("✅ COMPARISON COMPLETE")
print("="*70)
print("\n💡 Key Insights:")
print("  - Minimal (10): Fast, basic patterns, trained neural oracle compatible")
print("  - Enhanced (30): Deep understanding, semantic awareness, future-ready")
print("  - Backward compatible: Enhanced can convert to minimal anytime")
