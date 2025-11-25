"""
Quick test for dataset analyzer to verify it works correctly.
"""

import pandas as pd
import sys
sys.path.insert(0, 'src')

from analysis.dataset_analyzer import DatasetAnalyzer

# Create sample dataset
print("Creating sample dataset...")
df = pd.DataFrame({
    'customer_id': range(1, 101),
    'order_id': range(1001, 1101),
    'product_id': [i % 10 + 1 for i in range(100)],  # 10 products
    'price': [10 + i * 0.5 for i in range(100)],
    'quantity': [1 + i % 5 for i in range(100)],
    'revenue': [(10 + i * 0.5) * (1 + i % 5) for i in range(100)],  # price * quantity
    'category': ['A', 'B', 'C'] * 33 + ['A'],
    'region': ['North', 'South'] * 50
})

print(f"Dataset: {len(df)} rows, {len(df.columns)} columns\n")

# Initialize analyzer
analyzer = DatasetAnalyzer()

# Run analysis
print("Running dataset analysis...")
result = analyzer.analyze(df)

# Print results
print("\n=== ANALYSIS RESULTS ===\n")

print(f"Primary Key Candidates: {result.primary_key_candidates}")
print(f"Composite Key Candidates: {result.composite_key_candidates}")

print(f"\nNumeric Correlations:")
for col, corrs in result.numeric_correlations.items():
    print(f"  {col}:")
    for other_col, corr_value in corrs[:3]:  # Top 3
        print(f"    -> {other_col}: {corr_value:.3f}")

print(f"\nCategorical Associations:")
for col, assocs in result.categorical_associations.items():
    print(f"  {col}:")
    for other_col, cramers_v in assocs[:3]:  # Top 3
        print(f"    -> {other_col}: {cramers_v:.3f}")

print(f"\nSchema Type: {result.schema_type} (confidence: {result.schema_confidence:.2%})")
print(f"Schema Features: {result.schema_features}")

print(f"\nForeign Key Candidates: {result.foreign_key_candidates}")

print("\n✅ Dataset analyzer test complete!")
