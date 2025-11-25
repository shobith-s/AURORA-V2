import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.preprocessor import IntelligentPreprocessor
from src.symbolic.engine import SymbolicEngine

def test_phase2_integration():
    """Test integration of DatasetAnalyzer into Preprocessor."""
    print("\n=== Testing Phase 2 Integration ===")
    
    # Create sample dataset with known relationships
    df = pd.DataFrame({
        'user_id': range(1, 101),  # Primary Key
        'email': [f"user{i}@example.com" for i in range(1, 101)],  # Unique
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.randint(30000, 150000, 100),
        'department': np.random.choice(['HR', 'IT', 'Sales'], 100)
    })
    
    # Add correlation
    df['bonus'] = df['salary'] * 0.1 + np.random.normal(0, 1000, 100)  # Correlated with salary
    
    # Initialize Preprocessor
    preprocessor = IntelligentPreprocessor(confidence_threshold=0.8)
    
    # Run batch preprocessing (triggers DatasetAnalyzer)
    print("Running batch preprocessing...")
    results = preprocessor.preprocess_dataframe(df)
    
    # Verify Primary Key Detection
    print("\nVerifying Primary Key Context:")
    user_id_result = results['user_id']
    # We need to check if the context was passed correctly. 
    # Since we can't easily inspect the internal state of the engine during execution without mocking,
    # we'll check if the result context contains the metadata we expect.
    
    # Note: The result.context comes from stats.to_dict() which we updated to include is_primary_key
    if user_id_result.context.get('is_primary_key'):
        print("✅ 'user_id' correctly identified as Primary Key in context")
    else:
        print("❌ 'user_id' NOT identified as Primary Key in context")
        
    # Verify Correlation
    print("\nVerifying Correlation Context:")
    # We didn't set a target column, so correlation_with_target should be 0.0
    # Let's try with a target column
    
    print("Running batch preprocessing with target='salary'...")
    results_with_target = preprocessor.preprocess_dataframe(df, target_column='salary')
    
    bonus_result = results_with_target['bonus']
    correlation = bonus_result.context.get('correlation_with_target', 0.0)
    print(f"Bonus correlation with target (salary): {correlation:.4f}")
    
    if abs(correlation) > 0.5:
        print("✅ Correlation correctly detected and passed to context")
    else:
        print("❌ Correlation NOT detected or passed to context")

    print("\n=== Phase 2 Integration Test Complete ===")

if __name__ == "__main__":
    test_phase2_integration()
