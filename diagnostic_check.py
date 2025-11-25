"""
Diagnostic script to check neural oracle and explanation system status.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.preprocessor import get_preprocessor
from src.neural.oracle import get_neural_oracle
import pandas as pd
import numpy as np

print("=" * 70)
print("AURORA Diagnostic Check")
print("=" * 70)

# Check 1: Neural Oracle Model File
print("\n1. Checking Neural Oracle Model File...")
print("-" * 70)
model_path = Path("models/neural_oracle_v1.pkl")
if model_path.exists():
    print(f"✓ Model file exists: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")
else:
    print(f"✗ Model file NOT found: {model_path}")

# Check 2: Neural Oracle Loading
print("\n2. Checking Neural Oracle Loading...")
print("-" * 70)
try:
    oracle = get_neural_oracle(model_path)
    print("✓ Neural oracle loaded successfully")
    
    # Test prediction
    test_data = pd.Series(np.random.randn(100))
    from src.features.minimal_extractor import get_feature_extractor
    extractor = get_feature_extractor()
    features = extractor.extract(test_data)
    
    pred = oracle.predict(features)
    print(f"✓ Test prediction successful: {pred.action.value} (confidence: {pred.confidence:.2f})")
except Exception as e:
    print(f"✗ Neural oracle loading failed: {e}")
    import traceback
    traceback.print_exc()

# Check 3: Preprocessor Initialization
print("\n3. Checking Preprocessor Initialization...")
print("-" * 70)
try:
    preprocessor = get_preprocessor(
        confidence_threshold=0.75,
        use_neural_oracle=True,
        enable_learning=True
    )
    print("✓ Preprocessor initialized successfully")
    print(f"  use_neural_oracle: {preprocessor.use_neural_oracle}")
    print(f"  confidence_threshold: {preprocessor.confidence_threshold}")
    print(f"  enable_learning: {preprocessor.enable_learning}")
    
    # Check if neural oracle is actually loaded
    if preprocessor.neural_oracle:
        print("✓ Neural oracle is loaded in preprocessor")
    else:
        print("✗ Neural oracle is NOT loaded in preprocessor")
        
except Exception as e:
    print(f"✗ Preprocessor initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Check 4: Test Preprocessing Decision
print("\n4. Testing Preprocessing Decision...")
print("-" * 70)
try:
    # Test with numeric data
    test_column = pd.Series([1, 2, 3, 4, 5, 100, 200, 300])  # Has outliers
    result = preprocessor.preprocess_column(
        column=test_column,
        column_name="test_numeric"
    )
    print(f"✓ Preprocessing decision made")
    print(f"  Action: {result.action.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Source: {result.source}")
    
    if result.source == 'neural':
        print("✓ Neural oracle participated in decision!")
    else:
        print(f"⚠ Decision made by: {result.source} (not neural)")
        
except Exception as e:
    print(f"✗ Preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()

# Check 5: Explanation Endpoint Test
print("\n5. Testing Explanation Generation...")
print("-" * 70)
try:
    # Test explanation generation
    test_column = pd.Series([1.5, 2.3, 3.1, 4.8, 5.2])
    result = preprocessor.preprocess_column(
        column=test_column,
        column_name="test_ratings"
    )
    
    # Generate markdown report (simulating the endpoint)
    markdown_report = f"""
## Decision Analysis for "test_ratings"

### Recommended Action: **{result.action.value.replace('_', ' ').title()}**

**Confidence:** {result.confidence * 100:.1f}%  
**Source:** {result.source}

### Explanation

{result.explanation}
"""
    
    print("✓ Explanation generated successfully")
    print("\nSample explanation:")
    print(markdown_report[:200] + "...")
    
except Exception as e:
    print(f"✗ Explanation generation failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("\nIf neural oracle is not participating:")
print("  1. Check server logs for initialization errors")
print("  2. Ensure numba is installed: pip install numba")
print("  3. Restart backend server")
print("\nIf explanation is not working:")
print("  1. Check browser console for errors")
print("  2. Verify /api/explain/enhanced endpoint is accessible")
print("  3. Clear browser cache and hard refresh")
print("=" * 70)
