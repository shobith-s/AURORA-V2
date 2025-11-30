#!/usr/bin/env python3
"""
Test Evaluation Pipeline for AURORA V2

Quick sanity checks before running full evaluation.
Tests all major components to ensure they work correctly.

Usage:
    python scripts/test_evaluation_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loading():
    """Test 1: Check config file exists and is valid."""
    print("\n[TEST 1] Configuration Loading")
    print("-" * 40)
    
    config_path = Path(__file__).parent.parent / 'configs' / 'evaluation_config.yaml'
    
    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['evaluation', 'statistics']
        for key in required_keys:
            if key not in config:
                print(f"  ❌ Missing required key: {key}")
                return False
        
        print(f"  ✅ Config loaded successfully")
        print(f"     Datasets: {config['evaluation'].get('num_datasets', 'N/A')}")
        print(f"     Variants: {config['evaluation'].get('variants', [])}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load config: {e}")
        return False


def test_preprocessor_loading():
    """Test 2: Check AURORA preprocessor loads."""
    print("\n[TEST 2] AURORA Preprocessor")
    print("-" * 40)
    
    try:
        from src.core.preprocessor import IntelligentPreprocessor
        preprocessor = IntelligentPreprocessor()
        print(f"  ✅ Preprocessor initialized")
        return True
    except Exception as e:
        print(f"  ❌ Failed to load preprocessor: {e}")
        return False


def test_single_column_prediction():
    """Test 3: Test prediction on a single column."""
    print("\n[TEST 3] Single Column Prediction")
    print("-" * 40)
    
    try:
        from src.core.preprocessor import IntelligentPreprocessor
        preprocessor = IntelligentPreprocessor()
        
        # Create test column
        test_data = pd.Series([1.0, 2.5, np.nan, 4.0, 5.5, 6.0, 7.5, np.nan, 9.0, 10.0])
        
        start_time = time.time()
        result = preprocessor.preprocess_column(test_data, "test_column")
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"  ✅ Prediction successful")
        print(f"     Action: {result.action.value}")
        print(f"     Confidence: {result.confidence:.2f}")
        print(f"     Source: {result.source}")
        print(f"     Latency: {elapsed_ms:.2f}ms")
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        return False


def test_dataset_download():
    """Test 4: Test downloading a single dataset."""
    print("\n[TEST 4] Dataset Download (Titanic)")
    print("-" * 40)
    
    try:
        from sklearn.datasets import fetch_openml
        
        start_time = time.time()
        data = fetch_openml(data_id=40945, as_frame=True, parser='auto')
        elapsed = time.time() - start_time
        
        if hasattr(data, 'frame') and data.frame is not None:
            df = data.frame
        else:
            df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
        
        print(f"  ✅ Dataset downloaded")
        print(f"     Rows: {len(df)}")
        print(f"     Columns: {len(df.columns)}")
        print(f"     Time: {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def test_ablation_variant():
    """Test 5: Test one ablation variant."""
    print("\n[TEST 5] Ablation Variant (Random)")
    print("-" * 40)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        # Create simple test data
        np.random.seed(42)
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = np.random.randint(0, 2, 100)
        
        # Simple preprocessing (random baseline)
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col])
            else:
                scaler = StandardScaler()
                X_processed[col] = scaler.fit_transform(X_processed[[col]])
        
        # Cross-validation
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(model, X_processed, y, cv=3)
        
        print(f"  ✅ Ablation variant test passed")
        print(f"     Mean score: {np.mean(scores):.3f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Ablation test failed: {e}")
        return False


def test_figure_generation():
    """Test 6: Test figure generation."""
    print("\n[TEST 6] Figure Generation")
    print("-" * 40)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['A', 'B', 'C'], [0.8, 0.85, 0.9])
        ax.set_title('Test Figure')
        
        # Save to temp location
        test_path = Path('/tmp/aurora_test_figure.png')
        plt.savefig(test_path, dpi=100)
        plt.close()
        
        if test_path.exists():
            print(f"  ✅ Figure generation successful")
            test_path.unlink()  # Clean up
            return True
        else:
            print(f"  ❌ Figure file not created")
            return False
            
    except ImportError:
        print(f"  ⚠️ Matplotlib not available (optional)")
        return True  # Not a failure
    except Exception as e:
        print(f"  ❌ Figure generation failed: {e}")
        return False


def test_dependencies():
    """Test 7: Check all required dependencies."""
    print("\n[TEST 7] Dependencies Check")
    print("-" * 40)
    
    required = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', 'stats'),
        ('sklearn', 'sklearn'),
    ]
    
    optional = [
        ('tqdm', 'tqdm'),
        ('yaml', 'yaml'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('reportlab', 'reportlab'),
    ]
    
    all_ok = True
    
    print("  Required:")
    for name, alias in required:
        try:
            __import__(name)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ❌ {name} - MISSING")
            all_ok = False
    
    print("  Optional:")
    for name, alias in optional:
        try:
            __import__(name)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ⚠️ {name} - not installed")
    
    return all_ok


def test_results_directory():
    """Test 8: Check results directory can be created."""
    print("\n[TEST 8] Results Directory")
    print("-" * 40)
    
    try:
        results_dir = Path(__file__).parent.parent / 'results'
        figures_dir = results_dir / 'figures'
        checkpoints_dir = results_dir / 'checkpoints'
        
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  ✅ Results directory created: {results_dir}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create directories: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("AURORA V2 - Evaluation Pipeline Tests")
    print("=" * 70)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Preprocessor", test_preprocessor_loading),
        ("Single Prediction", test_single_column_prediction),
        ("Dataset Download", test_dataset_download),
        ("Ablation Variant", test_ablation_variant),
        ("Figure Generation", test_figure_generation),
        ("Dependencies", test_dependencies),
        ("Results Directory", test_results_directory),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Safe to run full evaluation.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review before running evaluation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
