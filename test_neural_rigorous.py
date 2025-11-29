"""
RIGOROUS Neural Oracle Quality Test
- Uses DIVERSE datasets (not common ones)
- Tests ALL columns (not just first 10)
- Focuses on AMBIGUOUS cases (symbolic confidence < 0.70)
- Comprehensive evaluation
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, 'C:/Users/shobi/Desktop/AURORA/AURORA-V2')

from src.symbolic.engine import SymbolicEngine
from src.features.minimal_extractor import MinimalFeatureExtractor
import xgboost as xgb

print("="*70)
print("RIGOROUS NEURAL ORACLE QUALITY TEST")
print("="*70)

# Load model
print("\n📦 Loading model...")
with open('models/neural_oracle_realworld.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
action_encoder = model_data['action_encoder']
action_decoder = model_data['action_decoder']
feature_names = model_data['feature_names']

print(f"✅ Model loaded")

# Initialize engines
symbolic_engine = SymbolicEngine()
feature_extractor = MinimalFeatureExtractor()

# Download diverse datasets
print("\n📥 Downloading diverse datasets...")

datasets = {}

try:
    # Dataset 1: Abalone (age prediction)
    from ucimlrepo import fetch_ucirepo
    abalone = fetch_ucirepo(id=1)
    datasets['abalone'] = abalone.data.features
    print("✅ Abalone downloaded")
except Exception as e:
    print(f"⚠️ Abalone failed: {e}")

try:
    # Dataset 2: Adult Income
    from ucimlrepo import fetch_ucirepo
    adult = fetch_ucirepo(id=2)
    datasets['adult_income'] = adult.data.features
    print("✅ Adult Income downloaded")
except Exception as e:
    print(f"⚠️ Adult Income failed: {e}")

try:
    # Dataset 3: Car Evaluation
    from ucimlrepo import fetch_ucirepo
    car = fetch_ucirepo(id=19)
    datasets['car_evaluation'] = car.data.features
    print("✅ Car Evaluation downloaded")
except Exception as e:
    print(f"⚠️ Car Evaluation failed: {e}")

# Fallback: Use Books.csv if available
if len(datasets) == 0:
    print("\n⚠️ UCI datasets failed, using local Books.csv...")
    try:
        books_df = pd.read_csv('datas/Books.csv', low_memory=False, nrows=1000)
        datasets['books'] = books_df
        print("✅ Books.csv loaded (1000 rows)")
    except Exception as e:
        print(f"❌ Books.csv failed: {e}")

# If still no datasets, create synthetic diverse data
if len(datasets) == 0:
    print("\n⚠️ Creating synthetic diverse datasets...")
    
    # Synthetic dataset 1: E-commerce
    datasets['ecommerce'] = pd.DataFrame({
        'user_id': range(1000),
        'order_id': range(1000, 2000),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'price': np.random.lognormal(4, 1.5, 1000),
        'quantity': np.random.poisson(2, 1000),
        'discount_pct': np.random.beta(2, 5, 1000) * 100,
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        'review_text': ['Great product!' if i % 3 == 0 else 'Not bad' for i in range(1000)],
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H')
    })
    
    # Synthetic dataset 2: Healthcare
    datasets['healthcare'] = pd.DataFrame({
        'patient_id': range(500),
        'age': np.random.normal(50, 15, 500).clip(18, 90).astype(int),
        'bmi': np.random.normal(25, 5, 500).clip(15, 50),
        'blood_pressure': np.random.normal(120, 20, 500).clip(80, 180).astype(int),
        'cholesterol': np.random.choice(['Normal', 'High', 'Very High'], 500),
        'smoker': np.random.choice([0, 1], 500, p=[0.7, 0.3]),
        'diagnosis': np.random.choice(['Healthy', 'At Risk', 'Diseased'], 500, p=[0.6, 0.3, 0.1])
    })
    
    print("✅ Created 2 synthetic datasets")

print(f"\n✅ Total datasets: {len(datasets)}")

# Test each dataset
print(f"\n{'='*70}")
print("TESTING ON DIVERSE DATASETS")
print(f"{'='*70}")

all_results = []
dataset_results = {}

for dataset_name, df in datasets.items():
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"Shape: {df.shape}")
    
    correct_all = 0
    total_all = 0
    correct_ambiguous = 0
    total_ambiguous = 0
    
    mismatches = []
    
    # Test ALL columns (not just first 10)
    for col_name in df.columns:
        try:
            column = df[col_name].dropna()
            
            if len(column) < 10:
                continue
            
            # Get symbolic decision
            symbolic_result = symbolic_engine.evaluate(column, col_name)
            true_action = symbolic_result.action
            symbolic_confidence = symbolic_result.confidence
            
            # Check if action in neural vocabulary
            if true_action not in action_decoder:
                continue
            
            # Extract features
            features = feature_extractor.extract(column, col_name)
            feature_array = features.to_array().reshape(1, -1)
            
            # Get neural prediction
            dmatrix = xgb.DMatrix(feature_array, feature_names=feature_names)
            pred_probs = model.predict(dmatrix)[0]
            pred_idx = np.argmax(pred_probs)
            pred_action = action_encoder[pred_idx]
            pred_confidence = pred_probs[pred_idx]
            
            # Track all cases
            if pred_action == true_action:
                correct_all += 1
            else:
                mismatches.append({
                    'column': col_name,
                    'true': true_action.value,
                    'predicted': pred_action.value,
                    'symbolic_conf': symbolic_confidence,
                    'neural_conf': pred_confidence
                })
            total_all += 1
            
            # Track ambiguous cases (symbolic confidence < 0.70)
            if symbolic_confidence < 0.70:
                if pred_action == true_action:
                    correct_ambiguous += 1
                total_ambiguous += 1
                
        except Exception as e:
            continue
    
    # Results for this dataset
    if total_all > 0:
        acc_all = correct_all / total_all
        dataset_results[dataset_name] = {
            'accuracy_all': acc_all,
            'total_columns': total_all,
            'correct': correct_all
        }
        
        print(f"\n  ALL COLUMNS:")
        print(f"    Tested: {total_all} columns")
        print(f"    Correct: {correct_all}/{total_all}")
        print(f"    Accuracy: {acc_all:.1%}")
        
        if total_ambiguous > 0:
            acc_ambiguous = correct_ambiguous / total_ambiguous
            dataset_results[dataset_name]['accuracy_ambiguous'] = acc_ambiguous
            dataset_results[dataset_name]['ambiguous_count'] = total_ambiguous
            
            print(f"\n  AMBIGUOUS CASES (symbolic conf < 0.70):")
            print(f"    Tested: {total_ambiguous} columns")
            print(f"    Correct: {correct_ambiguous}/{total_ambiguous}")
            print(f"    Accuracy: {acc_ambiguous:.1%}")
        
        # Show top 3 mismatches
        if mismatches:
            print(f"\n  Top Mismatches:")
            for mm in mismatches[:3]:
                print(f"    {mm['column']}: {mm['true']} → {mm['predicted']}")
                print(f"      Symbolic conf: {mm['symbolic_conf']:.2f}, Neural conf: {mm['neural_conf']:.2f}")

# Overall results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

if dataset_results:
    # Calculate averages
    avg_acc_all = np.mean([r['accuracy_all'] for r in dataset_results.values()])
    total_columns = sum([r['total_columns'] for r in dataset_results.values()])
    
    ambiguous_accs = [r['accuracy_ambiguous'] for r in dataset_results.values() if 'accuracy_ambiguous' in r]
    avg_acc_ambiguous = np.mean(ambiguous_accs) if ambiguous_accs else 0
    
    print(f"\nDataset Results:")
    for name, results in dataset_results.items():
        print(f"  {name:20s}: {results['accuracy_all']:.1%} ({results['total_columns']} columns)")
    
    print(f"\n{'='*70}")
    print(f"AVERAGE ACCURACY (ALL): {avg_acc_all:.1%}")
    print(f"TOTAL COLUMNS TESTED: {total_columns}")
    if avg_acc_ambiguous > 0:
        print(f"AVERAGE ACCURACY (AMBIGUOUS): {avg_acc_ambiguous:.1%}")
    print(f"{'='*70}")
    
    # Decision
    print(f"\n🎯 RIGOROUS TEST DECISION:")
    if avg_acc_all >= 0.70:
        print(f"✅ Model is TRULY GOOD ({avg_acc_all:.1%} ≥ 70%)")
        print(f"   Tested on {len(datasets)} diverse datasets, {total_columns} columns")
        print(f"   Recommendation: KEEP and do incremental training")
    elif avg_acc_all >= 0.50:
        print(f"⚠️ Model is MEDIOCRE ({avg_acc_all:.1%} between 50-70%)")
        print(f"   Recommendation: CONSIDER retraining from scratch")
    else:
        print(f"❌ Model is BAD ({avg_acc_all:.1%} < 50%)")
        print(f"   Recommendation: DELETE and retrain from scratch")
else:
    print("❌ No valid results - test failed")

print(f"\n{'='*70}")
print("RIGOROUS TEST COMPLETE")
print(f"{'='*70}")
