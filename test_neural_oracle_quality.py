"""
Test existing neural oracle model to determine quality.
Decision: Keep (>75%), Consider retraining (60-75%), or Delete (<60%)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes

# Add project root to path
sys.path.insert(0, 'C:/Users/shobi/Desktop/AURORA/AURORA-V2')

from src.symbolic.engine import SymbolicEngine
from src.features.minimal_extractor import MinimalFeatureExtractor
import xgboost as xgb

print("="*70)
print("NEURAL ORACLE MODEL QUALITY TEST")
print("="*70)

# Load existing model
print("\n📦 Loading existing model...")
with open('models/neural_oracle_realworld.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
action_encoder = model_data['action_encoder']
action_decoder = model_data['action_decoder']
feature_names = model_data['feature_names']

print(f"✅ Model loaded: {model.num_boosted_rounds()} boosters, {len(action_encoder)} actions")

# Initialize symbolic engine and feature extractor
symbolic_engine = SymbolicEngine()
feature_extractor = MinimalFeatureExtractor()

# Test datasets (sklearn built-in)
test_datasets = {
    'iris': load_iris(as_frame=True)['frame'],
    'wine': load_wine(as_frame=True)['frame'],
    'breast_cancer': load_breast_cancer(as_frame=True)['frame'],
    'diabetes': load_diabetes(as_frame=True)['frame']
}

print(f"\n🧪 Testing on {len(test_datasets)} datasets...")

# Track results
all_results = []
dataset_accuracies = {}

for dataset_name, df in test_datasets.items():
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")
    print(f"Shape: {df.shape}")
    
    correct = 0
    total = 0
    mismatches = []
    
    # Test each column
    for col_name in df.columns[:10]:  # Limit to first 10 columns
        column = df[col_name]
        
        # Get symbolic decision (ground truth)
        try:
            symbolic_result = symbolic_engine.evaluate(column, col_name)
            true_action = symbolic_result.action
            
            # Check if action is in neural oracle's vocabulary
            if true_action not in action_decoder:
                # Neural oracle doesn't know this action, skip
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
            
            # Compare
            if pred_action == true_action:
                correct += 1
            else:
                mismatches.append({
                    'column': col_name,
                    'true': true_action.value,
                    'predicted': pred_action.value,
                    'confidence': pred_confidence
                })
            
            total += 1
            
        except Exception as e:
            print(f"  ⚠️ Error on {col_name}: {e}")
            continue
    
    # Calculate accuracy
    if total > 0:
        accuracy = correct / total
        dataset_accuracies[dataset_name] = accuracy
        
        print(f"\n  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.1%}")
        
        # Show mismatches
        if mismatches and len(mismatches) <= 3:
            print(f"\n  Mismatches:")
            for mm in mismatches:
                print(f"    {mm['column']}: {mm['true']} → {mm['predicted']} (conf: {mm['confidence']:.2f})")
    else:
        print(f"  ⚠️ No testable columns (actions not in neural vocabulary)")

# Overall results
print(f"\n{'='*70}")
print("OVERALL RESULTS")
print(f"{'='*70}")

if dataset_accuracies:
    avg_accuracy = np.mean(list(dataset_accuracies.values()))
    
    print(f"\nDataset Accuracies:")
    for name, acc in dataset_accuracies.items():
        print(f"  {name:20s}: {acc:.1%}")
    
    print(f"\n{'='*70}")
    print(f"AVERAGE ACCURACY: {avg_accuracy:.1%}")
    print(f"{'='*70}")
    
    # Decision
    print(f"\n🎯 DECISION:")
    if avg_accuracy >= 0.75:
        print(f"✅ Model is GOOD ({avg_accuracy:.1%} ≥ 75%)")
        print(f"   Recommendation: KEEP and do incremental training")
        print(f"   Action: Add new datasets without deleting existing model")
    elif avg_accuracy >= 0.60:
        print(f"⚠️ Model is OKAY ({avg_accuracy:.1%} between 60-75%)")
        print(f"   Recommendation: CONSIDER retraining from scratch")
        print(f"   Action: User decision - keep or retrain")
    else:
        print(f"❌ Model is BAD ({avg_accuracy:.1%} < 60%)")
        print(f"   Recommendation: DELETE and retrain from scratch")
        print(f"   Action: Backup, delete, and train new model")
else:
    print("❌ No valid test results - model vocabulary mismatch")
    print("   Recommendation: DELETE and retrain from scratch")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
