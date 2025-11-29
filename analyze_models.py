"""
Analyze existing neural oracle models to determine training history.
"""
import pickle
import os
import sys

os.chdir('C:/Users/shobi/Desktop/AURORA/AURORA-V2')

print("="*70)
print("NEURAL ORACLE MODEL ANALYSIS")
print("="*70)

# Load both models
print("\n📦 Loading models...")
with open('models/neural_oracle_realworld.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('models/neural_oracle_v1.pkl', 'rb') as f:
    model2 = pickle.load(f)

print("✅ Both models loaded successfully")

# Analyze Model 1
print("\n" + "="*70)
print("MODEL 1: neural_oracle_realworld.pkl")
print("="*70)
print(f"Type: {type(model1)}")
print(f"Keys: {list(model1.keys())}")

if 'model' in model1:
    print(f"\nXGBoost Model Type: {type(model1['model'])}")
    print(f"Number of boosters: {model1['model'].num_boosted_rounds()}")
    
if 'action_encoder' in model1:
    print(f"\nAction Encoder ({len(model1['action_encoder'])} actions):")
    for idx, action in model1['action_encoder'].items():
        print(f"  {idx}: {action}")

if 'feature_names' in model1:
    print(f"\nFeature Names ({len(model1['feature_names'])} features):")
    for fname in model1['feature_names']:
        print(f"  - {fname}")

if 'metadata' in model1:
    print(f"\nMetadata:")
    for key, value in model1['metadata'].items():
        print(f"  {key}: {value}")
else:
    print("\n⚠️ No metadata found in model 1")

# Analyze Model 2
print("\n" + "="*70)
print("MODEL 2: neural_oracle_v1.pkl")
print("="*70)
print(f"Type: {type(model2)}")
print(f"Keys: {list(model2.keys())}")

if 'model' in model2:
    print(f"\nXGBoost Model Type: {type(model2['model'])}")
    print(f"Number of boosters: {model2['model'].num_boosted_rounds()}")

if 'metadata' in model2:
    print(f"\nMetadata:")
    for key, value in model2['metadata'].items():
        print(f"  {key}: {value}")
else:
    print("\n⚠️ No metadata found in model 2")

# Compare models
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Check if models are identical
if 'model' in model1 and 'model' in model2:
    model1_boosters = model1['model'].num_boosted_rounds()
    model2_boosters = model2['model'].num_boosted_rounds()
    
    print(f"Model 1 boosters: {model1_boosters}")
    print(f"Model 2 boosters: {model2_boosters}")
    
    if model1_boosters == model2_boosters:
        print("⚠️ Models have same number of boosters (likely identical)")
    else:
        print("✅ Models are different")

# Check action encoders
if 'action_encoder' in model1 and 'action_encoder' in model2:
    if model1['action_encoder'] == model2['action_encoder']:
        print("✅ Action encoders are identical")
    else:
        print("⚠️ Action encoders differ!")

# Check for training history file
print("\n" + "="*70)
print("TRAINING HISTORY CHECK")
print("="*70)

history_path = 'models/training_history.json'
if os.path.exists(history_path):
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"✅ Training history found: {len(history)} entries")
    for entry in history:
        print(f"\n  Date: {entry.get('date', 'unknown')}")
        print(f"  Datasets: {entry.get('datasets', 'unknown')}")
        print(f"  Model: {entry.get('model_path', 'unknown')}")
else:
    print("❌ No training history file found")
    print("⚠️ We don't know which datasets were used!")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
