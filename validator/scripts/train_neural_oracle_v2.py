"""
Improved Neural Oracle Training (Week 1)
- More data (40 datasets)
- Better features (20 vs 10)
- Ensemble (XGBoost + LightGBM)
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

def print_flush(msg):
    """Print and flush immediately for Colab"""
    print(msg)
    sys.stdout.flush()

print_flush("="*70)
print_flush("IMPROVED NEURAL ORACLE TRAINING (Week 1)")
print_flush("="*70)

# Load validated labels
labels_file = Path('validator/validated/validated_labels.json')

if not labels_file.exists():
    print_flush(f"❌ ERROR: File not found: {labels_file}")
    print_flush(f"Current directory: {Path.cwd()}")
    print_flush(f"Files in validator/: {list(Path('.').glob('*'))}")
    sys.exit(1)

print_flush(f"\n📂 Loading labels from: {labels_file}")

try:
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    print_flush(f"✅ Loaded {len(labels)} validated labels")
except Exception as e:
    print_flush(f"❌ ERROR loading labels: {e}")
    sys.exit(1)

if len(labels) == 0:
    print_flush("❌ ERROR: No labels found in file!")
    sys.exit(1)

# Extract features (20 features vs 10)
def extract_features_v2(label):
    """Extract 20 features (10 original + 10 new)"""
    features = label['features']
    
    return {
        # Original 10 features
        'null_percentage': features.get('null_pct', 0),
        'unique_ratio': features.get('unique_ratio', 0),
        'skewness': features.get('skewness', 0),
        'outlier_percentage': features.get('outlier_pct', 0),
        'entropy': features.get('entropy', 0),
        'pattern_complexity': features.get('pattern_complexity', 0),
        'multimodality_score': features.get('multimodality', 0),
        'cardinality_bucket': features.get('cardinality_bucket', 0),
        'detected_dtype': features.get('dtype_numeric', 0),
        'column_name_signal': features.get('name_signal', 0),
        
        # NEW: 10 additional features
        'kurtosis': features.get('kurtosis', 0),
        'coefficient_of_variation': features.get('cv', 0),
        'zero_ratio': features.get('zero_ratio', 0),
        'has_negative': features.get('has_negative', 0),
        'has_decimal': features.get('has_decimal', 0),
        'name_contains_id': 1 if 'id' in label.get('column', '').lower() else 0,
        'name_contains_date': 1 if any(x in label.get('column', '').lower() for x in ['date', 'time', 'year']) else 0,
        'name_contains_price': 1 if any(x in label.get('column', '').lower() for x in ['price', 'cost', 'amount']) else 0,
        'range_ratio': features.get('range_ratio', 0),
        'iqr_ratio': features.get('iqr_ratio', 0),
    }


# Prepare data
X = []
y = []

for label in labels:
    features_dict = extract_features_v2(label)
    X.append(list(features_dict.values()))
    y.append(label['action'])

X = np.array(X)
y = np.array(y)

print_flush(f"\n📊 Dataset shape: {X.shape}")
print_flush(f"   Features: {X.shape[1]} (vs 10 original)")
print_flush(f"   Examples: {X.shape[0]} (vs 149 original)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print_flush(f"\n✂️ Split:")
print_flush(f"   Train: {len(X_train)} examples")
print_flush(f"   Test: {len(X_test)} examples")

# Train ensemble (XGBoost + LightGBM)
print_flush(f"\n🎯 Training ensemble...")
print_flush("   This may take 30-60 seconds...")
sys.stdout.flush()

# Model 1: XGBoost
xgb = XGBClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbosity=0  # Suppress XGBoost output
)

# Model 2: LightGBM
lgb = LGBMClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=-1  # Suppress LightGBM output
)

# Ensemble (voting)
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb)],
    voting='soft'
)

print_flush("   Training XGBoost...")
sys.stdout.flush()
ensemble.fit(X_train, y_train)
print_flush("   ✅ Training complete!")

# Evaluate
print_flush("\n📈 Evaluating...")
train_pred = ensemble.predict(X_train)
test_pred = ensemble.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print_flush(f"\n{'='*70}")
print_flush("RESULTS")
print_flush(f"{'='*70}")
print_flush(f"Train accuracy: {train_acc:.1%}")
print_flush(f"Test accuracy:  {test_acc:.1%}")
print_flush(f"\n🎯 Target: ≥80% test accuracy")

if test_acc >= 0.80:
    print_flush(f"✅ SUCCESS! Achieved {test_acc:.1%}")
else:
    print_flush(f"⚠️  Close! Got {test_acc:.1%}, need ≥80%")

# Save model
output_dir = Path('models')
output_dir.mkdir(exist_ok=True)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = output_dir / f'neural_oracle_v2_improved_{timestamp}.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(ensemble, f)

print_flush(f"\n💾 Model saved: {model_path}")
print_flush(f"   Size: {model_path.stat().st_size / 1024:.0f} KB")

# Save training history
history = {
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'num_features': X.shape[1],
    'num_examples': X.shape[0],
    'model_type': 'ensemble_xgb_lgb',
    'timestamp': timestamp
}

history_path = output_dir / 'training_history_v2.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print_flush(f"📊 History saved: {history_path}")

print_flush(f"\n{'='*70}")
print_flush("COMPARISON")
print_flush(f"{'='*70}")
print_flush(f"Original model:")
print_flush(f"  - Features: 10")
print_flush(f"  - Examples: 149")
print_flush(f"  - Accuracy: 75.9%")
print_flush(f"\nImproved model:")
print_flush(f"  - Features: {X.shape[1]}")
print_flush(f"  - Examples: {X.shape[0]}")
print_flush(f"  - Accuracy: {test_acc:.1%}")
print_flush(f"\nImprovement: +{(test_acc - 0.759)*100:.1f}%")
print_flush(f"\n{'='*70}")
print_flush("✅ TRAINING COMPLETE!")
print_flush(f"{'='*70}")

